"""Fresh-process CPU proof at the production checkpoint/loader boundary.

No motion/text/renderer dependencies, downloaded models or external datasets.
Run with a stdlib-only coordinator so Torch workers never overlap::

    python3 tests/prove_exact_resume.py --python <cpu-env>/bin/python --out /tmp/proof
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def worker(args):
    import copy
    import hashlib
    import random
    from types import SimpleNamespace

    import accelerate
    import numpy as np
    import torch
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from torch.utils.data import DataLoader, Dataset

    from unimate.dataset.mixture.sampler import MixtureSampler
    from unimate.training.ema import EMAModel
    from unimate.training.exact_resume import (
        ExactDataLoader, capture_rng, preserve_rng, read_checkpoint,
        restore_checkpoint, save_checkpoint, input_identity, training_identity,
        runtime_identity, file_sha256,
    )
    from unimate.training.tracker import TrainingTracker

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    set_seed(917)

    class TinyData(Dataset):
        def __init__(self):
            names = [f"clip-{i}" for i in range(10)]
            self.motion_dataset = SimpleNamespace(
                train_name_list=names,
                train_motion_dict={n: {"dataset_type": "tiny", "object_type":
                                      "A" if i < 3 else "B"} for i, n in enumerate(names)})

        def __len__(self):
            return 10

        def __getitem__(self, index):
            # Exercise all three main-process augmentation RNG streams.
            x = torch.tensor([index / 10, random.random(), float(np.random.rand())])
            return {"id": index, "x": x + torch.rand(3), "y": torch.tensor([index / 20])}

    dataset = TinyData()
    sampler = MixtureSampler(dataset, seed=917) if args.mode == "balanced" else None
    original = DataLoader(dataset, sampler=sampler, shuffle=sampler is None,
                          batch_size=2, num_workers=0, drop_last=True)
    accelerator = Accelerator(cpu=True, gradient_accumulation_steps=2)
    loader = ExactDataLoader(original, accelerator.device, 917)
    model = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Dropout(0.3),
                                torch.nn.Tanh(), torch.nn.Linear(5, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, betas=(0.9, 0.999),
                                  weight_decay=0.01)
    model, optimizer = accelerator.prepare(model, optimizer)
    ema = EMAModel(model.parameters(), use_ema_warmup=True)
    tracker = TrainingTracker(max_epochs=2, steps_per_epoch=3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=6)
    identity = {"synthetic": "10-clips/2-batch/2-accum/2-epochs-v1", "mode": args.mode,
                "runtime": runtime_identity(),
                **{key:"synthetic-"+key for key in ("config","source","data","resolved_config",
                   "splits","stats","encoder_outputs","execution")}}
    args.out.mkdir(parents=True, exist_ok=True)
    if args.resume:
        restore_checkpoint(read_checkpoint(args.resume), accelerator, model, optimizer,
                           scheduler, ema, tracker, loader, identity)

    def rng_fingerprint():
        state = capture_rng()
        return {"python": hashlib.sha256(repr(state["python"]).encode()).hexdigest(),
                "numpy": hashlib.sha256(state["numpy"][1].tobytes()
                                         + repr(state["numpy"][2:]).encode()).hexdigest(),
                "torch": hashlib.sha256(state["torch"].numpy().tobytes()).hexdigest()}

    events = []
    while not tracker.training_finished():
        loader.set_epoch(tracker.current_epoch)
        for batch in loader:
            before = rng_fingerprint()
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    prediction = model(batch["x"])
                    loss = (prediction - batch["y"]).square().mean()
                accelerator.backward(loss)
                event = {"ids": batch["id"].tolist(), "epoch": tracker.current_epoch,
                         "rng": before, "loss": loss.item(),
                         "sync": accelerator.sync_gradients}
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    ema.step(model.parameters())
                    tracker.step()
                    event["step"] = tracker.current_step
                    event["lr"] = scheduler.get_last_lr()
                    event["weights"] = [p.detach().tolist() for p in model.parameters()]
                    event["ema"] = [p.tolist() for p in ema.shadow_params]
                    checkpoint_path = args.out / f"checkpoint_step_{tracker.current_step}.pt"
                    save_checkpoint(checkpoint_path, accelerator, model, optimizer,
                                    scheduler, ema, tracker, loader, identity)
                    # Real debug work can consume every RNG; it must not affect training.
                    saved = rng_fingerprint()
                    with preserve_rng():
                        random.random(); np.random.rand(); torch.rand(3)
                    assert saved == rng_fingerprint()
                events.append(event)
                if args.stop and tracker.current_step == args.stop:
                    break
            if args.stop and tracker.current_step == args.stop:
                break
        if args.stop and tracker.current_step == args.stop:
            break
        if not tracker.training_finished():
            tracker.next_epoch()
    def serializable(value):
        if torch.is_tensor(value):
            return value.tolist()
        if isinstance(value, dict):
            return {str(k): serializable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [serializable(v) for v in value]
        return value

    report = {"events": events, "final_rng": rng_fingerprint(),
              "optimizer": serializable(optimizer.state_dict()),
              "sampler_generator": hashlib.sha256(
                  loader.resume_sampler.state_dict()["generator"].numpy().tobytes()).hexdigest(),
              "tracker": vars(tracker), "scheduler": scheduler.state_dict(),
              "ema_step": ema.optimization_step, "ema_decay": ema.cur_decay_value,
              "next_batch": loader.resume_sampler.state_dict()["next_batch"],
              "runtime": identity["runtime"]}
    (args.out / "result.json").write_text(json.dumps(report, indent=2))

    if args.resume or args.stop:
        return

    # Rejections exercise the same production readers/restorer, without mocks.
    saved_path = args.resume or str(args.out / "latest.json")
    saved = read_checkpoint(saved_path)
    rejections = []

    def rejected(name, operation):
        try:
            operation()
        except (ValueError, FileExistsError):
            rejections.append(name)
        else:
            raise AssertionError(f"{name} was silently accepted")

    def restore(candidate, bound_identity=identity):
        # Compare the saved horizon against a fresh run's progress budget.
        fresh = TrainingTracker(max_epochs=2, steps_per_epoch=3)
        restore_checkpoint(candidate, accelerator, model, optimizer, scheduler,
                           ema, fresh, loader, bound_identity)

    for key in identity:
        changed = {**identity, key: "different"}
        rejected(f"identity-{key}", lambda changed=changed: restore(saved, changed))
    for field in ("ema_state_dict", "lr_scheduler_state_dict", "optimizer_state_dict",
                  "tracker", "rng", "sampler"):
        candidate = copy.deepcopy(saved)
        candidate[field].pop(next(iter(candidate[field])))
        rejected(f"incomplete-{field}", lambda candidate=candidate: restore(candidate))
    candidate = copy.deepcopy(saved)
    candidate["scaler_state_dict"] = {"scale": 1}
    rejected("scaler-presence", lambda: restore(candidate))
    candidate = copy.deepcopy(saved)
    candidate["sampler"]["next_batch"] = [-1, -1]
    rejected("next-batch-mismatch", lambda: restore(candidate))
    # EMA's visualization swap must release its temporary model-sized clones.
    ema.store(model.parameters())
    ema.copy_to(model.parameters())
    ema.restore(model.parameters())
    assert not hasattr(ema, "collected_params")
    rejections.append("ema-visualization-storage-released")
    # Even zero-valued gradients mean the optimizer boundary was not cleared.
    first_parameter = next(model.parameters())
    first_parameter.grad = torch.zeros_like(first_parameter)
    rejected("uncleared-gradients", lambda: save_checkpoint(
        args.out / "uncleared.pt", accelerator, model, optimizer, scheduler,
        ema, tracker, loader, identity))
    optimizer.zero_grad(set_to_none=True)
    rejected("worker-processes", lambda: ExactDataLoader(
        DataLoader(dataset, batch_size=2, num_workers=1), accelerator.device, 917))
    saved_file = Path(saved_path)
    if saved_file.suffix == ".json":
        saved_file = saved_file.parent / json.loads(saved_file.read_text())["checkpoint"]
    rejected("checkpoint-overwrite", lambda: save_checkpoint(
        saved_file, accelerator, model, optimizer, scheduler, ema, tracker, loader, identity))
    incomplete_path = args.out / "incomplete.pt"
    incomplete = dict(saved)
    incomplete.pop("rng")
    torch.save(incomplete, incomplete_path)
    rejected("incomplete-file", lambda: read_checkpoint(incomplete_path))
    incomplete_path.unlink()
    corrupt_receipt = args.out / "corrupt.json"
    corrupt_receipt.write_text(json.dumps({"format": saved["format"],
        "checkpoint": "result.json", "bytes": 0, "sha256": "wrong"}))
    rejected("receipt-corruption", lambda: read_checkpoint(corrupt_receipt))
    corrupt_receipt.unlink()
    # Bind the actual identity producer to consumed file/split/encoder bytes,
    # including motion directories linked from a feature root.
    import tempfile
    from unimate.configs.schema import MainConfig
    with tempfile.TemporaryDirectory(dir=args.out) as temporary:
        area = Path(temporary)
        features, motions = area / "features", area / "motions"
        features.mkdir(); motions.mkdir()
        (features / "motions").symlink_to(motions, target_is_directory=True)
        clip = motions / "clip.npz"
        clip.write_bytes(b"original cleared fixture")
        config = MainConfig.from_json(ROOT / "configs/truebones_60frames_graph_adaln.json")
        config.truebones.path = str(features)
        original = input_identity(config)
        clip.write_bytes(b"changed cleared fixture")
        changed = input_identity(config)
        assert changed["data"] != original["data"]
        assert changed["source"] == original["source"]
        config.training.learning_rate *= 2
        assert input_identity(config)["config"] != original["config"]
        stats = area / "stats.npy"
        np.save(stats, np.array([1.0]))
        md = SimpleNamespace(train_name_list=["clip"], eval_name_list=[],
             _joint_name_emb={"root": np.array([1.0])}, _caption_tokens={},
             _caption_emb={}, _extra_caption_tokens={}, _extra_caption_emb={})
        first = training_identity(original, config, SimpleNamespace(motion_dataset=md), stats)
        md.eval_name_list = ["held"]
        second = training_identity(original, config, SimpleNamespace(motion_dataset=md), stats)
        assert first["splits"] != second["splits"]
        md._joint_name_emb["root"][0] = 2.0
        third = training_identity(original, config, SimpleNamespace(motion_dataset=md), stats)
        assert second["encoder_outputs"] != third["encoder_outputs"]
        np.save(stats, np.array([2.0]))
        fourth = training_identity(original, config, SimpleNamespace(motion_dataset=md), stats)
        assert third["stats"] != fourth["stats"]
    rejections.append("actual-config-data-split-stats-encoder-binding")

    # Actual production restore/save with real CPU model/Adam/EMA/sampler/RNG.
    # ONLY the unavailable CUDA generator I/O/device-count API is simulated;
    # this proves admission and byte-state transfer, not cross-GPU arithmetic.
    from unittest.mock import patch
    parent = read_checkpoint(args.out / 'checkpoint_step_1.pt')
    parent['identity'] = copy.deepcopy(identity)
    parent['identity']['runtime']['cuda_devices'] = ['NVIDIA L40S']
    parent['rng']['cuda'] = [torch.arange(16, dtype=torch.uint8)]
    parent_path = args.out / 'simulated-cuda-parent.pt'
    torch.save(parent, parent_path)
    parent, parent_sha = read_checkpoint(parent_path, return_sha256=True)
    target = copy.deepcopy(parent['identity'])
    target['runtime']['cuda_devices'] = ['NVIDIA H100 80GB HBM3']
    admission = {'owner':'/root', 'parent_checkpoint_sha256':parent_sha,
        'parent_step':parent['step'], 'from_cuda_devices':['NVIDIA L40S'],
        'to_cuda_devices':['NVIDIA H100 80GB HBM3'], 'operation_id':'productive-h100',
        'runtime_sha256':'b'*64, 'acknowledged_numeric_divergence':True}
    context = {'parent_checkpoint_sha256':parent_sha, 'operation_id':'productive-h100',
               'runtime_sha256':'b'*64}
    def portable_restore(bound=target, allowed=admission, candidate=parent, **changes):
        fresh = TrainingTracker(max_epochs=2, steps_per_epoch=3)
        returned = restore_checkpoint(candidate, accelerator, model, optimizer, scheduler,
            ema, fresh, loader, bound, transition=allowed, **dict(context, **changes))
        return fresh, returned

    rejected('different-GPU-without-admission',lambda:portable_restore(allowed=None))
    wrong = {'owner':'other', 'parent_checkpoint_sha256':'a'*64, 'parent_step':2,
        'from_cuda_devices':['NVIDIA L4'], 'to_cuda_devices':['NVIDIA L4'],
        'operation_id':'wrong-operation', 'runtime_sha256':'c'*64,
        'acknowledged_numeric_divergence':False}
    for key,value in wrong.items():
        rejected('transition-'+key,lambda key=key,value=value:portable_restore(allowed=dict(admission,**{key:value})))
    for key in admission:
        changed=dict(admission);del changed[key]
        rejected('transition-missing-'+key,lambda changed=changed:portable_restore(allowed=changed))
    rejected('transition-extra-key',lambda:portable_restore(allowed=dict(admission,extra=True)))
    rejected('transition-context-runtime',lambda:portable_restore(runtime_sha256='c'*64))
    rejected('transition-context-operation',lambda:portable_restore(operation_id='wrong'))
    rejected('transition-context-parent',lambda:portable_restore(parent_checkpoint_sha256='a'*64))
    for key in target:
        if key == 'runtime': continue
        changed=copy.deepcopy(target);changed[key]='changed'
        rejected('transition-identity-'+key,lambda changed=changed:portable_restore(bound=changed))
    for key in target['runtime']:
        if key == 'cuda_devices': continue
        changed=copy.deepcopy(target);changed['runtime'][key]='changed'
        rejected('transition-runtime-'+key,lambda changed=changed:portable_restore(bound=changed))
    for level in (None,'runtime'):
        for kind in ('missing','extra'):
            changed=copy.deepcopy(target);node=changed if level is None else changed[level]
            if kind=='missing':del node[next(iter(node))]
            else:node['extra']='forbidden'
            rejected('transition-schema-'+str(level)+'-'+kind,lambda changed=changed:portable_restore(bound=changed))
    changed=copy.deepcopy(target);changed['runtime']['cuda_devices']*=2
    rejected('transition-two-target-devices',lambda:portable_restore(bound=changed))
    changed=copy.deepcopy(parent);changed['rng']['cuda']*=2
    rejected('transition-two-saved-generators',lambda:portable_restore(candidate=changed))
    rejected('transition-no-actual-CUDA-device',lambda:portable_restore())

    def equal_state(a,b):
        if torch.is_tensor(a):return torch.equal(a,b)
        if isinstance(a,np.ndarray):return np.array_equal(a,b)
        if isinstance(a,dict):return set(a)==set(b) and all(equal_state(a[k],b[k]) for k in a)
        if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal_state(x,y) for x,y in zip(a,b))
        return a==b
    cuda_bytes=[torch.zeros_like(parent['rng']['cuda'][0])]
    def set_cuda(values):cuda_bytes[:]=[value.clone() for value in values]
    with patch.object(torch.cuda,'device_count',return_value=1),\
         patch.object(torch.cuda,'is_available',return_value=True),\
         patch.object(torch.cuda,'get_rng_state_all',side_effect=lambda:[value.clone() for value in cuda_bytes]),\
         patch.object(torch.cuda,'set_rng_state_all',side_effect=set_cuda):
        for parameter in model.parameters():parameter.data.fill_(123)
        random.random();np.random.rand();torch.rand(3)
        fresh,returned=portable_restore()
        assert returned==admission
        assert equal_state(accelerator.get_state_dict(model),parent['model_state_dict'])
        assert equal_state(optimizer.state_dict(),parent['optimizer_state_dict'])
        assert equal_state(scheduler.state_dict(),parent['lr_scheduler_state_dict'])
        assert equal_state(ema.state_dict(),parent['ema_state_dict'])
        assert equal_state(vars(fresh),parent['tracker'])
        assert equal_state(loader.resume_sampler.state_dict(),parent['sampler'])
        assert accelerator.step==parent['accelerator_step']
        assert equal_state(capture_rng(),parent['rng'])
        child_path=args.out/'simulated-cuda-child.pt'
        save_checkpoint(child_path,accelerator,model,optimizer,scheduler,ema,fresh,loader,target)
        child=read_checkpoint(child_path)
        assert child['identity']==target
        assert all(equal_state(child[key],parent[key]) for key in parent if key!='identity')
        assert file_sha256(parent_path)==parent_sha
    (args.out/'device-transition-proof.json').write_text(json.dumps({
        'actual_restore_save_boundary':True,'CUDA_generator_IO_simulated':True,
        'real_GPU_execution':False,'all_CPU_state_sampler_nextbatch_and_all_RNG_equal':True,
        'parent_immutable':True,'child_actual_target_identity':True,
        'rejected_cases':len(rejections)},indent=2))
    parent_path.unlink();child_path.unlink()
    (args.out / "rejections.json").write_text(json.dumps(rejections, indent=2))


def coordinator(args):
    # The coordinator deliberately imports no numerical packages.
    environment = {**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
                   "MKL_NUM_THREADS": "2", "NUMEXPR_NUM_THREADS": "2",
                   "RAYON_NUM_THREADS": "2", "PYTHONDONTWRITEBYTECODE": "1"}
    results = []
    for mode in ("shuffle", "balanced"):
        def run(name, *extra):
            destination = args.out / mode / name
            command = [args.python, str(Path(__file__).resolve()), "--worker", "--mode", mode,
                       "--out", str(destination), *extra]
            result = subprocess.run(command, env=environment, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "process.log").write_text(result.stdout)
            if result.returncode:
                raise RuntimeError(f"{name} failed ({result.returncode}): {result.stdout}")
            return json.loads((destination / "result.json").read_text()), destination
        baseline, _ = run("uninterrupted")
        for stop in (1, 3, 6):
            prefix, prefix_dir = run(f"interrupt-{stop}", "--stop", str(stop))
            resumed, _ = run(f"resume-{stop}", "--resume", str(prefix_dir / "latest.json"))
            combined = {**resumed, "events": prefix["events"] + resumed["events"]}
            if combined != baseline:
                raise AssertionError(f"{mode} resume at {stop} differs from uninterrupted run")
            results.append({"mode": mode, "stop": stop, "equal": True,
                            "microbatches": len(combined["events"]),
                            "optimizer_steps": combined["ema_step"],
                            "runtime": baseline["runtime"]})
    (args.out / "summary.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--mode", choices=("shuffle", "balanced"), default="shuffle")
    parser.add_argument("--resume")
    parser.add_argument("--stop", type=int)
    arguments = parser.parse_args()
    arguments.out = arguments.out.resolve()
    worker(arguments) if arguments.worker else coordinator(arguments)
