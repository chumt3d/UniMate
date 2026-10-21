"""Bounded research comparison of released checkpoints and exact Fox captions."""

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
RUN = Path("/evaluation/unimate-20261021-escalation-b")
SOURCE = Path("/opt/unimate-evaluation/UniMate")
T5_ROOT = Path("/evaluation/unimate-20261021-compare-f/cpu")
FOX_ROOT = Path("/evaluation/unimate-20261021-upstream-fox-a")
MODEL_FOLDER = "unimate_truebones_f60_v2"
REVISION = "174fc755ec248d0c98f27759b7a95648e7f05905"
PROMPTS = ["An object walks in place with its tail swaying.", "An object bounds in place.",
           "An object trots forward.", "An object trots in place.", "An object walks in place."]
SEEDS = [1729, 2718, 31415]
FILES = {
    "config.json": (3171, "c253200a537638aed92f61741857c7b79e60bb4d1ec7c7fcc5b5c9af79212c55"),
    "dataset_stats.npy": (880, "b03a1a03869dda47db5064bb2ffbbbe0d68e33c3e13b0c23761dbd6c1f3d105b"),
    "checkpoints/checkpoint_step_80000.pt": (765119993, "2228eecf8d59f156a3d4509b8faaa312709a0b965415e342bb19fdd171fb8b51"),
}
TOOLS = "/opt/escalation/compare_checkpoints.py"
image = modal.Image.from_id("im-yvwuQXEyvg2yH1dd46r6k0").add_local_file(
    HERE / "compare_checkpoints.py", TOOLS, copy=True)
volume = modal.Volume.from_name("tin88-unimate-eval", environment_name="main")
app = modal.App("tin88-unimate-escalation")


def file_identity(path):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def environment():
    return {**os.environ,
        **{key: "2" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "RAYON_NUM_THREADS")},
        "PYTHONPATH": str(SOURCE) + ":/opt/unimate-evaluation/Motion",
        "HF_HOME": str(T5_ROOT / "hf"), "HF_HUB_CACHE": str(T5_ROOT / "hf/hub"),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
        "TORCH_FORCE_WEIGHTS_ONLY_LOAD": "1"}


def verify_model():
    for relative, (size, digest) in FILES.items():
        actual = file_identity(RUN / MODEL_FOLDER / relative)
        if actual["bytes"] != size or actual["sha256"] != digest:
            raise ValueError(f"Unexpected model bytes: {relative}")


def asset_args(asset):
    if asset == "native_fox":
        return ["--asset", "Fox", "--cond_path", str(FOX_ROOT / "truebones/cond.npy"),
                "--cond_dataset_type", "truebones"]
    if asset == "artist_fox":
        return ["--asset", str(FOX_ROOT / "asset/reviewed"), "--cond_dataset_type", "truebones"]
    raise ValueError("Unknown asset")


def cpu_work():
    import sys
    import urllib.request

    for relative, (size, digest) in FILES.items():
        target = RUN / MODEL_FOLDER / relative
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix(target.suffix + ".partial")
            url = f"https://huggingface.co/Linzhan/UniMate/resolve/{REVISION}/{MODEL_FOLDER}/{relative}"
            with urllib.request.urlopen(url, timeout=90) as response, partial.open("wb") as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            actual = file_identity(partial)
            if actual["bytes"] != size or actual["sha256"] != digest:
                raise ValueError(f"Download identity mismatch: {relative}")
            partial.rename(target)
    verify_model()
    os.environ.update(environment())
    sys.path[:0] = [str(SOURCE), "/opt/unimate-evaluation/Motion"]
    import torch
    from unimate.configs.schema import MainConfig
    from unimate.dataset.conditioning import create_sample_condition
    from unimate.inference.sample import InferenceArgs, _prepare_cases, _resolve_stats_path, _load_checkpoint
    from unimate.models.factory import create_model

    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    exp = RUN / MODEL_FOLDER
    config = MainConfig.from_json(str(exp / "config.json"))
    model = create_model(config.dataset, config.model)
    _load_checkpoint(model, str(exp / "checkpoints/checkpoint_step_80000.pt"), config)
    records = []
    for label in ("native_fox", "artist_fox"):
        reference = "Fox" if label == "native_fox" else str(FOX_ROOT / "asset/reviewed")
        args = InferenceArgs(exp_dir=str(exp), asset=(reference,),
                             cond_path=(str(FOX_ROOT / "truebones/cond.npy"),) if label == "native_fox" else (),
                             cond_dataset_type="truebones", prompt=tuple(PROMPTS), seed=1729)
        dataset, cases, assets = _prepare_cases(args, config, 3, _resolve_stats_path(str(exp)), torch.device("cpu"))
        _, cond = create_sample_condition(config, dataset,
            test_case_captions=[(obj, caption, encoding) for _, obj, caption, encoding, _ in cases])
        expected_joints = 39 if label == "native_fox" else 47
        if cond["n_joints"].tolist() != [expected_joints] * len(PROMPTS):
            raise ValueError(f"Unexpected conditioned joint counts: {label}")
        records.append({"asset": label, "jointCounts": cond["n_joints"].tolist(),
                        "shapes": {key: list(value.shape) for key, value in cond.items() if torch.is_tensor(value)},
                        "conditionFiles": [file_identity(Path(asset.cond_path)) for asset in assets.values()]})
    return {"checkpointAndEma": "passed", "model": [file_identity(exp / p) for p in FILES],
            "conditions": records, "prompts": PROMPTS, "seeds": SEEDS, "repetitions": 1,
            "note": "First two captions are exact native Fox training captions at UniML3D faaa817. Native conditioning is retained current data, not asserted byte-identical to old training condition."}


def gpu_work(seed):
    if seed not in SEEDS:
        raise ValueError("Seed outside the declared comparison")
    verify_model()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    records = []
    for label, exp in (("truebones80k", RUN / MODEL_FOLDER),
                       ("uniml3d_v3", T5_ROOT / "unimate_uniml3d_f60_v3")):
        for asset in ("native_fox", "artist_fox"):
            output = RUN / "samples" / label / asset / str(seed)
            if output.exists():
                raise FileExistsError(f"Refusing to overwrite samples: {output}")
            command = ["/opt/conda/bin/python", "-m", "unimate.inference.sample", "--exp_dir", str(exp),
                       *asset_args(asset), "--prompt", *PROMPTS, "--seed", str(seed), "--num_repetitions", "1",
                       "--only_save_motion", "--output_dir", str(output)]
            started = time.monotonic()
            log_path = RUN / f"{label}-{asset}-{seed}.log"
            with log_path.open("w") as log:
                subprocess.run(command, cwd=SOURCE, env=environment(), stdout=log, stderr=subprocess.STDOUT, check=True)
            import numpy as np
            paths = sorted((output / "motions").glob("*.npy"))
            if len(paths) != len(PROMPTS):
                raise ValueError("Incomplete comparison output")
            for path in paths:
                values = np.load(path, allow_pickle=False)
                expected = (60, 39 if asset == "native_fox" else 47, 12)
                if values.shape != expected or not np.isfinite(values).all():
                    raise ValueError(f"Invalid motion: {path}")
            records.append({"model": label, "asset": asset, "seed": seed, "command": command,
                            "seconds": time.monotonic() - started, "files": [file_identity(path) for path in paths]})
            volume.commit()
    return {"gpu": torch.cuda.get_device_name(0), "results": records}


def execute(stage, image_id, seed=1729):
    RUN.mkdir(parents=True, exist_ok=True)
    key = stage if stage == "cpu" else f"gpu-{seed}"
    receipt_path = RUN / f"{key}-receipt.json"
    if receipt_path.exists():
        raise FileExistsError("Preserve earlier run; choose a separately reviewed retry")
    if stage == "gpu":
        prerequisite = json.loads((RUN / "cpu-receipt.json").read_text())
        if prerequisite["status"] != "passed" or prerequisite["image"] != image_id:
            raise ValueError("GPU requires the same CPU-qualified image")
    receipt = {"stage": key, "image": image_id, "startedUnix": time.time(), "status": "running"}
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    volume.commit()
    try:
        receipt["result"] = cpu_work() if stage == "cpu" else gpu_work(seed)
        receipt["status"] = "passed"
    except Exception as error:
        receipt.update(status="failed", error=repr(error))
        raise
    finally:
        receipt["seconds"] = time.time() - receipt["startedUnix"]
        receipt["script"] = file_identity(Path(TOOLS))
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        volume.commit()
    return {"receipt": str(receipt_path), "status": receipt["status"], "seconds": receipt["seconds"]}


@app.function(image=image, volumes={"/evaluation": volume}, cpu=(2, 2), memory=(8192, 8192),
              timeout=600, retries=0, max_containers=1, scaledown_window=2)
def prepare(image_id: str):
    return execute("cpu", image_id)


@app.function(image=image, volumes={"/evaluation": volume}, gpu=["L4", "A10G", "L40S"],
              cpu=(2, 2), memory=(8192, 8192), timeout=600, retries=0,
              max_containers=1, scaledown_window=2)
def compare(image_id: str, seed: int):
    return execute("gpu", image_id, seed)


@app.local_entrypoint()
def main(stage: str = "cpu", seed: int = 1729):
    if stage == "cpu":
        print(json.dumps(prepare.remote(image.object_id)), flush=True)
    elif stage == "gpu":
        print(json.dumps(compare.remote(image.object_id, seed)), flush=True)
    else:
        raise ValueError("Unknown stage")
