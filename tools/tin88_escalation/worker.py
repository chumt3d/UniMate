"""Execute and record the author-provided Tiger baseline on retained runtime."""

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[key] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

SOURCE = Path("/opt/unimate-evaluation/UniMate")
MOTION = Path("/opt/unimate-evaluation/Motion")
CACHE_ROOT = Path("/evaluation/unimate-20261021-compare-f/cpu")
MODEL = CACHE_ROOT / "unimate_uniml3d_f60_v3"
TIGER = Path("/opt/escalation/tiger")


def identity(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def prepare_environment():
    spec = json.loads((Path(__file__).with_name("runtime-inputs.json")).read_text())
    for root, expected in ((SOURCE, spec["sourceCommit"]), (MOTION, spec["motionSourceCommit"])):
        pin = json.loads((root / ".source-pin.json").read_text())
        if pin["commit"] != expected:
            raise ValueError(f"Unexpected source commit: {root}")
        for relative, digest in pin["files"].items():
            if identity(root / relative)["sha256"] != digest:
                raise ValueError(f"Unexpected source bytes: {root / relative}")
    cache = CACHE_ROOT / "hf/hub"
    if (cache / "models--google--flan-t5-base/refs/main").read_text().strip() != spec["encoderRevision"]:
        raise ValueError("Unexpected text encoder revision")
    for name in ("config", "stats", "checkpoint"):
        expected = spec["model"][name]
        actual = identity(CACHE_ROOT / expected["path"])
        if actual["sha256"] != expected["sha256"] or actual["bytes"] != expected["sizeBytes"]:
            raise ValueError(f"Unexpected model bytes: {name}")
    os.environ.update({"HF_HOME": str(CACHE_ROOT / "hf"), "HF_HUB_CACHE": str(cache),
                       "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "TORCH_FORCE_WEIGHTS_ONLY_LOAD": "1"})


def preflight(output):
    import numpy as np
    import torch
    from unimate.configs.schema import MainConfig
    from unimate.dataset.conditioning import create_sample_condition
    from unimate.inference.sample import InferenceArgs, _load_checkpoint, _prepare_cases, _resolve_stats_path
    from unimate.models.factory import create_model

    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)

    paths = [MODEL / "config.json", MODEL / "dataset_stats.npy",
             MODEL / "checkpoints/checkpoint_step_150000.pt", TIGER / "asset/cond.npy"]
    expected = json.loads((TIGER / "source-sha256.json").read_text())
    for relative, sha256 in expected.items():
        actual = identity(SOURCE / relative)["sha256"]
        if actual != sha256:
            raise ValueError(f"Runtime source mismatch: {relative}")
    config = MainConfig.from_json(str(MODEL / "config.json"))
    args = InferenceArgs(exp_dir=str(MODEL), asset=(str(TIGER / "asset"),),
                         prompt=("An object walks in place.",), seed=0, num_repetitions=8, cfg_scale=3)
    dataset, cases, _ = _prepare_cases(args, config, 3, _resolve_stats_path(str(MODEL)), torch.device("cpu"))
    _, conditioning = create_sample_condition(config, dataset,
        test_case_captions=[(ot, caption, encoding) for _, ot, caption, encoding, _ in cases])
    if conditioning["n_joints"].tolist() != [36]:
        raise ValueError("Author Tiger conditioning must have 36 joints")
    model = create_model(dataset_config=config.dataset, model_config=config.model)
    _load_checkpoint(model, str(MODEL / "checkpoints/checkpoint_step_150000.pt"), config)
    cond = np.load(TIGER / "asset/cond.npy", allow_pickle=True).item()
    stats = np.load(MODEL / "dataset_stats.npy", allow_pickle=True).item()
    info = {"inputs": [identity(p) for p in paths], "sourceSha256": expected,
            "config": json.loads((MODEL / "config.json").read_text()),
            "statsKeys": list(stats), "conditioning": {}, "strictModelLoad": True,
            "conditionShapes": {key: list(value.shape) for key, value in conditioning.items()
                                if torch.is_tensor(value)},
            "casePrompts": [case[2] for case in cases],
            "versions": {key: importlib.metadata.version(key) for key in
                         ("torch", "numpy", "scipy", "transformers", "torchdiffeq", "imageio-ffmpeg")}}
    for name, entry in cond.items():
        info["conditioning"][name] = {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                                      if hasattr(value, "shape") else str(value)[:200]
                                      for key, value in entry.items()}
    for path in (Path("/opt/upstream-reproduction"), Path("/opt/unimate-evaluation")):
        info[str(path)] = [str(p) for p in path.iterdir()]
    from data_process.utils.plotting import _write_video
    video = output / "mp4-backend-check.mp4"
    _write_video(str(video), np.zeros((4, 64, 64, 3), dtype=np.uint8), 30)
    info["renderBackend"] = identity(video)
    (output / "preflight.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps({"strictModelLoad": True, "statsKeys": list(stats), "torch": torch.__version__}))


def baseline(output):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this bounded baseline")
    preflight = json.loads((output / "preflight-receipt.json").read_text())
    if preflight["status"] != "passed":
        raise ValueError("CPU preflight did not pass")
    command = [sys.executable, "-m", "unimate.inference.sample", "--exp_dir", str(MODEL),
               "--asset", str(TIGER / "asset"), "--prompt", "An object walks in place.",
               "--num_repetitions", "8", "--seed", "0", "--cfg_scale", "3",
               "--output_dir", str(output / "tiger-baseline")]
    (output / "baseline-command.json").write_text(json.dumps(command, indent=2) + "\n")
    (output / "baseline-device.json").write_text(json.dumps({"gpu": torch.cuda.get_device_name(0)}, indent=2) + "\n")
    subprocess.run(command, cwd=SOURCE, check=True)
    import numpy as np
    motions = list((output / "tiger-baseline/motions").glob("*.npy"))
    if len(motions) != 8:
        raise ValueError(f"Expected eight completed motions, found {len(motions)}")
    comparison = []
    for path in sorted(motions):
        generated = np.load(path)
        author = np.load(TIGER / "samples/motions" / path.name)
        if generated.shape != author.shape or not np.isfinite(generated).all():
            raise ValueError(f"Invalid motion {path.name}")
        delta = generated - author
        comparison.append({"name": path.name, "shape": list(generated.shape),
                           "maxAbsFeatureDelta": float(np.abs(delta).max()),
                           "rmsFeatureDelta": float(np.sqrt(np.mean(delta ** 2)))})
    (output / "author-comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    print(json.dumps(comparison), flush=True)


if __name__ == "__main__":
    stage, directory = sys.argv[1:]
    functions = {"preflight": preflight, "baseline": baseline}
    prepare_environment()
    functions[stage](Path(directory))
