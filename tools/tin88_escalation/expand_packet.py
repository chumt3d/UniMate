"""Unmodified upstream expansion, eight windows per independently seeded chain."""

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
SOURCE = Path("/opt/unimate-evaluation/UniMate")
RUN = Path("/evaluation/unimate-20261021-escalation-c")
BASE = Path("/evaluation/unimate-20261021-compare-f/cpu")
FOX = Path("/evaluation/unimate-20261021-upstream-fox-a")
MODELS = {"v3": BASE / "unimate_uniml3d_f60_v3",
          "truebones": Path("/evaluation/unimate-20261021-escalation-b/unimate_truebones_f60_v2")}
TOOLS = "/opt/escalation/expand_packet.py"
CASES = {
    "tiger": ("v3", "/opt/escalation/tiger/asset", "An object walks in place."),
    "artist-v3-walk": ("v3", str(FOX / "asset/reviewed"), "An object walks in place with its tail swaying."),
    "artist-v3-trot": ("v3", str(FOX / "asset/reviewed"), "An object trots in place."),
    "artist-truebones-walk": ("truebones", str(FOX / "asset/reviewed"), "An object walks in place with its tail swaying."),
    "artist-truebones-trot": ("truebones", str(FOX / "asset/reviewed"), "An object trots in place."),
    "native-truebones-walk": ("truebones", "Fox", "An object walks in place with its tail swaying."),
}
image = modal.Image.from_id("im-tJ0nhonL6Rq4IB7eAVbV3K").add_local_file(HERE / "expand_packet.py", TOOLS, copy=True)
volume = modal.Volume.from_name("tin88-unimate-eval", environment_name="main")
app = modal.App("tin88-unimate-escalation")


def identity(path):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def environment():
    return {**os.environ, "PYTHONPATH": str(SOURCE) + ":/opt/unimate-evaluation/Motion",
            "HF_HOME": str(BASE / "hf"), "HF_HUB_CACHE": str(BASE / "hf/hub"),
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TORCH_FORCE_WEIGHTS_ONLY_LOAD": "1",
            "TOKENIZERS_PARALLELISM": "false",
            **{key: "2" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                   "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "RAYON_NUM_THREADS")}}


def prepare_case(key):
    model, asset, prompt = CASES[key]
    extra = ["--asset", asset]
    if key != "tiger":
        extra += ["--cond_dataset_type", "truebones"]
    if asset == "Fox":
        extra += ["--cond_path", str(FOX / "truebones/cond.npy")]
    return model, asset, prompt, extra


@app.function(image=image, volumes={"/evaluation": volume}, cpu=(2, 2), memory=(4096, 4096),
              timeout=300, retries=0, max_containers=1, scaledown_window=2)
def prepare(image_id: str):
    import sys
    os.environ.update(environment())
    sys.path[:0] = [str(SOURCE), "/opt/unimate-evaluation/Motion"]
    import torch
    from unimate.configs.schema import MainConfig
    from unimate.inference.sample import InferenceArgs, _prepare_cases, _resolve_stats_path

    output = RUN / "cpu-receipt.json"
    RUN.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("Preserve the prior expansion preparation")
    receipt = {"image": image_id, "status": "running", "startedUnix": time.time(), "cases": []}
    try:
        for key in CASES:
            model, asset, prompt, _ = prepare_case(key)
            exp = MODELS[model]
            config = MainConfig.from_json(str(exp / "config.json"))
            args = InferenceArgs(exp_dir=str(exp), asset=(asset,), prompt=(prompt,) * 8,
                                 motion_expand=True, expand_overlap=10, seed=0 if key == "tiger" else 1729,
                                 cond_dataset_type=None if key == "tiger" else "truebones",
                                 cond_path=(str(FOX / "truebones/cond.npy"),) if asset == "Fox" else ())
            _, cases, assets = _prepare_cases(args, config, 3, _resolve_stats_path(str(exp)), torch.device("cpu"))
            if len(cases) != 1 or cases[0][2] != [prompt] * 8 or len(cases[0][3]) != 8:
                raise ValueError(f"Unexpected expansion case encoding: {key}")
            receipt["cases"].append({"id": key, "prompt": prompt, "windowCount": 8, "overlap": 10,
                "expectedFrames": 410, "config": identity(exp / "config.json"),
                "conditioning": [identity(Path(value.cond_path)) for value in assets.values()]})
        receipt["status"] = "passed"
    except Exception as error:
        receipt.update(status="failed", error=repr(error))
        raise
    finally:
        receipt["seconds"] = time.time() - receipt["startedUnix"]
        output.write_text(json.dumps(receipt, indent=2) + "\n")
        volume.commit()
    return {"status": receipt["status"], "seconds": receipt["seconds"], "receipt": str(output)}


@app.function(image=image, volumes={"/evaluation": volume}, gpu=["L4", "A10G", "L40S"],
              cpu=(2, 2), memory=(8192, 8192), timeout=600, retries=0, max_containers=1, scaledown_window=2)
def expand(key: str, seed: int, image_id: str):
    if key not in CASES or seed not in (0, 1729, 2718, 31415) or (key == "tiger") != (seed == 0):
        raise ValueError("Outside the declared expansion cases")
    prerequisite = json.loads((RUN / "cpu-receipt.json").read_text())
    if prerequisite["image"] != image_id or prerequisite["status"] != "passed":
        raise ValueError("Expansion requires the CPU-qualified image")
    output = RUN / key / str(seed)
    output.mkdir(parents=True, exist_ok=False)
    model, _, prompt, extra = prepare_case(key)
    command = ["/opt/conda/bin/python", "-m", "unimate.inference.sample", "--exp_dir", str(MODELS[model]),
               *extra, "--prompt", *([prompt] * 8), "--motion_expand", "--expand_overlap", "10",
               "--seed", str(seed), "--num_repetitions", "1", "--only_save_motion", "--output_dir", str(output)]
    receipt = {"case": key, "seed": seed, "image": image_id, "status": "running",
               "command": command, "startedUnix": time.time(), "script": identity(Path(TOOLS))}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    volume.commit()
    try:
        import numpy as np
        import torch
        receipt["gpu"] = torch.cuda.get_device_name(0)
        with (output / "sample.log").open("w") as log:
            subprocess.run(command, check=True, cwd=SOURCE, env=environment(), stdout=log, stderr=subprocess.STDOUT)
        paths = list((output / "motion_expand/motions").glob("*.npy"))
        if len(paths) != 1:
            raise ValueError("Expected one completed expansion chain")
        values = np.load(paths[0], allow_pickle=False)
        joints = 36 if key == "tiger" else 39 if key.startswith("native") else 47
        if values.shape != (410, joints, 12) or not np.isfinite(values).all():
            raise ValueError(f"Invalid expanded motion shape or values: {values.shape}")
        receipt.update(status="passed", motion=identity(paths[0]), shape=list(values.shape),
                       seamFrames=[60 + 50 * i for i in range(7)])
        if key == "tiger":
            author = np.load("/opt/escalation/tiger/samples/motions/Tiger_rig-an_object_walks_in_place-rep_0-0.npy")
            receipt["firstWindowAuthorMaxAbsDelta"] = float(np.abs(values[:60] - author).max())
    except Exception as error:
        receipt.update(status="failed", error=repr(error))
        raise
    finally:
        receipt["seconds"] = time.time() - receipt["startedUnix"]
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        volume.commit()
    return receipt


@app.local_entrypoint()
def main(stage: str = "cpu", case: str = "tiger", seed: int = 0):
    if stage == "cpu":
        print(json.dumps(prepare.remote(image.object_id)), flush=True)
    elif stage == "gpu":
        print(json.dumps(expand.remote(case, seed, image.object_id)), flush=True)
    else:
        raise ValueError("Unknown stage")
