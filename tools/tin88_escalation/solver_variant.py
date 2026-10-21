"""One discriminating numerical experiment: continuation Euler50 versus Euler200."""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] != "--upstream":
        raise ValueError("Use the Modal entry point or the explicit upstream subprocess mode")
    import functools
    import runpy
    from unimate.inference import motion_inbetweening
    motion_inbetweening.inbetween_sample_ode = functools.partial(motion_inbetweening.inbetween_sample_ode, num_steps=200)
    sys.argv = ["unimate.inference.sample", *sys.argv[2:]]
    runpy.run_module("unimate.inference.sample", run_name="__main__")
    raise SystemExit(0)

import modal

HERE = Path(__file__).resolve().parent
TOOLS = "/opt/escalation/solver_variant.py"
image = modal.Image.from_id("im-BpHCKNBqK61ryuoR5HzJV9").add_local_file(HERE / "solver_variant.py", TOOLS, copy=True)
volume = modal.Volume.from_name("tin88-unimate-eval", environment_name="main")
app = modal.App("tin88-unimate-escalation")


@app.function(image=image, volumes={"/evaluation": volume}, gpu=["L4", "A10G", "L40S"],
              cpu=(2, 2), memory=(8192, 8192), timeout=300, retries=0, max_containers=1, scaledown_window=2)
def compare():
    import numpy as np
    import torch
    output = Path("/evaluation/unimate-20261021-escalation-d/tiger-euler200")
    output.mkdir(parents=True, exist_ok=False)
    base = Path("/evaluation/unimate-20261021-compare-f/cpu")
    source = Path("/opt/unimate-evaluation/UniMate")
    command = ["/opt/conda/bin/python", TOOLS, "--upstream", "--exp_dir", str(base / "unimate_uniml3d_f60_v3"),
        "--asset", "/opt/escalation/tiger/asset", "--prompt", "An object walks in place.", "An object walks in place.",
        "--motion_expand", "--expand_overlap", "10", "--seed", "0", "--num_repetitions", "1",
        "--only_save_motion", "--output_dir", str(output)]
    receipt = {"status": "running", "startedUnix": time.time(), "gpu": torch.cuda.get_device_name(0),
        "command": command, "deviation": "Only inbetween_sample_ode num_steps=200; first free sample and every other source behavior unchanged",
        "scriptSha256": hashlib.sha256(Path(TOOLS).read_bytes()).hexdigest()}
    try:
        env = {**os.environ, "PYTHONPATH": str(source) + ":/opt/unimate-evaluation/Motion", "HF_HOME": str(base / "hf"),
            "HF_HUB_CACHE": str(base / "hf/hub"), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "TORCH_FORCE_WEIGHTS_ONLY_LOAD": "1", "TOKENIZERS_PARALLELISM": "false",
            **{k: "2" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}}
        with (output / "sample.log").open("w") as log:
            subprocess.run(command, check=True, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
        paths = list((output / "motion_expand/motions").glob("*.npy"))
        if len(paths) != 1:
            raise ValueError("Expected one chain")
        values = np.load(paths[0], allow_pickle=False)
        if values.shape != (110, 36, 12) or not np.isfinite(values).all():
            raise ValueError("Unexpected shape or nonfinite features")
        baseline_path = Path("/evaluation/unimate-20261021-escalation-c/tiger/0/motion_expand/motions/Tiger_rig-an_object_walks_in_place-rep_0-0.npy")
        baseline = np.load(baseline_path, allow_pickle=False)
        receipt.update(status="passed", motion=str(paths[0]), sha256=hashlib.sha256(paths[0].read_bytes()).hexdigest(),
            baselineSha256=hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
            firstWindowBaselineMaxAbsDelta=float(np.abs(values[:60] - baseline[:60]).max()),
            continuationFeatureRmsDelta=float(np.sqrt(np.mean((values[60:] - baseline[60:110]) ** 2))))
    except Exception as error:
        receipt.update(status="failed", error=repr(error))
        raise
    finally:
        receipt["seconds"] = time.time() - receipt["startedUnix"]
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        volume.commit()
    return receipt


@app.local_entrypoint()
def main():
    print(json.dumps(compare.remote()), flush=True)
