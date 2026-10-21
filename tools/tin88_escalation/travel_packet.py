"""Long Truebones artist-fox travel, after the three-seed short-clip screen."""

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import modal

image = modal.Image.from_id("im-BpHCKNBqK61ryuoR5HzJV9")
volume = modal.Volume.from_name("tin88-unimate-eval", environment_name="main")
app = modal.App("tin88-unimate-escalation")


@app.function(image=image, volumes={"/evaluation": volume}, gpu=["L4", "A10G", "L40S"],
              cpu=(2, 2), memory=(8192, 8192), timeout=300, retries=0, max_containers=1, scaledown_window=2)
def sample(seed: int):
    if seed not in (1729, 2718, 31415):
        raise ValueError("Outside the declared independent seeds")
    base = Path("/evaluation/unimate-20261021-escalation-b")
    for prerequisite in (base / "cpu-receipt.json", base / f"gpu-{seed}-receipt.json",
                         Path("/evaluation/unimate-20261021-escalation-c/cpu-receipt.json")):
        if json.loads(prerequisite.read_text())["status"] != "passed":
            raise ValueError(f"Unqualified prerequisite: {prerequisite}")
    root = Path("/evaluation/unimate-20261021-escalation-e/artist-truebones-forward") / str(seed)
    root.mkdir(parents=True, exist_ok=False)
    source = Path("/opt/unimate-evaluation/UniMate")
    cache = Path("/evaluation/unimate-20261021-compare-f/cpu/hf")
    command = ["/opt/conda/bin/python", "-m", "unimate.inference.sample", "--exp_dir", str(base / "unimate_truebones_f60_v2"),
        "--asset", "/evaluation/unimate-20261021-upstream-fox-a/asset/reviewed", "--cond_dataset_type", "truebones",
        "--prompt", *(["An object trots forward."] * 8), "--motion_expand", "--expand_overlap", "10",
        "--seed", str(seed), "--num_repetitions", "1", "--only_save_motion", "--output_dir", str(root)]
    receipt = {"status": "running", "startedUnix": time.time(), "command": command,
               "sourceCommit": "b78c7807df7cf4231d4e67f8d0ece29c5e11025b", "sourceBehavior": "unchanged upstream expansion"}
    try:
        import numpy as np
        import torch
        receipt["gpu"] = torch.cuda.get_device_name(0)
        env = {**os.environ, "PYTHONPATH": str(source) + ":/opt/unimate-evaluation/Motion", "HF_HOME": str(cache),
            "HF_HUB_CACHE": str(cache / "hub"), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "TORCH_FORCE_WEIGHTS_ONLY_LOAD": "1", "TOKENIZERS_PARALLELISM": "false",
            **{k: "2" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}}
        with (root / "sample.log").open("w") as log:
            subprocess.run(command, check=True, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
        paths = list((root / "motion_expand/motions").glob("*.npy"))
        if len(paths) != 1:
            raise ValueError("Expected one chain")
        values = np.load(paths[0], allow_pickle=False)
        if values.shape != (410, 47, 12) or not np.isfinite(values).all():
            raise ValueError("Incomplete or nonfinite chain")
        receipt.update(status="passed", motion={"path": str(paths[0]), "bytes": paths[0].stat().st_size,
            "sha256": hashlib.sha256(paths[0].read_bytes()).hexdigest()}, shape=list(values.shape))
    except Exception as error:
        receipt.update(status="failed", error=repr(error))
        raise
    finally:
        receipt["seconds"] = time.time() - receipt["startedUnix"]
        (root / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        volume.commit()
    return receipt


@app.local_entrypoint()
def main(seed: int = 1729):
    call = sample.spawn(seed)
    print(json.dumps({"callId": call.object_id, "seed": seed}), flush=True)
