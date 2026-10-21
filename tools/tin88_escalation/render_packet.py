"""Bounded CPU rendering of already diagnosed, unchanged joint trajectories."""

import gc
import hashlib
import json
import os
import time
from pathlib import Path

import modal

image = modal.Image.from_id("im-zbZcs3EcWvNACKYNyL0x6D")
volume = modal.Volume.from_name("tin88-unimate-eval", environment_name="main")
app = modal.App("tin88-unimate-escalation")


@app.function(image=image, volumes={"/evaluation": volume}, cpu=(2, 2), memory=(4096, 4096),
              timeout=600, retries=0, max_containers=1, scaledown_window=2)
def render(stage: str, group: int, sample: int):
    if "/" in stage or ".." in stage or not stage.startswith(("expand-", "compare-", "reference")):
        raise ValueError("Expected a retained analysis stage name")
    if group not in range(10) or sample not in range(9):
        raise ValueError("Outside bounded report indices")
    import sys
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "RAYON_NUM_THREADS"):
        os.environ[key] = "2"
    sys.path[:0] = ["/opt/unimate-evaluation/UniMate", "/opt/unimate-evaluation/Motion"]
    import numpy as np
    from PIL import Image
    from data_process.utils.plotting import render_skeleton_motion, _write_video

    root = Path("/evaluation/unimate-20261021-escalation-analysis") / stage
    if json.loads((root / "receipt.json").read_text())["status"] != "passed":
        raise ValueError("Analyze the actual motion before rendering")
    report = json.loads((root / f"{group:02d}/report.json").read_text())
    result = report["samples"][sample]
    arrays = np.load(result["raw_arrays"]["path"], allow_pickle=False)
    positions, parents = arrays["fk_positions"], arrays["parents"]
    if not 2 <= len(positions) <= 410 or not np.isfinite(positions).all():
        raise ValueError("Invalid or oversized render input")
    output = Path("/evaluation/unimate-20261021-escalation-renders") / stage / f"{group:02d}-{sample:02d}"
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"status": "running", "startedUnix": time.time(), "sample": result["input"],
               "rawArrays": result["raw_arrays"], "frames": len(positions), "fps": result["fps"],
               "coordinates": "Unchanged exact-source FK positions; no grounding, root replacement, cleanup, or skin",
               "views": []}
    try:
        tiles = []
        selected = np.linspace(0, len(positions) - 1, 8, dtype=int)
        for label, elev, azim in [("side", 0, 0), ("top", 90, 0), ("three-quarter", 20, -60)]:
            frames = render_skeleton_motion(parents, positions, elev=elev, azim=azim,
                title=f"{stage} / {label}\nRaw FK joints, no skin or cleanup", figsize=(4, 4), dpi=100)
            destination = output / f"{label}.mp4"
            _write_video(str(destination), frames, int(result["fps"]))
            tiles.append(np.concatenate(frames[selected], axis=1))
            receipt["views"].append({"view": label, "path": str(destination), "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()})
            del frames
            gc.collect()
        Image.fromarray(np.concatenate(tiles, axis=0)).save(output / "contact-sheet.png")
        receipt.update(status="passed", contactSheetFrames=selected.tolist())
    except Exception as error:
        receipt.update(status="failed", error=repr(error))
        raise
    finally:
        receipt["seconds"] = time.time() - receipt["startedUnix"]
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        volume.commit()
    return receipt


@app.local_entrypoint()
def main(stage: str, group: int = 0, sample: int = 0):
    print(json.dumps(render.remote(stage, group, sample)), flush=True)
