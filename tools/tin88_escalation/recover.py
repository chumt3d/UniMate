"""Recover existing evaluation evidence without starting compute."""

import hashlib
import json
import sys
from pathlib import Path

import modal


def main():
    destination = Path(sys.argv[1])
    destination.mkdir(parents=True, exist_ok=True)
    volume = modal.Volume.from_name("tin88-unimate-eval", environment_name="main")
    roots = sys.argv[2:] or ["unimate-20261021-upstream-go2-b", "unimate-20261021-upstream-fox-a"]
    inventory = []
    for root in roots:
        for entry in volume.iterdir("/" + root, recursive=True):
            inventory.append({"path": entry.path, "size": entry.size, "type": int(entry.type)})
    (destination / "volume-inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    selected = [row for row in inventory if Path(row["path"]).suffix in {".json", ".npy", ".npz", ".log", ".txt", ".png", ".mp4", ".f64le", ".u32le", ".glb"}
                and row["size"] < 32_000_000]
    print(json.dumps({"entries": len(inventory), "selected": len(selected),
                      "bytes": sum(row["size"] for row in selected)}), flush=True)
    receipts = []
    for row in selected:
        relative = Path(row["path"].lstrip("/"))
        if ".." in relative.parts:
            raise ValueError("Unexpected parent traversal in volume entry")
        output = destination / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        with output.open("wb") as file:
            for chunk in volume.read_file(row["path"]):
                digest.update(chunk)
                file.write(chunk)
        receipts.append({**row, "sha256": digest.hexdigest()})
    (destination / "recovery-manifest.json").write_text(json.dumps(receipts, indent=2) + "\n")
    print(json.dumps({"recovered": len(receipts), "bytes": sum(row["size"] for row in receipts)}), flush=True)


if __name__ == "__main__":
    main()
