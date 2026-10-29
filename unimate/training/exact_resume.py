"""Optimizer-boundary checkpoints for single-process, zero-worker training.

The loader deliberately has no lookahead: Accelerate's DataLoaderShard fetches
another augmented batch before yielding the current one. Its batch cursor alone
therefore cannot describe the dataset RNG at an interruption boundary.
"""

import hashlib
import json
import os
import platform
import random
import tempfile
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import accelerate
import numpy as np
import torch
from accelerate.data_loader import DataLoaderStateMixin
from accelerate.state import GradientState
from accelerate.utils import send_to_device
from torch.utils.data import DataLoader, RandomSampler

FORMAT = "unimate-exact-resume-v1"


def capture_rng():
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    if len(state["cuda"]) != torch.cuda.device_count():
        raise ValueError("Checkpoint CUDA device count differs from this process")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


@contextmanager
def preserve_rng():
    state = capture_rng()
    try:
        yield
    finally:
        restore_rng(state)


class ResumableBatchSampler:
    """Retain the actual epoch draws and next batch, never replay dataset work."""

    def __init__(self, batch_sampler, seed):
        self.batch_sampler = batch_sampler
        self.sampler = batch_sampler.sampler
        self.batch_size = batch_sampler.batch_size
        self.drop_last = batch_sampler.drop_last
        if isinstance(self.sampler, RandomSampler):
            self.sampler.generator = torch.Generator().manual_seed(seed)
        self.epoch = 0
        self.cursor = 0
        self.batches = None

    def __len__(self):
        return len(self.batch_sampler)

    def set_epoch(self, epoch):
        if epoch != self.epoch:
            if self.batches is not None and self.cursor != len(self.batches):
                raise ValueError("Cannot change epoch before consuming its batches")
            self.epoch, self.cursor, self.batches = epoch, 0, None
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)

    def __iter__(self):
        if self.batches is None:
            self.batches = list(self.batch_sampler)
        while self.cursor < len(self.batches):
            batch = self.batches[self.cursor]
            self.cursor += 1
            yield batch

    def state_dict(self):
        if self.batches is None:
            raise ValueError("Cannot checkpoint before the first batch")
        return {
            "epoch": self.epoch, "cursor": self.cursor, "batches": self.batches,
            "generator": self.sampler.generator.get_state(),
            "next_batch": (self.batches[self.cursor]
                           if self.cursor < len(self.batches) else None),
        }

    def load_state_dict(self, state):
        if set(state) != {"epoch", "cursor", "batches", "generator", "next_batch"}:
            raise ValueError("Incomplete sampler checkpoint")
        batches, cursor = state["batches"], state["cursor"]
        if len(batches) != len(self) or not 0 <= cursor <= len(batches):
            raise ValueError("Checkpoint batch order/cursor does not match loader")
        if any(len(batch) != self.batch_size for batch in batches):
            raise ValueError("Checkpoint batch size does not match loader")
        if any(not isinstance(i, int) or not 0 <= i < len(self.sampler.data_source)
               for batch in batches for i in batch):
            raise ValueError("Checkpoint contains invalid sample indices")
        next_batch = batches[cursor] if cursor < len(batches) else None
        if next_batch != state["next_batch"]:
            raise ValueError("Checkpoint next-batch identity is inconsistent")
        self.epoch, self.cursor, self.batches = state["epoch"], cursor, batches
        self.sampler.generator.set_state(state["generator"])
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(self.epoch)


class ExactDataLoader(DataLoader, DataLoaderStateMixin):
    """Torch loading with Accelerate's end-of-epoch accumulation bookkeeping.

    Uses Accelerate 1.14's DataLoaderStateMixin lifecycle, without the
    DataLoaderShard lookahead. Only the current batch is materialized.
    """

    def __init__(self, original, device, seed):
        if original.num_workers != 0:
            raise ValueError("Exact resume requires num_workers=0")
        self.resume_sampler = ResumableBatchSampler(original.batch_sampler, seed)
        super().__init__(original.dataset, batch_sampler=self.resume_sampler,
                         collate_fn=original.collate_fn, num_workers=0,
                         generator=torch.Generator().manual_seed(seed))
        self.device = device
        self.gradient_state = GradientState()
        self._drop_last = original.drop_last
        self.reset()

    def set_epoch(self, epoch):
        self.resume_sampler.set_epoch(epoch)

    def __iter__(self):
        self.begin()
        # Torch iterator initialization consumes a base seed even with zero
        # workers. This generator is separate from both sampling and training.
        seed_state = self.generator.get_state()
        iterator = super().__iter__()
        self.generator.set_state(seed_state)
        try:
            for batch in iterator:
                self.end_of_dataloader = self.resume_sampler.cursor == len(self)
                yield send_to_device(batch, self.device)
        finally:
            self.end()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def _config_dict(config):
    value = asdict(config)
    value["experiment"].pop("output_dir", None)
    return value


def input_identity(config):
    """Bind source and every file under the configured feature roots once/run."""
    source_root = Path(__file__).resolve().parents[2]
    source = {str(p.relative_to(source_root)): file_sha256(p)
              for p in sorted((source_root / "unimate").rglob("*.py"))}
    source["requirements.txt"] = file_sha256(source_root / "requirements.txt")
    data = {}
    for name, dataset in sorted(config.dataset.data_configs.items()):
        root = Path(dataset.path).resolve(strict=True)
        files = {}
        visited = set()
        for directory, subdirs, filenames in os.walk(root, followlinks=True):
            actual = Path(directory).resolve()
            if actual in visited:
                raise ValueError(f"Repeated/cyclic training directory: {directory}")
            visited.add(actual)
            subdirs.sort()
            for filename in sorted(filenames):
                path = Path(directory) / filename
                files[str(path.relative_to(root))] = file_sha256(path)
        data[name] = files
        if not data[name]:
            raise ValueError(f"No training files at {root}")
    config_dict = _config_dict(config)
    return {
        "config": _json_hash(config_dict), "source": _json_hash(source),
        "data": _json_hash(data),
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "numpy": np.__version__, "accelerate": accelerate.__version__,
                    "cuda": torch.version.cuda,
                    "cudnn": torch.backends.cudnn.version(),
                    "threads": torch.get_num_threads(),
                    "deterministic": torch.are_deterministic_algorithms_enabled(),
                    "cudnn_deterministic": torch.backends.cudnn.deterministic,
                    "cudnn_benchmark": torch.backends.cudnn.benchmark,
                    "matmul_precision": torch.get_float32_matmul_precision(),
                    "cuda_devices": [torch.cuda.get_device_name(i)
                                     for i in range(torch.cuda.device_count())]},
    }


def _embedding_hash(value, digest):
    if isinstance(value, dict):
        for key in sorted(value, key=repr):
            digest.update(repr(key).encode())
            _embedding_hash(value[key], digest)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _embedding_hash(item, digest)
    else:
        array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
        digest.update(str((array.dtype, array.shape)).encode())
        digest.update(np.ascontiguousarray(array).tobytes())


def training_identity(inputs, config, dataset, stats_path):
    md = dataset.motion_dataset
    embeddings = hashlib.sha256()
    for name in ("_joint_name_emb", "_caption_tokens", "_caption_emb",
                 "_extra_caption_tokens", "_extra_caption_emb"):
        embeddings.update(name.encode())
        _embedding_hash(getattr(md, name), embeddings)
    return {**inputs, "resolved_config": _json_hash(_config_dict(config)),
            "splits": _json_hash({"train": md.train_name_list, "eval": md.eval_name_list}),
            "stats": file_sha256(stats_path), "encoder_outputs": embeddings.hexdigest()}


def _atomic_write(path, write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_checkpoint(path, accelerator, model, optimizer, scheduler, ema, tracker,
                    dataloader, identity):
    if os.path.exists(path):
        raise FileExistsError(f"Checkpoint files are immutable: {path}")
    if not accelerator.sync_gradients:
        raise ValueError("Checkpoint requires an optimizer boundary")
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError("Checkpoint requires cleared accumulated gradients")
    checkpoint = {
        "format": FORMAT, "identity": identity,
        "model_state_dict": accelerator.get_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": scheduler.state_dict(),
        "ema_state_dict": ema.state_dict() if ema is not None else None,
        "scaler_state_dict": (accelerator.scaler.state_dict()
                              if accelerator.scaler is not None else None),
        "tracker": vars(tracker).copy(), "sampler": dataloader.resume_sampler.state_dict(),
        "accelerator_step": accelerator.step, "rng": capture_rng(),
        "step": tracker.current_step, "epoch": tracker.current_epoch,
    }
    _atomic_write(path, lambda stream: torch.save(checkpoint, stream))
    receipt = {"format": FORMAT, "checkpoint": Path(path).name,
               "sha256": file_sha256(path), "bytes": os.path.getsize(path),
               "step": tracker.current_step, "epoch": tracker.current_epoch}
    _atomic_write(Path(path).parent / "latest.json",
                  lambda stream: stream.write(json.dumps(receipt, sort_keys=True).encode()))


def read_checkpoint(path):
    path = Path(path)
    if path.suffix == ".json":
        receipt = json.loads(path.read_text())
        filename = receipt["checkpoint"]
        if Path(filename).name != filename:
            raise ValueError("Invalid checkpoint receipt path")
        path = path.parent / filename
        if (receipt["format"] != FORMAT or path.stat().st_size != receipt["bytes"]
                or file_sha256(path) != receipt["sha256"]):
            raise ValueError("Checkpoint receipt does not match the durable file")
    # Training checkpoints contain Python and NumPy RNG objects. Only load
    # trusted run artifacts; PyTorch's tensor-only loader cannot read these.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {"format", "identity", "model_state_dict", "optimizer_state_dict",
                "lr_scheduler_state_dict", "ema_state_dict", "scaler_state_dict",
                "tracker", "sampler", "accelerator_step", "rng", "step", "epoch"}
    if set(checkpoint) != required or checkpoint["format"] != FORMAT:
        raise ValueError("Incomplete or unsupported exact-resume checkpoint")
    return checkpoint


def restore_checkpoint(checkpoint, accelerator, model, optimizer, scheduler, ema,
                       tracker, dataloader, identity):
    if checkpoint["identity"] != identity:
        changed = sorted(k for k in identity if checkpoint["identity"].get(k) != identity[k])
        raise ValueError(f"Resume identity mismatch: {changed}")
    if (ema is None) != (checkpoint["ema_state_dict"] is None):
        raise ValueError("Checkpoint EMA presence does not match run")
    if (accelerator.scaler is None) != (checkpoint["scaler_state_dict"] is None):
        raise ValueError("Checkpoint scaler presence does not match precision")
    for name, expected in (("optimizer_state_dict", optimizer.state_dict()),
                           ("lr_scheduler_state_dict", scheduler.state_dict()),
                           ("tracker", vars(tracker)),
                           ("rng", capture_rng())):
        if set(checkpoint[name]) != set(expected):
            raise ValueError(f"Incomplete checkpoint state: {name}")
    if ema is not None and set(checkpoint["ema_state_dict"]) != set(ema.state_dict()):
        raise ValueError("Incomplete EMA checkpoint")
    if (accelerator.scaler is not None
            and set(checkpoint["scaler_state_dict"]) != set(accelerator.scaler.state_dict())):
        raise ValueError("Incomplete scaler checkpoint")
    if set(checkpoint["sampler"]) != {"epoch", "cursor", "batches", "generator", "next_batch"}:
        raise ValueError("Incomplete sampler checkpoint")
    saved_tracker = checkpoint["tracker"]
    for key, value in vars(tracker).items():
        if key not in ("current_step", "current_epoch") and saved_tracker.get(key) != value:
            raise ValueError(f"Checkpoint training horizon mismatch: {key}")
    if (saved_tracker["current_step"] != checkpoint["step"]
            or saved_tracker["current_epoch"] != checkpoint["epoch"]
            or checkpoint["sampler"]["epoch"] != checkpoint["epoch"]):
        raise ValueError("Checkpoint progress is inconsistent")
    accelerator.unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
    if ema is not None:
        ema.load_state_dict(checkpoint["ema_state_dict"])
        ema.to(accelerator.device)
    if accelerator.scaler is not None:
        accelerator.scaler.load_state_dict(checkpoint["scaler_state_dict"])
    vars(tracker).update(saved_tracker)
    dataloader.resume_sampler.load_state_dict(checkpoint["sampler"])
    accelerator.step = checkpoint["accelerator_step"]
    # Last operation: initialization and state loading may have consumed RNG.
    restore_rng(checkpoint["rng"])
