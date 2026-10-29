# Exact single-process training resume

New jobs use complete `unimate-exact-resume-v1` checkpoints. Run one process
with `--num_workers 0`; distributed execution and worker processes are refused
before dataset/model construction. Incomplete legacy checkpoints are refused.

```bash
python -m unimate.training.train --config configs/my-cleared-run.json \
  --num_workers 0 --output_dir experiments/my-run --stop_after_seconds 1200
python -m unimate.training.train --config configs/my-cleared-run.json \
  --num_workers 0 --output_dir experiments/my-run \
  --resume experiments/my-run/checkpoints/latest.json --stop_after_seconds 2400
```

Use the same original input config, source, feature files, splits, stats,
encoder outputs and runtime. Output-directory relocation and invocation time
allowance may change. The checkpoint binds hashes of config/source/data,
resolved config/splits/stats/encoder outputs and the numerical runtime. Missing
resumed stats fail; they are never recomputed. Only load trusted `.pt` artifacts:
the complete Python/NumPy RNG state needs PyTorch's pickle-capable loader.

The model/loss, AdamW settings, EMA schedule and original scheduler horizon
remain the configured training contract. A positive finite
`--stop_after_seconds` bounds an invocation: elapsed time starts before setup;
after the next completed optimizer/EMA update and gradient clear, training
saves and exits without changing the run's target. Setup or a single optimizer
step may exceed that allowance; it is not a hard process timeout. No automatic
resubmission is provided.

Recovery saves are checked after optimizer updates, targeting 120 seconds
(source: Tiny3D's approved 2026-10-29 `RETRAIN-PLAN.md`, “Checkpointed training
and first run”). A step already in flight can delay the boundary. Existing
`training.save_interval` also saves milestones; final updates and intentional
budget stops save. `--visualize` opts into initial and milestone debug rendering;
it is off by default and does not run on timed recovery saves. Visualization
preserves training RNG and releases EMA's temporary model-sized collection.

The exact loader materializes only the current batch. Accelerate 1.14's normal
DataLoaderShard fetches the next augmented batch before yielding the current
one; restoring only its cursor would miss those consumed RNG draws. The exact
loader uses the installed DataLoaderStateMixin lifecycle so Accelerate still
flushes a partial accumulation group at epoch end. The original balanced or
shuffled epoch draws, generator, cursor and next batch are persisted. Iterator
base-seed draws use a separate generator. Checkpoints require cleared gradients
and contain CPU/CUDA/Python/NumPy RNG, model, optimizer, scheduler, EMA, optional
scaler, full tracker and Accelerate accumulation counter.

Checkpoint files are immutable. Each write uses a temporary file, file fsync,
atomic rename and directory fsync; a small hashed `latest.json` receipt is then
written the same way. The previous completed file remains. Restarting from an
older point into a directory containing later step files fails on collision;
use a new output directory for that branch.

Local fsync does not establish remote-volume publication. Before paid training,
the worker/orchestrator must commit its private Modal volume, verify the new
checkpoint and pointer across invocations, then retire only identified task-owned
superseded recovery files. Preserve the preceding confirmed recovery checkpoint
and useful milestones; record checkpoint bytes/time and storage cost. Retention
and remote commit are orchestration responsibilities. This fork neither prunes
files nor launches a remote persistence service.

## CPU proof

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
NUMEXPR_NUM_THREADS=2 RAYON_NUM_THREADS=2 \
python3 tests/prove_exact_resume.py --python /path/to/cpu-env/bin/python \
  --out /tmp/unimate-resume-proof
```

The stdlib coordinator runs Torch children serially. The proof uses the actual
production loader/checkpoint/sampler/EMA/tracker helpers with real Torch AdamW,
Dropout, a cosine LR scheduler, and Python/NumPy/Torch stochastic dataset
transforms. It compares uninterrupted training against fresh-process
interrupt/resume at mid-epoch, the final partial accumulation group/epoch
boundary, and the final horizon, for both shuffled and balanced draws. It checks
batch identity, all RNG streams, loss, optimizer/model/EMA state, LR and tracker;
malformed state, identity drift, uncleared gradients and receipt corruption are
rejected.

Observed CPU proof: Python 3.12.3, Torch 2.9.1+cpu, Accelerate 1.14.0, NumPy 2.2.6;
all six comparisons matched exactly. The repository's production Torch pin is
2.5.1+cu124. This proof does not establish equivalence to that runtime, CUDA
numerical determinism, full `train_diffusion`/T5/Blender integration, the real v3
network, data admission or motion quality. The full configured scheduler/loss
remain source-inspected; the tiny proof's model and scheduler are intentionally
small. Paid-run preparation and acceptance remain separate.
