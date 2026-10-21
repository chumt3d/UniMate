# UniMate escalation, 21 October 2026

Owner: TIN-88 ROOT, delegated to UNIMATE_ESCALATION. Research and bounded repair;
no production release or admission. Total authorised GPU budget $4; no training
above $1 without owner approval. Scratch is disposable, evidence copied to the
delegate's Windows visualization folder and checkpoint source pushed without CI.

Outcome: a useful creature locomotion path on the artist's rig. Reproduce public
author examples first, then test failure hypotheses against the real fox and
the existing procedural baseline. Product/export acceptance stays distinct.

First fixed experiment: author issue #8 Tiger v3-150k, original conditioning,
seed 0, CFG 3, eight repetitions, prompt `An object walks in place.` Use the
retained successful Go2-b image, strict checkpoint/EMA loading, global stats,
and byte-verified b78c7807 upstream inference source. Compare all eight outputs
with the author's arrays. No conditioning, decoder, prompt, or sampler changes.

Before GPU: CPU checks model load, input identities, runtime source and imports.
Persist each run's output and failure receipt. A successful wrapper is not proof
of successful motion generation. Check eight finite arrays with correct shape.

Next questions (experiments declared after baseline, before their output):
- Can upstream expansion sustain three complete strides without seam drift?
- Does in-place versus travelling conditioning explain the fox gap?
- Do RIC target positions and rotation/FK skin positions disagree?
- Do caption-style prompt/CFG changes alter gait instead of only energy?
- Does existing E.5 or foot IK improve contacts without erasing the gait?

Prior dynamic stance detector was rejected. Its gait/duty/phase outputs are
telemetry, not F1-F3 acceptance. Report raw and corrected motion separately.

Checkpoint A result: CPU preparation17.82s; GPU generation/render106.32s;
eight author motions reproduced with maximum absolute feature error6.7726e-6,
RMS2.19e-7..4.99e-7. This validates the released inference path for this fixture,
not contact quality. App ap-hNhdDMjUoZfwWJuq5IsThd, imageim-yvwuQXEyvg2yH1dd46r6k0.

Experiment B, declared before output: compare released truebones_f60_v2/80k
against uniml3d_f60_v3/150k on native Fox39j and reviewed artist fox47j.
Use each exact config/stats/EMA; explicitly select truebones stats for the
source-specific model. Seeds1729,2718,31415, one repetition per seed, five
prompts: exact old training captions `An object walks in place with its tail
swaying.` and `An object bounds in place.`, plus `An object trots forward.`,
`An object trots in place.`, `An object walks in place.`. All use unchanged
adaptive dopri5/CFG3. Suppress only video generation for the numerical screen.
No contact, phase or gait admission inferred from a60-frame window. Checkpoint
source is HF174fc755, Truebones training caption source is UniML3Dfaaa81773b315247b03f183548e3898dbec2ce80/export/truebones/motion_captions.json.
Native conditioning comes from the prior retained pipeline; its equality to
old training preprocessing is unverified and cannot be claimed.

Experiment C, declared before GPU output: unchanged upstream expansion with
eight 60-frame windows and overlap10 produces410 fresh generated frames
(13.67s at30fps). First window uses release dopri5; seven constrained windows
use release Euler50. Reproduce Tiger seed0 first, then first-screen seed1729
artist v3/Truebones walk/trot and native Truebones walk. Extend promising
conditions to seeds2718/31415; single-seed screening cannot meet acceptance.
No repeated playback frames count as strides. Original48-joint artist-skin
decoding remains a separate consumer check. IMPL owns its source-based bundle.
Read-only GPT-6.1 Sol/high child experiment_complexity_review inspects the
experiment/source/resource contracts; escalation owner integrates findings.

Diagnostic length L is canonical rest-root-to-lowest-joint height, not the
rubric's per-leg length. Joint heuristics and spectral peaks remain telemetry;
they do not determine contact phase, gait pass/fail, or original-skin quality.

C first screen generated410finiteframes for all six declared cases. Tiger
first60features matched authorrep0 within3.40e-6; full chain took46.94sL40S.
Raw Tiger append boundaries show per-joint acceleration/interior-p99 ratios
7.50..17.39; artistv3walk2.57..8.65; artistTrueboneswalk9.31..17.94.
These are FK joint telemetry at canonical length, not the skin/leg AT2 verdict.
Same seed integers across model widths71/86 do not establish paired latent noise.

D hypothesis: is continuation Euler50 discretization responsible for seams?
Change only the existing inbetween sampler's num_steps to200, Tiger seed0,
two windows110frames. Keep every other behavior and initial free window.
Runtime override is explicitly research-only. Generation passed59.24sA10;
first60 versus C maxabs1.46e-6; continuation feature RMS difference.00315.
Paired FK seam comparison remains required before interpreting this difference.

Original-skin consumer closure from IMPL binds exact GLB7610eda5 and
geometryed5ed33e, sourcecf183, decoderb78. The diagnostic-only exact-asset guard
replaces broad generic Studio authoring admission; no arbitrary-asset or product
admission follows. Exposed procedural known-plant control uses diagnostic
unrestricted bend limits and is not a naturalness/professional-limit baseline.
