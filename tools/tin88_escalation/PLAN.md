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
