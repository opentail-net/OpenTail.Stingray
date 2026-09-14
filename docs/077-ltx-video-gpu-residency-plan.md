# LTX-Video GPU Residency Plan (2026-09-13)

## Read this warning before doing any GPU work here

Unlike FLUX (`docs/069`), Wan (`docs/072`/`docs/073`), Z-Image-Turbo (`docs/075`), and
SD3.5-medium (`docs/076`) — all of which have a **real, verified-correct CPU baseline** to build
GPU residency on top of and check GPU output against — **LTX-Video's own correctness is not yet
solid.** Per `PerformanceLeague.md`'s LTX-Video row: "only 1 of 6 total runs across this checkpoint
has ever produced a coherent image" — the rest, including repeat runs at the same explicit seed,
produced garbled visual noise, and the cause is explicitly **not root-caused** as of that entry.
`LtxVideoModel.cs`'s own class doc comment also states it is "NOT yet wired to a real T5-v1.1-XXL
encoder... or the VAE decoder" — both deferred, per the original implementation plan
(`docs/055-ltx-video-implementation-plan.md`).

**Building GPU residency on top of an unresolved correctness bug means every future GPU-vs-CPU
parity check is comparing against a baseline that might itself be wrong** — this doubles the
debugging surface (is a mismatch a GPU bug, or is it the CPU path's own known-unreliable output?)
exactly the situation this project's own `CLAUDE.md` rule 7 warns against ("Performance pass...
once a model's port is complete... do a performance pass"). **Recommended: treat the correctness
investigation as a prerequisite, not a parallel task** — either do it first, or at minimum flag
very clearly in any GPU-residency PR that correctness here was already shaky before the GPU work
started, so a future regression report doesn't waste time blaming the wrong change.

If the correctness bug gets fixed (or ruled unrelated to the DiT itself — e.g. traced to VAE/T5
wiring, which are explicitly not yet real per the class doc comment above) before this doc is
picked up, the rest of this plan is still accurate and can proceed normally.

**2026-09-14: root cause found and fixed.** `LtxVideoModel.TimestepEmbedder` (and, found alongside
it, `LtxVaeDecoder.TimestepEmbedMlp`) called `DiffusionOps.SinusoidalTimestepEmbedding` without
`flipSinToCos: true` — defaulting to this project's own `[sin,cos]` layout, when the model's real
reference (the shared `timestep_embedding` helper in
`examples/stable-diffusion.cpp/src/core/ggml_extend.hpp`, which BOTH Wan's and LTX-Video's real C++
ports call) defaults `flip_sin_to_cos=true` (`[cos,sin]`) — the exact same convention mismatch
already found and fixed for Wan's own timestep embedding earlier this session. Found via a new
diagnostic test (`LtxVideoTimestepEmbedGoldenDiagnosticTest.cs`) that closed a real gap in the
existing golden-fixture coverage: `TestData/LtxGolden/manifest.json` already had `embedded_timestep`
and `temb` dumps that `LtxVideoGoldenParityTests` never actually asserted against. Before the fix,
`embedded_timestep` measured cosine-sim **0.789** against the real dump (a genuine, structural
mismatch, not floating-point noise) — and since every block's AdaLN modulation derives from this one
value, it explains the previously-observed `block0 cosine-sim too low: 0.9893641` failure exactly
(a consistent, compounding but partial corruption of the modulation signal across 28 blocks, not
pure noise). After adding `flipSinToCos: true` to both call sites: `embedded_timestep` and `temb`
cosine-sim are both **1.0000000** (machine precision), and
`LtxVideoGoldenParityTests.LtxVideoModel_MatchesRealDiffusersReference_OnRope_Patchify_CaptionProj_Block0_FullOutput`
— previously failing at `block0 cosine-sim too low: 0.9893641` — now **passes outright** (both
`block0` and `full_out` above the `>0.9999` threshold). All 12 other real LTX-Video test-class
`[Fact]`s (`LtxT5EncoderGoldenParityTests`, `LtxVaeDecoderGoldenParityTests`,
`LtxVaeDecoderMultiFrameGoldenTests`, `LtxVideoRealScaleGoldenTests`, `LtxVideoRealWeightsTests`,
`LtxVideoTests`, `LtxVideoTrajectoryGoldenTests`) re-ran clean afterward (0 failures, 25.8s real
wall-clock — genuine runs, not silent no-ops per `CLAUDE.md` rule 12). A real end-to-end
`stingray image` generation run against the checkpoint (with this fix) is in progress to confirm
visual coherence per the "Success criterion" section below before this correctness item is
considered fully closed — see that section's own update once the run completes. Ask-before-commit
still applies; this fix (plus the earlier same-day double-precision RoPE angle fix, which remains a
real if not root-cause-relevant improvement) is uncommitted pending explicit confirmation.

**2026-09-14, same day — end-to-end visual re-run result: partial improvement, NOT yet fully fixed.**
Ran `stingray image -m models/ltx-video-2b-v0.9.1.safetensors -p "a red apple on a wooden table"
--steps 20` (512×512, real T5-XXL text conditioning found locally, 468.7s) with the timestep-flip fix
applied. **Output**: `docs/diffusion-samples/ltx_video_apple_flipfix.png` — honest assessment: this
is **NOT** a coherent red apple on a wooden table, but it is also clearly **NOT** the pure
unstructured static noise every prior run produced. The image shows real, repeating structure:
horizontal banding and coherent red/white color-block regions (vaguely cow-hide/Holstein-pattern-like),
consistent and non-random in character. **Interpretation**: the DiT transformer itself is now
numerically verified correct to machine precision on the golden fixture (see above), so this
remaining visual defect is very likely in a DOWNSTREAM stage the golden test doesn't cover — the VAE
decoder's own timestep conditioning (note `LtxVaeDecoder.TimestepEmbedMlp` had the identical
flip-sin-to-cos bug, fixed in the same pass, but its own decode-side correctness hasn't been
re-verified against a full real run this way before), the CFG/guidance combination in the CLI's
scheduler loop, or a patchify/unpatchify or frame-count convention mismatch between the DiT's raw
token output and what the VAE decoder expects as input. **This correctness item stays OPEN** — the
timestep-embedding fix is real, confirmed, and worth keeping (it fixed a genuine, machine-precision-
verified bug), but is not sufficient on its own to close out LTX-Video's long-standing "only 1 of 6
runs coherent" finding.

**2026-09-14, same day — VAE decoder ruled OUT as the source of the banding artifact.** Also
confirmed `vae.per_channel_statistics.std-of-means`/`mean-of-means` are genuinely present in the
checkpoint with the expected `[128]` shape (`LtxVaeStatsPresenceDiagnosticTest.cs`) — ruling out a
silent un-normalization skip as the cause. Then isolated the VAE decoder itself
(`LtxVaeNoiseIsolationTest.cs`, mirroring `WanVaeOnlyIsolationTests`'s approach for Wan): built a
standard-normal latent at the REAL 512×512 pipeline scale (patchH=patchW=16, 256 tokens — much larger
than the golden fixture's tiny 2×2/4×4 latent), ran it through the exact same real
un-normalize-then-decode logic `LtxVideoPipeline.GenerateVideo` uses, and saved the result
(`docs/diffusion-samples/ltx_vae_noise_isolation.png`). **Result: ordinary smooth multicolor blob
noise — no horizontal banding, no repeating structure.** This is a clean negative result: it rules
out the VAE decoder itself (at real scale, with real un-normalization) as the source of the banding/
color-block artifact seen in the actual end-to-end run
(`docs/diffusion-samples/ltx_video_apple_flipfix.png`) — decoding genuine random noise looks like
genuine random noise, not like that artifact's distinctive repeating-row character.

**Narrows the remaining search to the DiT-at-real-scale-or-scheduler path.** The DiT is verified
correct on the golden fixture (32 tokens, single block, single forward call) but that fixture never
exercises: (a) all 28 blocks in sequence (only block0 individually verified, plus one full 28-block
forward — but that full-block check only verifies the FINAL output's aggregate cosine similarity,
not that every intermediate block stays well-behaved), (b) the real 256-token spatial scale (16×16
vs the golden fixture's 4×4), (c) the multi-step Euler/RectifiedFlow denoising LOOP (each step's
output latents feed the next step's input — a per-step error, even a small one, could compound
across 20 steps in a way a single-forward golden check can't catch), or (d) the real T5-XXL prompt
encoding at its real ~256-padded-token length (the golden fixture's caption is a fixed, much shorter
16 tokens). The next concrete step for whoever picks this up: run the real pipeline at a MUCH lower
step count (e.g. 1-2 steps) to check whether the banding is already present after just one DiT
forward pass (implicates the DiT-at-real-scale or a single-step scheduler math issue) or only
emerges after several steps (implicates compounding/accumulation across the denoising loop) — this
directly distinguishes hypothesis (c)/(d) from (a)/(b) without needing new golden data.

**2026-09-14, same day — step-count bisection: decisive result, real progress.** Ran the same prompt
at `--steps 2 --seed 42` (101.1s, `docs/diffusion-samples/ltx_video_apple_2step_diag.png`). **Result:
a blurry but genuinely PLAUSIBLE image** — soft, warm red/tan/orange tones consistent with "apple on
wood," smooth gradients, zero banding, zero color-blocking. This is not just "less broken" than the
20-step run, it looks like a normal early-denoising-step preview of a real generation. **This rules
out hypotheses (a)/(b) above** (28-block-sequence bug, real-256-token-scale DiT bug) as the PRIMARY
cause — if either were the root cause, 2 steps would already show it, since each step is an
independent full 28-block/256-token DiT forward. **The remaining candidates are (c) the multi-step
denoising LOOP itself (compounding numerical drift, or a scheduler timestep-schedule issue that only
manifests as steps approach the low-noise end near t=0) and (d) the real ~256-padded-token T5
caption length** (constant across all step counts, so not directly distinguished by this test alone,
but a fixed per-step conditioning bug would show at 2 steps too — its absence there weakly argues
against (d) as well, though not conclusively). A 8-step bisection run is in progress to narrow
whether the artifact appears gradually (drift) or sharply past some step threshold (a
scheduler/timestep-schedule discontinuity) — see the next update for its result.

**2026-09-14, same day — 8-step result: dramatic, decisive, and inverted from normal diffusion
behavior.** `--steps 8 --seed 42` (224.9s, `docs/diffusion-samples/ltx_video_apple_8step_diag.png`)
produced a **clearly recognizable red apple with a visible stem/leaf** on a light background — real,
substantial subject coherence, comparable in quality to this project's other successfully-fixed
diffusion models' outputs (some residual noisy garnish/greenery texture around it, and mild banding
at the bottom, but the core subject is unambiguous). **This means: 2 steps → plausible blurry colors,
8 steps → a recognizable apple, 20 steps → destroyed into repeating red/white bands.** More steps
making the output WORSE, not better, is the inverse of normal diffusion sampling behavior (where more
steps should refine detail, not destroy it) — this is a strong, structural tell, not just "needs more
tuning." **Leading hypothesis, precise and testable**: `LtxVideoPipeline.TimeShift`'s real formula
`exp(mu) / (exp(mu) + (1/t - 1)^sigma)` has a `1/t - 1` term that grows without bound as `t -> 0`; the
schedule's minimum RAW timestep before shifting is `1/steps` (`tRaw = 1 - i/steps`, smallest at
`i=steps-1`), so **more steps directly means the schedule reaches smaller raw `t` values** — 0.5 min
at 2 steps, 0.125 at 8 steps, 0.05 at 20 steps — exactly the region where `1/t-1` and its numerical
sensitivity grow fastest. A likely failure mode: at small `t`, `TimeShift` output and/or the `dt`
between consecutive shifted timesteps at the LATE (low-noise, fine-detail) end of the schedule
becomes disproportionately large or unstable, causing the final several Euler steps to overshoot and
destructively overwrite the otherwise-good structure the earlier steps had already built up (matching
that the 8-step run, which never reaches as small a `t`, stays coherent). **Concrete next step for
whoever picks this up**: add a temporary debug dump of `shiftedTimesteps[]` and per-step `dt` for a
real 20-step run and inspect the last few entries for a non-monotonic jump, a `dt` spike, or a
near-NaN/extreme value — this should directly confirm or rule out this hypothesis without needing any
new golden data, and if confirmed, points straight at `TimeShift`/`GetNormalShift`'s formula or the
`sd3Shift` value itself as the fix target (matching the real reference's
`sd3_resolution_dependent_timestep_shift`/`time_shift` more precisely, e.g. a clamp or an off-by-one
in how `1/steps` vs `1/(steps-1)`-style boundary conventions are handled — this project's own formula
was ported from `ltx_video/schedulers/rf.py` per the existing code comment, so the boundary handling
there is the first thing to re-diff against).

**2026-09-14, same day — schedule-instability hypothesis directly checked and RULED OUT.** Dumped
the real `shiftedTimesteps[]`/`dt` values (re-deriving `GetNormalShift`/`TimeShift` exactly,
`LtxSchedulerDiagnosticTest.cs`) for the real 256-token case at 2/8/20 steps.  **Result: every value
at every step count is smooth, monotonic, and free of any spike, near-zero, or NaN** — `tShifted`
decreases smoothly from 1.0 toward 0 at all three step counts, and `dt` increases smoothly and
monotonically toward the END of the schedule (largest at the last step: 0.219 at 8 steps, 0.094 at
20 steps) rather than blowing up — this is the SD3-style shift's actual intended behavior (compress
steps near t=1, spread them near t=0), not an anomaly. **The schedule/`TimeShift` formula itself is
clean at every step count tested — this specific hypothesis is decisively ruled out**, a genuine
negative result, not just "didn't find it yet." Given the schedule is fine but degrades sharply with
more steps, the leading remaining candidate shifts to: (a) CFG combination amplifying a small
per-step modeling error across steps (`vPred = vPredUncond + guidance*(vPred-vPredUncond)` — each
step's output feeds the next step's input, so even a small consistent per-step bias could compound
multiplicatively through repeated CFG amplification, guidance=3.0 here), or (b) a genuine but subtle
DiT numerical-precision or attention-pattern degradation that's invisible on a single golden forward
pass (cosine-sim >0.9999 is not "exactly zero error") but compounds destructively over many
sequential calls with different, evolving inputs — a class of bug a single-forward golden check
structurally cannot catch. **Next concrete step**: re-run the 20-step case with `guidance=0`
(disables CFG entirely, single forward per step) to test hypothesis (a) directly — if that also still
degrades with more steps, CFG is ruled out and (b) becomes the leading candidate; not yet run this
pass.

**2026-09-14, same day — CFG-amplification hypothesis directly checked and RULED OUT.** Ran
`--steps 20 --seed 42 --cfg-scale 0` (CFG fully disabled — `guidance > 1.0f` gate false, single
forward per step, no uncond branch, no `vPredUncond + guidance*(vPred-vPredUncond)` combination at
all) — `docs/diffusion-samples/ltx_video_apple_20step_nocfg_diag.png`. **Result: still broken**, same
character of artifact as the CFG-enabled 20-step run (red/white/black blocky vertical banding, no
recognizable apple). **This rules out hypothesis (a) (CFG amplification) decisively** — the
degradation happens even in the simplest possible per-step case (one DiT forward, no guidance
combination, latents fed straight back in). **This narrows the bug to hypothesis (b): something in
the raw iterative loop itself** — `latents[i] -= dt * vPred[i]` repeated 20 times, each step's DiT
forward consuming the previous step's own output — that compounds a real, small, per-step error into
a large visible one specifically as the loop runs longer, and which a single-golden-forward check
(comparing one isolated DiT call against one isolated reference call) structurally cannot see. Two
sub-candidates remain open for whoever picks this up next: (i) the DiT itself IS slightly wrong in a
way too small to fail the golden cosine-sim gate (>0.9999 is not "exactly correct") but which
compounds multiplicatively feeding its own output back in 20 times — would need either a tighter
golden tolerance or a real multi-step trajectory-level golden comparison (not attempted yet for LTX;
`LtxVideoTrajectoryGoldenTests` exists and already passes, worth re-reading closely to confirm it
actually covers a multi-step loop and not just a single-step structural check) to catch; (ii) the
Euler update `latents[i] -= dt*vPred[i]` itself, or the real `RectifiedFlowScheduler.step()`'s exact
formula it's meant to mirror, has a subtle convention mismatch (sign, per-token vs per-latent scale,
or something in how frame-count/patch dimensions map back into the flat `latents` array across
steps) that only manifests as accumulated positional drift over many iterations rather than a
single-step error. **Given the step-count bisection's own shape (2-step good colors, 8-step
recognizable apple, 20-step destroyed), (ii) is the more likely of the two** — a per-step
compounding drift that's still small at 8 steps but has visibly taken over by 20 is a classic
symptom of an integration-scheme bug (wrong sign, wrong scale, or systematic per-step offset), not
of a tiny DiT numerical-precision gap (which would typically show as gradually increasing blur/noise,
not a sudden qualitative collapse into a totally different, highly structured banding pattern).
Re-reading `LtxVideoTrajectoryGoldenTests.cs` closely, and/or adding a debug dump of the latent
tensor's own per-step mean/variance/max-abs across a real 20-step run (to see exactly which step the
statistics first go anomalous) are the two most promising concrete next steps.

**2026-09-14, same day — per-step latent statistics dumped: decisive, quantified root cause found.**
Re-read `LtxVideoTrajectoryGoldenTests.cs` first: it DOES already exercise the real multi-step loop
(4 steps, with CFG) end-to-end and passes at >0.999 cosine-sim per step — but only at a tiny 4-token
(2×2 patch) scale, nowhere near the real 256-token (16×16) case where the bug appears, and over only
4 steps rather than 20. This is a real, plausible explanation for why it never caught this bug.
Added an env-gated (`STINGRAY_LTX_DEBUG_STEPSTATS=1`, zero cost when unset) per-step latent/`vPred`
statistics dump to `LtxVideoPipeline.GenerateVideo` and ran the real `--steps 20 --cfg-scale 0` case
again. **Result — the latent standard deviation grows monotonically and UNBOUNDED across the entire
20-step loop, never converging: 0.976 (step 0) → 0.911 (step 4, briefly dips) → 1.006 (step 8) →
1.522 (step 14) → 2.424 (step 19) — a 2.5× growth from its initial value, with `latMaxAbs` growing
from 4.72 to 11.93 in lockstep.** `vPred`'s own max-abs magnitude also grows substantially over the
same window (5.65 → a peak of ~13.7), meaning the model's own predicted velocity is NOT settling down
as the trajectory should approach a clean, low-noise image — real, well-behaved flow-matching
sampling should see the latent's statistics converge toward the training data's own distribution as
`t -> 0`, not diverge. **This directly and mechanistically explains the visual symptom**: feeding
a VAE decoder latent values 2-3× larger in magnitude than what it was trained on saturates its output
into flat, high-contrast, blocky regions — exactly the red/white/black banding character seen in
every broken run, and exactly why MORE steps (which let this runaway growth compound further) make
the result WORSE rather than better (2 steps: barely any compounding yet, still recognizable; 8
steps: partway there; 20 steps: fully diverged). **This is the real, quantified, root-level
mechanism of the bug** — narrowed from "something in the loop" to "the per-step latent update is
unstable and diverges instead of converging." **Two remaining candidates for WHY the update
diverges, not yet distinguished**: (i) the DiT's own v-prediction is genuinely miscalibrated at this
real scale/late-step regime (a subtle model-level bug too small for the golden single-forward check
to catch, e.g. an attention or normalization behavior that only misbehaves outside the golden
fixture's narrow tested range), or (ii) a scale/convention bug in how `dt`/`vPred` combine in the
Euler update itself — worth specifically re-checking against the real `RectifiedFlowScheduler.step()`
for any per-step rescaling, clipping, or renormalization this port might be missing (the pure
`x -= dt*v` Euler formula is the simplest possible integrator; some real schedulers apply additional
stabilization the plain formula wouldn't have). **Next concrete step**: add the same per-step
latent-statistics dump to the SMALL-scale case `LtxVideoTrajectoryGoldenTests` already exercises (4
tokens, 4 steps, known-good per the passing golden check) as a control — if that small case's std
also grows (just less dramatically, given fewer steps), the mechanism is scale-independent and
purely a step-count/formula issue (favors ii); if the small case's std stays flat/converges while the
real-scale case diverges, the bug is specifically triggered by the larger token count (favors i,
and would need direct model inspection at real scale to pin down further).

**2026-09-14, same day — scale-independence control run: CONFIRMED, root cause narrowed to (ii), a
formula/integration bug, not a model-quality issue.** Ran the exact same real pipeline code path
(`STINGRAY_LTX_DEBUG_STEPSTATS=1`, `--steps 20 --cfg-scale 0 --seed 42`) at `--width 64 --height 64`
(patchH=patchW=2, 4 tokens — the SAME tiny scale `LtxVideoTrajectoryGoldenTests` already golden-
verifies, just run for 20 steps instead of that test's 4). **Result: the SAME divergence pattern
appears, nearly the same shape**: `latStd` 1.001 (step0) → 1.025 (step4) → 1.098 (step8) → 1.450
(step14) → 2.062 (step19) — a ~2.06× growth, closely tracking the real-scale run's ~2.48× growth
(0.976→2.42) over the identical 20 steps. **This is scale-independent** — it happens even at 4
tokens, decisively ruling out hypothesis (i) (a DiT miscalibration specific to real 256-token scale)
and confirming hypothesis (ii): **this is a formula/integration-convention bug in the Euler update
or `dt`/timestep derivation itself, present at every scale, not a model-quality gap.**

**This also explains, precisely, why `LtxVideoTrajectoryGoldenTests` never caught it**: that test
checks `CosineSimilarity(latents, expectedTrajectory[step]) > 0.999` — cosine similarity is
magnitude-blind by construction (it only measures the angle between vectors, not their length). A
trajectory that grows 2× in magnitude while staying directionally close to the real reference can
pass a cosine-similarity check at every single step and still be numerically wrong in a way that
becomes visually catastrophic once the magnitude error compounds far enough (which needs more than
4 steps to become large — matching why that test's own 4-step check never saw it). **A real, concrete
follow-up worth flagging for the test itself** (not required to fix the bug, but a real gap now that
it's understood): add an L2-relative-magnitude check (e.g. `|latents|/|expected| ∈ [0.95, 1.05]`)
alongside the existing cosine-sim assertion in `LtxVideoTrajectoryGoldenTests`, since a magnitude-blind
metric structurally cannot catch this whole class of bug.

**Status at the point this investigation was handed off**: the divergence is real, precisely
quantified, and narrowed to the Euler-update/scheduler-integration formula rather than the DiT model
itself — but the EXACT line-level cause within `latents[i] -= dt * vPred[i]` / the `TimeShift`/
`GetNormalShift` derivation has not yet been found. Candidates still open: a missing renormalization
step the real `RectifiedFlowScheduler.step()` might apply that this port's plain Euler formula
lacks, a sign/scale convention mismatch specific to how this checkpoint's v-prediction relates to
`(data, noise)` (worth re-deriving from the real `ltx_video/schedulers/rf.py` source line-by-line
rather than trusting the existing code comment's summary, now that a real numeric discrepancy proves
something in that area is measurably wrong), or a subtly wrong `TimestepScale`/`timestep` value
being fed to `_transformer.Forward` that makes the model think it's at a different point in the
schedule than it actually is on each call.

**2026-09-14, same day — Euler/scaling formula re-derived against the real generic reference and
confirmed algebraically correct; refined final hypothesis.** Checked the real C++ reference's generic
rectified-flow denoiser (`examples/stable-diffusion.cpp/src/runtime/denoiser.hpp`'s
`DiscreteFlowDenoiser`, the same family LTX-family models route through): its `get_scalings(sigma)`
returns `c_skip=1, c_out=-sigma, c_in=1`, meaning the real k-diffusion-style Euler step computes
`denoised = x - sigma*model_output`, `d = (x-denoised)/sigma = model_output`,
`x_next = x + d*(sigma_next-sigma) = x - model_output*(sigma-sigma_next)` — which is EXACTLY
`latents -= dt*vPred` (this project's own formula) once `dt = sigma - sigma_next` is substituted.
**No discrepancy found here either** — the Euler/scaling math is confirmed algebraically identical
to the real reference's generic formula, ruling out a scaling-convention bug too.

**Refined final hypothesis, given everything else now ruled out**: the observed growth is
consistently ~2× over 20 steps at BOTH scales — equivalent to a small, consistent proportional
overshoot of roughly 3.5% per step (`1.035^20 ≈ 1.98`). A per-step error this small and this
*consistent* (same relative size at every step, at every scale) is not what a structural/convention
bug in the loop itself would typically produce (those tend to show as an obviously wrong formula
shape, a sign flip, or a scale factor off by a large clean multiple) — it is, however, exactly the
signature of a small, systematic magnitude bias in the DiT's own predicted-velocity output (e.g. a
missing or extra normalization/scale factor on the FINAL output projection, `head.head`, or a
residual-stream scale convention slightly off from what the real checkpoint expects) that is too
small to fail the golden test's >0.9999 cosine-similarity gate (which is angle-only) but compounds
multiplicatively once fed back into itself 20 times. **Concrete next step for whoever picks this up**:
re-check `LtxVideoModel.Forward`'s final `head.head` projection and the `AdaLN`-modulated
`norm_out`/`finalShift`/`finalScale` combination immediately before it against the real reference one
more time, specifically looking for a missing scale constant (e.g. a `1/sqrt(d)`-style final-layer
normalization, or a residual-stream growth-compensation factor some DiT architectures apply at their
last block) — and/or measure the exact per-step growth RATIO of `|vPred|` relative to `|latents|`
across a real run to see if it is a suspiciously round, easily-recognizable number (e.g. very close
to a `sqrt(2)`, `1/0.966`, or similar constant) that would point directly at a specific missing/
extra scale factor rather than requiring a blind search.

**2026-09-14, same day — final `norm_out`/`proj_out` layer also checked and confirmed clean.**
Compared `LtxVideoModel.Forward`'s final-layer code (`topTable`/`finalShift`/`finalScale`/
`LayerNormNoAffine`/`Modulate`/`proj_out` Linear) against the real reference's
`get_output_scale_shift`/`norm_out`/`LTXV::modulate`/`proj_out` (`ltxv.hpp` lines ~1537-1662):
shift/scale chunk ordering (`table[0:dim]`=shift, `table[dim:2dim]`=scale, both additively combined
with `embedded_timestep` before use) matches exactly; `norm_out` is confirmed non-affine LayerNorm
(`elementwise_affine=False`, matching `LayerNormNoAffine`); the normalize-then-modulate-then-project
sequence matches. **No discrepancy found here either** — every structural piece of the model this
session has been able to check against the real reference (RoPE, patchify, caption projection,
AdaLN modulation, self-attention, cross-attention, FFN, gating, the final output layer, the Euler
integration formula, the scheduler shift formula) checks out clean. The remaining, unconfirmed
candidate is a genuinely subtle numerical-magnitude bias inside the 28-block loop itself (per-block
residual-stream scale, or a cumulative floating-point drift specific to this port's exact operation
ordering) that structural comparison against reference pseudocode cannot reveal — finding it would
need either a new fine-grained golden dump (blocked: no Python in this environment) or a from-scratch
independent re-implementation of the block loop to diff against this port's own output block-by-block
across a real multi-step run. **Handing off at this point**: this investigation has produced a
precise, reproducible, well-evidenced characterization of the bug (unbounded ~2x per-20-step latent
magnitude growth, scale-independent, not caused by VAE/CFG/schedule/Euler-formula/final-layer) even
though the exact line has not yet been found — a significant narrowing from the turn's starting point
("only 1 of 6 runs ever coherent, not root-caused").

## Current state — confirmed less mature than the other four models

`src/OpenTail.Stingray.Diffusion/LTXVideo/LtxVideoModel.cs` (441 lines) has **zero references to
`IComputeBackend`, `Gpu`, or `Vulkan` anywhere** — confirmed by grep. This is a step earlier than
FLUX/Wan/Z-Image/SD3.5 were before their own residency work: those all had at least the
`MatQ`-style per-matmul GPU dispatch path already wired (upload/`Sgemm`/download/free per call);
LTX-Video has no GPU code at all yet, CPU-only from the ground up.

## Real architecture — confirmed from the file directly

- **`headDim=64` by default** (`DetectConfig`'s fallback `heads=32, headDim=64` for
  `hidden=2048`, confirmed real via `InferAttentionLayout` — checked against the actual checkpoint,
  not hardcoded-and-assumed). Same as SD3.5-medium (`docs/076`): use `MultiHeadAttentionTiled`
  (headDim=64), NOT `MultiHeadAttentionTiled128` — the two share no fused kernel this session's
  FLUX/Wan/Z-Image work already tuned.
- **Continuous 3D RoPE, confirmed INTERLEAVED pairing** (`LtxVideoRoPE.ComputeContinuous3DRoPE`'s
  own doc comment: "interleaved layout: index 2i and 2i+1 share one rotation angle") — the SAME
  convention FLUX's `Flux2DRoPE` GPU kernel already implements. Unlike Z-Image (`docs/075`, where
  this needed independent verification) this one's convention is already documented in the source
  file itself, so `Flux2DRoPE` is very likely directly reusable via a Wan-style compact
  frequency-table builder (mirror `WanRoPE.Compute3DRoPECompact`'s approach: build LTX-Video's own
  per-axis frequency bands in the compact `[tokens, headDim/2]` one-value-per-pair layout
  `Flux2DRoPE` expects) — **but confirm the actual per-axis dimension split** (LTX-Video is video,
  so likely frame/height/width like Wan, not FLUX's 2-axis image split) directly against
  `ComputeContinuous3DRoPE`'s implementation before assuming Wan's specific 44/42/42-style numbers
  apply — they won't; get LTX-Video's own real axis-dim split from the source.
- **`BasicTransformerBlock` with cross-attention, gated self/cross-attention, and an AdaLN-single
  timestep branch** (per the class doc comment, referencing the real `stable-diffusion.cpp`
  `ltxv.hpp` structure directly) — this has a real cross-attention sub-layer (against T5-XXL
  caption embeddings) in addition to self-attention, structurally closer to Wan's
  self-attn+cross-attn block shape (`docs/072`) than to FLUX's single joint-attention blocks.
  `CrossAttentionAdaln`/`SelfAttentionGated`/`CrossAttentionGated` are real, checkpoint-detected
  booleans (not assumed) — read `LtxVideoModel.cs`'s actual block-forward method to get the exact
  modulation/gating wiring right per these flags before porting; do not assume they're always true
  or always match another model's convention.
- **Intermediate-tensor capture fields already exist** (`LastProjInOut`, `LastCaptionProjOut`,
  `LastEmbeddedTimestep`, `LastTimestepProj`, `LastRopeCos`, `LastRopeSin`, `LastBlock0Out`) —
  populated unconditionally by the CPU `Forward()` for `LtxVideoGoldenParityTests`. **A GPU port's
  own parity test can piggyback on these same captured intermediates** rather than needing to
  invent new checkpoints — genuinely useful, already-built infrastructure for exactly this kind of
  verification work.

## Recommended approach

Given LTX-Video starts one step earlier than the other four models (no GPU code at all, not even
`MatQ`), consider doing this in two explicit phases rather than jumping straight to full
residency:

1. **Phase 0 (prerequisite, see warning above)**: confirm or fix LTX-Video's CPU correctness bug,
   or get explicit sign-off that GPU residency work should proceed anyway (e.g. if the bug is
   confirmed to live in the VAE/T5 wiring, not the DiT this doc covers).
2. **Phase 1**: add basic GPU dispatch (`_backend` field + constructor parameter, `MatQ`-style
   per-matmul upload/`Sgemm`/download/free — the SAME starting shape FLUX/Wan/Z-Image/SD3.5 all
   had before their own residency work) — gets a working, correctness-checkable GPU path with
   minimal new surface area, useful as an intermediate correctness checkpoint even if slow.
3. **Phase 2**: real residency — `LtxVideoGpuWeights`/`LtxVideoGpuWorkspace` (mirror
   `FluxGpuWeights.cs`/`WanGpuWeights.cs` directly), a real `ForwardGpu`/`BasicTransformerBlockGpu`
   wiring self-attention + cross-attention (against precomputed, GPU-resident caption K/V, the same
   invariant-caching pattern Wan's `PrecomputeCrossKvCacheGpu` already established) + gated
   residuals + FFN through GPU-resident weights and workspace buffers, batched one
   `BeginBatch()`/`EndBatch()` per block.
4. **Verify correctness with a real, cheap parity test first, using the existing captured
   intermediates** (`LastBlock0Out` etc.) as the comparison points, or the same
   `FluxGpuVsCpuForwardBisectDebugTest.cs`-style small-synthetic-scale approach if those fields
   don't cover what's needed.
5. **Real end-to-end re-verification is unusually important here** given the known correctness
   fragility — don't just check the GPU path matches the CPU path numerically; re-run the same
   visual-inspection discipline `PerformanceLeague.md`'s own LTX-Video row already established
   (view the actual output image, don't just trust a non-crash) before claiming anything works.
6. **Update `PerformanceLeague.md`** with real, measured before/after numbers AND an honest
   correctness note — if the underlying CPU correctness issue is still unresolved when this is
   picked up, say so explicitly in the same entry rather than silently reporting only a timing
   number next to a possibly-still-broken image.

## Practical constraints (same as every prior handoff this session)

- **No subagents** — do all work directly in the main session (`CLAUDE.md` rule 6).
- **Never remove or revert existing correct code.**
- **Ask before committing** — recent work in adjacent areas was committed with explicit
  authorization each time; confirm current expectations rather than assume standing permission.
- **Run one thing at a time, alone** — check `Get-CimInstance Win32_OperatingSystem | Select
  FreePhysicalMemory` (PowerShell) before a benchmark; a real incident earlier this session came
  from an unrelated concurrent `dotnet test` run contaminating a timing measurement.
- **Use small synthetic-scale tests for iteration**, not full end-to-end runs, until confident.

## Success criterion

Given the correctness caveat above, success here is **not just a speed number** — it's a real,
visually-confirmed-coherent output on both CPU and GPU paths, with the GPU path measured honestly
against CPU (win or not). Do not report a "GPU residency win" for this model without first
addressing whether the underlying image-coherence bug is still present.

## 2026-09-14 update — block0 golden-parity bisection (correctness investigation resumed)

`LtxVideoGoldenParityTests.LtxVideoModel_MatchesRealDiffusersReference_OnRope_Patchify_CaptionProj_Block0_FullOutput`
is a real, existing golden-parity test (dumped from real diffusers + the real
`ltx-video-2b-v0.9.1.safetensors` checkpoint) that currently **FAILS** at the block0 stage:
`block0 cosine-sim too low: 0.9893641` (threshold requires `>0.9999`). RoPE cos/sin, `patchify_proj`,
and `caption_projection` all pass independently at `>0.999`–`0.9999` cosine similarity — so the
divergence is introduced somewhere inside `TransformerBlock`'s own combined self-attn+cross-attn+FFN
forward, not in any of those three sub-stages checked in isolation.

**Line-by-line structural audit against the real reference** (`examples/stable-diffusion.cpp/src/
model/diffusion/ltxv.hpp`'s `BasicTransformerBlock::forward`, `CrossAttention::forward`,
`LTXV::modulate`/`apply_gate`, `FeedForward`/`GELU`) found every one of the following to **already
match exactly** — ruled OUT as the cause:
- AdaLN modulation formula (`x*(1+scale)+shift`), gate application (plain multiply, no `+1`).
- 6-way `scale_shift_table + timestepProj` chunk order (shift_msa, scale_msa, gate_msa, shift_mlp,
  scale_mlp, gate_mlp) and per-chunk `dim`-contiguous layout.
- Self-attn RMSNorm (non-affine, eps=1e-6) before modulation; QK-norm (affine RMSNorm, eps=1e-5,
  full-width `inner_dim` before head split) — both epsilons confirmed against the real
  `RMSNorm(inner_dim, 1e-5f)` construction.
- RoPE applied to Q/K as ONE "head" of width `d=2048` before the head split (matches
  `apply_hidden_rope` being called pre-split in the real reference).
- Cross-attention (attn2): raw `x` as query (no pre-norm), K/V from the same `caption_projection`
  output reused at every block, no RoPE, plain (ungated) residual add — matches the real
  `cross_attention_adaln=false` branch exactly (confirmed both `CrossAttentionAdaln` and
  `SelfAttentionGated`/`CrossAttentionGated` are `false` for this real checkpoint via
  `LtxVideoRealWeightsTests`, so the simpler code paths are the CORRECT ones to be using, not a gap).
- FFN: 2048→8192→2048 tanh-approximation GELU (`ggml_gelu`, matching this project's own
  `DiffusionOps.Gelu`) — same activation variant `caption_projection` already uses and independently
  passes with.
- `NumHeads=32`/`HeadDim=64` confirmed against the real checkpoint via `LtxVideoRealWeightsTests`.

**Tested and RULED OUT this session**: RoPE angle float32 argument-reduction precision. The test's
own comment already flagged that at the highest frequency channel the raw angle reaches ~15708 rad,
where `MathF.Cos`/`MathF.Sin`'s float32 argument reduction measurably diverges from the golden
reference (up to 0.02 max-abs-diff, while still passing the >0.9999 cosine-similarity check on the
full table). This looked like a strong candidate for how a "passing" RoPE table could still cause a
"failing" block0 (softmax attention is disproportionately sensitive to fine/high-frequency phase
error). **Fixed forward** (`LtxVideoRoPE.cs`'s `BuildFreqGrid`/`WriteAngle`/`ComputeContinuous3DRoPE`
now compute angles in `double` throughout, only casting to `float` at the final `Math.Cos`/`Math.Sin`
call) — real, harmless, permanent improvement, kept. **But it did not move the needle**: re-ran the
golden test after the fix and block0 cosine-sim was `0.9893642` (was `0.9893641`) — a change in the
7th significant digit, i.e. this candidate is decisively ruled out. The block0 divergence is
**deterministic and structural**, not accumulated floating-point noise.

**Not yet checked** (the real next steps, blocked this turn only by lack of finer-grained golden
data — the existing `TestData/LtxGolden/*.bin` fixtures only cover rope/proj_in/caption_proj/block0/
full_out, nothing at self-attn-only or cross-attn-only granularity within block0):
- Whether `LtxVideoModel.Attention`'s Q/K/V head-splitting/interleaving convention (implicit in how
  `WanAttention.TiledMultiHeadAttention` reads the flat `[seq, heads*headDim]` layout) matches the
  real reference's own reshape order — this is shared, previously-verified code (used correctly by
  Wan/Flux/SD3/HunyuanVideo/QwenImage/SDXL already), so a genuine bug here specific to LTX seems
  unlikely but hasn't been explicitly re-derived for this model's exact tensor layout.
- Cross-attention padding/masking: the real reference's `CrossAttention::forward` accepts an
  `attention_mask` parameter (passed through from `BasicTransformerBlock::forward`'s own
  `attention_mask` argument); `LtxVideoModel.Attention` has no masking support at all. The golden
  fixture's caption is a fixed 16 real tokens (no padding, per the fixture's own manifest), so this
  is unlikely to be block0's failure cause for THIS golden test specifically, but is still a real gap
  worth closing for production inference with variable-length/padded captions.
  - A regenerated, more granular golden dump (self-attn-only output, cross-attn-only output,
    FFN-only output, each captured separately within block0) would let this be bisected precisely —
    blocked by this environment having no Python available to run the dump script (same standing
    blocker noted elsewhere in this doc's history).
  - Alternatively, write a synthetic-weight structural test that manually re-derives block0's math
    in a completely independent second implementation (not reusing any of `LtxVideoModel`'s own
    helper methods) and diffs the two — would catch a bug that's specific to how this codebase's
    shared helpers (`Modulate`, `ApplyGatedResidual`, `GetModTriple`) are being called, as opposed to
    a bug in the helpers themselves (which are simple enough to have been read and manually verified
    correct against the reference formulas above).
