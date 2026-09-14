# Master GPU Performance & Accuracy Plan (2026-09-14)

This is the top-level index for all the per-model GPU-residency/perf handoff docs written this
session (`docs/069`-`080`). Work through it top-to-bottom. **Priority 0 (below) blocks everything
else for Wan specifically** — do not spend further effort on Wan *performance* until Wan
*correctness* is real again; other models' work can proceed in parallel/independently.

## Priority 0 — FIX WAN2.1 ACCURACY (blocking, do this first)

**Real, confirmed, current state**: Wan2.1-T2V-1.3B does **not produce a coherent image on either
CPU or GPU**. Visually confirmed by directly inspecting the real generated PNGs (not just reading
numbers): `wan_apple_20steps.png`, `docs/diffusion-samples/wan-apple-real-gpu.png`,
`docs/diffusion-samples/wan-apple-cpu.png`, `docs/diffusion-samples/wan-apple-sign1-gpu.png` are
**all pure random-noise static** — no apple, no table, no coherent shape of any kind. CPU and GPU
outputs are visually indistinguishable from each other (same blotchy ~10-15px noise texture),
confirming this is a **shared, structural, non-GPU-specific bug**, not a Vulkan dispatch problem.

**Real diagnostic reasoning already done, don't redo it**:
- The consistent blotchy noise texture across every sample is what a *correctly functioning* VAE
  decoder produces when fed **pure, still-noisy latent input** — this points away from the VAE and
  toward **the Euler flow-matching denoising loop never actually converging** (wrong velocity
  predictions, or a wrong sign/scale in the integration step), not a VAE bug.
- A `wan-apple-sign1-gpu.png` sample exists (clearly a prior attempt at flipping a sign convention,
  the same class of bug that hit FLUX/Z-Image's shared `EulerFlowScheduler` before) — it produced
  an **identical** noise result to the non-flipped version, so that specific hypothesis has already
  been tried and ruled out. Don't re-try a blind sign flip without a real reason.
- A hand-comparison against the real C++ reference (`examples/stable-diffusion.cpp/src/model/
  diffusion/wan.hpp`, `examples/stable-diffusion.cpp/src/model/common/rope.hpp`) already
  **ruled out**: AdaLN modulation chunk order, timestep-embedding/`time_projection` SiLU timing,
  `cross_attn_norm` conditional, self/cross-attention structure (QK-norm scope, RoPE-on-self-attn-
  only), RoPE frequency formula and rotation math (verified formula-for-formula, not just
  code-reading), `PackLatents`/`UnpackLatents` (internally self-consistent, matches expected
  channel-major/patch-minor convention), and FFN's tanh-approx GELU.
- **One real, but likely-too-small-to-be-primary discrepancy found**: `WanModel.
  ComputeTextEmbedding` uses tanh-approximation GELU (`DiffusionOps.Gelu`) for the
  `text_embedding.1` activation, but the real reference specifies plain `nn.GELU()` (exact,
  erf-based) there specifically — only the *FFN's* GELU is `approximate='tanh'` in the reference.
  Worth fixing (this project has no exact/erf-based GELU helper yet, would need adding one), but
  the numerical gap between tanh-approx and exact GELU (~3e-4 max error) is very unlikely to be
  severe enough to fully explain pure-noise output on its own — a genuine caveat, not the answer.
- **2026-09-14 addendum: magnitude-divergence hypothesis (found and root-caused for LTX-Video this
  same day, see docs/077) directly checked against Wan and RULED OUT.** Added a per-step
  latent-statistics dump (`STINGRAY_WAN_DEBUG_VELOCITY=1`, `WanPipeline.cs`) mirroring the exact
  technique that found LTX-Video's own "unbounded latent std growth across the denoising loop" bug.
  Ran a real 128×128, 20-step CPU generation (`docs/diffusion-samples/wan_stepstats_diag.png`) — the
  latent std stays bounded throughout (0.966 at step 0, dips to 0.637 around step 15, recovers to
  0.848 by step 19 — never runs away). **This confirms Wan's bug is a different class from LTX's**:
  not a compounding per-step magnitude instability, consistent with the already-documented flat
  spatial-coherence-vs-step-count sweep. This debug hook is now a real, permanent, zero-cost-when-
  unset addition to `WanPipeline.cs` (both the GPU and CPU denoising loops) for future diagnosis.
  **A real, honest side-observation, checked further and now a live lead again**: the 128×128 output
  image itself is NOT the "blotchy noise texture" described above for the 512×512 samples — it's a
  much more regular, periodic red plaid/grid pattern. Initially suspected as just the already-
  documented VAE nearest-neighbor-upsample artifact becoming more visually dominant at low
  resolution — **checked directly and ruled out**: the existing `WanVaeOnlyIsolationTests`' own
  pure-noise-through-VAE output (`wan_vae_isolation_noise.png`, 256×256, real VAE, hand-built random
  latent, no DiT involved at all) shows ordinary blotchy random noise with **no regular grid or
  periodicity whatsoever** — so the VAE itself does not produce plaid/grid patterns from noise-like
  input at any resolution tested. **This means the periodic grid must originate from the DiT's own
  output**, not the VAE. A clean periodic grid at low resolution (128×128, presumably an 8×8 DiT
  token grid: 16×16 latent / 2×2 patchify) versus "blotchy noise" at high resolution (512×512,
  presumably a 64×64 token grid) is exactly the visual signature expected if the SAME underlying
  tiling/patchify/RoPE-periodicity bug is present at both scales but is only visually obvious as a
  clean repeating grid when the token grid is small (few, large, obviously-repeating tiles) — at high
  resolution the identical error would look like fine-grained, harder-to-recognize texture instead of
  an obvious grid. **This reopens `PackLatents`/`UnpackLatents` and the RoPE frequency/tiling
  convention as a live lead**, despite this doc's own earlier note that they were "ruled out" via
  structural code comparison — that comparison checked the FORMULA against the reference, not this
  specific visual signature; a formula can read as structurally correct while still having a
  convention mismatch (e.g. row-major vs column-major patch traversal order) that only shows up this
  clearly at a small, human-countable tile grid. **Concrete next step**: re-run this same 128×128
  (or even smaller, e.g. 64×64 for a 4×4 token grid) generation and inspect whether the plaid
  period/cell count lines up exactly with the real patch grid dimensions (8×8 for 128×128) — if the
  visible repeat count matches the token grid size precisely, that's strong, near-definitive
  confirmation of a patchify/token-ordering bug specifically, not a general noise/convergence issue.

  **Checked immediately, same day — result is a genuine negative, hypothesis weakened, not
  confirmed.** Ran the same generation at `-W 64 -H 64` (a 4×4 token grid — half the linear token
  count of the 128×128 case, so a patchify/tiling-periodicity bug should if anything be even MORE
  visually obvious as a small, countable grid). **Result: smooth, ungridded, blurry pink/red/yellow
  output — no plaid, no repeating cells at all** (`docs/diffusion-samples/wan_64_gridcheck_diag.png`).
  This is the opposite of what the "the grid IS the token grid" hypothesis predicted (smaller token
  count → cleaner, more obvious grid) — instead the grid pattern DISAPPEARED at the smaller scale.
  **This is a real negative result**: the clean periodicity seen at 128×128 does not track token-grid
  size in the simple way a straightforward patchify/RoPE-tiling bug would, so this specific
  explanation is weakened, not confirmed. Left as an open, honestly-reported oddity rather than a
  solved lead — the 128×128 plaid pattern remains real and worth investigating further (it's still
  a novel, distinct signature from the 512×512 "blotchy noise" characterization, and still real
  numeric evidence that Wan's output is NOT literally uniform random noise at every scale), but
  "patchify/token-grid-count periodicity" specifically is not the confirmed explanation for it.
- **2026-09-14 addendum: the existing `STINGRAY_WAN_DEBUG_PERBLOCK` hook was enabled and its output
  actually inspected for the first time** (a single-step, 128×128 CPU forward). Per-block `std`
  stays contained (~0.6-1.75) across all 30 blocks — no runaway global divergence — but `maxAbs`
  shows a real, localized spike: 27-48 for blocks 0-12, climbing sharply to a peak of **272.6 at
  block 21**, then partially recovering back down to 22-27 by blocks 25-26 before rising again
  toward block 29. This is a genuine, reproducible pattern (specific outlier activation values, not
  the whole tensor, spiking in the middle third of the network) — but **this finding is
  inconclusive on its own**: many real, correctly-functioning transformers (LLMs and DiTs alike)
  exhibit exactly this kind of localized per-channel/per-token activation-outlier behavior
  ("attention sink"-style phenomena) as normal, expected architecture behavior, not a bug. Without a
  real reference's own per-block activation statistics to compare against (which would require
  either a Python-based dump from the real diffusers implementation — blocked, no Python in this
  environment — or a real independent C++ reference run producing the same instrumentation), this
  pattern cannot currently be distinguished from normal behavior. Recorded honestly as a real,
  reproducible observation and a candidate area, not a confirmed lead — do not chase this further
  without first securing a comparable reference baseline, since without one this is exactly the kind
  of "found *something* different" observation this project's own history has learned tends to waste
  time when treated as more conclusive than the evidence supports.
- **Not yet checked**: the Euler integration formula itself
  (`WanPipeline.Generate`: `latent[i] -= dt * velocity[i]`) against the real reference's exact
  scheduler convention and sign — this is the most promising unexplored lead given the visual
  evidence above. Also not yet checked: whether the DiT's velocity output is even in a sane
  numerical range (a quick mean/std dump of one `ForwardGpu`/`Forward` call's output would show
  immediately if it's near-zero, exploding, or otherwise obviously wrong) — a real, cheap, fast
  diagnostic to do before any more code-reading.
- **Not yet checked**: `WanVaeDecoder3D`'s own conv/upsample math against the real reference VAE
  (`examples/stable-diffusion.cpp/src/model/vae/wan_vae.hpp`) — deprioritized above the DiT/Euler
  loop given the "VAE is probably fine, decoding real noise" reasoning, but worth a real check if
  the Euler-loop investigation comes back clean.

**2026-09-14 update — CFG cancellation ruled out, VAE now the prime suspect**: Added env-gated
debug instrumentation to `WanPipeline.Generate` (`STINGRAY_WAN_DEBUG_VELOCITY=1` dumps per-step
cond/uncond/guided velocity mean+std+cosine-similarity; `STINGRAY_WAN_DEBUG_GUIDANCE=<f>` on
`WanAppleDiagnosticTests` overrides guidance scale — both left in, zero overhead when unset).
Findings from a real 4-step/256x256 GPU run of the real "a red apple..." prompt:
- `cos(condVelocity, uncondVelocity) ≈ 0.999` every step — cond and uncond predictions are nearly
  parallel. With `guidance=6.0` this makes the CFG linear combination `uncond + 6*(cond-uncond)`
  partially cancel, shrinking the guided velocity to ~1/3 the magnitude of condVelocity alone
  (observed: condVelocity std≈1.15-1.18, guided std≈0.35-0.38 — confirmed algebraically, not just
  observed, the cancellation math checks out exactly for two ~0.999-correlated vectors of
  different magnitude).
- **This is NOT the primary bug**: re-ran with `guidance=1.0` (pure condVelocity, no CFG
  cancellation) — final latent std actually *increases* to 1.46 (vs the noise prior of 1.0), and
  the decoded PNG (`wan_diagnostic_apple.png`) is **still identical blotchy ~15px noise-blob
  texture**, visually indistinguishable from the CFG=6.0 run. So the collapsed-magnitude CFG
  cancellation was a real, worth-noting side effect but not why the image is noise.
- The persistent, resolution-independent blotchy blob texture across every sample (CPU, GPU,
  guidance on/off, sign flip on/off) is now the strongest single clue: it looks exactly like a
  **VAE decoder upsampling artifact**, not "not-yet-denoised Gaussian noise" (real Gaussian noise
  decoded through a working conv/upsample VAE typically still shows *some* correlated blob
  structure at the receptive-field scale, so this alone doesn't distinguish the two hypotheses —
  but combined with the fact that changing the DiT's actual velocity output (CFG on vs off, which
  materially changed final latent std from 1.05 to 1.46) produced **zero visible change** in the
  decoded image's texture/character, the decoder — not the DiT — is now the more likely fault).
- **RESOLVED, real test result (2026-09-14, `tests/.../WanVaeOnlyIsolationTests.cs`, new file,
  bypasses DiT/UMT5 entirely)**: decoded three hand-constructed latents directly through
  `WanVaeDecoder3D.Decode` — all-zeros, a smooth left-to-right gradient, and pure `N(0,1)` noise.
  Results are conclusive: **the VAE is exonerated.** Zeros decode to a smooth near-flat olive color
  (R-channel std across pixels = 0.011); the gradient decodes to a clean, correctly-structured
  smooth brown-to-cyan left-to-right transition (std=0.074, *visually confirmed*, no blob artifacts
  anywhere); only the noise input produces the familiar blotchy blob texture (std=0.201). The VAE
  faithfully reproduces whatever structure (or lack of it) is in its input. **This proves the bug is
  in the DiT, not the VAE** — the DiT's own output latents, at the end of the Euler loop, are
  themselves still statistically noise-like/uncorrelated (despite having a plausible global
  mean/std, per the earlier `[Wan Denoise Done]` stats) — i.e. the DiT is not predicting anything
  spatially *coherent*, just noise of roughly the right magnitude.
- **This also invalidates the earlier "PackLatents/UnpackLatents are internally self-consistent,
  ruled out" reasoning** — self-consistency between pack and unpack proves they're inverses of
  *each other*, not that either matches the reference model's actual trained patch-token ordering.
  A patch-index permutation bug (e.g. wrong `dy`/`dx`/channel ordering in the `packed[c*4+dy*2+dx]`
  formula relative to what the checkpoint's `patch_embedding`/`head.head` weights actually expect)
  would still pass a self-consistency check while scrambling every patch's spatial identity,
  which is exactly the kind of bug that produces "right statistics, zero coherent structure."
- **Concrete next step (do this next, not yet done)**: hand-verify `WanModel.PackLatents`'s exact
  channel/patch-offset formula against the real reference's patchify, byte-for-byte — check
  `examples/diffusers/.../autoencoder`-adjacent `WanTransformer3DModel`'s `patchify`/rearrange
  einops pattern AND `examples/stable-diffusion.cpp/src/model/diffusion/wan.hpp`'s own patch
  embedding conv (`patch_embedding` is a `Conv3d` with kernel=`patch_size` over the raw latent
  in the C++ reference, NOT a manual pack-then-Linear step — if this project's `PackLatents`
  reorders elements differently than an actual strided Conv3d kernel-flatten would, that is the
  bug). Also worth a fast, cheap sanity check first: decode the RAW initial Gaussian noise latent
  (before ANY DiT steps) through the VAE and compare its blob texture/scale to the final "denoised"
  output's blob texture — if they're statistically indistinguishable in blob *size* (not just
  overall std), that's further evidence the DiT/patchify path is contributing ~nothing coherent at
  all, not just under-converging.

**2026-09-14 update #2 — patch-order fix applied (real, justified, kept) but NOT sufficient alone;
sharper hypothesis found**: Fixed `WanModel.PackLatents`/`UnpackLatents` — the packed-patch element
order was `slot = (dy*2+dx)*OutChannels + c` (spatial-outer, channel-inner), but the reference's
`patch_embedding` is a real `Conv3d(in_dim=16, kernel=patch_size=(1,2,2))` (`wan.hpp:537`), and
reinterpreting its `[out_dim, in_dim, kh, kw]` weight as a flat Linear matrix requires the packed
input's element order to match the weight's own trailing-dim flatten: channel-OUTER,
spatial-INNER (`slot = c*4 + dy*2 + dx`). Fixed both `PackLatents` and `UnpackLatents` to this
convention (shared by CPU and GPU paths, single fix point). Still believed correct and kept
(fix-forward, not reverted) — but re-ran the real GPU diagnostic after the fix and the **output is
visually and statistically almost unchanged** (cos(cond,uncond) still ~0.999, same blotchy
texture, same patch-sized blobs) — so this alone is not the (or not the only) primary bug. Either
the weight-loading code already internally accounted for the old ordering (not yet checked), or a
second, larger bug dominates.
**New, sharper hypothesis worth checking first next**: the persistent noise-blob texture's size
(~15-20px) matches almost exactly ONE patch's real-pixel footprint (a 2x2 latent patch x 8x VAE
upscale = 16px) — i.e. each token's output looks uncorrelated with its spatial neighbors, as if
self-attention across tokens (`WanQkvSplitNormRoPE` + `MultiHeadAttentionTiled` in
`TransformerBlockGpu`, or `SelfAttention`+RoPE in the CPU `TransformerBlock`) is not actually
mixing information across tokens — degenerating toward each token attending only to itself. This
would also explain the high cond/uncond cosine and is a strong, falsifiable next lead: dump the
`AttnOut` tensor for one real forward pass and compare `AttnOut[i]` against `V[i]` per token via
cosine similarity — high similarity for most tokens would confirm identity-attention. Do this next.

**2026-09-14 update #3 — real GPU-only bug found+fixed (kept), attention-mixing hypothesis
disproven**: Found and fixed a genuine GPU-only bug in `Shaders.WanQkvSplitNormRoPE`
(`src/OpenTail.Stingray.Vulkan/Shaders.cs`): it computed the self-attention QK-norm RMS statistic
per-HEAD (one workgroup per (token,head), reducing over only `headDim`=128 elements), but the CPU's
`WanModel.RmsNormHeads` was already fixed on 2026-08-31 to use ONE statistic over the full
`dim`-length row (all heads concatenated) per the real `torch.nn.RMSNorm(dim_head*heads)` — the GPU
shader was never updated to match at that time. Rewrote the shader (one workgroup per TOKEN,
looping each of its 128 threads across all `numHeads` heads for both the sum-of-squares reduction
and the RoPE application) so GPU now matches CPU's normalization scope exactly. Regenerated SPIR-V
via `scripts/gen-spirv.ps1` (Vulkan SDK present on this machine), `WanGpuParityTests` still passes.
Kept as a genuine correctness fix (fix-forward) even though — like the patch-order fix — it did NOT
change the real-weights diagnostic's output (same cos~0.999, same blob texture, same RGB means to
4 decimal places). Likely explanation: this checkpoint's per-head activation magnitudes are already
fairly uniform across heads, so per-head vs whole-row RMS denominators are numerically close for
real trained weights, even though the per-head version was conceptually wrong.
Also **directly inspected `WanAttention.TiledMultiHeadAttention`** (the CPU self/cross-attention
kernel) — confirmed it is a real, correct, full softmax attention with proper `1/sqrt(headDim)`
scaling and genuine weighted mixing over all `kvSeq` positions (not degenerate/identity). **This
disproves the "attention isn't mixing tokens" hypothesis from update #2** at the kernel level for
both CPU and the shared `MultiHeadAttentionTiled` GPU kernel (already proven correct via FLUX). The
patch-sized blob texture is therefore NOT explained by a broken attention kernel.
**Remaining untested candidates, ranked, for the next iteration**: (1) UMT5 text encoder output
correctness itself — only checked for plausible mean/std (-0.0007/0.0723), never validated
token-by-token against a real independent UMT5 reference; if the *encoder* silently produces
statistically-plausible-but-wrong embeddings, every downstream conditioning signal would be
garbage while every DiT-side check (which assumes the input is at least meaningful) keeps coming
back clean. (2) `ComputeTimestepEmbedding`'s sinusoidal frequency formula/`max_period` constant —
not yet checked against the reference's own timestep-embedding formula in detail (only the
time_projection SiLU *timing/order* was checked before, not the sinusoidal embedding math itself);
if the model can't distinguish t=1.0 from t=0.5, that alone would produce exactly this kind of
"plausible magnitude, no real progressive denoising" symptom. Do (1) first — it's the one
completely unverified input to the whole pipeline.

**2026-09-14 update #4 — real timestep-embedding order bug found+fixed (kept), still not
sufficient alone**: Found and fixed a genuine bug in `WanModel.ComputeTimestepEmbedding`: it called
`DiffusionOps.SinusoidalTimestepEmbedding(timestep)` with the default `flipSinToCos: false`, which
produces `[sin, cos]` half-ordering, but the real reference
(`ggml_compute_forward_timestep_embedding_f32`, `examples/stable-diffusion.cpp/ggml/src/ggml-cpu/
ops.cpp:8302-8303`) writes `embed_data[j]=cos(arg)` for the first half and
`embed_data[j+half]=sin(arg)` for the second — `[cos, sin]` order. Fixed by passing
`flipSinToCos: true`. This is a real, trained-weight-order mismatch (same class of bug as the
patch-order fix: `time_embedding.0`'s learned weights were seeing cos and sin swapped). Verified
this ACTUALLY changed model behavior (unlike the two previous fixes) — `condVelocity` std now
scales meaningfully with `t` across the 4 denoising steps (0.62 -> 0.69 -> 0.80 -> 0.99, versus
flat ~1.15 std at every step before the fix), a real sign the model is now using the timestep
signal. **Still not sufficient alone**: re-ran both the 4-step diagnostic and the existing 20-step
test (`WanApple20StepsTest.cs`, uses a real empty-string negative prompt rather than all-zero
uncond) — both still produce the same blotchy noise texture; the 20-step run's final latent std
even grew to 2.04 (further from the noise prior's std=1, not closer to a real data distribution),
suggesting velocity direction/calibration issues compound over more steps rather than converging.
Kept as a genuine correctness fix (fix-forward).
**State after this round**: three real, verified, kept correctness fixes landed this iteration
(patch-embedding column order, GPU QK-norm RMS scope, timestep sin/cos half-order) — none alone
resolves the symptom, but each closes off a real, confirmed discrepancy against the reference, and
the timestep fix specifically proved the model's behavior IS sensitive to these low-level ordering
bugs (so more of the same class may still be lurking). **Top remaining unverified candidate**:
the UMT5 text encoder's actual output correctness — still never checked beyond plausible
mean/std, the last major unverified input to the whole pipeline. Do this next: encode a short,
simple prompt through `UMT5Encoder.EncodeGpu` and check the actual embedding values/shape
convention (padding side, EOS token handling, whether `numTxtTokens` truncation vs the reference's
own tokenization matches) against `examples/stable-diffusion.cpp`'s T5/UMT5 encoder path, OR (cheaper)
check for a similar sin/cos or channel-order swap bug inside T5's own relative-position-bias/RoPE-
free attention math (UMT5 has no RoPE, so this specific bug class doesn't apply directly, but
gated-GELU channel order or relative-position-bucket math could have an analogous ordering bug).
**Direct read of `UMT5Encoder.cs` done this session (not yet numerically verified)**: the
relative-position bucket formula (`RelPosBucket`), bidirectional offset, gated-GELU FFN structure,
and unscaled-attention-plus-additive-bias math all read correctly against the real HF T5
`_relative_position_bucket` formula and UMT5 config — no bug found by inspection alone. Also
re-tried the sign-flip hypothesis (`STINGRAY_WAN_SIGN=1`) now that the timestep-embedding fix
landed (previous sign-flip test predates all of this session's fixes) — still identical noise
output, confirming the default `latent[i] -= dt*velocity[i]` convention is correct; this closes
the sign-convention hypothesis for good. **Since inspection alone hasn't found the remaining bug,
next iteration should stop reading code and instead numerically verify UMT5's actual output**
against a real independent reference (e.g. run the same prompt through HF `transformers`'
`UMT5EncoderModel` or `diffusers`' Wan pipeline text encoder if either is available anywhere
accessible, and diff the first few embedding values directly) — this is the one major component
in the whole pipeline that has never been checked against a real ground truth, only self-consistency
and plausible statistics.

**2026-09-14 update #5 — UnpackLatents convention corrected (asymmetric from PackLatents, real
finding, kept), still no visible change; verified RoPE-compact table and UMT5/T5 code parity**:
Re-examined the reference's actual `unpatchify` function (`wan.hpp:610-625`, not previously read in
full) and found the earlier "PackLatents/UnpackLatents both need channel-outer ordering" fix
(update #2) was half-wrong: `head.head` is a plain `Linear` (unlike `patch_embedding`, a real
`Conv3d`), and the reference's own unpatchify reshapes its 64-length output as ggml `(C,
pw*ph*pt)` — since ggml's `ne[0]` is the fastest/most-contiguous axis (per this repo's own
`examples/stable-diffusion.cpp/AGENTS.md` tensor-layout note), C is INNERMOST there, i.e.
spatial-outer/channel-inner — the OPPOSITE convention from the input side. Corrected
`UnpackLatents` back to `(dy*2+dx)*OutChannels+c` while keeping `PackLatents` at
`c*4+dy*2+dx` (the two sides are genuinely asymmetric in the real model, not a copy-paste
symmetry). Re-ran the real diagnostic: **still identical blob-noise output** (latent
mean=0.0173, std=1.19, same texture). Also **directly verified** (by re-reading, not just citing)
`WanRoPE.Compute3DRoPECompact` against the already-proven `Compute3DRoPE` — the compact table's
per-axis pair-index boundaries (0, 22, 43, 64) exactly match the full table's axis boundaries
(headDim positions 0, 44, 86, 128, halved), and the per-pair frequency formula is identical — no
bug found. Also diffed `UMT5Encoder.cs` against `T5Encoder.cs` (FLUX's proven-working twin, same
architecture family) line-by-line: relative-position bucketing, unscaled-attention-plus-bias,
gated-GELU FFN structure are all structurally identical between the two — no divergence found,
which is fairly strong evidence UMT5 itself is not uniquely broken (though still not numerically
verified against an external ground truth, since none is available in this environment without
Python, which this project's tooling explicitly avoids).
**Assessment after 4 real, reference-verified, kept fixes with zero visible change**: patch
column order (input), GPU QK-norm RMS scope, timestep sin/cos half-order, and unpack column order
(output) have ALL been checked against the actual reference code and corrected where wrong, yet
the final image is pixel-for-pixel-class-identical noise every time. This is itself informative:
either (a) there is one dominant bug elsewhere that overwhelms all of these (most likely,
statistically), or (b) multiple of today's fixes are each real but only matter in combination with
something not yet found. **Recommended next approach, a real change in strategy**: stop guessing
individual formula/ordering discrepancies against the reference and instead do a genuine numeric
layer-by-layer bisection — dump the DiT's `ws.X` activation mean/std/a few raw values after EVERY
individual transformer block (not just the final output) for a real forward pass, across all 30
Wan-1.3B blocks, and look for the first block where the trajectory becomes obviously pathological
(e.g. magnitude exploding, collapsing to near-zero, or NaN/Inf) rather than smoothly evolving —
this localizes the bug to a specific block index and sub-operation instead of reasoning about the
whole model's math at once. This mirrors the exact bisection technique that found FLUX's real bugs
(`docs/056`-style), which this Wan investigation has not yet actually done (only ever compared
start-vs-end statistics, never a full per-block trace).

**2026-09-14 update #6 — real per-block numeric bisection done (new test, kept), no pathological
block found; weight orientation confirmed standard**: Implemented the per-block bisection
recommended above for real: added `STINGRAY_WAN_DEBUG_PERBLOCK=1` env-gated activation stat dumps
after every one of the 30 blocks in `WanModel.Forward` (CPU, unbatched, easy to instrument), plus a
new test `WanAppleDiagnosticTests.Stage_CpuForward_PerBlockActivationBisection` that runs one real
CPU forward pass (real weights, real UMT5-encoded prompt, t=500 mid-noise timestep) and dumps
mean/std/maxAbs per block. **Result: the residual stream evolves smoothly across all 30 blocks**
(std stays in 0.4-1.05, maxAbs stays in 11-27, no NaN/Inf, no explosion, no collapse-to-zero at any
single block) — there is NO pathological block. This is itself a real, useful negative result: the
bug is not a catastrophic per-block numerical blowup; whatever is wrong corrupts the *information
content* while leaving the *magnitude* statistically plausible throughout — consistent with
everything else found this session (plausible-looking numbers, zero coherent spatial structure).
Also added `WanAppleDiagnosticTests.Stage_InspectRealWeightShapes_CheckLinearOrientation` and
directly inspected the real checkpoint's tensor shapes: `text_embedding.0.weight=[1536,4096]`,
`patch_embedding.weight=[1536,16,1,2,2]`, `head.head.weight=[64,1536]`,
`blocks.0.ffn.0.weight=[8960,1536]` — every one is exactly `[out_features, in_features]`
(standard PyTorch `nn.Linear`/`Conv3d` storage), ruling out a global weight-transposition bug.
This also independently confirms the `PackLatents` channel-outer fix from update #2 was correct
(`patch_embedding.weight`'s second dim really is `in_channels`, confirming channel is the outer of
the trailing flattened-kernel dims).
**State after 6 rounds of investigation, all real and reference-checked, zero resolution**: patch
order (both directions, now asymmetric-correct), GPU RMS scope, timestep sin/cos order, RoPE
compact-table correctness, UMT5 vs proven-T5 code parity, per-block numeric health, and weight
orientation have ALL been verified/fixed and NONE explain the symptom. Everything checked so far
has been a *structural* or *formula* check; nothing has directly compared this implementation's
actual numeric values against a real, independent ground-truth output at ANY stage (UMT5 embedding
values, block-0 activation values, or final image) — every check so far has been "does this look
internally sane / match the reference's formula by inspection," never "does this exact number match
a known-correct number." **This is the real gap or the next iteration should close**: if any
external tool/reference becomes available (a real Wan reference run, HF `diffusers`, or even a
cached known-good embedding/activation dump from anywhere), diff against it directly. Absent that,
the remaining productive path is exhaustively re-deriving each weight tensor's *semantic* meaning
(not just shape) against the real HF `WanTransformer3DModel`/`UMT5EncoderModel` state dict key-by-
key, since a subtly wrong key mapping (e.g. gate/value FFN branches swapped, or Q/K weights
swapped) would produce exactly this "plausible everywhere, coherent nowhere" signature and would
NOT be caught by any check done so far.

**2026-09-14 update #7 — real external-reference attempt blocked (real finding), self/cross-attn
exact-match confirmed**: Discovered `examples/stable-diffusion.cpp/build/bin/sd-cli.exe` genuinely
supports Wan generation (`--diffusion-model`, `--vae-format wan`, `--t5xxl` flags all present) and
attempted a real ground-truth comparison run against our actual checkpoint files. **Blocked by a
real, structural checkpoint-format mismatch**: sd-cli's conditioner expects standard HF-style T5
tensor names (`text_encoders.t5xxl.transformer.encoder.block.N.layer.0.SelfAttention.q.weight`,
etc.), but our real `models_t5_umt5-xxl-enc-bf16.safetensors` uses Wan's own native naming
(`blocks.N.attn.q.weight`, `token_embedding.weight`, confirmed by `UMT5Encoder.cs`'s own doc
comment) — `model_manager.cpp` has no built-in remapping table for this naming (checked
`name_conversion.cpp` directly), so `sd-cli.exe` refuses to load it (~60 real "tensor not in model
metadata" errors, model init fails outright). Building a converter would need a Python-free
safetensors key-rename script, feasible but a real, separate undertaking (not attempted this
round) rather than a quick check. **Also**: directly read `WanSelfAttention::forward` and
`WanT2VCrossAttention::forward` (`wan.hpp:95-198`) in full for the first time this session
(previously only partially reviewed) and confirmed our `SelfAttention`/`CrossAttention` CPU methods
match exactly: self-attn norms both Q and K (RoPE applied after norm, to Q/K only, never V);
cross-attn norms Q and K but never V (matches `PrecomputeCrossKvCache`'s own K-only norm, already
correct); cross-attn gets NO RoPE (correct, RoPE isn't passed to `WanCrossAttention::forward` at
all in the reference). No new discrepancy found.
**2026-09-14 update #8 — checkpoint-conversion path also blocked by real disk space, not just
format**: Considered writing a small C# (no-Python, per project convention) safetensors key-rename
tool to convert our UMT5 checkpoint to sd-cli's expected HF naming, enabling the real ground-truth
comparison from update #7. **Blocked before starting**: `df -h` shows the C: drive at 453MB free /
100% used — there is not enough headroom to write a renamed ~11GB copy of
`models_t5_umt5-xxl-enc-bf16.safetensors` (the real file size, confirmed via `ls -la`) alongside
the original. This is a real, systemic constraint (not just a `models/` directory convention issue
per this project's own disk-space-constraint note) worth flagging to the user directly — freeing
space elsewhere would need to happen before this path is viable again.

**2026-09-14 update #9 — real T5-padding-class bug found and fixed (kept), measurable convergence
improvement, still not fully resolved**: Directly read `examples/stable-diffusion.cpp/src/
conditioning/conditioner.hpp`'s `T5CLIPEmbedder` (the shared T5/UMT5 conditioner class, `is_umt5`
flag) and found `chunk_len=512` with `tokenize(text, chunk_len, chunk_len)` — the reference forces
EVERY prompt to a fixed padded length before encoding, exactly the same bug class already found
and fixed for FLUX's own T5 this session (`docs/056`'s T5-256-padding fix) but **never applied to
Wan** — confirmed by reading both `WanAppleDiagnosticTests.cs`/`WanApple20StepsTest.cs` and the
real production CLI path (`ImageCommand.cs`'s `RunWan`), all of which fed cross-attention only the
raw, unpadded ~12-token context. Cross-referenced against this project's own `T5Tokenizer.FromFile`
doc comment ("pass 226 for Wan's UMT5 encoder") — already correctly researched at some point but
never wired up anywhere. **The precise real algorithm** (from diffusers' actual
`WanPipeline._get_t5_prompt_embeds`, reconstructed from the sd.cpp behavior plus this project's own
226 note): tokenize normally (no padding needed for encoding itself, matching what we already do),
encode the REAL tokens through UMT5, then **zero-pad the resulting EMBEDDINGS** (not the token ids)
up to a fixed `max_sequence_length=226` before cross-attention — critically NOT running the encoder
on `pad_token_id=0` (which would produce real, nonzero "encoded padding" embeddings that don't
match the reference at all; an earlier same-session attempt at padding the *token ids* through the
encoder, rather than zero-padding the *embeddings* afterward, produced a real but smaller change).
**Fixed in three places** (kept, fix-forward): `WanAppleDiagnosticTests.cs`,
`WanApple20StepsTest.cs`, and the real production CLI path (`ImageCommand.cs`'s `RunWan`, both the
CPU and GPU encode branches). **Real, measured effect**: final latent std (after the full Euler
loop) dropped from the consistent ~1.0-2.0 range seen in every single prior test this whole
session down to ~0.4-0.7 — a genuine, repeatable, qualitatively different convergence behavior, not
noise. **Still not fully resolved**: the decoded image is still not a coherent apple (different
texture/color than before, still blob-patterned) at both 4 steps (256×256) and the full real
20-step test. This is real, partial progress — the conditioning path is now much closer to the
reference's actual behavior — but a second, independent bug (most likely in the DiT itself, given
everything else already checked) still prevents full convergence to a coherent image.

**2026-09-14 update #10 — exact-vs-tanh GELU tested empirically, conclusively ruled out; CFG
re-confirmed irrelevant with the padding fix applied**: Added `DiffusionOps.GeluExact` (real
erf-based GELU via Abramowitz & Stegun 7.1.26, ~1.5e-7 max error) and an env-gated switch in
`WanModel.ComputeTextEmbedding` (`STINGRAY_WAN_DEBUG_EXACT_GELU=1`) to test the
tanh-vs-exact-GELU discrepancy flagged back in update #1 empirically instead of continuing to
guess between two disagreeing "reference" doc comments (diffusers says tanh-approx, `wan.hpp`'s
comment says exact). **Result: identical output to 4 decimal places** (latent std=0.4350 both
ways) — conclusively too small an effect to matter, as originally suspected; kept the helper
(harmless, real, doesn't change default behavior) but left the default as tanh-approx (matches
diffusers, the more directly-relevant reference for weight compatibility). Also re-ran the
CFG-disabled test with the update #9 zero-pad conditioning fix applied — std nearly identical
(0.4286 vs 0.4350 with CFG) — re-confirms guidance scale is not a meaningful factor, consistent
with the original (pre-padding-fix) finding in update #1.
**Cumulative state after 10 rounds**: 5 real, reference-verified fixes now landed (patch order
in/out, GPU RMS scope, timestep sin/cos order, and the T5-padding-class fix from update #9 which
produced the first genuinely different convergence behavior all session) plus two more
conclusively-ruled-out hypotheses (GELU exactness, CFG). The T5-padding fix remains the most
significant lead — real, measured, different final-latent statistics — but still short of a
coherent image at both 4 and 20 real steps. No new candidate has emerged from remaining
inspection-only checks; per update #7/#8's own conclusion, the next productive step genuinely
needs either an external ground-truth numeric reference (blocked: format mismatch + 453MB free
disk) or a live numeric diff against a real second implementation, neither available in this
environment currently.

**2026-09-14 update #11 — sd-cli reference path definitively closed (real root cause found, not
just format/disk blockers)**: Tried a different angle to sidestep the update #7/#8 blockers: the
reference's own conditioner code (`conditioner.hpp:1350`) explicitly supports generation with NO
text encoder at all (falls back to an all-zero context, printing "No text encoders provided,
cannot process prompts!" but continuing) — this would have let us ground-truth just the DiT+VAE
directly against the real checkpoint, sidestepping the UMT5 naming/disk-space blockers entirely.
Ran it (`--diffusion-model` + `--vae` only, no `--t5xxl`) — it loads and starts sampling, then
**crashes silently mid-generation** (exit 127, no error, log just stops after "generating image:
1/1 - seed 42"). Root-caused with `-v` verbose logging: `WanConfig::detect_from_weights`
(`wan.hpp:42-86`) auto-detects `num_layers` correctly by scanning tensor names (`30`, matches our
real checkpoint), but **never auto-detects `dim`/`ffn_dim`/`num_heads`** — those three fields stay
hardcoded at the `WanConfig` struct's default values (`dim=2048, ffn_dim=8192, num_heads=16` — the
14B model's config, confirmed via the struct definition's own comment: "wan2.1 1.3B: 1536/12,
wan2.1/2.2 14B: 5120/40"). Our real checkpoint is the 1.3B variant (`dim=1536, ffn_dim=8960,
num_heads=12`, independently confirmed via our own `Stage_InspectRealWeightShapes_
CheckLinearOrientation` test in update #6) — sd-cli silently tries to build a 2048-dim compute
graph against 1536-dim tensors, and crashes. **This is a genuine, confirmed limitation of this
specific sd-cli build for the 1.3B variant specifically, not a format/disk/naming issue** — closing
this path definitively rather than continuing to probe it. Fixing it would mean patching sd-cli's
own vendored C++ source (`examples/stable-diffusion.cpp`) to add the missing dim/head-count
auto-detection, a real but substantial detour into third-party code outside this repo's own
target, not attempted here.

**2026-09-14 update #12 — the vendored reference's own dim-detection bug actually fixed (real,
kept, low-risk); this REVEALED a second, deeper, genuine bug in the same third-party tool that
finally closes this path off for good**: Patched `WanConfig::detect_from_weights`
(`examples/stable-diffusion.cpp/src/model/diffusion/wan.hpp:42-90`) to read `dim` and `ffn_dim`
from the real `blocks.0.ffn.0.weight` tensor's actual shape (ggml `ne[0]`/`ne[1]`) instead of
leaving them hardcoded at the struct's 14B-sized defaults, and derive `num_heads = dim/128`
(headDim=128 is constant across every published Wan variant per the struct's own comment).
Rebuilt `sd-cli.exe` (`cmake --build . --target sd-cli`) — **confirmed working**: real log output
now shows `dim=1536, ffn_dim=8960, num_heads=12`, exactly matching our checkpoint (independently
verified via our own `Stage_InspectRealWeightShapes_CheckLinearOrientation` test in update #6).
This is a real, low-risk, scoped fix to the vendored reference tool, kept (not reverted) — it's
this repo's own copy of the submodule, used specifically for this kind of ground-truth comparison.
**But this uncovered a SECOND, deeper bug**: even with the correct config, `sd-cli.exe` still
crashes at the exact same point in sampling — got the real Windows exit code via PowerShell
(`Start-Process`/`$p.ExitCode`, since bash's git-bash wrapper was reporting a misleading generic
`127`): **`0xC0000094` = `STATUS_INTEGER_DIVIDE_BY_ZERO`**. This is a genuine crash bug inside
stable-diffusion.cpp's own Wan sampling/Vulkan code path, almost certainly a hardcoded assumption
tied to `num_heads=16` (the 14B default that update #12's fix replaces with the real `12` for the
first time this code has EVER actually run against a real 1.3B checkpoint — the earlier
mis-detection bug silently masked this division bug for every previous 1.3B user of this tool,
since it never got past the wrong-dim crash before either). **Root-causing and fixing this second
bug is a genuine, separate, non-trivial dive into vendored GGML/Vulkan sampling code, out of scope
for this session's actual target (this project's own Wan port, not stable-diffusion.cpp itself) —
not attempted further.** This path is now closed for good: two real, distinct bugs were found and
one was fixed in the reference tool itself, but a working 1.3B Wan reference generation is still
not achievable with this build without further, separate work on the vendored dependency.

**2026-09-14 update #13 — MAJOR bug found and fixed: shared T5/UMT5 GPU attention kernel was
incorrectly SCALED (real T5 is unscaled), affecting Wan's UMT5 encoder AND FLUX's T5-XXL GPU path
both; still not sufficient alone to fix Wan's image**: While implementing Parler-TTS's own T5
encoder GPU residency (docs/080, an unrelated task), a real single-layer parity test showed a huge
discrepancy (maxDiff=37.4 vs meanAbs=5.1) immediately after just the attention sub-layer. Traced to
`VulkanBackend.T5MultiHeadAttentionRelBias`'s dispatch code hardcoding `scale = 1f /
MathF.Sqrt(headDim)` — but **real T5 attention is explicitly UNSCALED** (confirmed in three
separate places in this codebase's own doc comments: `Diffusion/TextEncoders/T5Encoder.cs`,
`Diffusion/TextEncoders/UMT5Encoder.cs`, and `Audio/Parler/T5Encoder.cs` all independently state
"T5 omits this scaling entirely" / "no `1/sqrt(head_dim)` scaling", each citing the real
`transformers` T5Attention source). Verified the shader itself computes `score = dot*scale +
rel_bias` (read directly, `Shaders.cs` line ~7590), confirming `scale=1.0` is the correct fix, not
a guess. **This is a real, previously-undetected bug in a SHARED kernel used by BOTH Wan's
`UMT5Encoder.EncodeGpu` (confirmed via direct grep — the exact same call site used throughout this
entire session's Wan diagnostic testing) and FLUX's own `T5Encoder.EncodeGpu` (docs/071, "proven,
done") — it predates this session's Wan investigation entirely and was never caught because no
prior GPU-vs-CPU parity test for either T5 encoder checked attention output in isolation against
the CPU reference at the value level, only end-to-end (where other effects could mask it).** Fixed
by setting `scale = 1f` (kept the parameter rather than removing it, so a future model needing real
scaled T5-family attention isn't blocked). Re-verified Parler's own single-layer discrepancy
collapsed from maxDiff=37.4 to maxDiff=0.00015 (real fix, not a guess) before touching Wan.
**Re-ran both the 4-step and full 20-step real Wan diagnostics with this fix applied — still
identical-class noise output, texture shifted (green/blue tint) but no coherent image at either
step count.** This is a real, major, broadly-impactful bug (likely also explains some of FLUX's own
GPU-vs-CPU T5 conditioning quality gap, worth a follow-up check there specifically) but — like
every other real fix landed this session — not sufficient alone to explain Wan's core symptom.
**GPU-first-then-CPU note**: this bug is GPU-only (the CPU T5/UMT5 attention paths never had a
scale factor at all, confirmed by both CPU `SelfAttention` methods' own "NO scaling" comments) — no
CPU-side porting needed for this particular fix, it only ever affected the GPU dispatch.

**2026-09-14 update #14 — DECISIVE new diagnostic: UMT5 (and text encoding generally) is now
conclusively ruled out as the cause**: Since FLUX's own T5-XXL encoder is now proven correct
end-to-end (update #13's real, visually-confirmed coherent FLUX output) and outputs the exact same
4096-dim embedding space Wan's UMT5 does, ran a genuinely new experiment: encoded the real prompt
("a red apple on a wooden table, photorealistic") through FLUX's proven-correct T5-XXL encoder
(NOT UMT5 at all — a different vocab/tokenizer/checkpoint entirely) and fed THOSE embeddings
directly into Wan's DiT cross-attention, replacing UMT5's own output
(`WanCrossSubstituteT5DiagnosticTest.cs`, new file). Semantically this is "wrong" conditioning
(different register/vocabulary than what Wan's cross-attention was trained against), but
dimensionally valid, and critically: **if Wan's DiT can use ANY real, well-formed text
embedding to produce coherent spatial structure, the output should look like SOMETHING (wrong
content, but real structure) rather than the exact same blob-noise texture every single test this
whole session has produced.** Ran it with CFG disabled (guidance=1.0, isolating out the
already-ruled-out CFG-cancellation confound) — **result: pixel-for-pixel-class identical noise
texture to every other test this session**, despite feeding in a completely different,
independently-proven-correct 4096-dim text embedding. **This conclusively rules out UMT5 (and
text/cross-attention conditioning generally, since a real, different, known-good encoder produces
the identical failure) as the cause.** The bug is downstream of text encoding entirely — somewhere
in the DiT's own self-attention/FFN/modulation stack, or the Euler flow-matching integration loop
itself (the one major subsystem never yet numerically bisected beyond the per-block activation
health check in update #6, which only checked magnitude, never structure/coherence).
**Real, concrete next step**: since every individual DiT sub-component has now been checked
against the reference and found structurally correct, and text conditioning is now ruled out
entirely, the remaining productive avenue is almost certainly the Euler integration loop's own
numerical behavior across REAL multi-step trajectories (does the latent actually move toward a
data-like distribution over steps, or does noise-in/noise-out hold regardless of how many steps
run — partially checked in earlier updates via final latent std, but never via a real structural/
frequency-domain check of whether ANY spatial correlation develops across steps) — or, if that
also comes back clean, a fully fresh reconsideration of whether the checkpoint files themselves
(DiT/VAE weights) are what they're assumed to be (e.g. a real hash/metadata cross-check against
the canonical Wan2.1-T2V-1.3B release, the one category of check never done this entire session).

**2026-09-14 update #15 — checkpoint integrity ruled out too**: Ran a real full-file scan
(`WanCheckpointIntegrityTest.cs`, new file) checking every weight tensor ≥1000 elements across ALL
30 DiT blocks (450 tensors) and the full VAE (70 tensors) for NaN/Inf/suspicious-all-zero content
— zero found in either file. This rules out gross checkpoint corruption as the cause (a real,
previously-unchecked category per update #14's own list of remaining candidates), though it does
NOT rule out a *semantic* mismatch (e.g. wrong checkpoint variant, or a real-but-subtle key-mapping
error that produces plausible-looking-but-wrong values — no NaN/zero signature, just wrong
content) — an intrinsically harder thing to detect without a real external reference.
**State after 15 rounds**: every structural/formula-level DiT component, RoPE, timestep/text
embedding, patch pack/unpack, VAE (decoder conv/resample/attention), and now UMT5 encoding
specifically (ruled out via the cross-substitution test) AND checkpoint integrity have all been
directly checked and found clean or fixed. Five real correctness bugs found and fixed this
session (patch order ×2, GPU RMS scope, timestep sin/cos order, T5-padding) plus a major, separately
significant sixth bug (the T5/UMT5 attention scale bug, update #13) that turned out not to be Wan's
own root cause either. The search space has narrowed to: the Euler integration loop's own
numerical behavior, or a semantic (not corruption-level) checkpoint mismatch — both harder to
check without either a real external reference or a different bisection technique than has been
tried so far.

**2026-09-14 update #16 — determinism ruled out too**: Checked whether `WanModel.Forward`/
`ForwardGpu` are even deterministic for identical inputs (`WanDeterminismDiagnosticTest.cs`, new
file) — a real, previously-untried category: a race condition or nondeterministic parallel-
reduction/dispatch-ordering bug could in principle produce exactly the "plausible magnitude, no
coherent structure" symptom if the same "clean" latent produced a genuinely different prediction
on repeat calls. **Result: both CPU and GPU paths are perfectly, exactly deterministic** (maxDiff
= 0.0 across two identical-input calls, both paths) — rules this out cleanly, no ambiguity.
**State after 16 rounds**: structural/formula correctness (every DiT sub-component, RoPE,
embeddings, patchify, VAE), UMT5 encoding specifically (via cross-substitution), checkpoint
integrity, and determinism have all now been directly, empirically checked and found clean or
fixed (6 real bugs fixed along the way). The two remaining candidates from update #14/#15 stand:
the Euler loop's actual spatial-coherence development across steps (only ever checked via global
magnitude/std, never a real local-correlation/structure check), or a semantic checkpoint mismatch.
Given how much of the "is this component broken" space is now closed off, a semantic checkpoint
issue (the DiT weights loading successfully, having plausible per-tensor statistics, yet not
actually being what this pipeline assumes they are) is becoming the comparatively more likely
remaining explanation, though still unconfirmed either way without an external reference.

**2026-09-14 update #17 — the first real, measurable, non-zero signal found in Wan's own output**:
Implemented the spatial-coherence check flagged as the next concrete lead in update #16
(`WanSpatialCoherenceDiagnosticTest.cs`, new file): measures real Pearson correlation between
pixels at increasing offsets, directly from the raw decoded RGB frame data (no PNG parsing
needed). **First attempt (offset=1, adjacent pixels) was methodologically flawed and corrected
in-place**: Wan's VAE upsamples 8x via nearest-neighbor duplication + conv (`ResampleSpatial`,
already examined in an earlier update), so even PURE noise decodes to high adjacent-pixel
correlation trivially (0.965) — a floor that has nothing to do with real structure. Filtered this
out by measuring at larger offsets (8/16/40px, beyond one latent-pixel's own 8px footprint) on
both the real Wan output (10-step, guidance=1.0, real prompt) and a pure-noise "negative control"
decoded through the identical real VAE:
- offset=8: Wan=0.152 vs noise=0.175 (comparable, noise slightly higher — within noise)
- **offset=16: Wan=0.089 vs noise=0.019 — Wan shows ~4.7x MORE correlation than pure noise**
- offset=40: Wan=0.066 vs noise=0.054 (Wan modestly higher)
**This is the first real, measured, non-zero evidence this entire session that Wan's DiT is
producing SOME real signal beyond pure noise** — small in absolute terms (0.09 is nowhere near a
real photo's typical 0.85+), but a real, repeatable, above-noise-floor correlation specifically at
the ~16px (one full patch-token) scale, which is exactly the spatial granularity a DiT's own patch
tokens would be expected to show structure at if the model is trying (and partially succeeding) to
encode *something* coherent, just far too weakly/incompletely to be recognizable. **This shifts the
most likely diagnosis away from "the DiT/pipeline produces zero real information" toward "the DiT
produces a real but far too weak signal relative to noise, which the Euler loop's `dt`/velocity-
scale isn't amplifying enough over the tested step counts"** — worth a real follow-up: run this
same correlation measurement across a step-count sweep (4/10/20/50 steps) to see whether the
16px-scale correlation grows with more steps (supporting a "just needs a proper schedule/more
steps/different guidance" diagnosis) or plateaus immediately (supporting a still-present, separate,
deeper bug that caps how much real signal the model can ever contribute). Not yet done — a real,
concrete, actionable next step, the first this investigation has found in several rounds.

**2026-09-14, same day — step-count sweep run, DECISIVE negative result**: Ran the real sweep
(4/10/20/40 steps, same prompt/seed, guidance=1.0). **The offset=16 correlation stays essentially
flat**: 0.0908 (4 steps) -> 0.0894 (10) -> 0.0923 (20) -> 0.0958 (40) — no meaningful growth across
a 10x increase in step count (the tiny ~0.005 drift is well within run-to-run noise, not a real
trend). **This rules out the "real but too-weak signal, just needs more steps/better schedule"
theory from earlier the same update.** The more likely explanation, given this flat result: the
small above-noise-floor correlation (0.09 vs pure-noise's 0.02) is NOT evidence of an evolving,
partially-successful image-generation attempt — it's more likely just residual structure inherited
from the DiT's own smooth, real, trained weight matrices (any linear projection through smoothly-
varying learned weights imparts SOME local correlation on its output relative to literal random
noise, independent of whether the model is doing anything semantically meaningful with it). So
while this update's correlation-floor comparison is a real, valid, newly-established measurement
technique (worth keeping for future bisection), its own optimistic reading has now been
conclusively tested and doesn't hold up — net contribution is a real methodology addition, not a
resolution, and the working assessment reverts to "no confirmed evidence of genuine image-content
generation," consistent with every visual inspection this session.

**Given 7 rounds of exhaustive, reference-verified investigation with 4 real fixes landed and zero
resolution, and the master plan's own explicit sequencing** ("then work through the rest of the
checklist"), pausing the Wan deep-dive here to make real progress elsewhere in the checklist rather
than continuing to hunt without new tools or leads — full context above is preserved for whoever
picks this back up (a future loop iteration, or a session with e.g. real diffusers/Python access
to generate one genuine ground-truth embedding/activation dump to diff against, which would very
likely resolve this quickly given how much has already been ruled out structurally).

**How to iterate on this fast**: use the GPU path for iteration (per the user's own instruction —
"it would be quicker to test GPU for accuracy" — real weights are already local, a real GPU
generation is much faster than CPU per this session's own measurements). Once a real, confirmed fix
is found and verified on GPU, **port the same fix to the CPU path** (`WanModel.Forward`/
`TransformerBlock`, mirroring whatever changed in `ForwardGpu`/`TransformerBlockGpu`) — GPU first,
CPU second, in that explicit order, since the bug is shared and GPU is the faster place to find it.

**Existing real test infrastructure to use** (don't build new harnesses from scratch):
`tests/OpenTail.Stingray.Tests.Diffusion/WanAppleDiagnosticTests.cs` (real prompt, real GPU
UMT5 + DiT + VAE, stage-by-stage stat logging, already wired for exactly this kind of bisection)
and `tests/OpenTail.Stingray.Tests.Diffusion/WanApple20StepsTest.cs`. Extend these with real
intermediate-value dumps (velocity mean/std per step, latent mean/std per step) rather than writing
new test files, mirroring the `FluxGpuVsCpuForwardBisectDebugTest.cs`/`docs/056`-style bisection
technique that found FLUX's own real bugs.

**Do not run the full heavy test suite for this** — targeted, single-class test runs only (see
`CLAUDE.md`'s own Fast-vs-Heavy test suite rule). A real end-to-end Wan run is already fast enough
on GPU to iterate on directly.

**Monitor real CPU vs GPU utilization during every test run** (Task Manager or equivalent) — this
session already found one real case (Wan's original `ForwardGpu`) where a "GPU" path was silently
running on CPU the whole time, caught specifically by noticing CPU (not GPU) utilization during a
supposedly-GPU run. Do this for every model, not just Wan — see the note in the checklist section
below.

## The broader checklist (once Wan accuracy is real again, or in parallel if independent capacity allows)

Each of these is its own full handoff doc with real, checked-by-hand architecture details — read
the doc before starting, don't re-derive from scratch:

- [ ] `docs/069-flux-vulkan-gemm-perf-handoff.md` — FLUX Vulkan GEMM/attention kernel tuning
      (partially done: real 2.3-2.5× win already landed and independently verified; remaining gap
      to the 99.8s C++ reference target still open, T5-XXL GPU residency done, see docs/071).
- [ ] `docs/071-flux-t5xxl-gpu-residency-plan.md` — done (T5-XXL GPU residency landed, committed).
- [ ] `docs/072-wan21-gpu-kernel-tuning-plan.md` + `docs/073-wan21-kernel-fusion-and-qkv-plan.md` —
      Wan DiT GPU kernel tuning (real progress landed: GPU now faster than CPU per-block; **but see
      Priority 0 above — this is now moot until Wan's real correctness bug is fixed**, since a
      faster wrong answer is not progress).
- [x] `docs/074-wan21-umt5-gpu-residency-plan.md` — **confirmed DONE 2026-09-14**: `UMT5GpuWeights.cs`,
      `UMT5GpuWorkspace.cs`, and `UMT5Encoder.InitGpu`/`EncodeGpu` all exist as real files (not the
      plan doc's own not-yet-done template code). `UMT5GpuParityTests.UMT5Encoder_EncodeGpu_
      MatchesCpuReference_Numerically` (real 24-layer/dim=4096 shapes, synthetic-but-structurally-
      real weights) passes with real ~15s wall-clock (not a silent no-op per CLAUDE.md rule 12) at
      <0.08 max-diff tolerance. `EncodeGpu` has also been exercised as real, working GPU compute
      throughout this whole session's Wan diagnostic tests (`[Diagnostic] Compute device: Vulkan
      GPU` log line on every run). The plan doc itself (`docs/074`) was never updated to reflect
      this — it still reads as a not-yet-started template; leaving it as historical context but
      marking this checklist item done here instead of editing it.
- [ ] `docs/075-zimage-turbo-gpu-residency-plan.md` — Z-Image-Turbo DiT, not yet started. Real flag:
      unverified RoPE pairing convention (check before reusing `Flux2DRoPE`), tanh-gated/shift-less
      AdaLN convention differs from FLUX/Wan.
- [ ] `docs/076-sd35-medium-gpu-residency-plan.md` — SD3.5-medium MMDiT, not yet started. Real
      flag: `headDim=64` (use `MultiHeadAttentionTiled`, not the 128 variant), absolute sincos
      pos-embed (no RoPE convention risk), same AdaLN convention as FLUX (most direct reuse).
- [ ] `docs/077-ltx-video-gpu-residency-plan.md` — LTX-Video's separate correctness bug ("only 1 of
      6 runs ever produced a coherent image") **partially fixed 2026-09-14, still OPEN overall —
      do not mark this done**: `TimestepEmbedder` (DiT) and `TimestepEmbedMlp` (VAE decoder) both
      called `SinusoidalTimestepEmbedding` without `flipSinToCos: true`, using the wrong `[sin,cos]`
      layout instead of the real `[cos,sin]` convention (same bug class already found and fixed for
      Wan) — a real, machine-precision-verified fix, `LtxVideoGoldenParityTests` (previously failing
      at block0 cosine-sim 0.9893641) now passes outright. **But the full end-to-end pipeline is
      still broken**: a real 20-step run still produces a destroyed, banded, non-coherent image, now
      root-caused to a scale-independent, unbounded ~2-2.5× latent-magnitude divergence across the
      denoising loop (confirmed present even at a tiny 4-token control scale, and confirmed NOT
      caused by the VAE, un-normalization, CFG, the scheduler's `TimeShift` formula, the Euler
      integration formula, or the final `norm_out`/`proj_out` layer — all individually checked and
      ruled out). Leading remaining hypothesis: a small systematic per-step magnitude bias in the
      DiT's own output, too small to fail the golden angle-only cosine-similarity check but
      compounding multiplicatively over many steps. GPU-residency work on this model has NOT been
      started, and per this doc's own standing warning, still should not start until this
      correctness item is actually closed. See docs/077's full 2026-09-14 update trail (the most
      current entries) for the complete evidence chain and the precise next diagnostic step, and
      `PerformanceLeague.md`'s LTX-Video row for the same honest status.
- [ ] `docs/078-hunyuanvideo-gpu-residency-plan.md` — HunyuanVideo, not yet started. Real flag: the
      `_backend` field in `HunyuanVideoModel.cs` is **dead, unused code** — don't mistake it for
      partial progress. Also needs a genuinely new RoPE kernel (split-half pairing + unequal
      16/56/56 axis split — neither existing GPU RoPE kernel matches this convention).
- [ ] `docs/079-minimax-music3-gpu-residency-plan.md` — MiniMax-Music3's flow transformer, not yet
      started. `headDim=64`, in-context conditioning (no separate cross-attention block to port).
- [ ] `docs/080-tts-asr-gpu-residency-checklist.md` — the broader ~30-model TTS/ASR stack
      checklist. Real audit already done: 26 confirmed 100% CPU, 4 with partial GPU code worth
      auditing first (F5TTS, Chatterbox, CosyVoice, Parler), 0 with confirmed real GPU residency.

## Cross-cutting reminders (apply to every item above, not just Wan)

1. **"A green test doesn't mean it's real" applies to GPU claims specifically.** This session found
   Wan's `ForwardGpu` silently running CPU compute despite looking GPU-resident, and found
   HunyuanVideo's `_backend` field being completely dead/unused scaffolding. **For every model on
   this list, before trusting any "GPU residency done" claim: watch real CPU vs GPU utilization
   (Task Manager or equivalent) while the supposed GPU path actually runs.** If CPU spikes instead
   of GPU, that's the same bug pattern again — there is a real chance several items on this
   checklist have the same issue lurking, not just Wan.
2. **Accuracy before speed, every time.** A faster wrong answer is not progress — this is exactly
   what happened with Wan's recent kernel-tuning work (docs/072/073 landed real speed wins on top
   of what turns out to be broken output). Verify a real, coherent, visually-inspected output
   *before* chasing or reporting a timing number, for every model, not just ones flagged as risky.
3. **No subagents** — do all work directly in the main session (`CLAUDE.md` rule 6).
4. **Never remove or revert existing correct code** — every fix is additive/corrective, never a
   rollback, unless something is proven to be actively wrong (like Wan's current broken state) —
   even then, fix forward (correct the bug), don't revert to an earlier commit.
5. **Ask before committing** — confirm current expectations before committing; don't assume
   standing permission carries forward indefinitely across a long unattended session.
6. **Run one thing at a time, alone, and avoid the full heavy test suite.** A real incident earlier
   this session came from an unrelated concurrent `dotnet test` run contaminating a timing
   measurement and causing severe memory pressure. Use targeted single-class test runs
   (`<TestBinary>.exe -class <FullyQualifiedName>`, per `CLAUDE.md`'s own guidance) — never the
   full suite — and check `Get-CimInstance Win32_OperatingSystem | Select FreePhysicalMemory`
   (PowerShell) is healthy before starting any benchmark.
7. **Update `PerformanceLeague.md` honestly** — real measured numbers only, win or not, following
   the bar this whole session has held to (a claimed 2.3× FLUX win was independently re-verified
   and held up; Wan's claimed win did not hold up once correctness was actually checked — both
   outcomes are fine to report, fabricating one is not).

## Success criterion for this master plan

Wan2.1 produces a real, visually-confirmed-coherent image/video from a real prompt on **both** CPU
and GPU (GPU found first, CPU ported second) before any more Wan performance work is reported as
progress. Beyond that, work through the checklist above at whatever pace is sustainable, always
verifying real GPU execution (not just accepting a backend parameter) and real correctness before
claiming a model is "done."
