# SD3.5 composition/left-shift bug — T5 encoder bisection plan

**Status as of 2026-09-21 (end of session): NOT fixed.** Real, substantial, numerically-verified
progress made on conditioning accuracy (see below), but the end-to-end composition/left-shift bug
remains open. This doc is the handoff for continuing tomorrow — read this before re-opening
`docs/094-diffusion-performance-plan.md`'s SD3.5 rows for the fuller history.

## The symptom (unchanged from `docs/094`)

Real, recognizable content (apple + wood table), but wrong composition: the C++ reference renders
a full-frame, centered, photorealistic apple; our port renders the apple small/cropped/shifted, or
(after today's conditioning fixes) sometimes a completely different, hallucinated wrong subject
(Scrabble tiles, fish shapes) in the corner instead. Reproduces identically on CPU and GPU with
identical injected noise — this is shared pipeline/conditioning math, not a backend bug.

## What actually moved the needle today — real, numerically verified

**Method that finally worked**: stop judging generated images. Dump the real reference's own
internal tensors from `examples/stable-diffusion.cpp` at exact comparison points (noise, assembled
context/pooledY, token IDs) via temporary env-var-gated hooks, and diff them byte-for-byte /
cosine-similarity against this port's own tensors at the same points. Every real bug found today
was found this way; every guess based on how the final image looked was wrong or unconfirmed.

### Diagnostic infrastructure built (all temporary, `examples/` is gitignored so the C++ side isn't
committed; the C# side uses committed real fixes plus scratch test files that stay untracked)

- `examples/stable-diffusion.cpp/src/stable-diffusion.cpp`: `SD_DUMP_NOISE_PATH` env var dumps the
  real Philox-RNG noise tensor right after `sd::randn_like` (before `std::move` into `sample()`).
- `examples/stable-diffusion.cpp/src/conditioning/conditioner.hpp`:
  - `SD_DUMP_CONDITION_PREFIX` env var dumps the real assembled `context`/`pooledY` tensors
    (`result.c_crossattn`/`result.c_vector`) right after `SD3CLIPEmbedder::
    get_learned_condition_common` builds them, for both the cond and uncond calls (a static call
    counter distinguishes them).
  - `SD_DUMP_TOKENS` env var prints the real `clip_l_tokens`/`clip_g_tokens`/`t5_tokens` integer
    arrays from `tokenize()`.
- `src/OpenTail.Stingray.Diffusion/SD3/Sd3Pipeline.cs`: `STINGRAY_SD3_INJECT_NOISE_PATH` env var
  (real, committed) reads a raw float32 file and uses it as the initial latent, completely
  bypassing `System.Random`+Box-Muller — lets an exact reference noise tensor be replayed.
- `tests/OpenTail.Stingray.Tests.Diffusion/ZZ_ScratchSd3ConditionTensorDiffTest.cs` (untracked
  scratch, still on disk): loads the real pipeline, calls `Sd3Pipeline.EncodePromptForTesting` for
  cond/uncond, and diffs the result against the C++ dumps with per-region cosine/RMS/maxAbsDiff
  stats (CLIP-L rows, CLIP-G rows, T5 rows of context; CLIP-L half, CLIP-G half of pooledY).
- `tests/OpenTail.Stingray.Tests.Diffusion/ZZ_ScratchSd3TokenDumpTest.cs` (untracked scratch, still
  on disk): dumps this port's own `ClipTokenizer`/`T5Tokenizer` output for direct comparison.
- Real reference dump files on disk (gitignored, `docs/diffusion-samples/`):
  `zz_scratch_cpp_noise_seed42_realt5.bin`, `zz_scratch_cpp_cond_realt5_{context,pooledY}_
  {cond,uncond}.bin` — produced WITH real T5 wired in (`--t5xxl models/flux1-schnell/
  t5xxl_fp8_e4m3fn.safetensors`). Earlier dumps without `_realt5` suffix were produced
  ACCIDENTALLY without T5 (see pitfall below) — don't reuse those.

### Real bugs found, fixed, and committed today

1. `01d16f3`, `eecaf7e` — flow-matching sigma schedule was missing the real SD3 `flow_shift=3.0`
   (was plain linear); T5 self-attention had zero padding mask (real, ~96%-padded 256-slot buffer
   fully participating in every layer). Both real, both fixed. Neither was the composition bug.
2. `0ee73ee` — **`T5MaxTokens` was 256, should be 77.** SD3's real reference
   (`SD3CLIPEmbedder::get_learned_condition_common`, `conditioner.hpp:857`) hardcodes
   `chunk_len = 77` and uses that SAME length for CLIP-L, CLIP-G, AND T5 (unlike FLUX, which
   really does use 256 — don't backport this fix there). Confirmed via the real reference's
   assembled context tensor shape: `[4096, 154, 1]` = 77+77, not 333. Fixed.
   Also in this commit: retracted an earlier same-session CLIP-L `text_projection` "fix" after
   reading `clip.hpp`'s `CLIPTextModel::init_params` directly — that param is only ever registered
   for CLIP-G (`OPEN_CLIP_VIT_BIGG_14`); CLIP-L's real graph never applies it regardless of what a
   given checkpoint file contains. Reverted; kept the separate (correct, verified) penultimate-
   hidden-state fix for CLIP-L.
   Also: `ImageCommand.cs`'s SD3 `stingray image` CLI path never wired `--t5xxl`/`--t5-tokenizer`
   through at all — every CLI-driven SD3.5 generation this session (and presumably before) ran
   with T5 conditioning silently zero-filled. Fixed. **If you use the CLI for SD3.5 from now on,
   you MUST pass `--t5xxl`/`--t5-tokenizer` explicitly or T5 conditioning is silently absent again
   for any *other* still-unwired path.**
3. `9b8d71d` — **the big one, real root cause of a huge tokenizer bug**: `ClipTokenizer.Bpe()`'s
   initial character split emitted the `</w>` end-of-word marker as its OWN separate list element,
   but real CLIP/GPT-2-style BPE fuses it onto the LAST character as a single atomic initial unit.
   For a single-character word this is catastrophic (no merge rule to recombine them exists in the
   real vocab, because the real vocab already seeds e.g. `"a</w>"` as a zero-merge base token) —
   confirmed the word "a" was resolving to the wrong base token id 64 instead of the real 320.
   Fixed by fusing the suffix onto the last character before the merge loop. Verified: token IDs
   for the whole test prompt now match the reference byte-for-byte.
   Also in this commit: `OpenClipGEncoder` was reusing the shared tokenizer's EOS-repeated padding
   (the real CLIP-L/HF convention) for CLIP-G too, but the reference's separate CLIP-G tokenizer
   pads with 0 instead — fixed by zeroing everything after the first EOS.

### Real measured impact (tensor diff against the reference, cosine similarity, same prompt/noise)

```
                                    before      after
pooledY, cond, CLIP-L:              0.638   →   0.9999   (essentially exact)
pooledY, cond, CLIP-G:              0.600   →   0.902    (improved, NOT exact -- separate bug, see below)
pooledY, uncond (empty prompt):      -          0.988    (was already fine, untouched)
context, CLIP rows (tokens 0-76):   0.941   →   0.992
context, T5 rows (tokens 77-153):   0.169   →   0.169    (UNCHANGED -- not a tokenizer bug, see below)
```

**T5 token IDs are byte-for-byte IDENTICAL to the reference** (dumped and diffed directly, both the
real prompt and the empty string) — this conclusively rules out T5 tokenization. The 0.169 context
cosine is a real bug **inside the T5 encoder itself** (`src/OpenTail.Stingray.Diffusion/
TextEncoders/T5Encoder.cs`).

**CLIP-G's pooled output (0.902, unchanged by the padding fix)** is also a real, separate,
still-unfixed bug — expected to be untouched by the padding fix specifically (CLIP uses causal
masking, so trailing-padding identity cannot affect the earlier EOS-position pooled readout; the
padding fix only ever improves the *full-sequence* context tensor, which is exactly what it did).
Now that tokens are proven correct, this is a real bug in `OpenClipGEncoder`'s own transformer math
— not yet investigated.

## Confirmed NOT the cause (checked and ruled out this session, don't re-check without new evidence)

- RNG/noise source mismatch (own `System.Random` vs. reference's Philox) — exact reference noise
  injected, still produces a different-but-still-wrong composition, not the reference's correct one.
- `pos_embed` center-crop math — traced line-by-line against `mmdit.hpp`'s `cropped_pos_embed`,
  confirmed identical; empirically confirmed too (forcing a wrong crop offset destroys coherence
  entirely rather than mildly shifting content, which is what a real offset bug would look like).
- Patchify / unpatchify channel-ordering conventions — verified line-by-line against
  `dit.hpp`/diffusers' `_unpatchify` einsum convention, both correct.
- VAE decode crop/scale — shared code path, already proven correct for FLUX.1/FLUX.2/Z-Image.
- Image/text token concatenation order for joint attention (ours: img-then-txt; reference:
  txt-then-img) — self-consistent (same offsets for Q/K/V construction and output splitting), and
  this model has no RoPE/position-dependent masking, so full bidirectional attention is provably
  invariant to this permutation.
- CFG blend formula — matches `guidance.cpp` exactly (`pred_uncond + scale*(pred_cond-pred_uncond)`).
- The "extra SiLU before modulation" theory — checked directly against `mmdit.hpp`: our single
  cached `SiLU(c)` (computed once since `c=t_embed+y_embed` is invariant across all 24 blocks) is
  mathematically identical to the reference's per-block `ggml_silu(c)` recomputed fresh each time,
  not a duplicate nonlinearity. `t_embedder`/`y_embedder`'s own INTERNAL SiLUs (inside their 2-layer
  MLPs) are separate, correct, and present in the reference too — don't confuse the two.
- T5 padding-mask *implementation difference* (ours: baked into the relative-bias tensor as -1e9;
  reference: separate additive mask) — mathematically equivalent for finite logits, not a real
  discrepancy per a second-opinion review; not the lead suspect for the T5 divergence.

## The concrete next-session plan (in priority order, per a second-opinion review of this doc's
## own findings — do these, don't re-derive `RelPosBucket` by hand again)

1. ~~Byte-compare the 77×77 relative-position bucket matrix~~ **CHECKED 2026-09-21, CLEARED, do not
   re-check.** Compared `T5Encoder.RelPosBucket` term-for-term against the reference's real
   `T5::_relative_position_bucket` (`t5.hpp:463-497`) by direct source read (no build/dump needed):
   identical constants (`num_buckets=16` post-halving, `max_exact=8`, `max_distance=128`), identical
   log-bucket formula (`max_exact + int(log(pos/max_exact)/log(max_distance/max_exact) *
   (num_buckets-max_exact))`, clamped to `num_buckets-1`), identical `+num_buckets` offset for
   positive relative position, and identical sign convention (`relative_position[i][j] =
   memory_position[j]-context_position[i]` = `j-i`, matching our own `RelPosBucket(j - i)` call
   site exactly). **Not a bug — formula matches the reference exactly.** (A sibling encoder, Wan's
   UMT5, DID have a real sign-flip bug in this exact formula before, already fixed there — that
   history is why this was worth checking here too, but this specific encoder's version is clean.)
2. ~~Verify the attention scale convention~~ **CHECKED 2026-09-21, CLEARED, do not re-check.** Read
   `examples/stable-diffusion.cpp/src/model/te/t5.hpp`'s real `T5Attention::forward` directly:
   `k = ggml_ext_scale(ctx->ggml_ctx, k, sqrtf(d_head), true);` then
   `ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k, v, num_heads, mask)`.
   `ggml_ext_attention_ext` is the generic SDPA primitive, which internally divides by `sqrt(d_head)`
   itself — pre-multiplying K by `sqrt(d_head)` exactly CANCELS that internal division, so the
   reference's real net attention score is **plain, unscaled `q·k`**, matching real T5's documented
   architecture quirk (T5 omits `1/sqrt(d)` scaling by design, due to its init scheme). This is
   EXACTLY what `T5Encoder.SelfAttention` already does (`dot = Dot(q,k)`, no scale factor anywhere).
   **Not a bug — our lack of scaling is correct.** (`ClipLEncoder`'s own explicit
   `1/sqrt(HeadDim)` scale is unrelated and correct for CLIP, which is a different, standard-scaled
   architecture — don't try to "fix" T5 to match it.)
3. ~~Verify T5's gated-GELU activation is `gelu_new` (tanh-approx), not exact erf GELU~~ **CHECKED
   2026-09-21, CLEARED, do not re-check.** `T5Encoder.FeedForward` calls `DiffusionOps.GeluInPlace`,
   which dispatches to `DiffusionOps.Gelu` in both its scalar and parallel-chunked branches — that
   function is explicitly the tanh approximation (`0.5*x*(1+tanh(0.7978845608*(x+0.044715*x^3)))`,
   its own doc comment says "matches PyTorch default"), i.e. real `gelu_new` — NOT
   `DiffusionOps.GeluExact` (the separate erf-based implementation used elsewhere in this codebase,
   e.g. for Wan's `text_embedding.1`). **Not a bug — activation matches the real T5-v1.1-XXL config
   (`dense_act_fn=gelu_new`) exactly.**
4. **If none of the above resolves it** (all three cheap checks above are now cleared — this is the
   next real thing to try), do the 4-point T5 bisection (real dump hooks needed on
   both sides, more infrastructure work than 1-3 above):
   - `E0` = token embedding output (before any blocks)
   - `A0` = output after block 0's self-attention + residual (before block 0's FFN)
   - `F0` = output after block 0's FFN + residual (= input to block 1)
   - `final` = output after all 24 blocks + final RMSNorm
   Compare each against the same point in the reference. The FIRST checkpoint whose cosine
   collapses tells you the exact subsystem (embedding lookup vs. attention vs. FFN vs. later-layer
   accumulation) — don't jump straight to a full 24-block dump.
5. **Separately, still open**: `OpenClipGEncoder`'s own pooled-output bug (0.902 cosine with tokens
   now proven correct) — not yet investigated at all. Lower priority than T5 (T5's divergence is
   far more severe and dominates the joint context tensor), but real and needs its own fix
   eventually before SD3.5 can be called golden-verified.
6. **After T5 (and ideally CLIP-G) match the reference numerically**: redo the step-0 `pred_cond`/
   `pred_uncond` injection experiment (dump the reference's real conditioning + noise + timestep,
   inject the SAME into a single C# `MMDiTModel.Forward` call, compare outputs directly) — this
   was the original plan before the conditioning-tensor audit took priority, and it's now the right
   move once conditioning itself is trustworthy. Only if THAT still diverges does the existing
   `MaxBlockIndexForDiagnostic`/`DiagnosticStopStage` block-bisection infrastructure
   (`docs/094`'s 2026-09-20 entries) become the right next tool.

## Pitfalls hit this session, don't repeat them

- **The CLI's SD3 path silently drops T5 conditioning if you don't pass `--t5xxl` explicitly** (now
  fixed for the SD3 code path itself, but double-check any NEW code path you add). Several hours
  of this session's early comparisons were accidentally confounded by this before it was caught —
  always sanity-check the log output for real T5 memory usage (~10GB+ for text encoders) before
  trusting a "no effect" or "regression" result from a CLI-driven test.
- **Don't run multiple heavy tests/generations concurrently** — real resource contention skews
  timings and wastes wall-clock for no extra evidence. One test at a time.
- **Prefer GPU (Vulkan) over CPU for iteration on this bug** — it's proven backend-shared (identical
  results on both with matched noise), and GPU is ~3-4x faster per generation (~110-150s vs.
  ~440-460s for a 20-step 256×256 run on this machine).
- **Judging composition from generated images is unreliable and was actively misleading** multiple
  times this session (a "worse-looking" image after a real, reference-verified-correct fix, and a
  "no visible change" image after a real, verified-necessary fix). Always prefer the real tensor
  diff over visual inspection when the infrastructure exists to do so.

## Full chronological log of what was tried this session (for context, not re-reading required)

In order, so the next session can see what's already been spent and why each thing was dropped or
kept, without re-deriving the reasoning from scratch:

1. **Started from `docs/094`'s existing state**: composition/scale bug already found (small/cropped
   apple vs. reference's full-frame centered apple), two candidates already named but unchecked
   (patchify/unpatchify canvas mapping, VAE decode crop/scale).
2. **Added the missing SD3 flow-shift sigma schedule** (`sigma(t)=shift*t/(1+(shift-1)*t)`,
   shift=3.0, mirroring `DiscreteFlowDenoiser`). Real bug, real fix, committed (`01d16f3`).
   Visually: fixed the apple's SCALE (was tiny-in-corner, became correctly-sized) but NOT its
   position (still left-shifted). This was genuine, measured progress, not a dead end.
3. **Manually re-verified patchify / pos_embed crop / unpatchify** line-by-line against
   `mmdit.hpp`/`dit.hpp` — all three confirmed correct, byte-for-byte matching convention. Also ran
   an empirical bisection test (forcing a deliberately wrong pos_embed crop offset via an env var)
   — wrong offsets destroy all coherence rather than mildly shifting content, which itself argues
   against a subtle pos_embed bug (a real off-by-some-amount bug there would look catastrophic, not
   like a mild shift). **Dead end, correctly ruled out, don't re-check.**
4. **Ruled out VAE decode** by the "shared code path already proven correct for other models"
   argument (FLUX/Z-Image both use the same `VaeDecoder.cs` and are golden-verified) — didn't
   require its own dedicated test. **Dead end, correctly ruled out.**
5. **Ruled out img/txt token concatenation order** by the "self-consistent + no RoPE + full
   bidirectional attention = provably invariant to this permutation" argument. **Dead end, correctly
   ruled out, confirmed by direct source comparison against `mmdit.hpp`'s `block_mixing`.**
6. **Checked and ruled out an "extra SiLU before modulation" theory** (raised by an external
   second-opinion review) by reading `mmdit.hpp` directly — the cached single `SiLU(c)` is
   mathematically identical to the reference's per-block fresh `ggml_silu(c)`, not a duplicate.
   **Dead end from a plausible-sounding but ultimately wrong external suggestion — always verify
   against the actual reference source before acting on a suggested fix, even a well-reasoned one.**
7. **Built the noise-injection experiment**: added a temporary dump hook to
   `stable-diffusion.cpp`'s real `sd::randn_like` call site, dumped the exact Philox-RNG noise
   tensor, added `STINGRAY_SD3_INJECT_NOISE_PATH` to `Sd3Pipeline.cs` to consume it directly,
   bypassing `System.Random` entirely. **Result: injecting the EXACT reference noise did NOT
   reproduce the reference's correct composition — produced a DIFFERENT wrong composition instead.
   This was a real, decisive, correctly-interpreted negative result: it proved RNG-source mismatch
   is not the (sole) cause, since matching the noise exactly didn't fix things.** Also
   cross-confirmed CPU and GPU agree exactly given identical injected noise (ruled out any
   backend-specific confound in every subsequent experiment this session).
8. **Found and fixed T5 self-attention's missing padding mask** (256-slot buffer, ~96% padding,
   fully attended with no mask at every layer). Real bug, real fix, committed (`eecaf7e`). **Tested
   with injected noise: pixel-IDENTICAL output before/after.** This was initially confusing (a real
   bug with zero visible effect?) until later understood: the bug was real but not dominant enough
   to move this particular trajectory's output visibly — a good reminder that "no visible change"
   after a real fix doesn't mean the fix was wrong, just that it wasn't the dominant error for that
   specific symptom.
9. **Found, applied, and then had to isolate/retract half of a CLIP-L fix.** First found the real
   penultimate-vs-final-hidden-state asymmetry (real bug, confirmed against both the diffusers
   pipeline convention AND `clip.hpp`'s exact `layer_idx = n_layer - clip_skip` arithmetic — kept,
   correct). At the same time, ALSO added a `text_projection` matmul to CLIP-L's pooled output,
   reasoning by analogy to CLIP-G's own correct pattern and the checkpoint file's physical
   `text_projection.weight` tensor. **Tested with injected noise: output changed dramatically —
   apple disappeared entirely, replaced by plain wood texture.** This looked like a regression.
   Isolated the two sub-changes via an env-var gate, discovered the penultimate-hidden-state change
   ALONE reproduced the same "no apple" result — meaning the projection wasn't the cause of that
   particular visual change. Then, independently, read `clip.hpp`'s `CLIPTextModel::init_params`
   directly and found the reference NEVER applies `text_projection` for CLIP-L regardless of
   checkpoint contents (CLIP-G only) — retracted the projection fix on correctness grounds (not
   because of the visual result, which turned out to be a red herring for THIS sub-change), kept
   the penultimate-hidden-state fix. **Lesson: a bad visual result doesn't tell you WHICH of several
   simultaneous changes caused it — isolate before concluding, and verify each change against the
   reference independently regardless of what the image looks like.**
10. **Discovered mid-session that CLI-driven tests were accidentally running with T5 completely
    disabled** (`ImageCommand.cs` never wired `--t5xxl` for SD3). This invalidated the
    interpretation of several image-based comparisons made earlier in the session (they were real
    experiments, but conducted on an under-conditioned pipeline, so "no apple, just wood texture"
    results from that period are confounded and shouldn't be over-interpreted). Fixed the CLI
    wiring; re-ran key experiments with real T5 after the fix.
11. **Found the real T5-sequence-length bug** (`T5MaxTokens` 256→77) by building the context/
    pooledY tensor dump-and-diff infrastructure (described above) and noticing the reference's real
    context tensor shape was `[4096,154,1]`, not the expected 333 tokens. This is the point where
    the investigation shifted from "guess a candidate, test with an image" to "diff real tensors,
    let the numbers point at the next thing" — much higher signal, should have started here.
12. **Built the token-ID dump** (described above), which cleanly separated T5 (tokens identical,
    bug is in the encoder) from CLIP (tokens completely different, bug was in the tokenizer) —
    found and fixed the real CLIP `</w>`-fusion BPE bug and the CLIP-G padding-convention bug from
    this. This was the single highest-value diagnostic built this session, by a wide margin —
    replicate this pattern (dump real reference values at a real internal boundary, diff directly)
    for the T5 encoder bisection before trying anything else.
13. **Session ends here** with T5's context-tensor cosine still at 0.169 (unimproved by any of
    today's fixes, all of which were real but touched CLIP or sequence-length, not the T5 encoder's
    own math) and CLIP-G's pooled output still at 0.902 (real, separate, unfixed bug in
    `OpenClipGEncoder`'s own transformer math, now cleanly isolated since tokens are proven
    correct). Both are now real, precisely-localized, numerically-defined open items — not vague
    "something's wrong somewhere" — which is real progress even though the end-to-end bug isn't
    fixed yet.
