# 092 — RoPE pairing-convention audit for HunyuanVideo and Qwen Image (+ a DRY consolidation)

**Audience**: an AI agent picking this up fresh. Read this whole file before touching code.

## The lead, stated plainly

FLUX.2's long-standing tiling/grid artifact was just resolved (`docs/087`/`docs/088`) by fixing a
RoPE **pairing convention** bug: the code rotated split-half pairs `(x[d], x[d+half])` ("NEOX"
style) when the real reference actually rotates adjacent pairs `(x[2i], x[2i+1])` ("interleaved"/
"GPT-J" style) — confirmed directly against `examples/flux2/src/flux2/model.py`'s
`apply_rope`'s `reshape(*xq.shape[:-1], -1, 1, 2)`.

**This exact bug, with this exact fix, already happened once before in this codebase** — for Wan,
found and fixed on 2026-08-31 (see `src/OpenTail.Stingray.Diffusion/Wan/WanRoPE.cs`'s own class
doc comment for the full derivation). Wan used to delegate to a shared
`Primitives/SplitHalfRoPE.cs` kernel; that kernel's own doc comment still claims it's
`"used by Wan, HunyuanVideo and QwenImage"` — **but that's now stale**. Wan moved off it after the
2026-08-31 fix and has its own local interleaved implementation. HunyuanVideo and Qwen Image never
got the same audit and still call `SplitHalfRoPE.FillFrequencies`/`SplitHalfRoPE.ApplyRoPE`
directly.

**Independently re-derived against the real references for this doc** (don't take this on faith —
re-verify yourself before changing anything, but this was checked, not guessed):

- `examples/diffusers/src/diffusers/models/embeddings.py`'s `apply_rotary_emb` — the
  `use_real_unbind_dim == -1` branch (`x.reshape(*x.shape[:-1], -1, 2).unbind(-1)`, i.e. adjacent
  pairs) is explicitly commented **"Used for flux, cogvideox, hunyuan-dit"**, and
  `examples/diffusers/src/diffusers/models/transformers/transformer_hunyuan_video.py`'s call sites
  (`apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)`) use this function's **default**
  `use_real_unbind_dim=-1` — i.e. **HunyuanVideo's real reference uses adjacent-pair RoPE**, not
  split-half.
- `examples/diffusers/src/diffusers/models/transformers/transformer_qwenimage.py`'s
  `apply_rotary_emb_qwen` is called with `use_real=False` at every real call site (`img_query`/
  `img_key`/`txt_query`/`txt_key`), which takes the complex-number path:
  `x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))` — **also adjacent
  pairs**, just implemented via complex multiplication instead of explicit cos/sin. Same
  convention, different mechanism.

Both models' current C# ports (`HunyuanVideoRoPE.cs`, `QwenImageRoPE.cs`) call
`SplitHalfRoPE.FillFrequencies`/`SplitHalfRoPE.ApplyRoPE` — split-half, the wrong convention per
the above. **This is a strong, well-evidenced candidate for the actual root cause of both models'
remaining artifacts** (Qwen Image's horizontal banding, HunyuanVideo's pure noise) — but it is a
candidate, not yet a confirmed fix. Verify with a real end-to-end run before declaring victory,
exactly as was done for FLUX.2.

## Do NOT just copy `WanRoPE.cs` into two more files — consolidate instead

`WanRoPE.cs`'s local `FillFrequenciesInterleaved`/`ApplyRoPE` and the shared
`Primitives/InterleavedRoPE.cs` (already used by FLUX.2/FLUX3) are **functionally identical
implementations of the same formula** — same `theta^(-2i/dim)` frequency, same
"write the value at both `2i` and `2i+1`" table layout, same
`out[2i] = x[2i]*c - x[2i+1]*s; out[2i+1] = x[2i]*s + x[2i+1]*c` rotation. They were independently
written twice instead of shared once. Before adding a THIRD and FOURTH copy for HunyuanVideo and
Qwen Image, do this consolidation first:

1. Confirm `WanRoPE.cs`'s `FillFrequenciesInterleaved`/`ApplyRoPE` and
   `Primitives/InterleavedRoPE.cs`'s `FillAxisFreqs`/`ApplyRoPE` produce bit-identical output for
   the same `(pos, dim, theta)` inputs (a cheap unit test, no real weights needed — same spirit as
   `Flux2RopePositionDifferentiationTests.cs`, which already exercises `InterleavedRoPE` via
   `Flux2RoPE.BuildContextFreqs`).
2. If confirmed identical (expected — the math is the same), refactor `WanRoPE.cs` to delegate to
   `Primitives.InterleavedRoPE` (mirroring how `Flux2RoPE.cs` already does:
   `InterleavedRoPE.ComputeInvFreqs`/`FillAxisFreqs`/`ApplyRoPE`), keeping only Wan's own
   axis-dim composition (`t=44, h=42, w=42`, no leading identity axis) local to `WanRoPE.cs`. Delete
   the now-redundant local `FillFrequenciesInterleaved`/`ApplyRoPE` methods.
3. Re-run Wan's real coherence tests after this refactor (the 5-run sweep from `docs/088`'s Wan
   section, or at minimum one real seed) to confirm zero numeric regression — this is a pure
   dedup, it must not change Wan's output at all. If it does, stop and figure out why before
   proceeding; that would mean the two implementations weren't actually identical and something
   about this whole premise needs re-checking.
4. Only then: implement `HunyuanVideoRoPE.cs` and `QwenImageRoPE.cs` by calling the SAME shared
   `Primitives.InterleavedRoPE` functions (own axis-dim composition per model, matching whatever
   `dimT`/`dimH`/`dimW` split each currently has — **do not change the axis-dim split, only the
   pairing/table-layout convention**, exactly per `WanRoPE.cs`'s own doc comment caveat: "Per-axis
   dims ... were already correct and unaffected by this fix"), instead of writing new model-local
   interleaved math a third and fourth time.

This leaves the codebase with ONE shared interleaved-RoPE kernel (used by Flux2, Flux3, Wan,
HunyuanVideo, Qwen Image) and ONE shared split-half kernel (`SplitHalfRoPE`, used only by whatever
genuinely needs it after this audit — re-check `T5GemmaEncoder.cs`'s own usage separately if
curious, but that's a text-encoder LLM, a different problem domain, out of scope here).

## Verification plan, in order

1. Do the consolidation above first (Wan refactor + regression check) — this is safe, mechanical,
   and builds confidence in the shared kernel before touching two more models' correctness.
2. Fix `HunyuanVideoRoPE.cs` to use the shared interleaved kernel. Re-run
   `tests/OpenTail.Stingray.Tests.Diffusion/HunyuanVideoRealConditioningCoherenceTests.cs` (real
   weights, already exists) — compare against the last real sample
   (`docs/diffusion-samples/hunyuanvideo_real_conditioning_256_4step_2026-09-18.png`, pure noise).
   If this fix is right, expect a real, visible change in character (per this session's own
   established diagnostic: a real structural fix changes the artifact, even if it doesn't fully
   resolve it on the first try — see Qwen Image's own flipSinToCos history for what "real fix,
   partial progress" looks like versus "no visible effect").
3. Fix `QwenImageRoPE.cs` the same way. Re-run
   `tests/OpenTail.Stingray.Tests.Diffusion/QwenImageRealConditioningCoherenceTests.cs` (real
   weights, already exists) — compare against the last real sample
   (`docs/diffusion-samples/qwenimage_real_conditioning_256_8step_2026-09-18.png`, horizontal
   banding).
4. Document the real result — whichever way it goes — in `docs/088`, `docs/089` (Qwen Image's own
   tracking doc), README's status matrix, and `PerformanceLeague.md` if timing is relevant,
   following this project's established per-finding documentation format (see any recent entry in
   those files for the expected level of detail: what was checked, what the real reference said,
   what changed, with dated citations). **If it doesn't fully resolve the artifact, say so plainly**
   — a real, well-reasoned fix that only partially helps (or doesn't help at all) is still worth
   recording precisely, exactly as this session's own `docs/088` already does for several other
   partial fixes (Wan's/LTX-Video's/Qwen Image's own `flipSinToCos` histories).

## Constraints

- This is a CPU-only investigation. Do not attempt Vulkan GPU work for either model — both are
  still gated on Pass 1 (correctness) per `docs/088`, and CLAUDE.md rule 7 (measure correctness
  before speed; don't GPU-port broken math).
- Watch memory. Real-weight tests for these two models are large (HunyuanVideo: 13GB DiT + 4.6GB
  LLaMA text encoder + VAE; Qwen Image: 8.3GB DiT + 4.7GB Qwen2.5-VL text encoder) — run one real
  end-to-end test at a time, not concurrently, and check `Get-CimInstance Win32_OperatingSystem`'s
  free memory before launching a big one if unsure.
- Don't fabricate or round up findings. If the real re-run shows no visible change (like
  HunyuanVideo's own flipSinToCos fix did), say exactly that — it's still useful, real, negative
  evidence, not a failure to hide.
