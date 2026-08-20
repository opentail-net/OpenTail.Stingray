# Vision encoder FFN/QKV scratch-buffer reuse — 2026-08-20

Second pass over `OpenTail.Stingray.Vision`, requested directly after the dtype-safety migration
(`docs/vl-migration-plan-2026-08-20.md`) closed out: with all 13 encoders freshly re-read, real
models locally available, and the migration still in recent memory, look for further optimization
and safety opportunities while it's cheap to verify them against real weights.

## Finding: per-layer scratch-buffer reallocation

Every encoder's `Forward()` allocates its main scratch buffers (`qBuf`, `kBuf`, `vBuf`, `attnOut`,
`normed`) once, before the per-layer loop, and reuses them across all layers — correct, established
practice. But 15 of the encoders broke that pattern for the **FFN intermediate buffer** (and, where
present, the fused-QKV buffer): these were declared *inside* the `for (l = 0; l < _layers; l++)`
loop, so a fresh `float[numPatches * intermediate]` was allocated and later garbage-collected on
**every layer**, not once per `Forward()` call.

Two encoders (`Granite4VisionEncoder`, `PaddleOcrVisionEncoder`, `CogVlmVisionEncoder`,
`Gemma4VVisionEncoder`) already had the correct pattern — precompute `maxIntermediate` across all
layers once, allocate the buffer once at that max size, and read/write only the first
`numPatches * intermediate` elements each layer (safe because the same `intermediate` value is used
for both the write and the very next read within one layer's iteration, even though the backing
array is sized for the largest layer). `Granite4VisionEncoder` was the template this fix was copied
from verbatim.

**Fixed the same way** (hoisted `ffnMid`/`gateBuf`/`upBuf`/`mid`, and the fused-QKV buffer where
applicable, outside the layer loop; changed the elementwise activation loops from `buffer.Length` to
an explicit `numPatches * intermediate` bound via `.AsSpan(0, n)` or a local `int ffnLen`, since the
buffer is now oversized for any layer with a smaller-than-max intermediate dim):

`InternVlVisionEncoder`, `DeepSeekOcrVisionEncoder`, `DotsOcrVisionEncoder`, `KimiVisionEncoder`,
`MiniCpmVisionEncoder`, `QwenVlVisionEncoder`, `Glm4VisionEncoder`, `LlavaVisionEncoder`,
`PixtralVisionEncoder`, `NemotronVisionEncoder`, `Exaone4VisionEncoder`, `HunyuanVlVisionEncoder`,
`MimoVlVisionEncoder`, `Step3VlVisionEncoder`, `YoutuVlVisionEncoder` — 15 files.

**Not touched**: `Gemma3VisionEncoder`, `Gemma4VVisionEncoder`, `Llama4VisionEncoder`. These use a
structurally different, deeper pattern — per-token `Parallel.For` bodies that each allocate their
own small `localFfnIn`/`localUp`/`localFfnOut` scratch (required for thread-safety, since threads
can't share a single mutable buffer). `Gemma4VVisionEncoder` already hoists everything it can
outside its layer loop; `Gemma3` and `Llama4` still allocate per-*token*-per-*layer* inside
`Parallel.For`, which is a real, larger optimization opportunity (`Parallel.For<TLocal>` with
`localInit`/`localFinally` would convert per-iteration allocation to per-thread allocation) but a
higher-risk one — both files are explicitly documented elsewhere
(`docs/done/vision-attention-vectorization-2026-08-20.md`) as already carefully tuned and measured,
and touching their hot loops without a dedicated benchmark pass risks a regression in code this
session doesn't have a numeric oracle for. Flagged as a follow-up, not attempted here.

## Why this matters (concrete numbers, not estimated)

Measured against `dots.ocr-Q8_0.gguf`'s real declared dimensions (`v.blk.*.ffn_up.weight`:
`[1536, 4224]`, 28 layers) and the test image's actual patch count (81 output tokens after 2x2
merge ⇒ 324 patches pre-merge):

- **Before**: 28 allocations of `324 × 4224 × 4 bytes` ≈ 5.47 MiB each ⇒ **≈153 MiB of `ffnMid`
  churn per single image encode**, all short-lived Gen0 garbage.
- **After**: 1 allocation of the same ≈5.47 MiB buffer, reused across all 28 layers.

That's a ~28x reduction in this buffer's contribution to GC pressure (exactly the layer count, as
expected — the fix removes the layer-count multiplier, not the buffer itself). Encoders with fused
QKV (Kimi, MiniCpm, QwenVl, InternVL, Llava, DeepSeekOcr, Nemotron) get a second, same-shaped
reduction on the `qkv` buffer. For a server processing many images per session, this is real,
compounding GC pressure removed, not a one-off.

## Verification

Pure allocation-scope refactor — no math changed, so re-ran the exact same real-weight end-to-end
checks the migration doc already established as the correctness bar, confirming byte-identical
output token counts before/after:

| Model | mmproj dtype | Before | After |
|---|---|---|---|
| `dots.ocr-Q8_0.gguf` | Q8_0 | `81 soft tokens (1536-dim)` | `81 soft tokens (1536-dim)` |
| `granite-4.0-3b-vision-Q4_K_M.gguf` | F16 | `576 soft tokens (2560-dim)` | `576 soft tokens (2560-dim)` |
| `InternVL3-2B.Q4_K_M.gguf` | Q8_0 | `256 soft tokens (1536-dim)` | `256 soft tokens (1536-dim)` |

`dotnet build -c Release` clean for `OpenTail.Stingray.Vision` and `OpenTail.Stingray.Cli` after
every file touched. No test suite exercises per-encoder `Forward()` numerically beyond these three
real-model fixtures (same gap the migration doc already documents — no oracle for exact output
values, only shape/token-count/no-crash), so this is the same verification bar the rest of the
migration used, applied consistently.

## Safety review (the other half of the ask)

Reviewed every `*Encoder.cs` and `VisionOps.cs` for remaining raw-pointer risk beyond what the
dtype-migration already closed:

- **`VisionOps.GetTensorPtr<T>`** (float-only now) stays a raw pointer by necessity — GC-tracked
  `Span<T>`/array fields can't live on a heap-allocated class the way these encoders are shaped
  (`Span<T>` is a ref struct, illegal as a class field), and the tensor data itself lives in
  memory-mapped native memory, not the CLR heap, so there's no GC-safety story to gain by wrapping
  it further. It remains dtype-guarded (throws `NotSupportedException` on a mismatch instead of
  silently misreading), which is the safety property that actually matters here.
- **Norm/bias `float*` fields**: still raw pointers into mmap'd tensor data, same reasoning as
  above. Genuinely read element-wise inside `dim`-bounded loops driven by the same `_embd`/`dim`
  value the tensor was declared with — no length mismatch risk found by inspection beyond what the
  dtype guard already catches.
- **Element-indexed tensors that previously caused a real crash** (position/class embeddings, fixed
  in the prior pass by converting to bounds-checked `float[]` via `DequantizeToFloat32`) were
  spot-checked again here — confirmed none regressed back to a raw-pointer read during this pass's
  edits (the buffer-hoisting changes only touched FFN/QKV scratch buffers, a disjoint set of
  fields).
- No new `Half*`/blind-cast pattern found; the sweep from the migration's cleanup step
  (`docs/vl-migration-plan-2026-08-20.md`) still holds.

No further safety issue found worth a code change beyond what's already fixed in the two prior
passes. The remaining unsafe surface (raw pointers into mmap'd GGUF memory) is structural to how
this codebase reads memory-mapped model weights without copying them, matches the pattern
`OpenTail.Stingray.Cpu`/`OpenTail.Stingray.Engine` already use for the main LLM engine, and isn't
something a targeted fix here should change without a much larger redesign.

## Rejected / deferred

- **Per-token `Parallel.For` scratch reuse in Gemma3/Llama4** (see above) — real opportunity, larger
  and riskier, deferred as a dedicated follow-up rather than squeezed into this pass.
- **`VisionOps.Attention`/`AttentionGqa`'s per-head `scores`/`temp` allocation** (inside
  `Parallel.For(0, heads, ...)`, one alloc per head per call, already vectorized via
  `TensorPrimitives` per `docs/done/vision-attention-vectorization-2026-08-20.md`) — same shape of
  win as this pass (`Parallel.For<TLocal>` local-init reuse), smaller in absolute terms since `heads`
  is typically ≤32 vs `layers × numPatches`'s much larger multiplier for the Gemma3/Llama4 case
  above. Noted, not implemented this pass — the FFN/QKV win above is the same technique applied
  where the multiplier was largest.
