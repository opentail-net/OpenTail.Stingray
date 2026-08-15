# Phase V4 — Vision Decoder Splice (To-Do, Not a Detailed Plan)

Rough estimate: 12–18 hours. Bigger than any single encoder port — cross-cutting across all
four forward-pass backends, not isolated to one file set.

Three vision encoders (`gemma3`, `gemma4v`, `llama4`) are built and structurally sanity-tested,
but none are wired into actual text generation. This is that missing wiring.

## To do

- Embedding splice: replace image-placeholder token embeddings with the vision encoder's output
  tokens before the decoder trunk runs.
- New attention-mask mode: bidirectional within the image-token block, causal everywhere else.
  Doesn't exist anywhere in the engine today.
- Wire into all four forward-pass backends: `ForwardPass` (CPU), `CudaForwardPass`,
  `GpuForwardPass` (Vulkan), `HybridForwardPass`.
- Tokenizer/prompt-side image-placeholder token handling (`<image>` or similar → N placeholder
  tokens matching the encoder's token count).
- CLI/server surface: `--image`/`--mmproj` plumbing through to whichever encoder matches the
  loaded model's `clip.projector_type`.
- First real end-to-end test harness for "prompt + image → sane completion" — none exists yet;
  the encoder tests are structural-sanity only (shape/no-NaN, not correctness).
- Decide per-model whether image-token mask semantics actually match across `gemma3`/`gemma4v`/
  `llama4` or need per-architecture handling — not assumed identical, needs its own check.

Not scoped here: numerical parity verification against llama.cpp for any of the three encoders
(no oracle exists yet for any of them — a separate, prerequisite gap).
