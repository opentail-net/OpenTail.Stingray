# ported-families-todo.md - completed items 2026-11

Ticked items moved from [../2-coverage/ported-families-todo.md](../2-coverage/ported-families-todo.md) on 2026-11-08.

> from: ### 1. Qwen 3.8 Flash Next (`qwen4exp`)
- [x] Real-weight execution, one prompt (paged run 2026-11-07; the layer-0 GDN mixer output matches llama.cpp). Full-stack numerical parity is **not** established: see the Qwen reconciliation in the programme plan.

## DeepSeek-OCR2 descriptor and vision encoder (from the DeepSeek section)

- [x] *(done 2026-11-03: the `deepseek2-ocr` descriptor (`DeepSeek2Architectures.cs`) and the vision encoder (`DeepSeekOcr2VisionEncoder`, `UnifiedVisionPipeline`) reference none of the `DeepSeek4`/`DeepSeek32`/`DeepSeek41` types, which live in their own forward-pass classes; and on current main the real-weight SAM parity tests against `llama-mtmd-debug` pass: `DeepSeekOcr2_Rainbow1024_SamMatchesLlamaMtmdDebug` 14.8 s and `DeepSeekOcr2_Rainbow768Tile_SamMatchesLlamaMtmdDebug` 5.5 s with `mmproj-deepseek-ocr-2-q8_0.gguf`.)* Audit `deepseek2-ocr` only, to confirm nothing in Phases 0-2 silently affects the
  existing working vision branch. Not a rebuild.
