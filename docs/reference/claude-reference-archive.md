# Claude Reference Archive

Archived reference material, model run examples, and extended design notes moved from `CLAUDE.md`.

---

## Detailed Model Commands & Example Invocations

### Perplexity Evaluation
```bash
# Perplexity over a corpus (accuracy gate for KV compression, issue #180). Supports
# --tq/--tq-mode exactly like the run command (auto = KVarN where supported, else Lloyd-Max).
dotnet run --project src/OpenTail.Stingray.Cli -c Release -- \
  perplexity -m model.gguf -f corpus.txt -c 2048 --tq
```

### Whole-Turn Structured Output (JSON Schema)
```bash
# Whole-turn structured output (grammar-constrained decoding, issues #423/#425). Mirrors
# llama.cpp's -j/--json-schema; the entire response is constrained to the schema. The root
# must be an object schema with at least one property; --json-schema-ordered emits keys in
# declared order. Server exposes the same via OpenAI/Anthropic response_format:json_schema.
dotnet run --project src/OpenTail.Stingray.Cli -c Release -- \
  -m model.gguf --temp 0 -p "Extract name and age from: Alice is 30." \
  -j '{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"]}'
```

### VibeThinker-1.5B (Qwen2 Math/Reasoning)
```bash
# VibeThinker-1.5B (Qwen2-based math/reasoning, issue #282). Loads as a standard
# qwen2 GGUF (QKV bias but no output-projection bias, no QK-norm, 28 layers / 2 KV
# heads, ChatML, tied embeddings). `download-model.ps1 -Model vibethinker` fetches the
# default Q8_0 (near-lossless); `-Model vibethinker-q4` is the smaller quant. Recommended
# sampling: temp 0.6, top_p 0.95, top_k 0, and no system prompt (the chat template supplies
# the math one). Emits a long <think> chain-of-thought then a \boxed{} answer.
dotnet run --project src/OpenTail.Stingray.Cli -c Release -- \
  -m models/VibeThinker-1.5B.Q8_0.gguf -g -1 \
  --temp 0.6 --top-p 0.95 --top-k 0 \
  -p "If 5x + 3 = 2x + 18, what is x? Show your reasoning."
```

### Multimodal Vision (Gemma 4, Gemma 3, Llama 4)
```bash
# Pass one or more images with --image and --mmproj. UnifiedVisionPipeline auto-detects projector type.
dotnet run --project src/OpenTail.Stingray.Cli -c Release -- \
  -m models/gemma-4-E4B-it.gguf --mmproj models/gemma-4-E4B-it-mmproj.gguf -g 0 \
  --image photo.png -p "Describe <image>"
```

### Ornith-1.0 (DeepReinforce, MIT)
```bash
# Ornith-1.0 (DeepReinforce, MIT) â agentic-coding "self-scaffolding" RL finetunes of
# Qwen3.5 / Gemma 4 bases, NOT a new architecture. Self-scaffolding is a training-time
# technique; at inference they're ordinary transformers. GGUF arches reduce to ones
# already dispatched: 9B = `qwen35`, 35B/397B = `qwen35moe`. Validated end-to-end
# (issue #411): the bartowski 9B Q4_K_M GGUF actually ships GDN tensors, so
# `_opentailllm.is_hybrid_ssm` auto-activates and it takes the SAME hybrid Gated-DeltaNet +
# attention path as the 35B/397B MoE variants (24 GDN + 8 full-attention layers,
# full_attention_interval=4) â not a plain dense transformer as the arch name alone
# suggests. Full CUDA offload (-g -1) fits comfortably in 8 GB VRAM (~3 GB weights
# uploaded; GDN state + dense FFN run on CPU by design of CudaHybridGdnForwardPass).
# Chat template loads via JinjaChatTemplate and tool calls parse via the qwen35moe-style
# QwenToolCallAdapter (Qwen3.6 XML `<function=..><parameter=..>` inside `<tool_call>`).
# `download-model.ps1 -Model ornith-9b` (Q4_K_M, 5.5 GB).
dotnet run --project src/OpenTail.Stingray.Cli -c Release -- \
  -m models/deepreinforce-ai_Ornith-1.0-9B-Q4_K_M.gguf -g -1 \
  --temp 0.6 --top-p 0.95 --top-k 20 -p "Write a Python LRU cache."
```

### Image Generation with Upscaling (Z-Image-Turbo + RRDBNet)
```bash
# ImageCommand auto-detects Z-Image vs FLUX from model. Z-Image uses Qwen3-4B text encoder; FLUX uses CLIP-L + T5.
dotnet run --project src/OpenTail.Stingray.Cli -c Release -- image \
  -m models/z_image_turbo-Q5_K_M.gguf \
  --vae models/z-image-turbo/vae \
  --qwen-encoder models/Z-Image-AbliteratedV1.Q5_K_M.gguf \
  --qwen-tokenizer models/z-image-turbo/tokenizer/tokenizer.json \
  --upscaler models/RealESRGAN_x4plus.safetensors \
  --upscale-blend 0.8 \
  -p "a serene mountain lake at sunrise" -W 512 -H 512 --steps 4 -o out.png
```

---

## Detailed Engine & Subsystem Notes

### Batched Forward Pass & Prefill Mechanics
- `ForwardPass.BatchForwardMulti(tokens[], positions[], caches[])` â batched multi-sequence decode; amortizes weight reads NÃ across concurrent users. Each sequence has its own `PagedKvCache`. Not supported for MoE or TurboQuant.
- `ForwardPass.PrefillWithCache(tokens, cache, startPos)` â prefills a per-sequence cache (used by `ContinuousBatchingEngine` during request admission). Admission is chunked (`STINGRAY_PREFILL_CHUNK`, default 256 tokens) and interleaved with decode steps; multiple in-flight prompts prefill as one packed pass via `ForwardPass.PrefillPackedMulti` and admission is gated by a KV token budget (`STINGRAY_KV_BUDGET_MB`) â issue #183.

### Speculative Decoding Subsystem
- `SpeculativeDecoder` (general draft-model speculation), `MtpDecoder` + `MtpBatchTail` (self-speculative Multi-Token Prediction / NEXTN heads, e.g. Qwen3.6-27B-MTP, with folded k-token batched verify, issue #207), `PromptLookupDraft` (prompt-lookup draft), and `DSparkDecoder` + `DSparkDraftModel`/`CudaDSparkDraftModel` (DeepSeek DSpark block-parallel safetensors draft heads, docs/dspark-plan.md / PR #413: EAGLE-3-style backbone conditioned on target hidden-state taps via `IForwardPass.EnableHiddenTaps` â CPU and dense-CUDA targets both capture; rank-256 Markov re-bias + confidence-trimmed verify on the host (`DSparkHostHeads`); greedy only â `--dspark-model <safetensors-or-dir> --temp 0` with `-g 0` or `-g -1`, placement via `DSparkPlacementPlanner` / `--dspark-place` / `STINGRAY_DSPARK_*`).
