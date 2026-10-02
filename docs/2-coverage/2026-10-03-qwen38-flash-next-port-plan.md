# Qwen 3.8 Flash Next port plan (`qwen4exp`)

**Status:** not started. **Policy:** port now, prove later; not admitted, not advertised
(CLAUDE.md rule 14; [ported-families-todo](ported-families-todo.md)).

## Architecture

Source: TensorSharp `docs/models/qwen38-flash-next.md`, `Models/Qwen4Exp/` (19 files, about 7.4k
lines including CUDA/graph code). Not in our local llama.cpp or the vendored b10306, so TensorSharp
is the only local reference.

A hybrid MoE: GatedDeltaNet recurrent layers interleaved with full-attention layers, some
behind **Qwen Sparse Attention (QSA)**'s indexer. It also has:
- a **PLE n-gram embedding block** (`Qwen4ExpModel.Ple.cs`);
- **×4 hyper-connection streams**;
- a **512-expert MoE**;
- an MTP head (`Qwen4ExpModel.Mtp.cs`);
- image/video input through a Qwen3.5-VL mmproj with (T,H,W) IMRoPE.

48 layers.

Weights: `unsloth/Qwen3.8-Flash-Next-GGUF` (multi-shard); quants about 15-48 GB. **The small
quants fit this machine** (64 GB RAM), so real-checkpoint sanity is possible.

## Reuse in Stingray

- `HybridGdnForwardPass`: the GDN recurrence (chunked + sequential), the hybrid layer schedule,
  MoE (`qwen35moe`), the MTP head, and the optimized prefill (ADR-0002).
- Hyper-connections: the DeepSeek-V4 alpha's Sinkhorn mHC code (`DeepSeek4Graph`), if the recipe
  matches; verify against TensorSharp.
- Sparse attention: the `deepseek32` lightning-indexer alpha concept (QSA may differ; read first).
- Vision: the Qwen3-VL tower and M-RoPE are in `UnifiedVisionPipeline`.

## New work (phases)

- [ ] **0. Read.** `Qwen4ExpModel.cs`, `.Layers.cs`, `.Forward.cs`, `.Ple.cs`, `.Qsa.cs`, `.Mtp.cs`.
  Write the exact per-layer spec here, with tensor names and metadata keys, as was done for
  Muse-Glimmer.
- [ ] **1. Hyperparams + tensor set:** a `qwen4exp` branch in `ModelGraph` (layer types, QSA layer
  set, PLE dims, HC streams, expert counts).
- [ ] **2. Forward, text only:** probably a new `Qwen4ExpForwardPass` deriving from or composing
  the hybrid-GDN pass. The ×4 streams change the residual structure everywhere, so bolting it on
  is risky. Then PLE, HC pre/post, QSA indexer + masked attention, MoE (512 experts).
- [ ] **3. MTP head** (optional; reuse `MtpDecoder`).
- [ ] **4. Vision** (later): Qwen3.5-VL mmproj + IMRoPE.
- [ ] **5. Gate:** a `// qwen4exp — NOT admitted` block in `ModelCompatibility`.

## Verification

1. Synthetic tiny `qwen4exp` GGUF vs an independent spec-written reference forward (PLE, HC streams,
   QSA masking past top-k, one GDN layer, one attention layer, a 4-expert MoE).
2. Real checkpoint (smallest quant, about 15 GB): coherence; then token comparison against
   TensorSharp running the same GGUF (its `ggml_cpu`/`cpu` backend), or a llama.cpp build that adds
   `qwen4exp`.
3. Admission via the normal path.

**Effort:** about 1-2 days to ported + synthetic-verified (text). It's the hardest family here.
