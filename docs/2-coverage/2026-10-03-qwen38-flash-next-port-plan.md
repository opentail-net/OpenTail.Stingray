# Qwen 3.8 Flash Next port plan (`qwen4exp`)

**Status:** not started. **Policy:** port now, prove later; not admitted, not advertised
(CLAUDE.md rule 14; [ported-families-todo](ported-families-todo.md)).

## Architecture

**Primary reference: llama.cpp `src/models/qwen4exp.cpp`** (1,478 lines), in the local source
checkout since the 2026-10-03 pull to `bed0a8566`; not in the vendored b10306 binaries. Secondary:
TensorSharp `docs/models/qwen38-flash-next.md`, `Models/Qwen4Exp/` (19 files, about 7.4k lines
including CUDA/graph code). The upstream comments also cite transformers
`configuration_qwen4_exp.py`, vLLM and SGLang.

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

## Known implementation hazards (confirmed in `qwen4exp.cpp` `load_arch_hparams`)

1. **×4 hyper-connections are not a residual wrapper.** The stream count comes from
   `hyper_connection.count` and must be greater than 1. It sits on llama.cpp's DeepSeek-V4 HC
   machinery (`dsv4_hc_mult`) plus a qwen4exp-specific **low rank** (`hyper_connection.low_rank`).
   The hidden width the layers see is `count × n_embd`, so this touches every layer.
2. **GDN recurrent state + PLE:** prefix/state semantics matter (no rewind; exact-extension reuse
   only, as for `qwen35`). PLE applies on recurrent layers only (check the graph).
3. **QSA is not the `deepseek32` indexer verbatim.** It pools indexer keys in blocks of
   `compress_ratio` cells, with **one block size for the whole model** (`indexer_kpool`). The ratio
   comes from the per-layer `attention.compress_ratios` array (0 = not a QSA layer). It also has its
   own indexer head count, key length and top-k.
4. **Metadata-driven layer schedule:** the recurrent/full-attention split and compression ratios
   come from per-layer arrays, not a fixed period.
5. **MTP/NextN is an extra block,** not just an output projection.
6. **Vision is a separate problem** (Qwen3.5-VL mmproj, M-RoPE sections from
   `rope.dimension_sections`).

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

## Deferred (not in the initial port)

Vision (Qwen3.5-VL mmproj, video), MTP speculative decoding, batched/GPU paths, and prefix-cache
reuse beyond exact extension.

## Verification (levels as in [ported-families-todo](ported-families-todo.md))

1. **Specification test (level 2):** a synthetic tiny `qwen4exp` GGUF against a test-side
   reimplementation (PLE, HC streams with low rank, QSA pooling and masking past top-k, one GDN layer,
   one attention layer, a 4-expert MoE). It catches transcription errors; it is not independent.
2. **Independent implementation (level 3):** the same synthetic GGUF through a llama.cpp build from
   `bed0a8566` or later (has `qwen4exp.cpp`).
3. **Real weights (level 4):** the smallest quant (about 15 GB): coherence, then `stingray admit-arch`
   against that newer `llama-server` (or TensorSharp on the same GGUF).
4. **Admission (level 5):** the normal text-LLM path.

**Effort:** port + specification test about 1-2 days (text). It's the hardest family here.
Real-weight verification, the closeout performance + DRY pass, and admission are separate.
