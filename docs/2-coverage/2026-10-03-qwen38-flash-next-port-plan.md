# Qwen 3.8 Flash Next port plan (`qwen4exp`)

**Status:** in progress (Phase 0/1). **Policy:** port now, prove later; not admitted, not advertised
(CLAUDE.md rule 14; [ported-families-todo](ported-families-todo.md)).

## Architecture

**Primary reference: llama.cpp `src/models/qwen4exp.cpp`** (1,478 lines), in the local source
checkout since the 2026-10-03 pull to `bed0a8566`; not in the vendored b10306 binaries. Secondary:
TensorSharp `docs/models/qwen38-flash-next.md`, `Models/Qwen4Exp/` (19 files, about 7.4k lines
including CUDA/graph code). The upstream comments also cite transformers
`configuration_qwen4_exp.py`, vLLM `config.py` and SGLang `configs/qwen4_exp.py`.

A hybrid MoE consisting of 48 layers in a strict 3:1 pattern (`full_attention_interval = 4`):
- `(i + 1) % 4 != 0` (36 layers): **GatedDeltaNet (GDN) linear attention + MoE**
- `(i + 1) % 4 == 0` (12 layers, i.e. 3, 7, 11, ..., 47): **Qwen Sparse Attention (QSA) + MoE**

Additional architectural components:
- **Hyper-connections (HC)**: 4 parallel residual streams (`dsv4_hc_mult = 4`). Internal residual width is
  `4 × 2560 = 10240`. Block inputs/outputs are 2560.
  - Grouped RMSNorm with per-stream weights `[2560, 4]`.
  - Low-rank bottleneck of 320 (`hc_low_rank`): `down(10240 -> 320) -> SiLU(scale 1/4) -> up(320 -> 10240) -> sigmoid gate`.
  - Stream average collapse (`10240 -> 2560`) provides the block input.
  - Learned injection projection `w_inject` (`10240 -> 4`): block output is scaled by `2.0 * sigmoid(inject / 4)` and added to all 4 streams.
  - Note: This is an explicit `GatedResidual` mechanism. It is **not** DeepSeek-V4's Sinkhorn matrix hyper-connection (mHC).
- **PLE n-gram embedding block**:
  - Injected at layer 2 (`ple_layers = [2]`), which is a recurrent GDN layer.
  - ~20M-entry / 51B-parameter n-gram embedding table (`ple_layer_multipliers`, `ple_head_offsets`, `ple_head_vocab_sizes`).
  - Gated query/key projection with signed square root scaling:
    `s = sum(key * query) / sqrt(d); gate = sigmoid(sgn(s) * sqrt(|s|))`.
  - Broadcast value with grouped RMSNorm.
  - Dilated depthwise causal 1D conv (`ple_conv1d`) with `dilation = ple_ngram_size`, `hist = (kern - 1) * dil`.
  - Conv history is maintained per sequence inside the recurrent cache.
- **QSA (Qwen Sparse Attention)**:
  - Key pooling across blocks of `compress_ratio` cells (`indexer_kpool`).
  - Dedicated indexer projections (`index_q_proj`, `index_k_proj`) and RMS norms.
  - Top-k block selection (`indexer_top_k`), with tail preservation (`indexer_kpool_select_tail = true`).
  - Single Q projection producing interleaved `[q | gate]` per head, with `sigmoid(gate)` applied to attention output.
  - Causal masking, IMRoPE (`sections`), and sparse KV gather.
- **MoE (Mixture of Experts)**:
  - 512 routed experts, top-10 routed experts per token.
  - 1 shared expert per token.
  - Expert intermediate size $d_{ff} = 640$ (`n_ff_exp = 640`).
- **SSM Numerics**:
  - `mamba_ssm_dtype = float32`: GDN recurrent state and updates run strictly in FP32.
- **MTP Head**:
  - Optional multi-token prediction head; fed by `[enorm(e) ; hnorm(h)_s] -> eh_proj` per HC stream.
  - **Explicitly out of scope** for the initial text port.
- **Vision**:
  - Qwen3.5-VL mmproj with (T,H,W) IMRoPE.
  - **Explicitly out of scope** for the text port.

## Checkpoint & Hardware Realities

- **Total Parameter Count**: ~180B total (125B language + 51B PLE n-gram table + 4B MTP).
- **Weights**: Smallest published GGUF (`UD-IQ1_S`) is **72.5 GB**. Standard quants range from 79 to 192 GB (BF16 is 354 GB).
- **Hardware Constraint**: This dev machine has 64 GB RAM and an integrated GPU. Real-checkpoint verification **does not fit in RAM**.
- **Strategy**: Per CLAUDE.md rule 14, **synthetic verification is our primary gate**. Real-weight verification is deferred until hardware capacity permits. The architecture will be ported and verified synthetically, but kept unadmitted.

## Architectural Design: Composition over Inheritance

`Qwen4ExpForwardPass` will **own and compose** primitives rather than inheriting from `HybridGdnForwardPass`:
- `Qwen4ExpForwardPass`
  - owns `GatedResidual` (HC pre-mix, post-combine, and head norm)
  - owns `PleNgramState` and `PleConv` kernels
  - owns `GdnKernels` (FP32 recurrence state)
  - owns `QsaIndexer` and `QsaAttention` kernels
  - owns `MoeRouter` and expert projection infrastructure

`HybridGdnForwardPass` has assumptions baked in for standard 1-stream residuals and Qwen 3.5 routing. Composing primitives guarantees clean boundaries and avoids state corruption across the 4-stream residual.

## New Work (Phases)

- [x] **0. Specification & Hyperparameters Lock**:
  - [x] Map exact GGUF metadata keys and tensor names matching `qwen4exp.cpp`.
  - [x] Add `qwen4exp` architecture constants and parameter bindings to `Qwen4ExpHyperparams`.
- [x] **1. GatedResidual (Hyper-Connections)**:
  - [x] Implement 4-stream grouped RMSNorm, low-rank bottleneck (320), SiLU, stream-average collapse, and $2\sigma$ injection combine.
  - [x] Unit test `Qwen4ExpGatedResidualTests`.
- [x] **2. GDN + MoE Trunk (36 layers)**:
  - [x] Integrate the FP32 recurrence kernel via `GdnKernels.GdnRecurrenceDecode`.
  - [x] Implement 512-expert MoE (top-10 + 1 shared expert, intermediate 640).
  - [x] Unit test `GdnRecurrenceDecode` integration and MoE structure.
- [x] **3. PLE Subsystem**:
  - [x] Implement multi-head hash mapping / gather over n-gram table.
  - [x] Implement grouped RMSNorm + signed square root gating.
  - [x] Implement dilated depthwise causal 1D conv with state storage in recurrent cache.
  - [x] Unit test `Ple_ComputeGate_PositiveAndNegativeDotProducts` and `Ple_DilatedConv_AccessesExpectedTaps`.
- [x] **4. QSA Subsystem (12 layers)**:
  - [x] 4a: Implement indexer key pooling and top-k block selection (`Qsa_PoolIndexerKeys_AveragesAndNormalizes`, `Qsa_ComputeBlockScore_RectifiesAndScales`).
  - [x] 4b: Implement interleaved Q+gate projection, sparse attention, and causal masking (`Qsa_SplitAndNormQGated_And_ApplyAttentionGate`).
- [x] **5. Forward Pass Composition (`Qwen4ExpForwardPass`)**:
  - [x] Assemble the full 48-layer stack: 36 GDN layers, 12 QSA layers, PLE at layer 2, HC wrapping all mixers and MoE blocks.
  - [x] Implement the final HC head mix (acting as RMS output norm).
- [x] **6. Synthetic Full-Stack Parity**:
  - [x] Add the end-to-end tiny synthetic test (`Qwen4Exp_SyntheticForwardPass_RunsEndToEnd_ProducesFiniteLogits`).
- [x] **7. Not-Admitted Gate**:
  - [x] Add `// qwen4exp — NOT admitted` block in `ModelCompatibility.cs` (CLAUDE.md rule 14).
  - [x] Update `ported-families-todo.md`.

## Deferred (Out of Scope for Initial Port)

- Vision (Qwen3.5-VL mmproj, video, IMRoPE).
- MTP speculative decoding (`Qwen4ExpModel.Mtp.cs` / `MtpDecoder`).
- Real-weight validation (blocked on >64 GB RAM hardware).
- Batched / GPU CUDA/Vulkan graph paths.
