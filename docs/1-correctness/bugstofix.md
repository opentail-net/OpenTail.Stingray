Resolved entries split out to
[done/bugstofix-resolved-2026-08.md](../done/bugstofix-resolved-2026-08.md), 2026-08-15 (most recently
updated 2026-08-27 with the `DeepSeekMoeGraph.cs`/`MlaAttention.cs` column-major layout fixes and
the `IqCodebooks.cs` entry, which had already been marked FIXED but was left here by mistake). This
file keeps only what's still open.

**Closed investigation moved out** (2026-08-27): the full DeepSeek-V2-Lite MLA/YaRN/MoE-routing
investigation (`ModelCompatibility.cs:460`/`461`, originally logged 2026-08-21) is now at
[done/032-deepseek2-mla-yarn-moe-routing-investigation.md](../done/032-deepseek2-mla-yarn-moe-routing-investigation.md).
tl;dr: multiple real bugs found and fixed along the way (YaRN/kq_scale, expert_weights_norm/scale,
RMSNorm/softmax/SiLU double-precision & ggml-exp fidelity, and — the final entry — a genuine
Q8_0 activation-quantization bug in `MatVecQ8_0` that had silently invalidated the earlier
"native Q8_0" measurement). None of them individually or together produce "Paris" from this
checkpoint+prompt; the router's top-6-of-64 routing decisions are chronically near-tied
(median margin ~0.002) regardless of quantization level (Q2_K through native Q8_0), which is a
property of the trained weights, not of numerical precision. Investigation closed; do not
restart a fourth round of kernel-level chasing on this checkpoint without new evidence.

## Tracked items

- [ ] **GLM-4.5 (`glm4moe`) perplexity 1.9% worse than llama.cpp; not admitted** (logged 2026-09-27; `docs/done/102-status-open-items-plan.md` #16).
  - **Checkpoint:** `cerebras_GLM-4.5-Air-REAP-82B-A12B-Q2_K.gguf`. It mixes quant types: attn_q
    Q2_K, attn_output Q5_K, expert gate/up Q2_K, expert down IQ4_NL.
  - **Result:** wikitext second-half PPL at -c 2048 is 8.7753 vs `llama-perplexity` 8.6125.
  - **Already fixed:** a tokenizer bug (`glm4` pre-type unmapped, commit 4349db3) that made it 37%.
  - **Ruled out:**
    - tokenisation: a 20 KB sample matches `llama-tokenize` exactly;
    - RoPE: the partial NeoX table is built over `rope.dimension_count`=64;
    - batched vs sequential MoE prefill: identical PPL;
    - IQ4_NL dequant and codebook: identical to ggml.
  - **Symptom:** the error grows with position. Next-token log-probs vs `llama-server` given the same
    ids differ by about 0.01 nats at 5 tokens, about 0.1 at 23, and up to 0.85 at 326, with the same
    top-5 order. So look at what accumulates over context: attention over many keys, or
    sigmoid+bias routing drifting as the hidden state drifts.
  - **Next step:** compare per-layer hidden states for the last token of the 326-token wikitext
    prompt (first 1400 bytes of `scripts/kvarn-gate/wiki.test.raw`).
    - llama.cpp side: `examples/llama.cpp/llama.cpp/build-eval/bin/llama-eval-callback.exe`, tensors
      `ffn_inp-N` / `l_out-N`.
    - Our side: `StageCapture` stages `post_attn_resid` / `post_ffn_resid`.
    - Find the first layer whose last-token values drift.
  - **Memory:** llama.cpp needs about 46 GB for this file, so run it alone.
  - **Layer bisection done 2026-09-27:** last token of the 326-token prompt.
    - llama.cpp `llama-eval-callback` `attn_norm/ffn_inp/l_out-N` vs our `StageCapture`, via the
      scratch harness `ZzLayerDumpTmp` (untracked).
    - **No single broken layer.** `attn_norm-0` matches exactly (0.0000). The first difference, about
      0.3%, appears after layer 0's attention. It then grows steadily: about 1% by layer 5, 2-5% by
      layer 15, 5-10% in layers 20-45.
    - **Leading hypothesis:** activation-quantisation scheme differences in every layer's
      attention/FFN matmuls, not a logic bug:
      - our Q5_K matvec keeps activations F32 where ggml quantises them to Q8_K (GLM's `attn_output`
        is Q5_K in every layer);
      - our batched prefill quantises activations per row, not ggml's per-block Q8_K, for the 325
        cached K/V positions.
    - **Next experiment:** dump the `attn_out` stage too (llama.cpp `kqv_out-0`) to split attention
      from `wo`; then make the decode path use ggml's per-dtype activation scheme (Q5_K -> Q8_K) and
      re-measure the layer-0 difference and the PPL.
- [ ] **Dequantize.cs / IqCodebooks.cs coverage gap**: Port `iq1s_grid` (NGRID_IQ1S=2048) and decoders for `IQ1_S`/`IQ1_M` (`IQ1S_DELTA=0.125f`, distinct sign/shift scheme) and `IQ2_XS`/`IQ2_S` when needed by future GGUF models.
- [ ] **ModelCompatibility.cs / Kernels missing op coverage** (2026-09-27: the Mamba-2 `SSM_SCAN`/`SSM_CONV` path is now implemented on CPU in `ForwardPass.Mamba2.cs` for Granite 4.0-H / Nemotron-H; Mamba-1 and GPU remain): Implement `GGML_OP_SSM_SCAN` (the selective-scan recurrence, distinct from `SSM_CONV`), `RWKV_WKV6`/`RWKV_WKV7`, and DeepSeek-V4 ops (`LIGHTNING_INDEXER`, `DSV4_HC_*`, `SOLVE_TRI`, `WIN_PART`/`WIN_UNPART`).
- [ ] **SpeculativeDecoder.cs StepSampled/PLD bugs**: Confirmed real defect in speculative decode step sampling; currently unreachable/latent as no wired call path exercises it yet.
- [x] **DeepSeekMoeGraph.cs:172 ExpertOffsets off-by-index**: Resolved 2026-08-27 (moved to `docs/done/bugstofix-resolved-2026-08.md`).
- [x] **KvMemoryGovernor TOCTOU race & InferenceSession unguarded Fork() on CUDA**: Moot / resolved 2026-08-27 when superseded session types were removed.

```json
[
  {
    "file": "src/OpenTail.Stingray.Cpu/Dequantize.cs",
    "line": 0,
    "summary": "Drift check requested 2026-08-21: diffed examples/ggml/src/ggml-common.h's quant tables against ours to look for newer formats/moved tables since our IqCodebooks.cs/Dequantize.cs were derived. RESULT: MXFP4/NVFP4 (kvalues_fp4/kvalues_mxfp4) and IQ4_NL/IQ2_XXS/IQ3_XXS/IQ3_S (kvalues_iq4nl/iq2xxs_grid/iq3xxs_grid/iq3s_grid) match ggml's current literals exactly -- no drift, no fix needed there. HOWEVER: ggml-common.h also defines iq2xs_grid (512 entries, for IQ2_XS), iq2s_grid (1024 entries, for IQ2_S), and iq1s_grid (NGRID_IQ1S=2048 entries, for IQ1_S/IQ1_M) -- none of which exist anywhere in IqCodebooks.cs or Dequantize.cs. This isn't drift, it's a format that was never ported: DType.IQ1_S/IQ1_M/IQ2_XS/IQ2_S are declared in the enum (Tensor.cs) but Dequantize.ToFloat32 has no case for any of them (falls to the `default: throw NotSupportedException`), and ModelCompatibility.cs's IsSupportedWeightDType allowlist correctly does not admit them either -- so this is a real coverage gap, not a live correctness bug (nothing can load a GGUF using these dtypes today).",
    "failure_scenario": "Currently none -- these four dtypes are unreachable (no allowlist entry, and the decoder would throw NotSupportedException rather than silently produce wrong output, unlike the previous IQ2_XXS/IQ3_XXS/IQ3_S/IQ4_XS bug this session fixed). Becomes relevant only if a future GGUF is admitted with IQ1_S/IQ1_M/IQ2_XS/IQ2_S weights: would need iq1s_grid/iq2xs_grid/iq2s_grid copied verbatim from ggml-common.h into IqCodebooks.cs plus new decoders in Dequantize.cs (IQ1_S/IQ1_M also need IQ1S_DELTA=0.125f and a sign/shift scheme distinct from the IQ2/3 family), following the same real-table-not-fabricated-formula approach as this session's IQ2_XXS/IQ3_XXS/IQ3_S/IQ4_XS fix."
  },
  {
    "file": "src/OpenTail.Stingray.Engine/ModelCompatibility.cs",
    "line": 0,
    "summary": "Op-list drift check requested 2026-08-21: diffed ggml.h's `enum ggml_op` (examples/ggml/include/ggml.h) against what this engine actually implements. This engine already covers the ops it needs for its admitted architectures: GGML_OP_GATED_DELTAN_ET/SSM_CONV (GdnKernels.cs), GGML_OP_TOP_K (used by MoE routing), GGML_OP_FLASH_ATTN_EXT-equivalent attention, standard RMS_NORM/ROPE/MUL_MAT/GLU/SOFT_MAX paths, etc. Found ONE real, load-bearing gap: GGML_OP_SSM_SCAN (the actual Mamba/Mamba2 state-space recurrence -- distinct from SSM_CONV, which this engine does implement) has NO implementation anywhere (`grep -ri mamba|ssm_scan src/` is empty), and no Mamba/Mamba2/Jamba/Zamba/FalconMamba/Codestral-Mamba architecture string appears in ModelCompatibility.cs's allowlist at all -- consistent (not admitted), not a silent bug, but a real missing capability if any pure-SSM or hybrid-SSM architecture is ever wanted. Also unimplemented and unadmitted: GGML_OP_RWKV_WKV6/RWKV_WKV7 (RWKV6/7 architectures), GGML_OP_LIGHTNING_INDEXER/DSV4_HC_COMB/DSV4_HC_PRE/DSV4_HC_POST (DeepSeek-V4-generation ops -- newer than this session's deepseek2/V2-Lite investigation, ggml added these very recently), GGML_OP_SOLVE_TRI, and GGML_OP_WIN_PART/WIN_UNPART (Swin-style windowed vision attention -- OpenTail.Stingray.Vision covers Gemma3/Gemma4/Llama4 today, not window-attention vision transformers).",
    "failure_scenario": "None today -- every one of these is correctly un-admitted, so no architecture silently produces wrong output via a missing op; requests for these architectures simply fail model-compatibility checks, which is the intended fail-closed behavior. This is purely a capability/future-proofing note: Mamba-family (hybrid or pure SSM) support specifically would need a real GGML_OP_SSM_SCAN kernel (the actual selective-scan recurrence, not just SSM_CONV's causal conv1d prelude that already exists) added to SimdKernels/GdnKernels plus a new ForwardPass graph and ModelCompatibility allowlist entries -- a substantial new architecture family, not a small fix. RWKV6/7 and DeepSeek-V4's DSV4_HC_*/LIGHTNING_INDEXER ops would each similarly require their own dedicated kernel + graph work. None of this blocks anything currently supported."
  }
]
```

**Cut for the cap but still real and worth a look**: `SpeculativeDecoder.cs` StepSampled/PLD bugs — confirmed as a real defect but currently unreachable/latent, no wired call path exercises it yet.

**Historical, now moot** (both files were deleted 2026-08-27 when `InferenceSession`/`InferenceRuntime` and their superseded predecessors were removed — see `docs/030-delete-inferencesession-todo.md`): the `KvMemoryGovernor` TOCTOU race and the `InferenceSession` double-release/unguarded-`Fork()`-on-CUDA findings no longer apply to anything in the codebase.

**Resolved 2026-08-27** (moved to `docs/done/bugstofix-resolved-2026-08.md`): the `DeepSeekMoeGraph.cs:172` `ExpertOffsets[MaxExperts]` off-by-index this note used to flag.
