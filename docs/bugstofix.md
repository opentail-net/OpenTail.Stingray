Resolved entries split out to
[done/bugstofix-resolved-2026-08.md](done/bugstofix-resolved-2026-08.md), 2026-08-15. This file
keeps only what's still open.

```json
[
  {
    "file": "src/OpenTail.Stingray.Cpu/DeepSeekMoeGraph.cs",
    "line": 36,
    "summary": "STILL UNFIXED, but no longer on the exercised code path. This standalone file's column-major router bug was never touched this session -- the new CPU MLA work (ForwardPass.cs) reuses the codebase's existing, generic MoE routing (MoeFfn/MoeFfnBatched, already used correctly by Qwen3-MoE/OLMoE/qwen35moe) rather than this file's helpers, gated per-layer on the new IsMoeLayer(layer) so deepseek2's leading_dense_block_count is respected. Whether that generic path is itself fully correct for deepseek2's routing wasn't specifically re-verified as part of the numerical-bug hunt below.",
    "failure_scenario": "Unchanged from before: any caller that DOES route through this file's SelectTopKExperts would still get random expert selection. Not currently reachable from CPU deepseek2 inference, but the file itself remains broken and undeleted -- worth removing or fixing if anything ever calls it."
  },
  {
    "file": "src/OpenTail.Stingray.Cpu/MlaAttention.cs",
    "line": 42,
    "summary": "STILL UNFIXED, but no longer on the exercised code path. The new CPU MLA work (ForwardPass.cs: MlaComputeQkv/MlaComputeQkvBatched) is freshly written, quantization-aware code that does NOT call this file's CompressKvLatent/UncompressKv helpers at all -- their column-major transpose bug was never exercised or fixed.",
    "failure_scenario": "Unchanged from before for any caller of this standalone helper class. Not currently reachable from CPU deepseek2 inference (see ModelCompatibility.cs:460 entry below for what IS reachable, and its own unresolved numerical bug) -- worth removing or fixing if anything ever calls it."
  },
  {
    "file": "src/OpenTail.Stingray.Cpu/IqCodebooks.cs",
    "line": 33,
    "summary": "IQ2_XXS/IQ3_XXS/IQ3_S/IQ4_XS grid tables are procedurally fabricated from a bit-pattern formula rather than the real calibrated GGML codebook constants, and the matching decoders skip the real format's 9-bit index/7-bit sign-mask/per-group scale.",
    "failure_scenario": "Any GGUF quantized with these IQ formats dequantizes to plausible-looking but unrelated weight values, silently degrading generation to incoherence; the existing test only checks shape/NaN-absence, not reference values."
  },
  {
    "file": "src/OpenTail.Stingray.Engine/ModelCompatibility.cs",
    "line": 460,
    "summary": "PARTIALLY ADDRESSED, CPU only, STILL BROKEN. ForwardPass.cs (CPU dense path, the one RunCommand constructs when no GPU offload is requested) now loads the MLA-specific attn_kv_a_mqa/attn_kv_a_norm/attn_kv_b tensors, implements compress->RMSNorm->decompress K/V (legacy unsplit wkv_b layout only, matching DeepSeek-V2-Lite's actual GGUF), a YaRN-scaled RoPE table (SimdKernels.BuildYarnRopeTable), the deepseek2-specific kq_scale/mscale attention-score correction (ModelGraph.cs), and per-layer dense/MoE dispatch for leading_dense_block_count. Verified end-to-end against models/DeepSeek-V2-Lite-Chat.Q2_K.gguf: the model loads, prefills, and decodes with zero crashes across 27 layers -- a first for this architecture in the codebase. THREE real bugs were found and fixed in the YaRN/kq_scale formula chain by tracing the exact derivation through examples/llama.cpp/llama.cpp's src/models/deepseek2.cpp and src/llama-context.cpp (the raw rope.scaling.yarn_log_multiplier GGUF value needs the same /0.1f 'cancel the convert script' correction llama.cpp applies at load time [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]; the YaRN RoPE table's attn_factor and the kq_scale formula's attn_factor_org were both computed from a wrong constant instead of correctly reproducing llama-context.cpp's deliberate pre-divide-then-cancel derivation of cparams.yarn_attn_factor). Despite all three fixes (each independently verified formula-by-formula against the reference source), generated output is still numerically wrong for trivial prompts (e.g. 'The capital of France is' does not produce 'Paris'), while the same GGUF via a prebuilt llama.cpp binary (tools/llama.cpp/llama-cli.exe, b10306, no compiler needed -- see scripts/setup-llamacpp.ps1) correctly generates coherent, correct text. Extensively checked and ruled out as the remaining cause: Q/K/V tensor layout offsets against deepseek2.cpp's exact view offsets, the partial-RoPE pairing convention, MHA-not-MQA at the cache level (numKvHeads), no double-application of the attention scale, the Flash-64 fast-attention path correctly excluding headDim=192, and the Q2_K dequantization/matmul kernels at every batched-dispatch tier (verified byte-for-byte against ggml's block_q2_K struct and dequantize_row_q2_K). Root cause NOT identified. GPU backends (GpuForwardPass/CudaForwardPass/hybrid variants) remain entirely untouched -- CPU-only was the explicitly agreed scope, and was deliberately not extended given CPU itself isn't correct yet.",
    "failure_scenario": "No longer throws on load (the original Missing-tensor crash this entry described is resolved). Current failure: silently wrong generation with no crash, no NaN, no exception -- structurally valid tokens, numerically incorrect content. Next step if resumed: get real ground-truth intermediate tensor values (e.g. building llama.cpp's eval-callback tool, which needs the MSVC C++ build tools this session deliberately avoided installing in favor of the faster prebuilt-binary path) to diff against this codebase's actual intermediate values, since formula-level re-derivation against the reference source has been exhausted without finding the remaining bug."
  }
]
```

**Cut for the cap but still real and worth a look**, roughly in priority order: `DeepSeekMoeGraph.cs:172` (MoE `ExpertOffsets[MaxExperts]` off-by-index — silently drops expert gather/scatter on workspace reuse). Two candidates were explicitly REFUTED by verification: the `KvMemoryGovernor` TOCTOU race (safe — serialized through the session's own mutex with re-validation) and the `InferenceSession` double-release (safe — `Dispose()` is idempotency-guarded). Two more (`SpeculativeDecoder.cs` StepSampled/PLD bugs, `InferenceSession.cs` unguarded `Fork()` on CUDA) were confirmed as real defects but currently unreachable/latent — no wired call path exercises them yet.
