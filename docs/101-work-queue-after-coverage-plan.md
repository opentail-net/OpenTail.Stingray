# Work queue after the coverage plan (written 2026-09-26)

Order of work (the user's direction, 2026-09-26):

1. Finish `docs/100-actionable-models-coverage-plan.md` (Phi-3.5-MoE LongRoPE, GPT-OSS parity +
   admission, EXAONE 4.5 hybrid post-norm + vision CLI, DeepSeek V2-Lite).
2. Then the optimisation targets below (user's assessment, paraphrased).
3. Only once 1 and 2 are 100% done: plan and implement Vulkan/GPU support for the LLM models
   (write the plan into this file's "GPU for LLMs" section first, then work through it).

Progress notes go under each item with dates and measured numbers.

## Status log

- 2026-09-26: coverage-plan reference points captured.
  - GPT-OSS 20B: `GptOssForwardPass` greedy matches llama-server token-for-token on 12/12 tokens
    ("The capital of France is" → 12650,14396,271,1069,4674,483,261,2893,130142,3011,198,271;
    llama-server continues 3100,314,12011,3567,18249,13263,16500,568,2500,3325,28133,2874).
    11.25 t/s CPU decode. Not wired into the CLI (the CLI built the generic `ForwardPass`, which
    crashed on MXFP4 experts). YaRN (factor 32, orig ctx 4096) was missing; added.
  - `llama-completion.exe` diverges from `llama-server` after token 9 on gpt-oss; use llama-server
    (`/completion` with `return_tokens:true`) as the oracle.
  - Folded MoE decode (f14b954) threw on Q2_K/MXFP4/IQ* experts; fixed with a per-expert fallback
    (fd94fae).
  - DeepSeek V2-Lite: wrong from token 0 on BOTH Q2_K and Q8_0 (llama-server: " Paris is the
    capital of France."; Stingray: "lum"/zero-width chars). Not a router-margin effect as
    docs/done/032 concluded — a real bug. `STINGRAY_MOE_BATCHED_PREFILL=0` does not change it.
  - DONE Phase 2 (6480e9d): gpt-oss admitted. YaRN added; CLI/server route to GptOssForwardPass.
    Teacher-forced vs llama-server: 22/24 exact argmax, worst gap 0.063 (step 1 is a 0.02 tie inside
    llama.cpp itself; llama.cpp's own -fa on/off moves logits up to 0.13). 11.1 t/s CPU decode.
  - DONE Phase 1 (dba59ad): phimoe was allowlisted but broken. Root causes were NOT just LongRoPE:
    RMSNorm+bias (engine used LayerNorm because bias tensors exist), missing output.bias, missing
    top-k renorm, LongRoPE factors (chosen by CONTEXT SIZE like llama.cpp, not per position as
    docs/100 said) + rope.scaling.attn_factor 1.19024. 24/24 exact at -c 4096, 32/32 on a 214-token
    prompt at -c 8192. Phi-3.5-MoE Q3_K_M lives at K:\_other_models (F: is full).
  - DONE Phase 3 (93332cf, 69141da): EXAONE 4.5 33B. CPU: 64-layer SWA 3:1 + NoPE global layers
    (was missing entirely; the 1.2B receipt never exercised it), PrefillCore no-pre-norm guard (was
    crashing). Vulkan hybrid: optional pre-norm, post-norms, SWA window, RoPE-only-on-SWA,
    rope_freqs. 12/12 + 16/16 exact vs llama-server on CPU and -g 16. Vision: adapter markers were
    `<image>` (not in vocab) -> `<vision>`/`<|image_pad|>`; no structured-message work needed.
    iGPU: hybrid decode = CPU (1.8-1.9 t/s); hybrid prefill 1.8 vs CPU batched 10.3 t/s.
  - DONE Phase 4 (DeepSeek V2-Lite), admitted. Second bug: MLA decode reordered Q [nope,rope] ->
    [rope,nope] IN PLACE (q == _q), clobbering nope channels; decode diverged from prefill from the
    2nd position (n=1 matched because attention over one position returns V regardless of Q).
    Fixed via _mlaQRaw. Q2_K 16/16 exact; Q8_0 195-token prompt top-5 same order within 0.17
    logits, 14/24 exact then a 0.15-logit near-tie. -g falls back to CPU (no MLA on GPU).
    COVERAGE PLAN (docs/100) COMPLETE: all four phases done 2026-09-26.
  - (history) Phase 4 first finding: a gross bug — the shared expert is
    n_shared x expert dim wide (2 x 1408 = 2816) but every CPU path ran it at 1408 (half the rows,
    wrong down-proj stride). Fixed via ModelHyperparams.SharedExpertIntermediateDim (ForwardPass,
    HybridForwardPass/CudaHybridForwardPass CPU layers). First token now " Paris" (was "lum"/
    zero-width garbage); still diverges at token 2 on Q2_K (ours " (" 25.66 vs " is" 25.52; llama
    " is" at logprob -0.33, " (" not in its top 5) — investigating. The same bug likely hit
    qwen2moe (shared 5632 vs expert 1408). GPU shared-expert paths size by buffer length: audit
    in the GPU plan.
  - FIXED (b1019ee) — were pre-existing failures (identical on clean HEAD 49cc225
    ForwardPass.Fast 14 fails — MatMulBatchedEquivalenceTests.Q4K_* (e.g. batch=1 64x256 index 0
    batched -36.21 vs ref -35.98), BatchedMatVecTierTests.TieredFallback_DoesNotMisattributeSlots,
    MatMulBatchedQ8EquivalenceTests.GateOff_MatMulBatchedNeverCallsTheQ8Path; heavy
    MoeBatchedPrefillParityTests.BatchedMoePrefill_MatchesSequential_F32 and
    DecodePathParityTests.SingleSequenceBatchForwardMulti_VsPlainForward_ForTheSamePosition.
    Root causes: 8195547 conflated MatMulBatched's allowQ8 with activation quantization (new floatActivations param), and f14b954's folded MoE used F32 activations + non-FMA accumulation. All green now.
  - Test-harness trick (no admin needed): `models/_models` is a symlink to `F:\_models`; tests that
    only search `<ancestor>/models/<file>` can be pointed at it by running the test exe through a
    directory junction in a scratch dir whose `models` is a junction to `F:\_models`.

## Optimisation targets (from the user)

### SmolLM2 prefill (CPU)
~0.24–0.27x llama.cpp across 267→3218 tokens: flat ratio, so it's a fixed Q4_K GEMM disadvantage,
not KV/attention/length. llama.cpp's edge: `block_q4_Kx8` 8-row interleave + integer-domain scale
folding. Stingray's Q4Kx8 repack moved 0.33x→0.38x only. A better small-batch Q4_K GEMM benefits
many models. Learn from the layout, don't copy the implementation.

**Progress 2026-09-26:** the league's 0.24-0.27x was stale (OT ~50 t/s then). Measured now,
991-token raw prompt, CPU: OT 160-166 t/s vs llama-bench pp1024 260.6 t/s (-t 16) / 208.7 (-t 8).
- DONE (f801243, +15%): MatMulBatchedDualCached sent the FFN gate/up (largest Q4_K GEMMs) to the
  F32 dequant cache + BLAS whenever the cache was on (CLI default), bypassing the repacked Path-2
  GEMM; Q6_K likewise preferred F32 BLAS over the int8 tier. Now 183-187 t/s (~0.71x of llama's
  best thread count; ~0.81x at equal 8 threads). .NET thread sweep: 16 > 12 > 8.
- Profile after (5.2 s trunk): FFN 64.5%, QKV 19.5%, out-proj 5.7%, attention 5.0%, RoPE 2.8%
  (148 ms — scalar per-token rotation, vectorizable), RmsNorm 1%.
- DONE (b85bd33, +12%): Path-2 GEMM restructure (RHS shuffles staged in a 1 KB L1 buffer, LHS
  streamed) — bit-identical (output hash equal at batch 1/3/17/65/991); JIT spills 91/154 -> 14/17.
  201-226 t/s.
- DONE (b316550, +~5%): int8 prefill tier (Q6_K/Q3_K/Q4_0) tiled row-block x token-group so an
  8-token activation slice stays in L2; bit-identical. A/B 6 runs: 203-217 -> 217-237 t/s.
- STATUS: ~220 t/s vs llama 260.6 (-t 16) = ~0.85x (was ~0.62x at the start of this item).
- NEXT LEAD: Q6_K compute rate. Microbench (noisy on this machine, +-15%): Q6_K int8 tier
  ~160-350 GMAC/s vs repacked Q4_K ~400-600. llama.cpp has NO x86 Q6_K repack
  (repack.cpp: q4_0, q4_K, iq4_nl, mxfp4, q2_K only), so a Q6_K x8 repacked GEMM (8 rows in lanes,
  pre-decoded scales, no per-row hsum) would be original work and could beat llama here. Q6_K is
  ffn_down/attn_v on half the layers of every Q4_K_M model. Smaller: RoPE 150 ms (scalar,
  ~3%), attention 280 ms (~6%).
- (superseded) Earlier lead (kernel): RepackedGemmPath2 keeps 32 shuffled RHS vectors (sp1/sp2 x 16) live across
  the per-row-group `rp` loop — twice the 16 YMM registers. Check RyuJIT's spill code (JitDisasm)
  against the C original; consider precomputing the shuffled RHS into a 1 KB L1 stack buffer per
  (b, sb) so the rp loop uses memory operands deliberately. Also: attn_v/ffn_down Q6_K go through
  the generic int8 tier, not a repacked GEMM (a Q6_K x8 repack would cover 1/6 of FFN flops).

### Wan2.1 GPU (Vulkan)
End-to-end ~122s vs C++ Vulkan 60.3s. Stages: DiT 79.4 vs 45.2s (1.76x), UMT5 38.7 vs 13.0s
(2.98x), VAE 6.05 vs 2.02s (3.0x).
- UMT5 first: Stingray streams unquantized bf16 layers from disk sequentially; C++ keeps it
  resident/quantized. A residency/weight-loading architecture issue, not a kernel issue.
- VAE second (smaller share).
- DiT: 1.76x is respectable on the iGPU; don't attack it blindly.

**Progress 2026-09-26** (Wan2.1-T2V-1.3B, 256x256, 1 frame, 20 steps, seed 42, CLI `image`):
- Fresh baseline: Vulkan (`--device 0`; Wan's default is CPU) 105.3 s — UMT5 ~40 s, DiT 57.5 s
  (2.88 s/step), VAE 6.8 s. CPU 88.6 s.
- DONE cf37910: UMT5 converted the whole 4 GB token-embedding table per encode (twice per run) —
  now only the looked-up rows from the mmap; CLI uses EncodePairGpu (one layer stream for
  cond+uncond). UMT5 GPU 14.1 s (C++ 13.0), CPU 6.5 s. Output PNG byte-identical.
- DONE aad52f6 + efa1dda: VAE causal conv3d and the spatial resample conv via im2col +
  PackedSgemmF32 (packed weights cached). VAE 6.85 -> 1.83 s (C++ 2.02 s). Pixel diff max 1/255.
- NOW: CPU end-to-end 77.3 s; Vulkan ~88 s (DiT loop 57-62 s vs C++ 45.2 s — left alone per the
  "don't attack DiT blindly" note). On this iGPU the GPU UMT5 path (14.1 s) loses to CPU (6.5 s)
  because EncodePairGpu converts every BF16 layer to F32 on the host and uploads ~9 GB; lead:
  upload BF16 and convert on-GPU, or keep UMT5 on CPU when it is faster (needs a discrete-GPU
  measurement before changing any default — CLAUDE.md rule 13).

### HunyuanVideo GPU
CPU works (256² ≈19s/step, 512² ≈55s/step); GPU not attempted. Needs model-specific RoPE/data
handling. High potential, high effort.

### Cross-cutting: GPU command batching / graph execution
Per-dispatch overhead (upload→dispatch→wait→download per op) keeps hurting small LLMs, SDXL,
diffusion, TTS/DiT. Missing piece: record N ops into one command buffer, submit once, sync once.
Highest-leverage GPU infrastructure left.

### Cross-cutting: CPU safetensors packed GEMM audit
PackedSgemmF32 (load once, pack once, reuse panel) gave huge wins on Wan, LTX, SD3.5. Audit every
major safetensors-backed diffusion/audio model for the read-F32/convert/dot/discard anti-pattern.

## GPU for LLMs

DRAFT (2026-09-26, written between coverage-plan steps; refine before starting, execute only after
everything above is done).

Goal: every admitted text-generation architecture runs on Vulkan (and CUDA where available) with
output that matches its own CPU path, and it is at least not slower than CPU on this iGPU where the
workload is large enough to amortize dispatch (CLAUDE.md rule 13: an iGPU loss is not evidence
against the GPU path — judge by FLOP/bandwidth reasoning or a discrete GPU).

Known today (from code reading, to be confirmed by step 1):
- Dense llama-family: GpuForwardPass (Vulkan full offload) and HybridForwardPass (partial `-g N`),
  CUDA equivalents. GpuForwardPass has post-norm (sandwich) support; HybridForwardPass hardcodes
  pre-norm names (the EXAONE 4.5 gap, docs/100 Phase 3).
- Hybrid GDN (qwen35/qwen35moe): VulkanHybridGdnForwardPass / CudaHybridGdnForwardPass, MoE FFN
  stays on CPU (SLRU).
- Gemma 4: explicitly rejected by HybridForwardPass on Vulkan.
- gpt-oss: CPU only (GptOssForwardPass, own class) — no GPU path at all.
- MLA (deepseek2), phimoe LongRoPE factors / RMSNorm+bias / output.bias: unknown on GPU.
- Speculative --draft-model: not on Vulkan.

Steps:
1. Audit (measure, don't guess): for each admitted arch with a local checkpoint, run the CLI at
   `-g 0`, `-g -1` (Vulkan) and a partial `-g N`, raw prompt "The capital of France is", greedy
   24 tokens, and record: runs / refuses / crashes / wrong tokens (vs its own -g 0 output), plus
   prefill and decode t/s. Produce the architecture x path matrix in this doc.
2. Fix correctness gaps first, smallest blast radius first: shared norm/bias/rope features missing
   on GPU (post-norm in HybridForwardPass, RMSNorm+bias, output.bias, LongRoPE factor tables,
   RopeAttnFactor) — each verified GPU vs CPU logits on real weights.
3. MoE on GPU for the dense ForwardPass family (phimoe, olmoe, granitemoe, qwen3moe, deepseek2):
   routed-expert matvec shaders over the existing raw-quant matvec kernels (Q4_K/Q3_K/Q2_K/IQ*/
   MXFP4), experts resident in VRAM where they fit, CPU-expert fallback otherwise.
4. gpt-oss GPU path: sinks + SWA + biased MoE + OAI SwiGLU as shaders, or fold gpt-oss into the
   shared GPU pass behind feature flags if step 3 made MoE generic.
5. Dispatch batching (shared with the "GPU command batching" item above): record a whole layer
   (or the whole token) into one command buffer; this is what makes small-model decode on GPU
   worth it. Measure before/after on at least two model sizes.
6. Gemma 4 on Vulkan hybrid; MLA on GPU; speculative decoding on Vulkan — only after 1-5.
7. Every step: GPU-vs-CPU logit parity test on real weights (timing-checked per CLAUDE.md
   rule 12), README matrix row updated with the dated evidence.
