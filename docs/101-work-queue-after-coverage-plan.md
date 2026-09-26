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
- DONE (2026-09-26, Q6KPrefillGemm): group-paired Q6_K prefill GEMM over the stock GGUF bytes
  (no repacked copy). `vpunpck{l,h}qdq` of the stock-decoded sextet vectors lines up elements
  0-7 / 8-15 of the same four 16-element scale groups, so two `maddubs` results add in int16
  (max 32004) before ONE scale `madd` — 6 multiplies per 128 weights per token instead of 8 — and
  the row-outer / token-inner loop keeps 4 weight + 2 scale vectors in registers with activations
  as memory operands (the row-major `_8In` spills). A first version repacked Q6_K into a permuted
  copy; dropped because the one-off repack (~10 ms per 13 MB ffn_down) ate the prefill gain.
  Kernel microbench, b512, 16 threads, 3 rounds each: 2048x8192 row-major 155-286 -> paired
  233-315 GMAC/s; 2048x2048 203-245 -> 270-371. Single thread 2048x2048: 41 -> 68 GMAC/s.
  vs `TryMatMulBatchedQ8` on REAL SmolLM2 blk.0 ffn_down/attn_v at N=256: max abs diff 7.6e-5 on
  rms 11.4 (6e-6 rel; int dot exact, only per-lane float order differs).
  End-to-end, 1018-token prompt, 6 alternating runs each (`STINGRAY_Q6K_GEMM=0` A/B):
  225.9 -> 231.0 t/s mean (+2.3%; Q6_K is ~16% of MACs). Greedy 40-token output identical.
  PPL (wikitext-2, --batched): -c 2048 6.8611 -> 6.8865, -c 16384 (8191 tokens) 9.7624 -> 9.7661
  (+0.04%) — the 2k delta is re-quantisation noise concentrated in the first 255 tokens (bucket
  [256,1024) moves the other way), not a kernel error (see the real-weight diff above).
  Now ~231 t/s vs llama 260.6 = ~0.89x.
- DONE (2026-09-26): prefill per-token Q/K transforms (QK-norm, V-norm, RoPE) run across tokens
  in parallel; the KV append stays sequential. Bit-identical (same ops per token). Profile "RoPE"
  stage, 3 runs each: 159-161 -> 138-143 ms (~0.4% of prefill). The remaining ~140 ms is the
  serial append into freshly allocated KV pages (first-touch faults on ~400 MB for SmolLM2, no GQA).
- NEXT LEAD (remaining gap ~11%): Q4_K Path-2 GEMM itself (65% of trunk time), RoPE (scalar,
  ~3%), attention (~6%). Previous Q6_K note kept below for history.
- (superseded) Q6_K compute rate. Microbench (noisy on this machine, +-15%): Q6_K int8 tier
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
- DONE (2026-09-26): new Vulkan `SgemmBf16W` (fp32 activations x raw bf16 weights, SgemmF16's
  tiling; bf16->fp32 is a 16-bit shift, 32-bit loads only, no extension needed) +
  `IComputeBackend.SupportsBf16WeightSgemm`. UMT5 GPU weights now upload the mmap'd safetensors
  bf16 bytes directly — no host bf16->fp32->fp16 passes, and no fp16 range loss. Idle A/B, same
  build, alternating (3 each): fp16 21.6 (cold) / 14.7 / 14.5 s -> bf16 7.52 / 7.54 / 7.59 s
  (~1.95x; C++ Vulkan 13.0 s, our CPU path 6.5 s). 1-step 256x256 image vs CPU: mean abs 0.01678
  (bf16) vs 0.01681 (fp16) levels, max 1. Parity: VulkanRawWeightSgemmTests (vec4 + odd-K paths).

### HunyuanVideo GPU
CPU works (256² ≈19s/step, 512² ≈55s/step); GPU not attempted. Needs model-specific RoPE/data
handling. High potential, high effort.

**DONE 2026-09-26:** double + single blocks on Vulkan (`HunyuanVideoModel.Gpu.cs`), FLUX.1-style
GPU blocks with Hunyuan's differences applied as the CPU path has them (affine-free LayerNorm
modulation, image-first sequence, RoPE on image rows only via the compacted interleaved table,
biases everywhere); token refiner / embeddings / final layer stay on CPU (~0.6 s/step). Weights stay
fp8 in VRAM through the new `Shaders.SgemmFp8W` (fp32 x raw E4M3, exact in-shader decode, 32-bit
loads, no float8 extension) — ~13 GB instead of 26 GB fp16. `UseGpu` defaults on for a Vulkan
backend; `STINGRAY_HUNYUAN_GPU=0` forces CPU.
- Parity, real fp8 checkpoint, 64 image + 24 text tokens: cosine 1.000000, rel L2 1.6e-6
  (`HunyuanVideoGpuParityTests`, heavy). fp8/bf16 GEMM parity: `VulkanRawWeightSgemmTests` 6/6.
- End to end, 256², 1 frame, 8 steps, real LLaMA-3 + CLIP-L, seed 42 (ZzHunyuanProfTmp harness):
  CPU 18.0 s/step (168 s generate) -> GPU 8.8 s/step (99 s generate incl. one-off ~13 GB upload);
  images match (mean abs 0.00015 levels, max 1). A step is ~6.9 TFLOP (2 x 13B x 265 tokens) in
  ~8.0 s of blocks = ~0.86 TFLOP/s, ~42% of this Vega-8 iGPU's fp32 peak: compute-bound, so the
  remaining lever is the GEMM kernel itself (or fp16 math), not dispatch/upload.
- One OOM seen when the parity test ran concurrently with the full Diffusion suite (GPU weights
  live in shared system RAM on this APU); clean on rerun alone.

### Cross-cutting: GPU command batching / graph execution
Per-dispatch overhead (upload→dispatch→wait→download per op) keeps hurting small LLMs, SDXL,
diffusion, TTS/DiT. Missing piece: record N ops into one command buffer, submit once, sync once.
Highest-leverage GPU infrastructure left.

**Status check 2026-09-26 (measured, not assumed):** the "missing piece" already exists and is in
use. `VulkanBackend.BeginBatch/EndBatch` + `DispatchOrRecord` record into one command buffer with a
barrier per dispatch and submit/wait once; 24 diffusion/audio models call it, and GpuForwardPass /
HybridForwardPass / VulkanHybridGdnForwardPass record whole tokens. `STINGRAY_PROFILE_GPU_SPLIT=1`
now also reports per phase from `stingray run`: SmolLM2-1.7B `-g -1` decode = **1 queue submit per
token** (7 tokens -> 7 submits, ~88 ms/token of GPU time on this iGPU); prefill of 35 tokens = 225
submits (weight uploads + per-chunk). So for LLM decode the remaining cost is GPU execution, not
dispatch/sync overhead. Remaining per-op-sync users, if any, should be found the same way (count
submits per step) before building anything new.

### Cross-cutting: CPU safetensors packed GEMM audit
PackedSgemmF32 (load once, pack once, reuse panel) gave huge wins on Wan, LTX, SD3.5. Audit every
major safetensors-backed diffusion/audio model for the read-F32/convert/dot/discard anti-pattern.

**Progress 2026-09-26:**
- DcAeDecoder, TinyVaeDecoder, TinyVaeEncoder: their scalar direct convs now delegate to
  `DiffusionOps.Conv2D` (im2col + GEMM; same NCHW / "same"-padding / stride semantics, checked in
  the im2col index code). Note: none of the three has a production caller today (only shape tests
  with zero weights), so this is DRY + future-proofing, not a measured pipeline speedup.
- `DiffusionOps.Conv2D`'s shared 1x1 path (every 1x1 conv in the diffusion stack: VAE shortcuts
  and attention q/k/v/proj, UNet, ControlNet, LTX/Wan VAEs) was the dot-product anti-pattern
  (one `TensorPrimitives.Dot` per output pixel/channel, then a strided unpack pass). Now one packed
  GEMM with the roles swapped — kernel as the row operand, activations packed straight from NCHW
  into 16-pixel panels — so the output lands channel-major with no transposes. ms after warm-up:
  512x512 @4096 58-73 -> 10-14; 128->256 @65536 89 -> 28-39; 64x64 @65536 14.7 -> 7-10;
  32->64 @65536 14.8 -> 6-8; 256x256 @16384 ~16 vs ~17 (tie). SD1.5 256² image old vs new:
  99.87% pixels identical, max 1 level.
- Hunyuan VAE `Linear1x1` and pointwise `CausalConv3D` (per-channel axpy) now go through that
  shared 1x1 path: zero-latent decode pixel-identical, smooth latent 1 level at one pixel.
  Remaining: the VAE mid-block causal self-attention is still scalar seq x seq x ch (measure its
  share before touching it; at 32x32x1 latent the whole decode is a few seconds).
- Tried and reverted (2026-09-26): SD-family `VaeDecoder.LinearHW` (mid-block attention q/k/v/out,
  per-(token, output) Dot) as a packed GEMM. SD1.5 512², 2 steps, CPU, 2 alternating runs each:
  old 25.6/25.7 s vs new 28.6/26.4 s — no gain; the projections are not a measurable share here.
- Swept Audio/Diffusion for the weight-row Dot anti-pattern: remaining hits (HiggsCodecDecoder,
  VibeVoice ConvNeXt, XttsConditioningEncoder, LTX VAE timestep Linear) are single-vector matvecs,
  not multi-token GEMMs — nothing left there to pack.

## Found along the way

- 2026-09-26: `HybridGdnForwardPass_Ornith9B_SnapshotRestore` fails (2608/248320 non-finite
  logits, "giochi giochi..." from the CLI). NOT an engine bug: vendored llama.cpp prints "GGGG..."
  on the same file, clean HEAD and a 2026-09-19 build reproduce it, and the local
  `F:\_models\deepreinforce-ai_Ornith-1.0-9B-Q4_K_M.gguf` is 6,912,246,464 bytes vs the HF file's
  5,910,782,656 (etag 5035e767…). FIXED: re-downloaded (sha256 5035e767… matches HF), bad copy
  kept at `K:\_other_models\deepreinforce-ai_Ornith-1.0-9B-Q4_K_M.gguf.BAD-6912246464`. The test now
  passes on real weights (8.7 s, 0 non-finite logits); CLI output coherent.

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

### Step 1 — audit (2026-09-26)

CLI, "The capital of France is", greedy 24 tokens, `-g 0` vs `-g -1` (Vulkan, iGPU). "same" = same
greedy text; parity numbers are from `VulkanArchLogitParityTests` (real prompt + 8 teacher-forced
decode steps, CPU vs Vulkan logits).

| arch (file) | Vulkan result | notes |
|---|---|---|
| llama (SmolLM2-360M, Falcon3-3B, Mistral-7B) | same; Mistral parity cos 0.99974, 0 flips | SmolLM2 decode 29.7 vs 24.9 t/s CPU; prefill 30 vs 78 t/s |
| qwen2 (qwen2.5-0.5B) | same; cos 0.99966 | |
| qwen3 (0.6B) | same; cos 0.99847, 1 near-tie flip | |
| ernie4_5, maincoder, smollm3, xverse, hunyuan-dense | same | |
| gemma3 (4B) | same text; cos 0.99857 | see Gemma CPU finding below |
| gemma4 (E4B) | GPU was RIGHT, CPU prefill wrong | fixed on CPU, cos now 0.99893, 0 flips |
| phi3 (Phi-3-mini) | CRASHED (fused attn_qkv) | FIXED: fused qkv / gate+up row split; cos 0.99890, 0 flips |
| olmoe | same text; cos 0.9856 at 1 decode step, 0 flips | open: likely a routing near-tie, not yet proven |
| qwen3moe (Coder-30B-A3B) | same | |
| qwen35 hybrid (Ornith 9B) | same (Vulkan hybrid GDN) | |
| stablelm, cohere2 | SILENT GARBAGE | GPU pass has no LayerNorm / parallel residual -> now CPU with a note |
| gpt2, gptneox, phi2-file*, starcoder2, apertus | CRASHED (missing attn_q / ffn_gate) | now CPU with a note (LayerNorm, learned pos, non-gated FFN) |
| deepseek2 (MLA), gpt-oss | CPU fallback (by design) | |
| jais | not admitted (CPU too) | out of scope |

\* phi-2.Q4_K_M reports `general.architecture = phi3` in this file.

- `GpuForwardPass.UnsupportedReason(model, hp)` (CLI + server loader) replaces the MLA-only check:
  MLA, LayerNorm/biased norms, parallel residual, learned position table, non-gated FFN -> CPU with
  a one-line note instead of a crash or wrong logits. Next GPU work (step 2) = implement those.
- Gemma-4 E4B finding (the audit's biggest result): the Vulkan pass matched llama-server 16/16
  greedy tokens; the CPU PREFILL was wrong. Two CPU bugs, both fixed:
  1. prefill staged K/V at a `_maxHeadDim` head stride while every cache reader (decode K, the
     transposed V store) uses the layer's own head dim — half the heads read zero padding
     (token-0 attn_out cosine 0.7071 = 1/sqrt2). Now `StageCompactKv` (compact + zero tail).
  2. KV-shared layers (24+) left the cache length at `startPos` after the per-layer TruncateTo, so
     they attended to nothing from the chunk; now `TruncateTo(startPos + N)`.
  (A third suspect — Path-2 Q4_K GEMM, Q8_K one scale per 256, flipping a 2.4-logit greedy
  decision on a BOS-less prompt — was tried as a gemma Path-1 override and REVERTED on
  perplexity: wikitext-2 -c 2048, E4B batched Path 1 45.6086 / Path 2 45.3548 / sequential
  45.5941; gemma3 Path 1 21.1517 / Path 2 20.9569. Path 2 does not hurt aggregate quality. At
  kernel level with 50x outliers Path 2's rel err is ~3.2% vs Path 1 ~1.3%, equal otherwise.)
  CPU prefill vs sequential cos 0.908 -> 0.99929; CPU vs Vulkan 0.885 -> 0.99929; CPU greedy vs
  llama-server 16/16. `perplexity --batched` had refused per-layer-head-dim models on the stale
  claim that PrefillCore falls back to sequential for them — the guard hid these bugs; removed.

