# Inference improvement leads (plan, 2026-10-10)

Status: **proposed, not started.** Order of work: first let the sample-image queue finish (Wan 2.2, HunyuanVideo, FLUX.2; see
[the samples and Qwen plan](2026-10-10-diffusion-samples-and-qwen-image-performance-plan.md)), then take the leads below in
order. These leads came from reading the performance league ([PerformanceLeague.md](../../PerformanceLeague.md)), the backlog
([00-current-work.md](../00-current-work.md)) and the vendored reference code in `examples/`. Several league rows date from
September, so **each lead starts by re-measuring today's number** before any code changes.

Ground rules (from CLAUDE.md and the Qwen work): measure before and after with alternating runs and medians; a change is
kept only if it is measurably faster and numerically acceptable; every change keeps an environment-switch rollback; one model
job at a time (shared-memory iGPU); iGPU timings describe this APU, not a discrete GPU (rule 13).

## Deferred on purpose

**Reference per-op timings for the video/large diffusion models (Wan 2.2, HunyuanVideo, FLUX.2) are explicitly deferred until Leads 1 to 5 below are all done** (Leads 1 to 5 themselves wait for the sample images to finish). They are not to be started before that, even after the
sample images exist. The method is ready for that day: run `sd-cli --backend vulkan0`
with `GGML_VK_PERF_LOGGER=1` on the same workload (it prints per-op GPU time and GFLOP/s by shape), compare op by op with our
stage timings, as was done for Qwen Image. Notes already gathered for that day:
- Wan's GPU weights (`WanGpuWeights`) are dequantized to FP32 and uploaded as FP16 for every block, with no residency
  budget; Wan 2.2 14B therefore holds about 28 GB (the live run showed a process near 27 GB). A single 256x256 frame is
  compute-bound, so quantized weights would mainly cut memory (to about 9 GB), not time. Worth doing as a footprint fix.
- HunyuanVideo (fp8) and FLUX.2 (Q4_K plus a 24B Mistral encoder) are the two heaviest runs; stage timings from the sample
  runs will say where their time goes.

## The leads, ranked

### Lead 1: Vulkan raw-quant matvec bandwidth (backlog item 6)
- **Evidence:** decode is bandwidth-bound and reaches only about 13-18 GB/s here. Our M=1 quantized matvec moved about 13 GB/s
  (3.9 MB in 0.3 ms); gpt-oss decode is MXFP4-matvec-bound at about 18 GB/s. A sustained-memory-bandwidth ceiling for this
  APU has not been measured, so "how far from the limit" is unknown.
- **Reference:** ggml's `mul_mat_vec_q3_k/q4_k/q5_k/q6_k.comp` (`examples/stable-diffusion.cpp/ggml/src/ggml-vulkan/vulkan-shaders/`),
  timed with the vendored `test-backend-ops` (`examples/stable-diffusion.cpp/build/bin/test-backend-ops.exe perf`).
- **First step (cheap, no code change):** measure a streaming-read bandwidth ceiling on the device, then time ggml's
  mat-vec for Q3_K/Q4_K at the model shapes and compare effective GB/s with ours (same bytes-read-over-time convention).
- **Upside:** up to about 2x Vulkan decode if the reference is near the memory ceiling; also speeds every M=1 projection
  (Qwen modulation, Wan/FLUX conditioning). **Stop rule:** if the reference is within about 10% of ours, park the lead.

### Lead 2: batched prefill on the Vulkan LLM path (backlog item 5)
- **Evidence:** `VulkanLayerSplitForwardPass` prefills one token at a time; Vulkan prefill lost to CPU prefill on every
  model in the September sweep (Qwen3-0.6B: 23.6 vs 226 t/s). The code today hands long prompts to the CPU
  (`HybridCpuPrefill`) as a workaround.
- **Reference:** ggml's `mul_mm` path, and the quantized GEMM kernels already improved this session (large tile, small-M
  tile, `STINGRAY_VK_QGEMM_SMALLM`).
- **First step:** re-measure today's Vulkan prefill vs CPU prefill on two models (one dense, one MoE) and read what the
  per-token path does per layer; write down which kernels a batched path can reuse.
- **Upside:** large for GPU-resident models, especially where the CPU hand-off cannot apply (model larger than RAM share,
  or when the CPU is busy). **Risk:** correctness across 10+ architectures; gate with the existing prefill-vs-decode logit
  parity tests per architecture. **Stop rule:** a measured win on both models, otherwise park.

### Lead 3: hybrid-GDN (Qwen3.5/3.6) prefill
- **Evidence:** prefill is sequential single-token evaluation; league ratios 0.05x (35B) and 0.16x (27B) against C++,
  the worst of any architecture measured.
- **Reference:** llama.cpp's delta-net / gated-delta-net implementation in `examples/llama.cpp` (chunked prefill form).
- **First step:** re-measure, and confirm whether the existing `VulkanBatchedPrefill` / GDN chunked-prefill work
  (see the env switches `STINGRAY_VULKAN_GDN_CHUNKED_PREFILL`, `STINGRAY_VULKAN_BATCHED_PREFILL`) already changed the
  picture; the league number may be stale.
- **Upside:** several times faster prefill on those models. **Gate:** teacher-forced logit parity against `llama-server`.

### Lead 4: MoE decode gap (OLMoE 0.50x)
- **Evidence:** already investigated in [2026-10-04-moe-decode-gap-investigation.md](2026-10-04-moe-decode-gap-investigation.md);
  that document lists what was tried.
- **First step:** read it, re-measure, and only reopen if a not-yet-tried idea remains (for example the per-expert Q4_K GEMM
  plan in [2026-10-04-q4k-expert-gemm-plan.md](2026-10-04-q4k-expert-gemm-plan.md)). **Upside:** moderate.

### Lead 5: FLUX.2 GPU (backlog item 8)
- **Evidence:** open in the backlog: production-shape GEMM ladder and the `DoubleBlockGpu` row-offset audit. Overlaps with
  the quantized-GEMM work from the Qwen thread, which shares the same kernel family (FLUX.2 uploads raw Q4_K).
- **Reference:** the PyTorch and sd.cpp references already in the repo; the `sd-cli` per-op logger (see Deferred).
- **First step:** after the FLUX.2 sample run, read its stage timings and check whether the kernel-family changes already
  help it (a switch A/B with `STINGRAY_VK_QGEMM_SMALLM`).

### Lead 6 (deferred follow-on): diffusion-side mat-vec kernels
- **Evidence:** the `MatVecDq*` family (Qwen, FLUX, SD3.5 and other diffusion M=1 projections) trails ggml too: Q3_K 15.8 vs 29.0 GB/s,
  Q4_K 23.6 vs 42.6, Q5_K about 26 vs 42.9 (same 4096x14336 shape, kernel time only, 2026-10-10). A 16-weights-per-lane variant was tried
  and was slower, so decode amortisation alone does not help.
- **Why deferred:** it is a separate body of work that would double the effort, and its end-to-end effect on diffusion is small today
  (Qwen's M=1 products are about 1% of a forward). Taken after Leads 1-5, using whatever structure the LLM-decode work settles on
  (aligned word loads, packed scale decode, scale factored out of the dot product, several rows per workgroup, whole blocks per lane).

## Not worth chasing now
CPU SmolLM2 prefill (about 0.7-0.9x per the notes), Qwen Image non-GEMM ops and VAE (at parity with the reference).

## Order and checkpoints
1. Image queue finishes (images committed, README updated).
2. Leads 1 to 5, in that order, only after step 1 (the images) is complete. Leads 4 and 5 may be parked with a written reason, but the reference timings in step 4 still wait until all five are done or parked.
3. After each lead: log measured numbers (dated) here, update `PerformanceLeague.md` rows, and commit.
4. Only after Leads 1 to 5 are all done: the deferred reference per-op timings for Wan 2.2, HunyuanVideo and FLUX.2 (and the Wan quantized-weights footprint fix, if wanted).

## Progress log

### Lead 1, first measurement, 2026-10-10 (done): the reference is about 1.8x faster, with large headroom
Method: ggml `test-backend-ops perf -o MUL_MAT -b Vulkan0` (vendored in `examples/stable-diffusion.cpp/build/bin/`) for the
reference; our own standalone benchmark for Stingray (one-row quantized mat-vec, 50 calls per submission so fixed overhead
drops out, effective GB/s = quantized-weight bytes / kernel time). Same GPU (Radeon iGPU, `uma 1, fp16 1, warp 64`).

Sustained-bandwidth ceiling on this machine (ggml, 4096 x 14336 mat-vec): F16 42.6, BF16 44.7, F32 44.1, Q8_0 45.1, Q4_0 41.4
GB/s, so about 44 GB/s is the practical ceiling.

| Type | Ours | ggml | Ours as share of ceiling | ggml as share |
|---|---|---|---|---|
| Q3_K, 4096x14336 | 1596 us, 15.8 GB/s | 868 us, 29.0 GB/s | 36% | 66% |
| Q4_K, 4096x14336 | 1400 us, 23.6 GB/s | 775 us, 42.6 GB/s | 54% | 97% |
| Q5_K, 4096x14336 | 1551 us, 26.0 GB/s | not in the log | 59% | n/a |
| MXFP4 | about 18 GB/s (league doc, gpt-oss) | 35.9 GB/s | about 41% | 82% |

Our rates are flat across the Qwen shapes (3072x3072, 3072x12288, 12288x3072: 15.6-15.9 GB/s Q3_K, 23.5-25.4 Q4_K), so it is the
kernel's inner structure, not a shape effect. Verdict: Lead 1 is confirmed, with up to about 1.8x available on Q3_K/Q4_K
mat-vec and about 2x on MXFP4. Next: compare our `MatVecDqQ3K/Q4K/Q5K` shaders with ggml's `mul_mat_vec_q3_k/q4_k.comp`
(workgroup size, rows per workgroup, vector loads, reduction), try the cheapest structural change, and A/B it.

### Lead 1, correction and the real picture, 2026-10-10: LLM-decode kernels measured
The first table above measured the diffusion-side `MatVecDq*` family. The Vulkan LLM decode path uses a different, older
family (`MatVecQ3K/Q4K/Q5K/Q6K/Q8_0/Q4_0/Mxfp4`, entry `VulkanBackend.MatMul(..., weightDType)` and `MatVecMxfp4`). Measured with
the same method (4096 x 14336 mat-vec, 50 calls per submission, effective GB/s on the quantized bytes) against ggml on the same GPU
(ggml's mat-vec ceiling on this machine is about 44 GB/s):

| Type | Ours | ggml | ggml faster by | Notes |
|---|---|---|---|---|
| Q4_K | 42.2 GB/s | 42.6 | 1.0x | parity: the most common LLM quant needs nothing |
| Q5_K | 35.8 | 42.9 | 1.2x | |
| Q6_K | 36.6 | 45.0 | 1.23x | part of every Q4_K_M / Q5_K_M model (ffn_down, output) |
| Q8_0 | 32.7 | 45.1 | 1.38x | many small models ship as Q8_0 |
| Q4_0 | 31.2 | 41.4 | 1.33x | |
| Q3_K | 16.1 | 29.1 | 1.8x | the Q3_K_S/M/XL checkpoints, for example Qwen3.6-27B |
| MXFP4 | 14.3 | 35.9 | 2.5x | gpt-oss-20B decode is MXFP4 mat-vec-bound |

Impact: any Vulkan LLM using these weight types, in proportion to the bytes they hold; Q4_K-only models are unaffected. Priority by
gap x how many models: MXFP4, then Q8_0, Q3_K, Q4_0, Q6_K, Q5_K. ggml's Q4_K kernel (for reference, reading it): aligned 32-bit
word loads, all block scales decoded with a few packed bit operations, the scale factored out of the dot product (q.y first,
then one multiply per sub-block) instead of dequantizing each weight, and several rows per workgroup so activation loads are
shared. The experiment of decoding 16 weights per lane in our diffusion-side mat-vec was slower (13.8-25 GB/s vs 15.8-26),
so decode amortisation alone does not help; access pattern and ALU per weight are what to change.

### Scope note, 2026-10-10: diffusion-side mat-vec kernels are parked
The `MatVecDq*` family (diffusion M=1 projections) also trails ggml (Q3_K 15.8 vs 29 GB/s, Q4_K 23.6 vs 42.6) and could use the same
ggml-style rewrite, but it is a separate body of work that would double the effort and matters little for diffusion speed today
(Qwen's M=1 products are about 1% of a forward). Deferred; revisit after the LLM-decode kernels below.

### Lead 1, MXFP4 first result, 2026-10-10
New `MatVecMxfp4V2` (`Shaders.cs`): one 17-byte block per lane (five unaligned 32-bit loads), the 32 values decoded with a packed 4-bit
magnitude constant (no dynamically indexed array), input read as aligned vec4s, same push constants and dispatch. Standalone, 4096x14336:
**14.1-14.3 GB/s -> 24.3-25.1 GB/s (about 1.75x)**, output sum of |c| identical to 8 digits (summation order differs, first element agrees to
6 digits). ggml reaches 35.9 GB/s, so further room remains. Switch: `STINGRAY_VK_MXFP4_V2=1` (experiment; default flips after end-to-end
gpt-oss parity and speed checks below).

### Lead 1, MXFP4 kept as the default, 2026-10-10
- Real model, gpt-oss-20B (`gpt-oss-20b-MXFP4.gguf`), Vulkan GPU, greedy, 48 tokens, alternating A/B (old kernel / new kernel, two rounds):
  decode **6.7 / 6.7 t/s -> 7.5 / 7.6 t/s (+12-13%)**, prefill 6.7-6.9 -> 7.7 t/s, generated text **identical** in both rounds.
- Correctness gate added: `VulkanMxfp4MatVecTests` compares the Vulkan mat-vec with the CPU MXFP4 dequantizer and a double-precision dot
  (rows 37-70, cols 32 / 96 / 1024 / 2880 / 14336, row offsets 0-11); both the old and new kernels pass; the shader-drift gate passes.
- `STINGRAY_VK_MXFP4_V2=0` restores the old kernel. Remaining gap to ggml on this shape: 25 vs 35.9 GB/s. Ideas not yet tried: load the
  17-byte block as six consecutive words and funnel-shift once instead of five unaligned two-word reads; several rows per workgroup to share
  the activation loads.

### Lead 1, Q8_0 and Q4_0, 2026-10-10: kept
`MatVecQ8_0V2` and `MatVecQ4_0V2` (`Shaders.cs`): one block per lane, block bytes read as aligned or 16-bit-shifted 32-bit words (block starts are even),
Q8_0 bytes converted with `bitfieldExtract`, input read as vec4s, scale read once per block. Same bindings, push constants and dispatch.
- Standalone, 4096x14336 (ggml in brackets): **Q8_0 33.0 -> 43.0-43.3 GB/s (+31%) [45.1]**, Q4_0 31.2-31.4 -> 33.5-33.9 (+7%) [41.4].
  Q8_0 is within 4% of the reference; Q4_0's per-nibble integer decode is ALU-bound and still about 18% behind.
- Real model, Qwen3-0.6B-Q8_0, Vulkan, greedy, 96 tokens, alternating A/B: decode **23.6 / 23.6 -> 25.3 / 25.9 t/s (+7-10%)**, prefill 27.4-27.6 -> 29.6-30.8
  t/s (+8-12%), text identical in both rounds. A 0.6B model is overhead-heavy, so larger Q8_0 models should gain more (not yet measured).
- Correctness gate added: `VulkanQuantMatVecTests` (Q8_0, Q4_0, Q3_K, Q4_K, Q5_K, Q6_K against the CPU dequantizer with a double-precision dot; 16 cases,
  new and old kernels both pass). Rollback: `STINGRAY_VK_MATVEC_V2=0`.
- Remaining Lead 1 order: Q3_K (1.8x behind), Q4_0 (18%), Q6_K (1.23x), Q5_K (1.2x); MXFP4 still 25 vs 35.9 GB/s.

### Lead 1 status, 2026-10-10: the large wins are shipped; the rest is parked with reasons
Shipped: MXFP4 (1.75x kernel, gpt-oss decode +12%) and Q8_0 (+31% kernel, within 4% of ggml; Qwen3-0.6B-Q8_0 decode +7-10%), plus Q4_0 (+7%).
Q4_K is already at parity with ggml (42.2 vs 42.6 GB/s). Parked:
- **Q3_K (1.8x behind):** ggml's kernel uses 16 cooperating threads per block with the 16 sub-block scales decoded cooperatively into shared memory
  (barrier per block); even so it reaches only 29 GB/s. Our kernel issues about a dozen small, partly unaligned loads per 8 elements. A faithful port is a
  restructuring, for an audience limited to Q3_K checkpoints (for example Qwen3.6-27B-UD-Q3_K_XL, which does not fit the GPU budget anyway).
- **Q6_K, Q5_K (about 1.2x behind), Q4_0 (about 1.2x):** each is a modest share of a typical model's bytes (Q6_K is roughly a fifth of a Q4_K_M model), so the
  end-to-end gain is a few percent each.
- **MXFP4 residual (25 vs 35.9 GB/s):** untried ideas are six-word funnel loads and several rows per workgroup.
Revisit if a profile shows one of these types dominating a model we care about. Next: Lead 2 (batched prefill on the Vulkan LLM path).

### Lead 2, first measurement, 2026-10-10 (Vulkan iGPU vs CPU, same build, ~500-token benchmark prompt, 8 generated tokens)

| Model | CPU prefill | Vulkan prefill | CPU decode | Vulkan decode |
|---|---|---|---|---|
| Qwen3-4B-Q4_K_M (dense, 36 layers) | 102.8 t/s | 38.0 t/s (2.7x slower) | 12.2 t/s | 6.0 t/s |
| OLMoE-1B-7B-Q4_K_M (MoE) | 175.0 t/s | 155.7 t/s (near parity) | 30.8 t/s | 14.2 t/s |

Reading: the September finding (dense-model Vulkan prefill much slower than CPU) still holds, while MoE prefill is near parity. Vulkan decode is about half
of CPU decode on both models even though the Q4_K mat-vec kernel is at ggml parity (42.2 vs 42.6 GB/s), so Vulkan decode on this machine looks overhead-bound
(per-token dispatches, syncs, non-mat-vec kernels, lm_head), not mat-vec-bound. Caveat: only 8 decode tokens, which is a noisy basis; re-measure with 48+.
No decode profiler exists for `GpuForwardPass` (`STINGRAY_PROFILE_DECODE` is the hybrid-GDN path's), so the first step of Lead 2 is to add one: per-stage
GPU/host time for a decode token and for a prefill chunk, then decide between batched prefill, fewer syncs, or kernel work.

### Lead 2, diagnosis, 2026-10-10: the batched prefill exists but runs at about 0.3 TFLOP/s; the fix is a prefill-sized tiled quantized GEMM
Correction to the backlog wording: `GpuForwardPass` already has a weight-amortising batched prefill (chunked trunk, `_canBatchedTrunk`: dense models with
Q4_K/Q6_K weights) and records each decode token into one command buffer; "prefills per token" describes `VulkanLayerSplitForwardPass` and the MoE path (which hands
prompts to the CPU). The slow part is the matmul kernel.
- Measured (Qwen3-4B-Q4_K_M, 493-token prompt, Vulkan, `STINGRAY_VULKAN_MM_STATS=1`): Path 1 (default) 38.3 t/s, Path 2 (the existing tiled `MatMulTiledQ4K/Q6K`,
  capped at 16 tokens per dispatch) 38.8 t/s. Weight traffic is 64,197 MiB (130 MiB/token) in both, versus a GEMM floor of about 5 MiB/token.
- That traffic is not the bottleneck: 64 GB at the measured 44 GB/s ceiling is about 1.5 s, but the prefill takes 12.9 s. About 4 TFLOP in 12.9 s is **0.3 TFLOP/s**, so both
  paths are compute/kernel-bound. A tiled quantized GEMM with a 128-row tile (`SgemmQ3K/Q4K/Q5K`, which reach about 1 TFLOP/s at M >= 128 in the Qwen Image work)
  would take the same prefill to roughly 4 s (about 120 t/s), CPU parity (CPU: 102.8 t/s).
- **Design:** (1) add a Q6_K decoder and `SgemmQ6K` (Q4_K_M models keep Q6_K for ffn_down in many layers and the output matrix; without it those layers pin the chunk at 16);
  (2) implement Path 2 for Q3_K/Q4_K/Q5_K/Q6_K by routing to `VulkanBackend.Sgemm` (activations [nTok, K] and outputs [nTok, N] already match its layout);
  (3) raise `Path2MaxTokensPerDispatch` from 16 to a prefill-sized chunk (128-256) and check the scratch buffers and batched attention/KV-append kernels accept it
  (the prefill chunk already follows `MaxTokensPerDispatch`; the speculative-decode verify constant `MaxBatchVerifyK` stays 16); (4) keep Path 1 as the default until the A/B proves Path 2.
- **Gates:** per-kernel CPU-parity tests (extend `VulkanQuantMatVecTests` style to the GEMM), then end-to-end greedy generation identical to Path 1 on Qwen3-4B and one more dense model,
  then the prefill A/B (target: Vulkan prefill at least 2x today's, i.e. at or above CPU) and a Vulkan decode regression check. Rollback: `STINGRAY_VULKAN_MM_PATH=1`.

### Lead 2, Path 3 first result, 2026-10-11 (Qwen3-4B-Q4_K_M, 493-token prompt, Vulkan iGPU, greedy)
Implemented (all opt-in, `STINGRAY_VULKAN_MM_PATH=3`; the default stays Path 1):
- `SgemmQ6K` (Q6_K decoder in the shared tiled GEMM; large and small tile) and `VulkanBackend.SgemmQuant` (Q3_K/Q4_K/Q5_K/Q6_K, explicit weight type for the LLM's generic raw tensors);
  CPU-parity tests `VulkanSgemmQuantEntryTests` (6 cases, M from 1 to 130, N not a tile multiple).
- `VulkanMatMulPath.Path3` (prefill chunk 128) routed in `MatMulBatched`.
- Sliced flash attention in the batched trunk (`GpuForwardPass.RecordBatchedFlashAttention`): the flash kernel holds one register accumulator per query and serves at most
  16 per dispatch, so a 128-token chunk previously fell back to the per-(head, query) `AttentionBatched` kernel, which re-streams the K/V cache for every query.
  Chunks above 16 now run as 16-query slices of the same flash kernel (`basePos` offset per slice; K/V for the whole chunk are already appended).
- Measured by elimination (matmuls skipped, cumulative ms for the whole prefill): Path 1 = 10.6 s matmul + 2.15 s other; Path 3 before the slicing fix = 5.1 s matmul + **14.0 s other**
  (the fallback attention); Path 3 after = 5.1 s matmul + about 1.8 s other.

| Qwen3-4B-Q4_K_M, 493-token prefill, Vulkan | Prefill |
|---|---|
| Path 1 (default) | 38.4 t/s (12.8 s) |
| Path 3 (128-token chunks, tiled quantized GEMM, sliced flash attention) | **71.3 t/s (6.9 s), +86%** |
| CPU, same prompt (2026-10-10) | 102.8 t/s |

Greedy generation is identical to Path 1. Matmuls are now about 0.8 TFLOP/s effective (4 TFLOP in 5.1 s). Weight traffic fell from 64.2 GB to 8.3 GB (16.8 MiB/token).
Next: larger chunks (256), a second dense model, decode-regression check, then decide whether Path 3 earns the default.
