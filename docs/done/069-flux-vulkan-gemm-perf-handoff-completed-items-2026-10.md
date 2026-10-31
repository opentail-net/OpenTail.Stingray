# Completed items split from 069-flux-vulkan-gemm-perf-handoff.md (2026-10-30)

Finished items moved verbatim out of [../4-performance/diffusion/069-flux-vulkan-gemm-perf-handoff.md](../4-performance/diffusion/diffusion-performance.md). Open items stay there. Each block notes the section it came from.

> from: ## External analysis (ChatGPT, 2026-09-13, given the table above)
- [x] 1. **Build a per-dispatch, per-shader-type GPU timing breakdown first**, before changing anything.
   Use real Vulkan timestamp queries (`vkCmdWriteTimestamp`), not just the existing
   submit-to-fence-signaled wall-clock (`STINGRAY_PROFILE_GPU_SPLIT`), so recording/queue-wait time
   is separated from actual GPU execution time per dispatch. (Done: detailed profiling breakdown measured).

> from: ## External analysis (ChatGPT, 2026-09-13, given the table above)
- [x] 2. **Build a standalone GEMM microbenchmark** using the *real* FLUX matrix shapes (not synthetic
   square matrices) — the actual `[M, K] × [N, K]` shapes that occur in `FluxDiT`'s double/single
   blocks (`d=3072`, `d*3`, `d*4`, `d*6`, `d*7`) at real sequence lengths (`nImg=1024`, `nTxt=256`, `nSeq=1280`).
   (Done: measured 591-611 GFLOP/s across Lin1/Lin2/MLP/QKV shapes).

> from: ## External analysis (ChatGPT, 2026-09-13, given the table above)
- [x] 3. **Confirm the actual arithmetic path**, not just the type declarations. Inspect SPIR-V for FP16
   storage with FP32 accumulation and avoid activation overflow. (Done: SgemmF16 optimized with 128 threads and transposed LDS).

> from: ## External analysis (ChatGPT, 2026-09-13, given the table above)
- [x] 4. **Consider concrete kernel changes**: tile size, register-tile shape, workgroup size, and attention kernels.
   (Done: `SgemmF16` 64×128 tile, 128 threads + `MultiHeadAttentionTiled128` 32×16 tile; overall time improved 858s → 374.7s, then to 132.0s with register-tiled flash attention).

> from: ## External analysis (ChatGPT, 2026-09-13, given the table above)
- [x] 5. **Look at what GGML's own Vulkan backend actually does** for the equivalent matmul on this exact
   device (`sd-cli.exe` comparison).

