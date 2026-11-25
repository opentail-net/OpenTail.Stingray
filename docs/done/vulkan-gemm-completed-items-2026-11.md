# Completed items split from vulkan-gemm.md (2026-11-25)

Finished items moved verbatim out of [../4-performance/gpu/vulkan-gemm/vulkan-gemm.md](../4-performance/gpu/vulkan-gemm/vulkan-gemm.md). Open items stay there. Each block notes the section it came from.

> from: ### Open items

- [x] Larger prefill chunks (256); check scratch buffers and batched attention/KV-append accept them. **Done 2026-11-24 (AFK loop): no gain beyond 128, default unchanged.** The Path 3 cap is already 256 by default (`STINGRAY_VK_PATH3_CHUNK` overrides it). A 1759-token SmolLM2-1.7B prefill on the full GPU path: chunk 128 gives 250.9 t/s, default 256 gives 250.5 t/s, within noise; greedy text identical in 4 of 4 pairs, so the 256-token scratch and attention path accepts the larger chunk. Controls: chunk 16 gives 70.3 t/s and chunk 64 gives 143.0 t/s, so the knob is effective and the rate saturates at 128. Record in [PerformanceLeague-LOG.md](../../PerformanceLeague-LOG.md).

> from: ### Open items

- [x] Decode regression check on Vulkan. **Done 2026-11-24 (AFK loop).** The change to check is the A31 host-cached download staging, which is on by default and sits on the decode download path. Interleaved A/B, SmolLM2-1.7B Q4_K_M, `-g -1` (the whole model on the integrated Vulkan GPU; every run's banner was `GpuForwardPass`), 128 new tokens, greedy, six pairs: staging not host-cached 27.97 t/s (27.1 to 28.5); host-cached default 28.87 t/s (28.6 to 29.3). The default is not slower, and the text is identical in 6 of 6 pairs. No regression. Record in [PerformanceLeague-LOG.md](../../PerformanceLeague-LOG.md).

