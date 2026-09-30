# GLM-4.5 Q5_K×Q8_K activation-quantization implementation

> **Closed scope from bugstofix item 09:** implementation and synthetic-kernel verification are complete. This does **not** resolve the GLM-4.5 perplexity gap or admit the model. Real-checkpoint evaluation remains open as [docs/103-quickest-first-plan.md item 19](../103-quickest-first-plan.md).

## Original issue and evidence

- Checkpoint: `cerebras_GLM-4.5-Air-REAP-82B-A12B-Q2_K.gguf`; second-half wikitext PPL at `-c 2048` was 8.7753 versus llama.cpp's 8.6125 (1.9% worse).
- The layer-0 326-token investigation found `attn_out` / `kqv_out-0` matched the reference, while the first visible divergence (~2e-4) appeared after `wo` / `node_26`.
- The working hypothesis was that ggml's Q5_K matmul quantizes the activation to Q8_K, whereas the previous CPU Q5_K decode dot consumed F32 activations. This was not established as the cause of the full-model PPL gap.

## Closed implementation scope

- Added a separate `DotQ5K_Q8K` implementation and scalar fallback using the ggml Q5_K scale/min layout and Q8_K `bsums` correction. Existing `DotQ5K` numerics and paired-dot invariants remain unchanged.
- Added `Q5KDecodeQ8KActivations`, default off, at only the non-MLA CPU `ForwardPass.Decode` attention-output projection (`_wo[layer]`). Other Q5_K matvecs and GPU/hybrid dispatch are unchanged.
- `ZzLayerDumpTmp` can opt in for controlled runs with `ZZ_Q5K_Q8K=1` and restores the process-wide flag after each run.
- Synthetic oracle receipt: 5 cases (cols 256, 512, 1024, 2048, 4096), 8 seeded inputs per case. Dispatched and scalar kernels matched a test-local translation of `ggml_vec_dot_q5_K_q8_K_generic` within the predeclared `2e-4 + 2e-5 * |reference|` bound. The float-input overload matched the prequantized-Q8_K result.
- Focused regressions passed: `SimdKernelsQ5KQ8KTests` 5/5, `SimdKernelsQ8KSTests` 18/18, `SimdKernelsQ3KQ8KTests` 4/4. The CPU, fast-forward-pass, and ForwardPass test projects built successfully.

## Still open

No gate-on/off real-checkpoint layer-0 comparison, small real-Q5_K model run, or GLM PPL remeasurement has been performed. The gate stays off, and GLM-4.5 remains unadmitted. Complete the raw layer-0 comparison, conditional PPL run, and real-weight regressions under [item 19](../103-quickest-first-plan.md) using the [detailed experiment plan](../1-correctness/09-glm45-q5k-activation-quant-plan.md).
