# Plan: close the CPU-prefill handoff, one model family at a time (2026-10-04)

Status: **plan only, nothing downloaded or implemented.** Context: [coverage matrix](2026-10-04-moe-handoff-coverage.md), [batched MoE prefill plan](2026-10-03-batched-moe-prefill-plan.md).

## What "closed" means

A family is closed when `PrefillHandoffFamilies.Receipts` has a line for it, backed by a passing real-weight run, and its STATUS and coverage rows are updated. A receipt needs **two independent things**, and the matrix hides that the first is often missing:

- **Stage A, the CPU path is admitted.** The CPU forward pass produces the right tokens on a real checkpoint against an independent reference (`llama-server` from `tools/llama.cpp`), per `docs/reference/numerics-investigation-method.md` and the pattern of `PhiMoeGreedyParityTests`. The handoff copies K/V computed by that path, so a wrong CPU path would be handed over faithfully. Today **no real-weight CPU parity test exists in `tests/` for `qwen2moe`, `glm4moe`, `llama4` text, or Mixtral-style `llama` + experts** (grep, 2026-10-04); only `phimoe` has one.
- **Stage B, the handoff is exact for it.** Add the checkpoint as an `InlineData` row to `HybridCpuPrefillHandoffTests` (already a Theory over model files): K/V byte-identical to the CPU pass, logits equal, 40 decode tokens after the handoff within the near-tie rule, boundary lengths, `startPos > 0`. Then the receipt line, naming that test and the checkpoint.

Stop rule per family: if a stage fails and the cause is not understood within one working session, record the finding in the coverage doc, leave the family `Unverified`, and move to the next. No receipt without a passing run; no downloads in bulk.

## Constraints that shape the order

- **Disk:** C: has about 66 GB free, and `models/` already holds 79 GB. One candidate at a time; delete the previous candidate's weights when its receipt is written (standing rule). Anything above about 45 GB needs something else removed first, which I would ask about, not do.
- **Sizes below are from memory of published GGUFs, approximate, and must be confirmed with `stingray pull` before downloading.**
- No discrete GPU: Stage B proves correctness only, on the iGPU (rule 13). CUDA stays untested.
- Pacing: Stage A runs are CPU-bound and long on 30B+ models; run them in the background and check, do not poll.

## Step 0, shared work (about half a day, no download)

1. **Model-path override for the handoff tests:** let `STINGRAY_HANDOFF_TEST_MODEL=<file>` add one more row to `HybridCpuPrefillHandoffTests` and `GpuCpuPrefillHandoffTests`, so a new checkpoint is tested without editing code; the permanent InlineData row is added only with the receipt. (A new env var needs registering in `KnownEnvironmentVariables` and the inventory.)
2. **A Stage A test template:** a `MoeGreedyParity` helper shaped like `PhiMoeGreedyParityTests` (teacher-forced against `llama-server`, near-tie rule) taking the file name and prompts; one small test class per family reuses it.
3. A fixed three-prompt set (short, about 170 tokens, about 700 tokens) so results are comparable across families.

## The families, in order

### 1. Qwen2-MoE (`qwen2moe`): first, the cheapest honest proof
- **Why first:** ordinary KV, softmax top-k, a **shared expert** (the shape OLMoE and Qwen3-MoE do not exercise), already in `ModelCompatibility`, and the shared-expert width issue is already known in the work queue.
- **Checkpoint:** the smallest published `qwen2moe` GGUF (Qwen1.5-MoE-A2.7B, 14B total, about 8-9 GB at Q4_K_M from memory; confirm). Qwen2-57B-A14B has the same family key and is not needed for the receipt.
- **Stage A:** greedy parity vs `llama-server` on the three prompts, plus a short perplexity check if the greedy run shows near-ties. Expect to find at least one bug (shared-expert gate, width or normalisation): budget a day.
- **Stage B:** one InlineData row. This is also the first time the handoff meets a shared expert.
- **Exit:** receipt line, STATUS row for `qwen2moe`, weights deleted.
- **Effort:** 1-2 days. **Risk:** low to medium.

### 2. Mixtral (`llama` header with `expert_count > 0`)
- **Why second:** the archetypal generic MoE; it proves the design is not OLMoE-shaped. It is already its own family key (`llama+experts`), so it cannot borrow dense Llama's standing.
- **Checkpoint:** Mixtral-8x7B-Instruct Q4_K_M, about 26 GB (fits; a Q3 or Q2 variant if disk is tight). 8 experts, top-2 is very different from OLMoE's 64 and 8, so it also tests whether the expert warm-up (sized for many small experts) makes sense.
- **Stage A:** greedy parity vs `llama-server`; confirm the router renormalisation convention (the code comment says Qwen3-MoE/Mixtral convention).
- **Stage B:** InlineData row; with 8 large experts the slot cache is small, so test `STINGRAY_MOE_SLOTS` at 2 and 4 and measure the warm-up cost (if it is pure overhead here, gate it by expert count).
- **Exit:** receipt line for `llama+experts`, STATUS row, weights deleted.
- **Effort:** 2-3 days (mostly 26 GB CPU runs). **Risk:** medium (size, slow iteration).

### 3. Phi-3.5-MoE (`phimoe`): blocked by LongRoPE in the hybrid, not by the handoff
- **State:** Stage A is done on CPU (24/24 and 32/32 vs `llama-server`, `PhiMoeGreedyParityTests`). The hybrid refuses it (`HybridForwardPass.cs:187`) because LongRoPE (per-dimension short/long factors, `attn_factor`) exists only on the CPU pass.
- **Work:** LongRoPE in the Vulkan rope kernels (factor array as a buffer, short or long chosen by context length, attention scale) and in `GpuForwardPass`; then lift the refusal for it, and relax the LongRoPE rule in `PrefillHandoffFamilies` for backends that have it (keep it for the CUDA hybrid until that has it too). Shader constants change, so `scripts/gen-spirv.ps1` (Vulkan SDK) is required: **check the SDK is installed before committing to this one.**
- **Stage B** as usual, with a long-context boundary: the short/long factor switch must happen at the same position in the CPU pass and in decode.
- **Checkpoint:** Phi-3.5-MoE Q3_K_M (the file `PhiMoeGreedyParityTests` expects), about 20 GB.
- **Effort:** 3-4 days. **Risk:** medium to high (kernel change, SPIR-V regeneration, two passes). Benefit beyond this family: dense Phi-3 128k models also unlock for the Vulkan hybrid.

### 4. GLM-4.5-Air (`glm4moe`)
- **Why here:** a large real-world MoE (128 experts, top-8, shared expert, sigmoid gating, leading dense layers). The code knows it (`ModelGraph`: gating, norm, scale), but admission is open and CLAUDE.md's queue lists it as a top family.
- **Blocker:** size. Air is roughly 60-70 GB at Q4 from memory; a Q2 or Q3 is about 40-50 GB. That needs disk made available and is more than this machine runs comfortably: it doubles as the larger-than-RAM streamed-residency test.
- **Stage A:** a Q2/Q3 file vs `llama-server`; the custom QK-norm and partial-rotary details are the likely bug sites.
- **Stage B:** only after A; correctness over a few hundred tokens suffices.
- **Decision for you:** whether to free the disk, and which model to drop. I will not delete anything unasked.
- **Effort:** 3-5 days plus download time. **Risk:** high.

### 5. Llama 4 Scout / Maverick (`llama4`)
- **Why last of the conventional ones:** the architecture is odd (interleaved MoE, top-1 plus a shared expert, chunked attention and NoPE layers, QK-norm quirks), the checkpoints are large (Scout about 40 GB at Q2, 65 GB at Q4 from memory), and only `mmproj-llama-4-scout` is on disk, not the text weights.
- **Gate first, no weights:** read the `llama4` forward path for per-layer RoPE/NoPE and chunked-attention handling. Per-layer differences are exactly what the handoff's uniform-head check does not see, so it must refuse chunked attention until a receipt proves otherwise.
- **Effort:** 4-6 days. **Risk:** high. Treat as optional.

### Not on this list, on purpose
- `hunyuan-moe`: not admitted on CPU at all; admission comes first (a separate item).
- MLA (`deepseek2`), recurrent (`lfm2moe`, `granitehybrid`, `qwen35*`) and `gpt-oss`: a different class, each needing its own handoff mechanism (latent cache, state snapshot). None is a gate fix. Worth a separate plan only if a user need appears.

## Tracking

Update the coverage table in [2026-10-04-moe-handoff-coverage.md](2026-10-04-moe-handoff-coverage.md) after each family, and keep this checklist:

- [ ] Step 0 shared test plumbing
- [x] 1. `qwen2moe` (2026-10-04): Stage A (found and fixed 3 bugs, below), Stage B on the Vulkan hybrid at 1 and 4 GPU layers, receipt `(qwen2moe, VulkanHybrid)`. Not done for this family: Vulkan full-GPU path (shared-expert gate and scratch width are not implemented in `GpuForwardPass`, so it stays unverified), CUDA, router-level check (Stage A2; PPL 4.7838 against 4.8144 ± 0.78 stands in for it). Weights kept at `H:\_models` (F: is full; `models/_models` points at F:) until Mixtral needs the room.
- [x] 2. `llama+experts` (Mixtral, 2026-10-04): Stage A, Stage B on the Vulkan hybrid at 1 and 4 GPU layers, receipt `(llama+experts, VulkanHybrid)`. Not done: Vulkan full-GPU, CUDA, Stage A2 routing check (PPL stands in), the 2- and 4-slot `STINGRAY_MOE_SLOTS` boundary and warm-up cost question (only 16 and 4 slots were run). Weights kept in `H:\_models`.
- [x] 3. `phimoe` (2026-10-04): LongRoPE, norm biases and LM-head bias in the Vulkan hybrid, Stage B at both RoPE regimes, receipt `(phimoe, VulkanHybrid)`. Not done: Vulkan full-GPU (`GpuForwardPass` does not add the RMSNorm biases or `output.bias` for phimoe), CUDA hybrid (no LongRoPE), Stage A2.
- [ ] 4. `glm4moe`: disk decision, Stage A, Stage B, receipt
- [ ] 5. `llama4`: forward-path reading, then decide

Realistic total for 1-3: about a week. 4 and 5 depend on disk and on whether they are wanted.

## Results log

### 1. `qwen2moe` (Qwen1.5-MoE-A2.7B-Chat Q4_K_M, 2026-10-04)
Three real bugs, none visible without a real checkpoint (before any fix the CPU path emitted `!!!!`, i.e. all-zero logits):
1. **Routed-expert width fell back to the dense width.** This older GGUF has no `expert_feed_forward_length`; `feed_forward_length` (5632) is the dense width, the experts are 1408. `ModelGraph` now reads the width from the stacked `ffn_gate_exps` tensor when the key is missing (as llama.cpp does).
2. **The shared expert's sigmoid gate was never applied** (`ffn_gate_inp_shexp`, an F32 `[embDim]` vector): `out = sigmoid(w . x) * shared_ffn(x)`. Added to the CPU pass (per-token and batched) and to the Vulkan hybrid's CPU and GPU layers. Any other family with that tensor (qwen35moe uses the same gate) now gets it too; not verified there, no checkpoint on disk.
3. **Hybrid GPU scratch too narrow for the shared expert.** The hybrid's GPU FFN scratch is sized to the routed-expert width, and a matmul takes its row count from its output buffer, so a 5632-wide shared expert computed 1408 rows. Dedicated shared-expert scratch added when the widths differ.

Evidence: `Qwen2MoeGreedyParityTests` (teacher-forced against `llama-server`, 1.5-nat confident margins: 17 of 22 and 5 of 32 positions confident, all matched; free-running greedy diverged at a 0.05-nat near-tie, which is why the contract is teacher-forced). Second-half wikitext PPL at -c 512: ours 4.7838, `llama-perplexity` 4.8144 ± 0.78. Handoff: `HybridCpuPrefillHandoffTests` rows at 4 and 1 GPU layers, byte-exact K/V, prefill cosine 0.9995, decode cosines 0.998-0.9998 (the baseline is the sequential hybrid, which needed fixes 2 and 3 to be right). Regression: ForwardPass.Fast 1029 passed / 0 failed, `VulkanHybridOlmoeParityTests` 7/7.

Also found while checking Mixtral (not fixed, belongs to step 2): `NormalizeMoeTopKWeights` defaults to false for any architecture without an `expert_weights_norm` key, which includes `llama`; Mixtral needs renormalised top-2 weights (llama.cpp `build_moe_ffn` with `norm_w = true`).

### 2. Mixtral-style (`llama` + experts) (2026-10-04)
- **Checkpoint search:** the common Mixtral GGUFs (TheBloke, MaziyarPanahi, 2023) use the legacy per-expert tensor layout (`blk.N.ffn_gate.E.weight`, 995 tensors) that this engine does not load; stacked-layout (`ffn_gate_exps`) ones exist from newer converts. Header-sniffed with HTTP range requests before downloading. Legacy-layout support would be a separate loader task.
- **Bug fixed:** `NormalizeMoeTopKWeights` defaulted to false for `llama` (no `expert_weights_norm` key), but llama.cpp's `llama` builder passes `norm_w = true` for experts. Fixed in `ModelGraph`.
- **Stage A on TinyLlama-4x1.1B-MoE Q4_K_M** (4 experts top-2, 1.9 GB): `MixtralStyleGreedyParityTests` teacher-forced against `llama-server` (BOS added), 7 of 22 and 11 of 32 confident positions all match; second-half PPL (-c 512) 10.5418 vs `llama-perplexity` 10.1558 ± 1.71.
- **Stage B on the tiny model: not receipted.** The handoff is byte-exact (K/V and prefill logits identical to the CPU pass, 1 and 4 GPU layers), but decode after the handoff differs from decode after the sequential hybrid prefill at one step in both splits (cosine 0.973 and 0.975; at 4 GPU layers an argmax flip with a 4.7% gap). The suspected cause is a near-tied router in this mergekit-made model (a hybrid-GPU versus CPU K/V difference of a few ulps flips the chosen expert), not the handoff; not proven. The two test rows were removed again.
- **Real Mixtral-8x7B** (Nous-Hermes-2-Mixtral-8x7B-DPO i1-Q4_K_S, stacked layout, 26.7 GB, 8 experts top-2, 32 layers; downloaded from mradermacher after sniffing headers): `MixtralGreedyParityTests` teacher-forced against `llama-server`, all 24 confident positions match (18 of 32 and 6 of 22); second-half PPL (-c 512) 3.2040 vs `llama-perplexity` 3.1935 ± 0.41. Handoff on the Vulkan hybrid at 4 and 1 GPU layers: byte-exact K/V, prefill cosine 0.9998, decode cosines 0.99982-1.00000 (no near-tie argmax flips). That settles the tiny model: the decode dip there was the merge's router, not the handoff. Receipt added; `STINGRAY_HYBRID_CPU_PREFILL=all` was needed to run the rows before it existed.

### 3. `phimoe` (Phi-3.5-MoE-instruct Q3_K_M, 2026-10-04)
The hybrid refused LongRoPE, and while reading it for that, two more phimoe features were found missing from it (the CPU pass had them since 2026-09-26): the RMSNorm **bias** added after the attn, ffn and output norms, and the LM-head `output.bias`. All three implemented in `HybridForwardPass`:
- **LongRoPE:** GPU layers through `RoPEFactorsBatched` with the factor tensor and the `rope.scaling.attn_factor` scale; CPU layers through the cos/sin table with the same factors and scale. The factor set is chosen once by the context size (long when it exceeds `original_context_length` 4096), exactly as `ForwardPass` and the CPU handoff pass do; both are built with the hybrid's context, so they cannot disagree. A cos/sin attention factor without factor tensors (YaRN) is still refused.
- **Norm biases and output bias:** tensors on the GPU layers (`AddInPlace` after the norm), host vectors on the CPU layers, bias added to the downloaded logits.
- **Acceptance (review item, made an explicit gate):** `HybridCpuPrefillHandoffTests` runs ctx 1024 (short factors, 4 and 1 GPU layers) and ctx 8192 (long factors, 4 GPU layers), K/V byte-exact, prefill cosine 0.99998, decode cosines 0.99955-0.99998. The CPU reference side is unchanged and still passes `PhiMoeGreedyParityTests` (short and long factors against `llama-server`, 2/2, 264 s). The full 14-row handoff class passed afterwards (740 s), so the earlier receipts did not regress.
- **Cost note:** Q3_K expert weights are dequantised to F32 when uploaded to the GPU slot cache (only Q4_K and Q6_K go up raw), so this checkpoint is slower and heavier on GPU memory than a Q4_K_M one would be.

## Review feedback, parked for later consideration (2026-10-04)

An outside review of this plan, recorded here and **not yet acted on**. I checked point 1 against the code: it is correct as stated, and it also means the CUDA hybrid path (never run) is currently admitted for the same three families as the Vulkan hybrid.

| Priority | Change | Notes from checking the repo |
|---|---|---|
| Must | **Receipts per family and path** (`VulkanHybrid`, `VulkanFullGpu`, `CudaHybrid`), so a Vulkan `-g N` receipt cannot authorize another path. | `PrefillHandoffFamilies.Receipts` is family-only and all three `*PrefillHandoff.cs` call it. Real evidence today: Vulkan hybrid has OLMoE, Qwen3-Coder and Qwen3-0.6B; Vulkan full-GPU has OLMoE only (`GpuCpuPrefillHandoffTests`); CUDA hybrid has none. Smallest change: key receipts by `(family, path)`, pass the path from each caller, and the CUDA path becomes unverified until it runs. |
| Must | **Stage B must include a mixed CPU/GPU layer split** (for example `-g 4`), plus `-g -1` for the full-GPU path, optionally `-g 1`. | The existing hybrid theory already uses 4 and 8 GPU layers; make it a stated requirement of every receipt. |
| Strong | **Fingerprint the proven checkpoint in each receipt**: architecture, expert count, top-k, expert FFN width, shared-expert count and width, leading dense layers, gating function, top-k renormalisation, KV heads, head dimension, RoPE mode. | Records what "admitted" actually covered. Informational: a checkpoint whose fingerprint differs would be reported, not refused. Qwen1.5-MoE's own values (60 experts, top-4, 5632-wide shared expert against 1408 routed) are the kind of fact to freeze. |
| Strong | **Stage A2, routing check**: compare selected expert ids, routing weights, top-k count, renormalisation and shared-expert contribution against the reference for the first 8 layers and 32 prompt tokens. | A wrong router can still give the right argmax; token parity alone is weaker evidence for a discrete mechanism. Diagnostic run, no permanent dump. |
| Strong | **Phi-3.5-MoE: make LongRoPE threshold crossing an explicit gate.** | The CPU pass and the GPU must pick the same short/long factor regime on both sides of `original_context_length`; use a prompt that crosses it. |
| Minor | **License checkbox in the checklist.** | Qwen1.5-MoE-A2.7B is under the Tongyi Qianwen license, not Apache or MIT; check the project's checkpoint-license policy before a checkpoint becomes a permanent test fixture. |
| Order | **GLM-4.5-Air a little higher.** | The review reports 106B total / 12B active and an MIT license (not checked by me; confirm before relying on it). Still after Qwen2-MoE, Mixtral and Phi-MoE. |

Proposed redefinition of "closed" if adopted: a family/path is closed when that path has a real-weight receipt that shows (1) CPU correctness against an independent reference, (2) exact K/V transfer from that CPU path, (3) stable decode after the handoff, (4) at least one mixed CPU/GPU split, (5) the model's distinctive MoE features, and (6) the relevant context and RoPE boundaries. No family-level admission authorizes an untested path.

The review's verdict was that the plan is sound and its order (Qwen2-MoE, Mixtral, Phi-MoE) is a good first three; the changes above strengthen the evidence standard rather than redesign it.
