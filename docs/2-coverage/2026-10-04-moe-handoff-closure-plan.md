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
- [ ] 1. `qwen2moe`: Stage A, Stage B, receipt, weights removed
- [ ] 2. `llama+experts` (Mixtral): Stage A, Stage B, receipt, weights removed
- [ ] 3. `phimoe`: Vulkan LongRoPE, Stage B, receipt
- [ ] 4. `glm4moe`: disk decision, Stage A, Stage B, receipt
- [ ] 5. `llama4`: forward-path reading, then decide

Realistic total for 1-3: about a week. 4 and 5 depend on disk and on whether they are wanted.
