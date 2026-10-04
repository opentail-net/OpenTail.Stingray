# Plan: close the CPU-prefill handoff, one model family at a time (2026-10-04)

Status (2026-10-04, later): **steps 1-3 closed; step 4 (GLM-4.5-Air) closed on the Vulkan layer split: CPU path admitted, `-g 4` works, receipt `(glm4moe, VulkanLayerSplit)` from a CPU-prefill handoff test (more than 4 GPU layers still needs the expert slot cache: all 128 experts per resident layer are uploaded); step 5 (Llama 4) Vulkan-hybrid handoff receipt, closed past 8192 too (chunked attention and NoPE temperature tuning implemented d8ca10da; 9430-token CPU run matches llama-server, hybrid handoff at 9429 tokens passes, 2026-10-04); step 6 (speed league) done: ours is below llama.cpp on CPU for all five MoE models (0.21-0.84x), see PerformanceLeague.md.** 7 families now have path-specific receipts (6 MoE on the Vulkan hybrid). Context: [coverage matrix](2026-10-04-moe-handoff-coverage.md), [batched MoE prefill plan](2026-10-03-batched-moe-prefill-plan.md).

## What "closed" means

A family is closed when `PrefillHandoffFamilies.Receipts` has a line for it, backed by a passing real-weight run, and its STATUS and coverage rows are updated. A receipt needs **two independent things**, and the matrix hides that the first is often missing:

- **Stage A, the CPU path is admitted.** The CPU forward pass produces the right tokens on a real checkpoint against an independent reference (`llama-server` from `tools/llama.cpp`), per `docs/reference/numerics-investigation-method.md` and the pattern of `PhiMoeGreedyParityTests`. The handoff copies K/V computed by that path, so a wrong CPU path would be handed over faithfully. Today **no real-weight CPU parity test exists in `tests/` for `qwen2moe`, `glm4moe`, `llama4` text, or Mixtral-style `llama` + experts** (grep, 2026-10-04); only `phimoe` has one.
- **Stage B, the handoff is exact for it.** Add the checkpoint as an `InlineData` row to `HybridCpuPrefillHandoffTests` (already a Theory over model files): K/V byte-identical to the CPU pass, logits equal, 40 decode tokens after the handoff within the near-tie rule, boundary lengths, `startPos > 0`. Then the receipt line, naming that test and the checkpoint.

**Decode length:** the permanent closure tests teacher-force **40** decode steps (raised from 6 on 2026-10-04 review feedback: six catch gross corruption, forty also catch a small CPU-KV vs GPU-KV difference accumulating into a later router/attention divergence). Receipts earned before that used 6 steps and are being re-run at 40.

**Fingerprint:** a mismatch is diagnostic, not an admission gate. A checkpoint whose header differs from the proven one is still admitted and the difference is reported, so variants do not need their own entry.

**Stage A2 (router-level check):** parked for the three finished families (PPL stands in); mandatory for any new family with novel routing semantics.

Stop rule per family: if a stage fails and the cause is not understood within one working session, record the finding in the coverage doc, leave the family `Unverified`, and move to the next. No receipt without a passing run; no downloads in bulk.

## Constraints that shape the order

- **Disk:** C: has about 66 GB free, and `models/` already holds 79 GB. One candidate at a time; delete the previous candidate's weights when its receipt is written (standing rule). Anything above about 45 GB needs something else removed first, which I would ask about, not do.
- **Sizes below are from memory of published GGUFs, approximate, and must be confirmed with `stingray pull` before downloading.**
- No discrete GPU: Stage B proves correctness only, on the iGPU (rule 13). CUDA stays untested.
- Pacing: Stage A runs are CPU-bound and long on 30B+ models; run them in the background and check, do not poll.

## Step 0, shared work (about half a day, no download)

1. **Model-path override for the handoff tests** (implemented 2026-10-04 for `HybridCpuPrefillHandoffTests` as `Handoff_EnvironmentChosenModel`, with optional `STINGRAY_HANDOFF_TEST_GPU_LAYERS` and `STINGRAY_HANDOFF_TEST_CTX`; not yet for `GpuCpuPrefillHandoffTests`): let `STINGRAY_HANDOFF_TEST_MODEL=<file>` add one more row, so a new checkpoint is tested without editing code; the permanent InlineData row is added only with the receipt. (A new env var needs registering in `KnownEnvironmentVariables` and the inventory.)
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
- `hunyuan-moe`: now admitted on CPU (2026-10-04, see the extension-wave results entry); GPU paths pending.
- MLA (`deepseek2`), recurrent (`lfm2moe`, `granitehybrid`, `qwen35*`) and `gpt-oss`: a different class, each needing its own handoff mechanism (latent cache, state snapshot). None is a gate fix. Worth a separate plan only if a user need appears.

## Tracking

Update the coverage table in [2026-10-04-moe-handoff-coverage.md](2026-10-04-moe-handoff-coverage.md) after each family, and keep this checklist:

- [x] Step 0 shared test plumbing (receipts keyed by (family, path); fingerprints; Theory over model files)
- [x] 1. `qwen2moe` (2026-10-04): Stage A (found and fixed 3 bugs, below), Stage B on the Vulkan hybrid at 1 and 4 GPU layers, receipt `(qwen2moe, VulkanHybrid)`. Vulkan full-GPU (2026-10-04, later): `GpuForwardPass` now applies the shared-expert sigmoid gate and uses own shared-expert scratch; `GpuCpuPrefillHandoffTests` on Qwen1.5-MoE passes (prefill cosine 0.9995, decode 0.998-1.0), receipt `(qwen2moe, VulkanFullGpu)`. Not done for this family: CUDA, router-level check (Stage A2; PPL 4.7838 against 4.8144 ± 0.78 stands in for it). Weights kept at `H:\_models` (F: is full; `models/_models` points at F:) until Mixtral needs the room.
- [x] 2. `llama+experts` (Mixtral, 2026-10-04): Stage A, Stage B on the Vulkan hybrid at 1 and 4 GPU layers, receipt `(llama+experts, VulkanHybrid)`. Not done: Vulkan full-GPU, CUDA, Stage A2 routing check (PPL stands in), the 2- and 4-slot `STINGRAY_MOE_SLOTS` boundary and warm-up cost question (only 16 and 4 slots were run). Weights kept in `H:\_models`.
- [x] 3. `phimoe` (2026-10-04): LongRoPE, norm biases and LM-head bias in the Vulkan hybrid, Stage B at both RoPE regimes, receipt `(phimoe, VulkanHybrid)`. Vulkan full-GPU (2026-10-04, later): `GpuForwardPass` now adds the RMSNorm biases and `output.bias`; `GpuCpuPrefillHandoffTests` on Phi-3.5-MoE (ctx 1024, short factors only) passes (prefill cosine 0.99995, decode >= 0.9987), receipt `(phimoe, VulkanFullGpu)`. Not done: full-GPU at the long-factor regime (ctx > 4096), CUDA hybrid (no LongRoPE), Stage A2.
- [x] 4. `glm4moe` (2026-10-04, partial by design): Stage A done and the CPU path admitted; the first "no GPU path to hand to" gap was closed later the same day: Vulkan layer split `-g 4` plus a layer-split CPU-prefill handoff, receipt `(glm4moe, VulkanLayerSplit)` (see the last GLM results entry). Disk: GLM-4.5-Air-Q2_K (45.3 GB) is on `H:\_models`, nothing was deleted (H: had 235 GB free).
- [x] 5. `llama4` (2026-10-04, CPU Stage A only): reading done, Scout Q3_K_M in `H:/_models/Q3_K_M/`, Stage A passes (below). No handoff receipt: the Vulkan paths were not exercised, so the handoff stays Unverified for this family.
- [x] 6. **Speed league for the MoE lot** (done 2026-10-04: rows in PerformanceLeague.md "MoE lot"; ours is 0.21-0.84x of llama.cpp on CPU on all five, llama.cpp has no GPU backend here; found and fixed the GLM `-g -1` crash by capping auto offload to a 4-layer split) (added 2026-10-04 at the user's request): measure, then record dated rows in [PerformanceLeague.md](../../PerformanceLeague.md) (its existing table format and methodology: `tools/llama.cpp/llama-bench.exe -m <gguf> -p 512 -n 128 -t 6`, versus our CLI on `docs/reference/benchmark-prompt.txt -n 24 --temp 0 --single-turn --no-display-prompt`). **Never run while another heavy test or download is running** (the numbers would be contention, not speed); one model at a time, several repetitions, median.
  - Models: Qwen1.5-MoE-A2.7B Q4_K_M, Mixtral-8x7B i1-Q4_K_S, Phi-3.5-MoE Q3_K_M, GLM-4.5-Air Q2_K, Llama-4-Scout Q3_K_M (all in `H:\_models`); OLMoE and Qwen3-Coder-30B already have rows and are re-run only if the code under them changed.
  - **Part A, llama.cpp alone, three modes:** (1) CPU (`-ngl 0`); (2) mixed CPU+GPU (`-ngl N`) and (3) GPU-only (`-ngl 99`), each *only if the vendored `tools/llama.cpp` has a GPU backend that works on this machine*. It has the RPC backend and no Vulkan backend at the time of writing (the backfill plan says building one is a separate project), so modes 2 and 3 are recorded as "not available here" unless that is built; do not substitute a different build without saying so. The machine has only an integrated GPU sharing system RAM (CLAUDE.md rule 13), so a GPU row here is not evidence about discrete GPUs.
  - **Part B, our engine, after all pending code changes in this plan are committed and verified:** the same three modes: CPU (`-g 0`), mixed (`-g N`: Vulkan hybrid, or the layer split for GLM, with N stated) and GPU-only (`-g -1`, which falls back to the hybrid when the model does not fit; say which path actually ran). Same prompt lengths, same thread count, reported as prefill and decode tokens per second and as a ratio to llama.cpp in the same mode.
  - Rules: report what ran (path, thread count, quantisation, context), keep failures and "not available" as rows rather than omitting them, and do not conclude "the GPU path is slow" from the iGPU alone.

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

- **GLM-4.5-Air-Q2_K (`glm4moe`, 2026-10-04)**: `stingray admit-arch` gave 24 of 24 exact greedy tokens (repetitive prompt, so weak evidence alone). `GlmMoeGreedyParityTests` is the real Stage A: teacher-forced against `llama-server` (same GGUF, CPU, no prefix token), short prompt 14 of 22 positions confident and a 190-token prompt (batched prefill) 16 of 32, **all 54 matched, 0 near-tie differences**. Second-half wikitext PPL at -c 512: ours 3.3709, `llama-perplexity` 3.4260 +/- 0.41. `glm4moe` is now in the `ModelCompatibility` allowlist (CPU only; only the Q2_K quantisation has been run). CPU speed: about 2.7 tok/s on this machine.
  - **Stage B is blocked, and it is a feature gap rather than a handoff question.** `-g 4` crashed with `Missing tensor: blk.0.ffn_gate_inp.weight`: the Vulkan passes have no leading-dense-layer (dense FFN, then MoE) mix and no selection-bias routing (select on `sigmoid + exp_probs_b`, weight by the unbiased probabilities). The hybrid also refuses partial RoPE. I made `GpuForwardPass.UnsupportedReason` report "MoE with leading dense layers (CPU only)" and "selection-bias expert routing (CPU only)", so `-g N` now says so and runs on the CPU instead of crashing. No receipt line was added: with no GPU path there is nothing to hand K/V to.
  - To get a receipt: add leading-dense layers, selection-bias routing, `expert_weights_scale` and partial RoPE to the hybrid, then run the Stage B row. That is a feature port, not part of this closure.

- **Llama 4 reading step (2026-10-04, no weights)**: what the engine has: NoPE every 4th layer (`noRopeStep = 4` in `ModelGraph`), parameter-free L2 QK-norm (`UseL2QkNorm`), sigmoid expert gating, top-1 plus shared expert. What a grep of `src` does not find: **chunked attention** (the 8192-token attention chunk on the RoPE layers) and **attention temperature tuning** (floor-scaled Q scale on the NoPE layers). Both are the identity below 8192 positions, so the CPU path should be right for contexts up to 8192 and wrong beyond, silently. Consequences: Stage A must stay under 8192 tokens, any Llama 4 claim must say so, and the handoff must not be admitted for longer contexts without those two mechanisms. The full-GPU pass excludes `UseL2QkNorm` models from its batched trunk by assertion, so a Llama 4 receipt on a Vulkan path is a separate piece of work from the CPU one. `pull` now creates the quant subfolder of a sharded repo (it failed with "path not found" before).

- **Llama 4 Scout Q3_K_M (`llama4`, 2026-10-04)**: `LlamaFourGreedyParityTests` teacher-forced against `llama-server` (same two-shard GGUF, CPU, BOS added): short prompt 17 of 22 positions confident and a 190-token prompt 14 of 32, **all 31 confident positions matched**, 5 near-tie differences of 54 (margins under 1.5 nats, allowed by the rule). Second-half wikitext PPL at -c 512: ours 6.8583, `llama-perplexity` 6.8058 +/- 1.25. This covers the CPU path below 8192 tokens only (chunked attention and attention-temperature tuning are not implemented). Not done: any Vulkan/CUDA run (the full-GPU batched trunk asserts out L2 QK-norm models), the handoff receipt, and anything beyond 8192 tokens. CPU speed about 2.9 tok/s.

- **GLM-4.5-Air on Vulkan (2026-10-04, later)**: closed the gap I had wrongly left as "CPU only". `GpuForwardPass` (and so the Vulkan layer split behind `-g N`) now has: leading dense layers (own wider dense scratch), selection-bias routing (select on sigmoid + `exp_probs_b`, weight by the unbiased probabilities, renormalise, scale), GLM's `post_attention_norm`-as-pre-FFN-norm naming, and a CPU-side embedding row lookup for tables whose F32 dequant exceeds 1 GiB (GLM's 2.5 GB table crashed the upload with `ErrorOutOfHostMemory`). `VulkanLayerSplitParityTests` row `GLM-4.5-Air-Q2_K.gguf -g 4`: worst cosine 0.9983, 0 argmax flips in 9 steps. CLI `-g 4` answers coherently. **Limit found:** `-g 6` and `-g 8` fail with `ErrorOutOfHostMemory` in a fresh process, because the layer split uploads all 128 experts of every resident layer as separate device buffers; an expert-slot-cached hybrid for GLM is the way to fit more layers. The layer split has no CPU-prefill handoff, so there is no handoff receipt for GLM. `PartialOffloadUnsupportedReason` (hybrid and CUDA) now names leading dense layers and selection-bias routing as full-Vulkan-only.
- **Llama 4 on Vulkan (2026-10-04, later)**: I had not run it. The Vulkan hybrid runs it (`-g 4` answers "Paris"), and `HybridCpuPrefillHandoffTests.Handoff_LargeFamilies` passes at 4 and 1 GPU layers (prefill cosine 0.9999, decode cosines >= 0.998 over 40 steps), receipt `(llama4, VulkanHybrid)`. `-g -1` on Scout selects the hybrid automatically (30 GPU + 18 CPU layers); there is no pure full-GPU pass at 52 GB on this machine.

- **Llama 4 past the 8192 chunk (2026-10-04, later)**: CPU, 9430-token prompt: `LlamaFourLongContextParityTests` matches `llama-server` on all 24 positions (16 confident, 0 near-tie differences). Vulkan hybrid: `HybridCpuPrefillHandoffTests.Handoff_Llama4_PastTheChunkBoundary_DecodeAgreesWithCpu` (CPU-prefill handoff at 9429 tokens into 4 GPU + 44 CPU layers, then 6 teacher-forced decode steps against the all-CPU pass): decode cosines 0.99990, 0.99996, 0.99992, 0.99960, 0.99970, 0.99939; real run, 3130 s. The sequential GPU prefill at this length was not run (hours); the short-prompt rows cover it. The "contexts up to 8192" caveat is dropped from the receipt, STATUS and coverage rows. Still not done: CUDA. The earlier "below 8192 only" and "not implemented" statements in the Llama 4 entries above are historical.

- **GLM-4.5-Air handoff on the Vulkan layer split (2026-10-04, later)**: rather than port leading dense layers, selection-bias routing and partial RoPE into `HybridForwardPass` (the expert-slot-cache route), `VulkanLayerSplitForwardPass` got its own CPU-prefill handoff: the CPU pass prefills every layer into its own F32 KV cache, `GpuForwardPass.ImportSplitKv` uploads rows of the resident layers, decode continues split. New `HandoffPath.VulkanLayerSplit`; refusals: startPos != 0, under 32 tokens, per-layer head dims, bf16 KV store, no family receipt. `LayerSplitCpuPrefillHandoffTests` (GLM-4.5-Air Q2_K, `-g 4`, 205 tokens, 367 s real run): GPU K/V equal to the CPU pass (F32 exactly, packed fp16 to fp16 rounding), prefill logits equal, 40 teacher-forced decode steps cosine 0.9946-0.9997, 0 argmax flips of 41. Receipt `(glm4moe, VulkanLayerSplit)`. Not done: `-g` above 4 (the layer split uploads every expert of each resident layer; needs the slot cache), CUDA, Stage A2, a prompt over the context boundary or `startPos > 0` (refused by design).

## Extension wave: further conventional-KV MoE families (added 2026-10-04, after review)

The first wave (steps 1-5) proved the mechanism across materially different routing designs. Later families are **extensions of this plan, not new handoff projects**, unless inspection shows a different state or cache representation. Do this only after the open items above settle (Llama 4 long-context result, GLM expert-slot-cached hybrid for its handoff, step 6 speed league, CUDA evidence).

Rule: *recognised in `ModelGraph`* is not *CPU path works* is not *handoff works*. Admission needs a real-weight receipt for `(family, path)`; no family-wide "MoE is safe" switch.

**Workflow per candidate**
- **A. Structural classification** (no weights): ordinary per-head K/V? recurrent or conv state? MLA or another compressed cache? expert count, top-k, shared experts, routing semantics, special attention/RoPE/norm/residual features. Outcome: *conventional KV, extension candidate* / *conventional KV, needs a feature port first* / *different state, separate handoff class* / *not worth it on this hardware*.
- **B. CPU admission (Stage A):** real checkpoint, independent `llama-server` reference, teacher-forced greedy with the confident-margin rule, perplexity cross-check; record every architecture-specific fix. No GPU receipt before this is clean.
- **C. Is hybrid support already compositional?** Read `GpuForwardPass.PartialOffloadUnsupportedReason` / `UnsupportedReason` as the to-do list; prefer generalising an existing implementation (leading dense layers, selection-bias routing, partial RoPE, shared expert scratch, chunked attention are already in `GpuForwardPass`) over a family-specific branch.
- **D. Real-weight handoff (Stage B):** K/V byte identity, logits, 40 teacher-forced decode steps, mixed split, boundary contexts, `startPos > 0`.
- **E. Admit:** `(family, HandoffPath.VulkanHybrid)` only after D passes.

**Backlog (priority order)**
1. `hunyuan-moe` (Stage A done 2026-10-04, CPU admitted; GPU handoff pending): conventional KV. Needs Stage A first (Hunyuan-specific attention/RoPE/QK-norm behaviour against the real reference). Check disk before choosing a quant.
2. `afmoe` (Arcee Trinity Mini): attention output gate (`blk.N.attn_gate.weight`) is rejected by `GpuForwardPass.UnsupportedReason` today; port the gate into the Vulkan paths with identical semantics, then Stage A/B.
3. `cohere2moe`: compound extension; LayerNorm, parallel residual and sliding-window/NoPE mixing are refused for partial offload today.
4. `step35`, 5. `exaone-moe`: checkpoints too large for this machine; classify only.
- *inspect first:* `grovemoe` (expert layout may not fit the generic slot-cache model).

**Stay outside this class** (keep the structural refusals): `deepseek2` (MLA, needs a latent-cache handoff), recurrent/conv-state families (`lfm2moe`, `qwen35moe`, `qwen3next`, `granitehybrid`, Nemotron-H), `gpt-oss` (own forward pass).


### Extension wave results

- **`hunyuan-moe` Stage A (Hunyuan-A13B-Instruct Q3_K_S, DevQuasar, 34.8 GB on `H:\_models`, 2026-10-04)**: two `ModelGraph` changes from memory of the HF/llama.cpp references, both confirmed by the real run: QK-norm after RoPE (as `hunyuan-dense`) and renormalised top-k expert weights (no `expert_weights_norm` key). `HunyuanMoeGreedyParityTests` teacher-forced against `llama-server` (same GGUF, CPU, no prefix token): all 39 confident positions matched (27 of 32 on a 196-token prompt whose continuation is repetitive, 12 of 22 on "The capital of France is"), 0 near-tie differences, 37 s real run. Second-half wikitext PPL at -c 512: ours 219.8 (bucket [256,1024)), `llama-perplexity` 201.57 +/- 53.71. I did not run the code without the two changes, so which of them was individually required is not shown. `hunyuan-moe` is in the `ModelCompatibility` allowlist (CPU only, Q3_K_S only). Not done: any GPU path, a handoff receipt (QK-norm after RoPE is refused on the partial-offload paths, so the layer split is the candidate), Stage A2.
- **`hunyuan-moe` Stage B (2026-10-04): not run.** `LayerSplitCpuPrefillHandoffTests` has an `InlineData` row for `tencent.Hunyuan-A13B-Instruct.Q3_K_S.gguf` at `-g 4` (QK-norm after RoPE is refused on the partial-offload hybrid, so the layer split is the candidate path). The first attempt was stopped by the harness because the system was critically low on memory while the 35 GB checkpoint and the GPU upload were live; no result exists, so there is no `(hunyuan-moe, VulkanLayerSplit)` receipt. Re-run it alone on an otherwise idle machine (`STINGRAY_HYBRID_CPU_PREFILL=all`, class `LayerSplitCpuPrefillHandoffTests`).
