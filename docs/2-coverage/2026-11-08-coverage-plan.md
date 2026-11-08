# Model coverage plan

Status 2026-11-08. This is the one plan for model coverage: what to do next for each family, and what is blocked. Family detail stays in each family's owner doc, linked below. Verification levels and the status ladder are in [ported-families-todo.md](ported-families-todo.md). Qwen 3.8 has its own plan: [2026-11-08-qwen38-remaining-work-plan.md](2026-11-08-qwen38-remaining-work-plan.md).

**Rules that apply to every family:** rule 14: a family ported before a real checkpoint could be checked is **not admitted and not advertised** until it has a real checkpoint, an independent reference, timed runs, and then a STATUS row. The admission path is in [adding-an-architecture](../reference/adding-an-architecture.md). A parity claim names its reference and its revision. A checkpoint may only be downloaded if the item says so, to a drive with space (never F: for files over 3 GB).

---

## Part A: Runnable now, in order

### A1. DeepSeek-V4 (`deepseek4`): synthetic specification test, then the real-weight run

- **Why:** the port is structurally written, with 15 unit tests. The CSA and HCA attention completion and the real-weight verification are open. The one synthetic test uses `compress_ratios = [0, 0]`, so the non-zero case (the first real CSA test) has never run. The checkpoint is 86 GB, paged.
- **Next step:** (1) add the synthetic specification test with non-zero `compress_ratios`, against the reference's attention. (2) Only then the paged real-weight run, correctness only, on the checkpoint on disk.
- **Done when:** the synthetic test passes against the reference, and the paged run either matches a recorded reference output or names the first divergent step.
- **Owner doc:** [2026-10-03-deepseek-v4-review-plan.md](2026-10-03-deepseek-v4-review-plan.md). Rule 14 applies until verified.
- **Status 2026-11-08 (step 1 done, step 2 partly done):** the synthetic test with non-zero ratios is in (`DeepSeek4SyntheticTests`, 5 tests, all passing). It checks CSA and HCA block counts and that the compressed key changes the logits. It also found a real gap: a CSA layer without its compressor tensors crashed on the first token. The constructor now refuses that with the tensor's name. Reference check: the pinned llama.cpp source (`examples/llama.cpp/llama.cpp`, `bed0a8566`) was read against our CSA and HCA code. Everything matched except the RoPE position of a compressed block (block index `k`, reference `ratio * k`). That is fixed. Still open: a numeric comparison. Attempted 2026-11-08 and blocked: the pinned loader accepts a scratch GGUF once the keys listed in the DS4 plan are added, then exits on ratio-0 layers or asserts in `ggml_reshape_2d` on compressed layers. Details in the DS4 plan, Phase 2. The paged real-weight run is not started.

### A2. DiffusionGemma: the enc_layer_output_scale question

- **Why:** the parity ladder cannot be finished until it is known what `enc_layer_output_scale` and `layer_output_scale` mean, either in the prefill or the canvas pass. The Hugging Face `config.json` does not have these keys. The GGUF metadata does.
- **Next step:** read the HF or vLLM source for those names, and write down the answer with its source. If it is not there, record that and ask the owner.
- **Done when:** the meaning is recorded with its source, and the parity ladder step that depends on it is unblocked or documented as blocked.
- **Owner doc:** [2026-10-03-diffusiongemma-port-plan.md](2026-10-03-diffusiongemma-port-plan.md).
- **Status 2026-11-08: done.** Both are the same per-layer scalar. The upstream safetensors index, vLLM, and the GGUF tensor values (30 of 30 layers equal) agree, read by range request with no download. The GGUF has the scale as tensors, not metadata. The port should fail on a missing scale, not default it to 1.0. The parity ladder step is unblocked. Details, revisions, and the consequence are in the owner doc. The Q4_K_M download and the parity ladder itself are still open.

### A3. Muse-Glimmer: independent synthetic comparison (level 3)

- **Why:** the synthetic-GGUF comparison against llama.cpp was never run. It is level 3 of the ladder.
- **Next step:** build the synthetic GGUF, run the comparison against the pinned llama.cpp revision recorded in the plan, and record the result.
- **Done when:** the comparison is recorded with the revision, and the family's status row reflects it.
- **Owner doc:** [2026-10-03-muse-glimmer-port-plan.md](2026-10-03-muse-glimmer-port-plan.md).

### A4. Architecture semantics: final real-checkpoint verification

- **Why:** the user asked for a final real-checkpoint verification on 2026-10-08. It is the only open box of the semantics plan.
- **Next step:** list the checkpoints the plan names that are on disk. Run the verification on each one that is present, and record the missing ones as missing.
- **Done when:** each named checkpoint has a verification record or a "not on disk" note.
- **Owner doc:** [2026-10-08-architecture-semantics-admission-plan.md](2026-10-08-architecture-semantics-admission-plan.md).

### A5. Architecture follow-ups that are still code work

These were the open items of the admission follow-ups, now collected here:

- **Heavy suites never run:** the Cuda, Vulkan, Sessions and Server heavy projects, and the ForwardPass classes outside the migrated architectures. Run each on this machine where it can run; list each skip with its reason. Merged plan item A3 covers the finder part.
  - **Status 2026-11-08:** all four projects build (Release, 0 warnings). Test counts: Server 24 facts, Sessions 13, Vulkan 201, Cuda 463. The heavy gate is inconsistent: only Sessions and Vulkan check `STINGRAY_RUN_HEAVY_TESTS`; Server and Cuda have no such check, so they run whatever they contain. Runs use `STINGRAY_RUN_HEAVY_TESTS=1`, one project at a time, with free RAM at 42-44 GB and no other test process running. Outputs are in the scratchpad, `a5/`.

    | Project | Gate in code | Total | Passed | Failed | Skipped | Time | Skip reasons |
    |---|---|---|---|---|---|---|---|
    | Server | none | 26 | 6 | 9 | 11 | 3.5 s | Missing checkpoints: SmolLM2-135M, SmolLM2-360M, SmolLM2-1.7B, Qwen3-0.6B, Qwen3.5-0.8B. One needs `STINGRAY_FAIR_GATE_MEASURE=1`. |
    | Sessions | yes | 13 | 1 | 0 | 12 | 0.75 s | SmolLM2-1.7B missing (11), one model fixture not present (1). |
    | Vulkan | yes | 201 | incomplete | incomplete | incomplete | killed at 1504 s | No summary: my own `timeout 1500` killed the run (exit 124) while real models were loading. Skips seen before the kill: 10 "model fixture not present", 3 `scripts/kvarn-gate/wiki.test.raw` not found, SmolLM2 (135M and 360M) and SmolLM2 GGUF not present, DeepSeek-V2-Lite-Chat Q2_K not present. A full rerun is needed, without a short timeout. |
    | Cuda | none | 463 | pending | pending | pending | pending | pending |

    **Server, 9 failures, one product bug.** All nine are the DSpark refusal, not model loading. The planner sets `DsParkRequested` when `request.DSparkRequired` is true (`ExecutionPlanner.cs:138-140`, added 2026-11-06 in `6e17333c7`). The server loader hard-codes `DSparkRequired = true` (`InferenceEngineLoader.cs:235`, since 2026-10-07). So every server load requests DSpark, even with no DSpark head configured. `ForwardPassSelection` then refuses it: "requires --dspark-model" (5 accelerator-residency tests) or "not supported with continuous batching" (4 prefill-chunk tests). **Fixed in `15db0ac9` (2026-11-08).** The planner now requests DSpark only from a configured path or an explicit DSpark spec type, which matches `BuildSpeculationPlan`. The loader is unchanged. Two new synthetic planner tests in `ExecutionPlannerTests` (no weights): one reproduces the refusal before the fix and passes after it, the other pins the explicit-spec-type refusal. `ExecutionPlannerTests` 27/27, `ForwardPassSelectionTests` 69/69. The Server heavy rerun, which confirms the nine failures are gone, is pending a quiet machine.

    The per-class time is not printed by this runner. The 3.5 s total is too short for a real-weight run, so no real-model test ran in Server. The same holds for Sessions (0.75 s).
- **Legacy architecture shim:** delete `ModelArchitectureTraits.Legacy` (12 literals; `qwen3vlmoe` lives only there) once the hybrid layer topology, RWKV head size and NEOX table are keyed on the descriptor. Refactor only; each step keeps the golden tests passing.
- **Golden plan tests not automatable here:** busy-port race and Ctrl-C during `capture-golden`. Record as not automated.

### A6. GLM-5.x: K-pool selection drives attention (`glm5next`)

- **Why:** the K-pool indexer builds and stores its pooled keys, but selection does not yet drive attention. The full-attention shortcut stays until it does.
- **Next step:** make selection gate the attention reads, remove the full-attention shortcut, then add the isolated numerical tests (KDA recurrence, Sinkhorn mHC, pool construction and selection, sequence isolation).
- **Done when:** the isolated tests pass against the reference, and the shortcut is gone.
- **Owner doc:** [2026-10-03-glm5-port-plan.md](2026-10-03-glm5-port-plan.md) (item 2d). The full checkpoint (236 GiB) is blocked (Part C).

### A7. Research facts that are still unchecked

- **Why:** the coverage-facts file has nine open checks. Each one was recorded from a source and not yet confirmed against it.
- **Next step:** check each against its source, or delete the line with a note.
- **Done when:** no unchecked line remains in the file.
- **Doc:** [research/2026-10-31-coverage-facts.md](research/2026-10-31-coverage-facts.md).
- **Status 2026-11-08:** 8 of 9 checks closed, each with its source (HF tree listings, licence texts, the Llama 4 mirror config). One stays open: the llama-server logprob field names, which need the pinned `bed0a8566` source or a real build. The local `llama-server.exe` is a 9 KB stub, so that item is blocked on the source, not on work here.

---

## Part B: Decisions

1. **Batched MoE prefill, phase 0 and 1.** Phase 1 is done. Approve phase 0 and phase 1 as the first implementation, so the measurements can start. Phases 2 to 5 are gated on those measurements. The phase 2 gate needs a discrete GPU (Part C). Owner doc: [2026-10-03-batched-moe-prefill-plan.md](2026-10-03-batched-moe-prefill-plan.md).
2. **Admission follow-up 10: `FallbackChat` and `ChatProtocolId` disagree for six architectures.** The disagreement is by design (CLI and server no-template fallback versus engine API). Decide: retire `FallbackChat`, or keep it as a derived convenience.
3. **Gemma 2 beyond the 2B file.** Merged plan item B4 holds the disk decision and the golden.
4. **Admission follow-up 2, Llama 4 heavy parity.** Merged plan item A13 holds the quiet-machine run. The owner deferred it on the grounds of RAM; decide whether to run it at all on this machine.

---

## Part C: Blocked

Each item needs a checkpoint that does not fit, a discrete GPU, a second machine, or a reference that does not exist here. Do not start them.

- **MiniMax-H3** (9 open boxes, [2026-10-03-minimax-h3-port-plan.md](2026-10-03-minimax-h3-port-plan.md)): the DiT denoiser, the real-checkpoint fixture parity at level 3, long-clip VAE verification on real video, and the I2V, FL2V and Ref2VA conditioning extensions. Real-checkpoint parity needs the checkpoint, which is not on any drive.
- **Streamed residency** (5 open boxes, [2026-10-03-streamed-residency-plan.md](2026-10-03-streamed-residency-plan.md)): prefill on the streamed path, leases for the Vulkan hybrid pass, the residency abstraction, automatic split planning, and the larger-than-RAM GLM-5.3 run (236 GiB, paged, correctness only). Blocked on disk and on a design for the prefill path.
- **DiffusionGemma Level 4 paged run and the Q4_K_M download** (16.8 GB): the download is done (2026-11-08, sha256 verified). The CLI refuses the architecture (NotAdmitted). With the port's own override, a bounded smoke run (3-token prefill plus one 2-token canvas step) executes and gives finite logits in about 2 seconds; that is not a correctness result. The parity ladder (A2 is now answered) and the paged run stay open. Details: owner doc, "Real-checkpoint run, 2026-11-08".
- **Batched MoE prefill, phases 2 to 5:** the phase 2 gate needs GPU attention to beat CPU batched attention on the target hardware. Rule 13.
- **TensorSharp takeaways: many-core measurement.** This host has 16 logical CPUs. A measurement here says nothing about 100+ cores.
- **Ported-but-unverified families in general** (rule 14): [ported-families-todo.md](ported-families-todo.md) holds the ladder and the per-family boxes (16 open). Each family's blocker is on its line there.

---

## Owner docs kept as detail

These hold each family's design and evidence. Their open items are listed above:

- [ported-families-todo.md](ported-families-todo.md) (status ladder)
- [2026-10-03-deepseek-v4-review-plan.md](2026-10-03-deepseek-v4-review-plan.md)
- [2026-10-03-diffusiongemma-port-plan.md](2026-10-03-diffusiongemma-port-plan.md)
- [2026-10-03-minimax-h3-port-plan.md](2026-10-03-minimax-h3-port-plan.md)
- [2026-10-03-muse-glimmer-port-plan.md](2026-10-03-muse-glimmer-port-plan.md)
- [2026-10-03-glm5-port-plan.md](2026-10-03-glm5-port-plan.md)
- [2026-10-03-streamed-residency-plan.md](2026-10-03-streamed-residency-plan.md)
- [2026-10-03-batched-moe-prefill-plan.md](2026-10-03-batched-moe-prefill-plan.md)
- [2026-11-07-coverage-completion-programme-plan.md](2026-11-07-coverage-completion-programme-plan.md) (cross-cutting index and the GLM and DeepSeek phases)
- [2026-10-31-gemma-family-plan.md](2026-10-31-gemma-family-plan.md)
- [2026-10-08-architecture-semantics-admission-plan.md](2026-10-08-architecture-semantics-admission-plan.md)
- [2026-10-02-tensorsharp-takeaways-plan.md](2026-10-02-tensorsharp-takeaways-plan.md)
