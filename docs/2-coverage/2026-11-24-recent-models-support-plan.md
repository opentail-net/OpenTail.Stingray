# Recent 2026 models: support plan (popular, fits this host)

**Status: PLAN, nothing started (2026-11-24).** Evidence and numbers: [research/2026-11-24-recent-open-weight-models-research.md](research/2026-11-24-recent-open-weight-models-research.md). Policy: CLAUDE.md rule 14 (a ported family is not admitted and not advertised until a real checkpoint, an independent reference, timed runs and a STATUS row all exist), rule 15 (scout first), rule 9 (no scratch files in the repo root), rule 10 (STATUS.md stays honest).

## 1. Goal and scope

Add the popular 2026 text-generation architectures that llama.cpp master implements and this engine does not, where the weights fit this host. Port internally first; admission follows the admission path in section 4.

**In scope.** Architectures that meet every item below:
- popular under assumption A1 (family sum of 30-day downloads across the sampled GGUF repos is at least 100,000, or the primary repo alone is);
- not admitted here (absent from the registry, or NotAdmitted);
- implemented in llama.cpp master (2026-11-24);
- at least one main-file quant at or under 55 GiB (A2).

**Out of scope.** Files over 110 GiB; the 55 to 110 GiB paged band; non-LLM pipelines (audio, speech, image, video, music); drafters (`dflash`, `dspark`, `mtp-`); vision projectors (`mmproj`); Bonsai-2 Q2_0 (already on the backlog).

## 2. Assumptions and owner decisions

These are my assumptions, labelled as the research note labels them. The owner confirms or changes each before Phase 3 starts.

- [ ] **O1 (A1) Popularity.** Family sum of 30-day downloads at least 100,000, or primary repo at least 100,000. Decide whether all-time downloads should count too. Under the current rule, mistral4 (88k), step35 (39k), mistral-large4 (53k) and kimi-k31 (28k) are out.
- [ ] **O2 (A2) Host fit.** Resident limit 55 GiB (63 GiB RAM minus scout's 8 GiB reserve). Confirm the reserve, and say whether the 55 to 110 GiB paged band is ever in scope. The KV cache is not in the file-size test, so confirm that a measured run is the real gate.
- [ ] **O3 Reference revision.** Which llama.cpp build is the independent oracle for goldens: the local source at `bed0a8566` (built here), the vendored b10306, or a newer master build? Each port records the revision it used. See blocker B1.
- [ ] **O4 `qwen4exp` status.** The code says Admitted (2026-11-09, UD-IQ1_S). `ported-families-todo.md` and `00-current-work.md` §2 say not admitted. Reconcile before any other item: either the docs are stale, or the code admission must be reverted under rule 14.
- [ ] **O5 Licences.** Scout reports `other` for Nemotron-3.5-Lightning (an NVIDIA model, in the `ggml-org` repo) and for LiquidAI's LFM3-Omni, and `openmdw-1.1` for Poolside's Laguna. Read each licence before any STATUS row or public text. Licence wording is an owner decision (AFK rule), not a coding step. Apache-2.0 or MIT (Spark, K2-Horizon, North-Mini, Qwen3-Coder-Next, Inkling, Mellum, Ling-3.0-tiny) needs no decision.
- [ ] **O6 Omni inputs.** LFM3-Omni: admit the text path only, with image and audio deferred? (Proposed: yes.)
- [ ] **O7 Non-LLM popular families** (audiocpp, qwen_image, mert2, acestep-lm, and others in research §5): do any need a pipeline? Not decided here.

## 3. Blockers and preconditions

- [ ] **B1 No independent reference on this machine.** `tools/llama.cpp/llama-server.exe` and `llama-cli.exe` are 9,216-byte stubs (checked 2026-11-24). Goldens cannot be captured until a real llama-server exists at the revision chosen in O3. The local `examples/llama.cpp` checkout is pinned to `bed0a8566`. Building it is heavy: check free RAM and running processes first, and do not stop a process this task did not start.
- [ ] **B2 Disk.** Checkpoints go to `E:\_models` (121.8 GB free at check). C: has 39.2 GB and is not used for models. One candidate at a time (memory: disk constraint). Largest planned file: Qwen3-Coder-Next Q4_K_M, 45.2 GiB.
- [ ] **B3 Local stale docs.** Until O4 is resolved, do not edit STATUS.md for any architecture.

## 4. Contracts this must satisfy

Every port, and every admission, must meet these. Sources: `docs/reference/adding-an-architecture.md`, `src/OpenTail.Stingray.Engine/ModelCompatibility.cs`, `src/OpenTail.Stingray.Engine/Architectures/`.

| Contract | What it promises or requires | Where enforced |
|---|---|---|
| Descriptor | `static readonly ArchitectureDescriptor` with `Id` (unique across aliases), `Status` (NotAdmitted until admission), `EvidenceDoc`, exactly one of `StatusAnchor` / `StatusExemption`, and `CreateForwardPass` (`CommonForwardPassFactory.CreateDense`, or `CreateHybridGdn` with `ForwardPassFamily.HybridGdn`). `Validate()` runs at registry build. | `ArchitectureRegistry`, `ArchitectureDescriptorContractTests` (`tests/OpenTail.Stingray.Tests.Core`) |
| Manifest | One `yield return` line in `BuiltInArchitectures.cs`. Static registration only (NativeAOT, no reflection). | build |
| Planner reads capabilities only | Planning code reads `descriptor.Capabilities`, `Id`, `BackendLimitation`, nothing else. | `PlannerAndSelection_ReadOnlyCapabilities_FromTheDescriptor` |
| Admitted-set snapshot | Admitted ids are exactly the snapshot, kept in two copies. Admitting adds the id to both. | `AdmittedSet_IsExactlyTheSnapshot` in `ArchitectureDescriptorContractTests` and `ArchitectureRegistryTests` |
| Facts baseline | Admission adds one line to `Baselines/ArchitectureFacts.txt`. Regenerate with `STINGRAY_UPDATE_BASELINE=1` and review the new line. | `ArchitectureCapabilitiesBaselineTests`, `ArchitectureFacts_MatchBaseline` |
| Literal guard | A quoted architecture id outside `Engine/Architectures/` may not exceed the checked-in count. Shrink only. | `ArchitectureLiterals_OutsideTheArchitectureFolder_OnlyShrink` |
| Refusal path | `ValidateForTextGeneration` refuses NotAdmitted architectures with the descriptor's refusal message. `STINGRAY_DIAGNOSTIC_ALLOW_UNSUPPORTED_ARCH=1` is diagnostic only and never counts as support. | `ModelCompatibility` |
| Quant gate | `IsSupportedWeightDType` must list every tensor type in the file. Checked for the planned quants: Q3_K, Q4_K, Q5_K, Q6_K, IQ4_XS, IQ4_NL, MXFP4, NVFP4, Q4_0, Q4_1 are all listed. Each file's tensor types are rechecked at admission. | `ModelCompatibility.cs` |
| Golden parity | `stingray capture-golden` records llama-server reference tokens with the file's hash. `stingray admit-arch --golden` must exit 0 (exact or near-tie). The golden is committed under `tests/OpenTail.Stingray.Tests.ForwardPass/Goldens/`. | `GoldenParityTests` (heavy: `STINGRAY_RUN_HEAVY_TESTS=1`) |
| Checkpoint pull | `stingray pull -r <owner/repo>` verifies the published SHA-256. Use the same revision for the golden. | `stingray pull` |
| Evidence before advertising | The `docs/STATUS.md` row cites a dated finding in the port's plan, and comes in the same pass as the admission. | CLAUDE.md rule 10 |

**Mapping existing behaviour.** Each port's notes carry one row per llama.cpp branch it reproduces (per hyperparameter, per tensor, per graph branch). Written from the reference source, never from memory (CLAUDE.md rule 8).

## 5. Common steps (every port uses these)

- [ ] **C1 Map.** Read the llama.cpp file(s) at the revision chosen in O3. Write the branch-by-branch table into the port's own plan file (`docs/2-coverage/2026-11-24-<name>-port-plan.md`, one plan per system).
- [ ] **C2 Descriptor and manifest** with `Status = NotAdmitted`, plus the `// <arch> — NOT admitted` block in `src/OpenTail.Stingray.Engine/ModelCompatibility.cs` that rule 14 asks for. ported-families-todo.md says no such block exists for the existing families yet, so this is also an owner question (fold into O4).
- [ ] **C3 Structural and synthetic tests**, labelled as such (ported-families-todo.md verification levels 1 and 2): metadata, tensor set, a tiny forward pass with finite logits. No claim of independent verification.
- [ ] **C4 Real checkpoint.** `stingray scout -r <repo> --quants --budget 63G -c 4096` before the pull (rule 15). Then `stingray pull`, then `capture-golden`, then `admit-arch --golden`.
- [ ] **C5 Admission gates** (only after C4 exits 0): admitted-set snapshot in both contract tests, facts baseline regenerated and reviewed, literal guard checked.
- [ ] **C6 Timed runs.** Several samples each side at a realistic prompt length (not one run). Numbers go in the port plan. Rule 12: check the per-class time, so a silent no-op cannot pass as a real run.
- [ ] **C7 STATUS row** in `docs/STATUS.md`, sourced, in the same pass as C5.
- [ ] **C8 Disk.** Delete the checkpoint after verification if space is needed (memory: checkpoint deletion is allowed for re-downloadable weights).

## 6. Phase 1: evidence (no code, no weights)

- [x] 2026-11-24: HF listings (1,158 GGUF repos), llama.cpp master architecture lists, engine registry, and scout on 23 repos. Recorded in the research note.
- [ ] P1.1 Re-run scout at the revision chosen in O3 for each Tier 1 and Tier 2 repo. The 2026-11-24 runs used each repo's default head at that time; the commit hash is recorded in each scout JSON. Record the revisions used in the research note.
- [ ] P1.2 Re-fetch the llama.cpp files at the chosen revision. Compare line counts and base classes with the master figures in research §7.
- [ ] P1.3 Confirm the popularity numbers on the day of each port (30-day figures move).
- [ ] P1.4 Read the licence for each candidate flagged in O5.

## 7. Phase 2: Tier 0 (already supported architecture, no new code)

- [ ] **T0.1 Nanbeige4.3-33B** (`bartowski/Nanbeige_Nanbeige4.3-33B-GGUF`, arch `llama`, admitted; 158,031 downloads in 30 days). Scout: Q3_K_L 16.1 GiB to Q8_0 32.8 GiB all fit. Capture a golden with the O3 reference, then decide with O1 whether to advertise it. No code expected. If the golden fails, record the divergence and stop (rule 14: do not patch the llama descriptor to pass).

## 8. Phase 3: Tier 1 ports (popular, fits, reuses an admitted family)

Order is my recommendation: smallest and most reuse first. Each item uses section 5 (C1 to C8).

- [ ] **T1.1 `spark2_5`: Spark-X2.6-4B** (`XHToken/Spark-X2.6-4B-GGUF`, 1,004,919 in 30 days; Apache-2.0 per scout). Reference: llama.cpp `spark2-5.cpp` (137 lines; gated attention `wqkv_gate`, dense FFN). Files: Q4_K_M 3.3 GiB, Q8_0 4.1 GiB, full 7.7 GiB. Spark-X2.5-4B (282,113 in 30 days; 1,047,759 all-time) comes after, with its own scout.
- [ ] **T1.2 `k2-horizon` dense**: 0.9B, 3.7B, 7B (IFM repos; Apache-2.0 per scout; family sum 1,429,658). Reference: `k2-horizon.cpp` (356 lines), dense path only. MoVA is T2.3. Files: 0.6 to 16.8 GiB.
- [ ] **T1.3 `lfm3`: LFM3-Omni-2B, text path** (`LiquidAI/LFM3-Omni-2B-GGUF`, 566,642 in 30 days; licence `other`, O5; O6 for omni inputs). Reference: `lfm3.cpp` (211 lines). Base: `lfm2` (admitted). Files: 1.0 to 3.2 GiB.
- [ ] **T1.4 `cohere2moe`: North-Mini-Code-1.0** (`unsloth/North-Mini-Code-1.0-GGUF`, 124,324 in 30 days; Apache-2.0). Reference: `cohere2moe.cpp` (413 lines). Base: `cohere2` (admitted) plus the MoE block. Files: Q3_K_M 13.2 GiB to Q5-class.
- [ ] **T1.5 `nemotron_h_moe`: Nemotron-3.5-Lightning-30B-A3B** (`ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF`, 207,662 in 30 days; licence `other`, O5). Reference: `nemotron-h-moe.cpp` (154 lines), which builds on `nemotron-h.cpp` (admitted `nemotron_h`). Files: Q4_0 17.6, NVFP4 18.4, Q8_0 31.3 GiB. The `mtp-`, `dflash-` and `dspark-` files are excluded.
- [ ] **T1.6 `laguna`: Laguna-XS-2.1** (`poolside/Laguna-XS-2.1-GGUF`, 12,886 in 30 days; 665,924 all-time; licence `openmdw-1.1`, O5). Reference: `laguna.cpp` (328 lines), which shares MoE and gate structure with `afmoe` (admitted). Q4_K_M 18.9 GiB. Laguna-S-2.1 (117.6 B; Q3_K_M 50.3 GiB) only after XS passes, and only at the edge of A2.
- [ ] **T1.7 `qwen3next`: Qwen3-Coder-Next** (`unsloth/Qwen3-Coder-Next-GGUF`, 217,217 in 30 days; Apache-2.0 per scout; base release date not checked). Reference: `qwen3next.cpp` (35.7 KB). Hybrid Gated-DeltaNet, the same family as admitted `qwen35`/`qwen35moe`, so it may reuse `GdnKernels` and `HybridGdnForwardPass`. Q4_K_M is 45.2 GiB, which is at the edge of A2: run only after a free-RAM check, and record the peak. Q5_K_M (52.9 GiB) comes after Q4 passes.

## 9. Phase 4: Tier 2 ports (popular, fits, large or unclear base)

- [ ] **T2.1 `inkling2_loop`: Inkling-2-Mini** (`unsloth/Inkling-2-Mini-GGUF`, 1,913,407 in 30 days; family sum 3,912,873; Apache-2.0 per scout; created 2026-11-20). Largest family in the sample. References: `inkling2.cpp` (458 lines) and `inkling2-loop.cpp` (123 lines). The base has about ten mechanisms not in any admitted family (research §7). Split by mechanism, one sub-plan each, each with its own synthetic test and golden check, in this order: (a) plain attention and FFN with SWA and full layers, (b) MoE router with expert bias, (c) K/V short convolutions, (d) relative attention with clipping, (e) attention and FFN residual queries, (f) the loop wrapper with shared physical layers. Files: IQ1_S 7.2 GiB to Q8_K_XL 21.3 GiB.
- [ ] **T2.2 `bailingmoe3`: Ling-3.0-tiny** (`bloomer010/Ling-3.0-tiny-GGUF`, 243,858 in 30 days; MIT). Reference: `bailingmoe3.cpp` (511 lines). `bailingmoe2` is not admitted either, so there is no nearby admitted base. Files: 3.6 to 14.7 GiB. Ling-3.0-flash (124.4 B; smallest 67.4 GiB) is out of scope.
- [ ] **T2.3 `k2-horizon` MoVA 36B-A4B** (IFM; family sum above). Adds value experts (`attn_v_gate`, `attn_v_exps`) to T1.2. Q4_K_M 20.8 GiB to Q8_0 37.1 GiB fit; BF16 69.8 GiB does not.
- [ ] **T2.4 `mellum`: Mellum2.1-12B-A2.5B-Thinking** (`JetBrains/Mellum2.1-12B-A2.5B-Thinking-GGUF`, 242,427 in 30 days; Apache-2.0). Reference: `mellum.cpp` (373 lines); the base architecture is not yet read, so C1 comes first. Files: MXFP4 6.6 GiB to BF16 22.6 GiB; the `MTP-` files are excluded.

## 10. Phase 5: excluded and deferred (decided here, revisit only when the owner asks)

Excluded by file size (smallest main file, from scout):

| Model | Params | Smallest main file | Reason |
|---|---|---|---|
| Kimi-K3 (`unsloth/Kimi-K3-GGUF`) | 2,780 B | 434.3 GiB | over 110 GiB |
| Mistral-Large-4 (`unsloth/Mistral-Large-4-1T-A52B-GGUF`) | 1,049 B | 217.0 GiB | over 110 GiB |
| Hy4 family (`AngelSlim/Hy4-preview-GGUF` and others) | 770 B | 213.7 GiB | over 110 GiB; arch names not in llama.cpp master (research §8) |
| MiniMax-M3.1-Flash (`unsloth/MiniMax-M3.1-Flash-GGUF`) | 426 B | 120.5 GiB | over 110 GiB |
| Step-3.5-Flash (`ggml-org/Step-3.5-Flash-GGUF`) | 197 B | 110.6 GiB | over 110 GiB |
| Ling-3.0-flash (`AtomicChat/Ling-3.0-flash-GGUF`) | 124.4 B | 67.4 GiB | 55 to 110 GiB band with no resident quant; not planned |

Excluded by popularity: Mistral-Small-4-119B (`mistral4`, family sum 87,901; Q3_K_M 50.6 GiB on the edge), `mistral-large4` (52,674), `step35` (39,349), `kimi-k31` (28,136).

Deferred for every port (not part of the initial port):
- Omni inputs (LFM3, O6), MTP heads (`mtp-` files), speculative drafters (`dflash-`, `dspark-`), vision projectors (`mmproj`), and any paged run for the 55 to 110 GiB band.

## 11. Verification at a glance

- Structure and contracts: run the `tests/OpenTail.Stingray.Tests.Core` project (`dotnet test`, never with `--nologo`, CLAUDE.md rule 1). Every admission changes the snapshot and baseline (section 4).
- Goldens: `GoldenParityTests`, heavy (`STINGRAY_RUN_HEAVY_TESTS=1`). Check the per-class wall-clock time, not only pass/fail (CLAUDE.md rule 12).
- Timed runs: state the host (Ryzen 7 5700G, 63 GiB RAM, no discrete GPU, CLAUDE.md rule 13) and the free RAM at the time. Rule 13 applies: an iGPU-only timing says nothing about a discrete GPU.

## 12. Open items (not started)

Everything above is open. Nothing is committed to this plan's phases yet, and no item is checked beyond the Phase 1 evidence line.
