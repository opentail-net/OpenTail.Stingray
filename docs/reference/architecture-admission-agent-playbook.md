# Architecture admission: agent playbook

Operational contract for a coding model admitting or triaging a GGUF architecture. It points at the canonical docs and commands; it does not repeat them.
Background and plan: [checkpoint scout plan](../3-product-and-runtime/2026-10-09-checkpoint-scout-and-ai-admission-plan.md). Mechanics of adding a descriptor: [adding-an-architecture.md](adding-an-architecture.md). Tool usage: [061-coverage-tooling.md](061-coverage-tooling.md).

## Source of truth, highest first
1. Repository code and tests, and the reference implementation you actually have (llama.cpp or another), with its revision recorded. `examples/` and `tools/llama.cpp` are git-ignored in this repo: a maintainer may have them, other contributors do not, so never write a step that assumes they exist.
2. Hash-pinned goldens and timed test runs.
3. Scout output, model cards, names, and earlier conversation claims. These are **hypotheses**, never proof of numerical semantics.

## The loop

1. **Scout first.** `stingray scout -m <gguf> --format json -o <scratch>/scout.json` (add `--budget 64G` to see the execution decision). Read blockers, tokenizer facts, tensor irregularities and architecture resolution before anything else. A `confirmed` blocker means the engine's own gate would refuse; a `suspected` one is a lead to check.
2. **Check the context.** Active backlog (`docs/00-current-work.md`), memory budget, which checkpoints are installed (`models/`, `models/_models/`), matching reference binaries, licence terms, and whether the architecture or dtype is already supported. Missing checkpoints in the ranked backlog: start the download proactively.
3. **Classify the problem before coding.** Exactly one of: tokenizer, weight dtype/dequantization, tensor layout, architecture semantics, runtime/cache, backend. Do not fix several by adding broad aliases or loosening a parity test.
4. **Read the real implementation** of each semantic difference in the reference you have. Record file, revision and the operation you are porting. If ported maths looks wrong, diff it against the reference first (`CLAUDE.md` rule 8).
5. **Use existing machinery.** Descriptor plus manifest line; touch Core or central dispatch only when the evidence shows the abstraction cannot express the model.
6. **Run the real thing, bounded.** `admit-arch -m <gguf>` for an unregistered architecture (explicit, unverified, memory-bounded). Read the gate from scout (`--budget 64G -c <ctx>`): `allowed` means the upper-bound CPU estimate plus reserve fits (a safety gate, not proof); `blocked` or an Unknown estimate (MLA, hybrid, RWKV and unregistered families are not modelled) means **park it**: write down the blocker and take the next backlog item. Never override the gate to run an oversized checkpoint on this host.
7. **Independent reference.** `capture-golden` when a compatible oracle exists (it needs a llama.cpp you supply locally; **checking** an already-recorded golden with `admit-arch --golden` needs none, so goldens contributed by others are verifiable by everyone); compare prompt tokenization separately from forward-pass parity; record exact / near-tie / diverged with the reference margins. Never call a mismatch "near-tie" without the margin evidence.
8. **Test.** Targeted tests first, then broader ones, one heavy process at a time. Build the test project, then run the `.exe` with the fully qualified class name if `dotnet test` filters misbehave. A zero-test run is not evidence.
9. **Check the timing.** A real-weight run takes seconds and logs a weight-loading line; "passed" in ~0.1-0.4 s means the test silently no-op'd (`CLAUDE.md` rule 12).
10. **Promote only on evidence.** Descriptor status, golden, test evidence and `docs/STATUS.md` change together, after: real checkpoint, independent reference, timed run, licence noted. Missing any of these: report the blocker, do not claim completion.
11. **Finish with a performance pass and a DRY pass** (`CLAUDE.md` rule 7) when a model's port is complete.

## Hard rules
- No subagents in this project (`CLAUDE.md` rule 6).
- A family ported before a real checkpoint could verify it stays internal: `NOT admitted` block, absent from STATUS, README and catalogs (rule 14).
- Scout similarity and feature hypotheses never decide admission and never substitute for a golden. If a scout rule is wrong or noisy, say so and fix or remove it.
- Investigation output (scout reports, receipts, dumps) goes to scratch, not the repo root (rule 9). Commit only synthetic, path-free examples.
- No Python reference scripts.
- A GPU result from this iGPU-only machine is not evidence about GPUs in general (rule 13).

## Conclusion to write when you stop
What changed; what was verified (with commands and timings); what was not tested; the exact remaining admission gate.
