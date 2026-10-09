# Prompt: registry step 7, STATUS / "Ported, not verified" contract test (for another AI agent)

Copy everything below the line into the agent. Plan: `docs/2-coverage/2026-10-05-architecture-registry-plan.md` (step 7).

---

You are working in `C:\Git-Public\OpenTail.Stingray` (C# 14 / .NET 10, Windows, PowerShell/Git Bash). Read `CLAUDE.md` first
and obey it: `TreatWarningsAsErrors`, never pass `--nologo` to `dotnet test`, no scratch files in the repo root, no
subagents, and especially **rule 10** (the `docs/STATUS.md` matrix must stay honest and sourced) and **rule 14** ("ported,
not verified" families stay internal: not in STATUS.md, README, WHAT-YOU-CAN-DO, RUNNING or the catalogs).

## Background

Admission now lives in descriptors under `src/OpenTail.Stingray.Engine/Architectures/` (`ArchitectureDescriptor`,
`ArchitectureRegistry`, `BuiltInArchitectures`). The point of this step is to make the docs and the registry unable to
drift apart, mechanically:

- every **Admitted** descriptor should be traceable to a row in `docs/STATUS.md`;
- every **NotAdmitted** descriptor must be listed in the "Ported, not verified" table in
  `docs/2-coverage/2026-10-02-tensorsharp-takeaways-plan.md` and must **not** appear in `docs/STATUS.md`.

`docs/STATUS.md` is a free-text capability matrix (about 150 table rows, "Capability | Status | Confidence | Backend |
Notes"), not a per-architecture list. Many admitted architectures are covered by a generic row (e.g. "LLM inference
(GGUF)") or by a row naming a model family rather than the GGUF architecture id. So matching must be explicit, not
fuzzy.

Currently **44 admitted descriptors use `EvidenceDoc = "docs/STATUS.md"`** as a weak fallback, chosen when the real
receipt wasn't obvious; the real receipt is usually named in the comment above each descriptor.

## Task

1. **Add a `StatusAnchor` to `ArchitectureDescriptor`** (nullable string): a substring that must appear in a table row of
   `docs/STATUS.md` for this family (e.g. `"Gemma 4"`), plus a `StatusExemption` (nullable string, required when the
   anchor is null) giving a one-line reason the family has no row of its own (e.g. "covered by the generic
   'LLM inference (GGUF)' row"). `Validate()` must require exactly one of the two for `Admitted` descriptors, and neither
   for `NotAdmitted`/`Experimental`.
2. **Fill them in for all Admitted descriptors by reading `docs/STATUS.md` and each descriptor's own comment.** Do not
   invent anchors: if a family is genuinely not in STATUS.md, use the exemption and say why. Do not edit STATUS.md rows to
   make an anchor match, except to fix an *unsourced or stale* row you can justify from the descriptor's cited evidence
   (rule 10); list any such edit in your report.
3. **Contract tests** (new file `tests/OpenTail.Stingray.Tests.Core/ArchitectureDocsContractTests.cs`; locate the repo root
   by walking up from `AppContext.BaseDirectory` to `CLAUDE.md`, as `ArchitectureRegistryTests` does):
   - every Admitted descriptor with a `StatusAnchor` has that text in at least one `docs/STATUS.md` table line (a line
     starting with `|`), case-insensitive;
   - every Admitted descriptor without one has a non-empty `StatusExemption`;
   - every NotAdmitted descriptor id appears (backticked) in the "Ported, not verified" section of the takeaways plan doc
     and appears **nowhere** in `docs/STATUS.md`, `README.md`, `docs/WHAT-YOU-CAN-DO.md`, `docs/RUNNING.md` (rule 14; use
     whole-word, case-insensitive matching on the architecture id, and report any hit rather than silently deleting it);
   - every Experimental descriptor satisfies the same "not advertised" rule;
   - every `EvidenceDoc` still exists (an existing test does this; keep it).
4. **Fix the weak evidence pointers where cheap and certain.** For each of the 44 descriptors that use
   `EvidenceDoc = "docs/STATUS.md"`: if the comment above it names a specific doc under `docs/` that exists (e.g.
   `docs/done/01-gguf-model-coverage-plan.md §1b`), point `EvidenceDoc` there. If it names none, leave it and list the family
   in your report as "needs a human to find the receipt". Never guess a path.
5. **Expect the tests to find real drift** (e.g. `deepseek32` may be missing from the "Ported, not verified" table, which
   currently lists `deepseek4` and `deepseek41`). Fix doc gaps by adding the missing table row in the takeaways plan doc,
   in the same format and with dated, sourced content from the descriptor's own comment. Do not weaken a test to make it
   pass, and do not change any admission status.

## Constraints

- Behaviour-preserving for the engine: no change to what is admitted or refused, no model runs, no checkpoints.
- Keep `ArchitectureRegistryTests.AdmittedSet_IsExactlyTheSnapshot` and the other registry tests green.
- Use the Edit tool for source edits. Do not generate C# through Python strings with `\n` escapes (a past attempt wrote
  literal newlines into a char literal). Files are LF in the working copy.
- Tick step 7 in the plan doc and add one line describing the new contract.

## Verify

```
dotnet build src/OpenTail.Stingray.Cli -c Release
dotnet build tests/OpenTail.Stingray.Tests.Core -c Release
tests/OpenTail.Stingray.Tests.Core/bin/Release/net10.0/OpenTail.Stingray.Tests.Core.exe -class OpenTail.Stingray.Tests.Core.ArchitectureRegistryTests
tests/OpenTail.Stingray.Tests.Core/bin/Release/net10.0/OpenTail.Stingray.Tests.Core.exe -class OpenTail.Stingray.Tests.Core.ArchitectureDocsContractTests
dotnet build tests/OpenTail.Stingray.Tests.Server.Fast -c Release
tests/OpenTail.Stingray.Tests.Server.Fast/bin/Release/net10.0/OpenTail.Stingray.Tests.Server.Fast.exe
```

0 warnings. Class names must be fully namespace-qualified with the `.exe`; check that each class actually ran tests (a
`Total: 0` means the name didn't match). The full `Tests.Core` run takes ~3 minutes and has **3 known, unrelated failures**
in `KnownEnvironmentVariablesTests`; do not count them as regressions and do not fix them here.

## Commit

Commit only the files you changed (the tree has unrelated modified files: `.gitignore`, two docs under
`docs/1-correctness` and `docs/2-coverage/ported-families-todo.md`, `stage_diagnostics_report.txt`, `tools/engine-compare/`).
End the commit message with the attribution line the session requires, if any.

## Final report

List: how many Admitted descriptors got an anchor vs an exemption (and the exemption reasons, grouped); every drift the new
tests found and how you resolved it; STATUS.md edits (if any) with justification; the evidence-pointer fixes and the
families still needing a human to find their receipt; test output with per-class timings.
