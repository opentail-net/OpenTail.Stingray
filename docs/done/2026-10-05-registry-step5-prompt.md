# Prompt: registry migration step 5, `Experimental` status (for another AI agent)

Copy everything below the line into the agent. (Called "#4" in chat; it is **step 5** in
`docs/2-coverage/2026-10-05-architecture-registry-plan.md`. Step 4 is already done.)

---

You are working in `C:\Git-Public\OpenTail.Stingray` (C# 14 / .NET 10, Windows, PowerShell/Git Bash). Read `CLAUDE.md` first
and obey it (`TreatWarningsAsErrors`, never pass `--nologo` to `dotnet test`, no scratch files in the repo root, no
subagents, and rule 14: "ported, not verified" families stay internal and unadvertised).

## Background

The registry lives in `src/OpenTail.Stingray.Engine/Architectures/` (`ArchitectureDescriptor`, `ArchitectureRegistry`,
`BuiltInArchitectures`). `AdmissionStatus` has `Admitted`, `NotAdmitted` and `Experimental`; an `Experimental` descriptor
is usable only when its `ExperimentalEnvVar` is `"1"` (`ArchitectureDescriptor.IsUsable()`). **No descriptor uses
`Experimental` yet, and the path is untested end to end.** Today's real gates are hand-written:

- Bonsai2 PRISM: `ModelCompatibility.ValidateForTextGeneration` checks the *tensor type*
  (`BonsaiQuant.IsBonsaiType`) and `STINGRAY_EXPERIMENTAL_PRISM`. It is a quant-type gate, **not** an architecture
  (the file is `qwen35`, which is admitted).
- `STINGRAY_DIFFUSIONGEMMA_ENABLE_REAL` in `Diffusion/DiffusionGemma/DiffusionGemmaForwardPass.cs` (a pipeline-level
  override on a NotAdmitted family).
- `STINGRAY_DIAGNOSTIC_ALLOW_UNSUPPORTED_ARCH` (a global diagnostic bypass; keep as is).

## Task

Make `Experimental` a real, tested, behaviour-preserving mechanism, and route the PRISM gate through the same shape so
the "env var + reason + evidence" logic is declared in one place instead of hard-coded in `ModelCompatibility`.

1. **Fix the refusal message for `Experimental`.** In `ValidateForTextGeneration`, an `Experimental` descriptor whose env
   var is unset currently gets the generic "not admitted ... status Experimental" text. It must say: the family is ported
   but not verified, name the env var to set (`set <VAR>=1 to try it`), say outputs are unverified, and cite
   `EvidenceDoc`. `NotAdmitted` text stays as it is today.
2. **Declare the PRISM gate declaratively.** Add a small type in `Engine/Architectures/` (suggested name
   `ExperimentalQuantGate`: tensor-type predicate, env var, reason, evidence doc) with a registry list, and have
   `ValidateForTextGeneration` iterate it instead of the hard-coded `BonsaiQuant`/`STINGRAY_EXPERIMENTAL_PRISM` block.
   The user-visible message and the "also exempt these dtypes from the unsupported-format check when enabled" behaviour
   must be **identical** to today (diff the strings; capture them in a test before refactoring). Keep it small; do not
   build a general plugin system.
3. **Do not promote any architecture to `Experimental`.** No family is changed from `NotAdmitted`. A family becomes
   `Experimental` only if its forward pass genuinely runs end to end behind an env gate today, and none does. If you
   believe one qualifies, say so in the report and leave it alone.
4. **Keep the env-var inventory honest.** `KnownEnvironmentVariablesTests` scans `src/` for `STINGRAY_*` literals. Because
   descriptors read `ExperimentalEnvVar` dynamically, add a test that every `Experimental` descriptor's and every quant
   gate's env var is listed in `KnownEnvironmentVariables.All`; register `STINGRAY_EXPERIMENTAL_PRISM` there if missing.
   Do **not** touch the two unrelated pre-existing gaps (`STINGRAY_DIFFUSIONGEMMA_ENABLE_REAL`, `STINGRAY_MOE_PHASE_TIMING`)
   or "fix" those 3 failing tests.
5. **Tests** (in `tests/OpenTail.Stingray.Tests.Core/ArchitectureRegistryTests.cs` or a new sibling file):
   - an `Experimental` test descriptor is refused with the env var named, and usable once the variable is set (use a
     variable name **not** starting with `STINGRAY_`, and restore it in `finally`);
   - message text for PRISM unchanged (golden strings from before your change);
   - quant-gate with env unset refuses, with env set exempts the dtype from the unsupported-format error;
   - existing 28 registry tests stay green (including `AdmittedSet_IsExactlyTheSnapshot`).

## Constraints

- Behaviour-preserving for every real model: no checkpoint, no model run, no change to what is admitted or refused.
- Use the Edit tool for source edits. Do not write C# through Python strings with `\n` escapes (a past attempt put literal
  newlines into a char literal). Files are LF in the working copy.
- Update `docs/2-coverage/2026-10-05-architecture-registry-plan.md`: tick step 5 and note that PRISM remains a quant gate
  declared through `ExperimentalQuantGate`, not an architecture.

## Verify

```
dotnet build src/OpenTail.Stingray.Cli -c Release
dotnet build tests/OpenTail.Stingray.Tests.Core -c Release
tests/OpenTail.Stingray.Tests.Core/bin/Release/net10.0/OpenTail.Stingray.Tests.Core.exe -class OpenTail.Stingray.Tests.Core.ArchitectureRegistryTests
tests/OpenTail.Stingray.Tests.Core/bin/Release/net10.0/OpenTail.Stingray.Tests.Core.exe -class OpenTail.Stingray.Tests.Core.BonsaiPrismTests
dotnet build tests/OpenTail.Stingray.Tests.Server.Fast -c Release
tests/OpenTail.Stingray.Tests.Server.Fast/bin/Release/net10.0/OpenTail.Stingray.Tests.Server.Fast.exe
```

0 warnings. Use fully namespace-qualified class names with the `.exe`; if `BonsaiPrismTests` lives in another test project,
find it with grep and run that project's `.exe`. The full `Tests.Core` run takes ~3 minutes; its 3
`KnownEnvironmentVariablesTests` failures predate this task.

## Commit

Commit only the files you changed (the tree has unrelated modified files: `.gitignore`, two docs,
`stage_diagnostics_report.txt`, `tools/engine-compare/`). End the commit message with the attribution line the session
requires, if any.

## Final report

List: files changed, the before/after PRISM message (should be identical), any family you think could be `Experimental`
and why you left it, and test output with per-class timings.
