# Prompt: registry migration step 4 (for another AI agent)

Copy everything below the line into the agent. Context doc: `docs/2-coverage/2026-10-05-architecture-registry-plan.md`.

---

You are working in `C:\Git-Public\OpenTail.Stingray` (C# 14 / .NET 10, Windows, PowerShell/Git Bash). Read `CLAUDE.md` first
and obey it (in particular: `TreatWarningsAsErrors`, never pass `--nologo` to `dotnet test`, no scratch files in the repo
root, do not use subagents).

## Task

Finish step 4 of the architecture-registry plan: move every remaining **admitted** architecture out of the legacy
allowlist `s_textGenerationArchitectures` in `src/OpenTail.Stingray.Engine/ModelCompatibility.cs` and into
`ArchitectureDescriptor`s under `src/OpenTail.Stingray.Engine/Architectures/`, then delete the legacy `HashSet`.

Look at the existing examples first: `ArchitectureDescriptor.cs`, `ArchitectureRegistry.cs`,
`BuiltInArchitectures.cs`, `GraniteArchitecture.cs` (an evidence comment moved from the allowlist into its descriptor
file), `MuseGlimmerArchitecture.cs` (with an alias).

## This is a behaviour-preserving move. No checkpoints, no model runs.

Nothing about what is admitted may change. You are relocating a decision and its evidence, not re-judging it. Do **not**
download models, run real-weight tests, "fix" an evidence comment, or admit/refuse anything new. If a comment looks wrong
or contradictory, keep it verbatim and list it in your final report.

## The 59 names that must stay admitted (and nothing else)

qwen, qwen2, qwen2moe, qwen3, qwen3moe, qwen2vl, qwen35, qwen35moe, mimo, mimo2, gemma, gemma2, gemma3, gemma3n, phi2,
phi3, phimoe, olmoe, rwkv7, rwkv6, gpt-oss, deepseek2, smollm3, apertus, gptneox, falcon, olmo2, exaone, orion, ernie4_5,
paddleocr, qwen3vl, deepseek2-ocr, granitehybrid, nemotron_h, lfm2, lfm2moe, internlm2, starcoder2, cohere2, glm4, glm4moe,
stablelm, hunyuan-dense, hunyuan-moe, afmoe, gpt2, granitemoe, olmo, starcoder, codeshell, jais2, jais, maincoder, exaone4,
mistral3, ministral, xverse, minicpm

(Already migrated, do not touch: gemma4, granite, llama, llama4, muse-glimmer/muse_glimmer, and the seven NotAdmitted
families.) Verify this list against the file before starting; it is the snapshot at commit `de903217`.

## Rules

1. **One descriptor per GGUF architecture string.** Use `Aliases` only where the existing comment already treats two
   spellings as the same family (e.g. `muse_glimmer`). Do not merge `qwen2`/`qwen3` etc. into one descriptor: they
   differ in chat behaviour. When several architectures share one receipt, give each its own descriptor and point each
   at the same `EvidenceDoc`; put the shared comment in one file and reference it from the others.
2. **Move evidence comments verbatim** into the descriptor's file (above the `Descriptor` field), including dates,
   receipts and caveats ("DO NOT MODIFY THIS ARCHITECTURE'S CODE PATH..."). Light reflow is fine; no rewording of claims.
3. **`EvidenceDoc` must be a real repo-relative path** that exists (a test checks it). Use the doc the comment cites
   (`docs/done/01-gguf-model-coverage-plan.md`, a `*GreedyParityTests` is a test file, not a doc: then use the
   closest doc cited, else `docs/STATUS.md`). Do not invent paths.
4. **Do not set `ThinkingDefaultOff` or `FallbackChat`** on anything new. The defaults (`false`, `ChatMl`) reproduce
   today's behaviour for every architecture except the ones already migrated.
5. **File layout:** group by trunk, one file per descriptor is fine but large families may share a file that exposes
   several `static readonly ArchitectureDescriptor` fields (qwen*, gemma*, phi*, rwkv*, deepseek2*, granite* variants,
   glm4*, hunyuan*, exaone*, jais*, starcoder*, olmo*). Add each to `BuiltInArchitectures.Create()`.
6. **Delete the legacy path** once empty: remove `s_textGenerationArchitectures`, make
   `IsTextGenerationArchitectureSupported` registry-only, and update the "Supported profiles" list in
   `ValidateForTextGeneration` to come from the registry. Keep `STINGRAY_DIAGNOSTIC_ALLOW_UNSUPPORTED_ARCH` behaviour.
   An architecture in neither place is still refused with the same message as today.
7. **Tests** (in `tests/OpenTail.Stingray.Tests.Core/ArchitectureRegistryTests.cs`):
   - add `AdmittedSet_IsExactlyTheSnapshot`: the set of admitted ids+aliases equals the 59 + 4 + muse names above, so a
     dropped or accidental entry fails loudly;
   - update `MigratedDescriptorIds_AreNotAlsoInTheLegacyAllowlist` (it scans the legacy list; remove or adapt it when the
     list is deleted);
   - keep all existing tests green.
8. Do not use `\n` escapes inside Python-generated C# strings; a past attempt wrote literal newlines into a char literal.
   Prefer the Edit tool for source edits. Files in this repo are LF in the working copy; keep that.

## Verify

```
dotnet build src/OpenTail.Stingray.Cli -c Release
dotnet build tests/OpenTail.Stingray.Tests.Core -c Release
tests/OpenTail.Stingray.Tests.Core/bin/Release/net10.0/OpenTail.Stingray.Tests.Core.exe -class OpenTail.Stingray.Tests.Core.ArchitectureRegistryTests
dotnet build tests/OpenTail.Stingray.Tests.Server.Fast -c Release
tests/OpenTail.Stingray.Tests.Server.Fast/bin/Release/net10.0/OpenTail.Stingray.Tests.Server.Fast.exe
```

Build with 0 warnings. (Class names must be fully namespace-qualified when invoking the `.exe`.) The full
`Tests.Core` run takes ~3 minutes and currently has **3 known, unrelated failures** in `KnownEnvironmentVariablesTests`
(`STINGRAY_DIFFUSIONGEMMA_ENABLE_REAL`, `STINGRAY_MOE_PHASE_TIMING` missing from `KnownEnvironmentVariables.All`); do not
"fix" them as part of this task and do not count them as regressions.

## Commit

Commit only the files you changed (the working tree has unrelated modified files: `.gitignore`, two docs,
`stage_diagnostics_report.txt`, `tools/engine-compare/`). Tick step 4 in the plan doc. Commit message ends with the
attribution line required by the session, if any.

## Final report

List: descriptors created, any comment you found stale or contradictory (kept verbatim), any architecture whose evidence
doc you had to guess, and the test output.
