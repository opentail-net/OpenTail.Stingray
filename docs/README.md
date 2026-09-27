# Documentation

Start with the [project README](../README.md). This page sorts the rest of `docs/` by who it is
for: most files here are engineering records, not user guides.

## Using Stingray

| I want to… | Read |
|---|---|
| Pick and download a model | [MODELS.md](MODELS.md) |
| Know what works today, and how well it was checked | [STATUS.md](STATUS.md) |
| Use the command line | [CLI README](../src/OpenTail.Stingray.Cli/README.md), [cli-option-inventory.md](cli-option-inventory.md) (every option) |
| Run the OpenAI/Anthropic-compatible server | [Server README](../src/OpenTail.Stingray.Server/README.md) |
| Use the NuGet package from C# | [Package README](../src/OpenTail.Stingray/README.md) |
| Generate images or video | [Diffusion README](../src/OpenTail.Stingray.Diffusion/README.md) |
| Choose settings for my hardware | [reference/recommended-configurations.md](reference/recommended-configurations.md), [profiles/](profiles) |
| Look up an environment variable | [env-var-inventory.md](env-var-inventory.md) |
| Query capabilities from code | [reference/capabilities.md](reference/capabilities.md) |
| Run a new GGUF from Hugging Face | [061-coverage-tooling.md](061-coverage-tooling.md) (`pull`, `admit-arch`, `gen-vision-scaffold`) |
| Compare Stingray with other engines | [reference/POSITIONING.md](reference/POSITIONING.md) |
| See performance numbers | [../PerformanceLeague.md](../PerformanceLeague.md) (full, measured, hardware-specific) |

## Understanding the design

- [reference/OpenTail.Stingray-Design.md](reference/OpenTail.Stingray-Design.md): subsystem architecture
- [reference/inference-state-architecture.md](reference/inference-state-architecture.md) and
  [reference/adr-0001-session-cache-lifecycle.md](reference/adr-0001-session-cache-lifecycle.md): sessions and state
- [reference/023-native-skills-architecture-and-design-principles.md](reference/023-native-skills-architecture-and-design-principles.md): skills, instructions and tools
- Everything else in [reference/](reference): inventories, release gates and troubleshooting guides

## Engineering (active work)

For maintainers and coding agents. Numbered files are plans, handoffs and investigations, named in
the order they were written.

- **Backlog:** [00-current-work.md](00-current-work.md) (ordered runway and open findings),
  [102-status-open-items-plan.md](102-status-open-items-plan.md) (non-green STATUS rows),
  [101-work-queue-after-coverage-plan.md](101-work-queue-after-coverage-plan.md),
  [90-external-hardware-work.md](90-external-hardware-work.md) (needs hardware this machine lacks),
  [bugstofix.md](bugstofix.md)
- **Model coverage:** 02, 03, 07, 08, 050, 054, 058, 064, 066, 086
  (01, 055, [vl-migration-plan-2026-08-20.md](done/vl-migration-plan-2026-08-20.md) done — see [done/](done))
- **Runtime and serving:** 010, 032, 04, 051
- **Performance:** 05, 049, 052, 053, 069, 071, 073, 077-080, 082, 084, 088, 090, 092-094,
  [perf-sweep-plan.md](perf-sweep-plan.md), [cpu-performance-baseline.md](done/cpu-performance-baseline.md),
  the `PerformanceLeague-*` plans, [perf-loop-project-review-progress.md](perf-loop-project-review-progress.md),
  [vulkan-gemm/](vulkan-gemm)
  (067, 068, 072, 074-076, 081 done — see [done/](done))
- **Audio:** [audio-review-new-progress.md](audio-review-new-progress.md),
  [tts-performance-baseline-and-plan.md](tts-performance-baseline-and-plan.md)
  ([qwentts-cosyvoice3-handoff.md](done/qwentts-cosyvoice3-handoff.md),
  [2026-09-24-audio-recheck-plan.md](done/2026-09-24-audio-recheck-plan.md) done — see [done/](done))
- **Front door:** [103-front-door-design.md](103-front-door-design.md) (README and first-run design),
  [nuget-release-checklist.md](nuget-release-checklist.md)

## Archive

[done/](done) keeps completed work, measurements, negative results and superseded plans. Nothing
there is deleted; each archived document has a banner saying what closed and what, if anything,
carried forward to the backlog. The archive rule is at the end of
[00-current-work.md](00-current-work.md).
