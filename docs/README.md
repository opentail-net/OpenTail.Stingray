# Documentation

Start with the [project README](../README.md). This page maps `docs/`: user guides and reference
first, then the engineering backlog, grouped by theme in priority order.

## Using Stingray

| I want to… | Read |
|---|---|
| Pick and download a model | [MODELS.md](MODELS.md) |
| Know what works today, and how well it was checked | [STATUS.md](STATUS.md) |
| Use the command line | [CLI README](../src/OpenTail.Stingray.Cli/README.md), [reference/cli-option-inventory.md](reference/cli-option-inventory.md) (every option) |
| Run the OpenAI/Anthropic-compatible server | [Server README](../src/OpenTail.Stingray.Server/README.md) |
| Use the NuGet package from C# | [Package README](../src/OpenTail.Stingray/README.md) |
| Generate images or video | [Diffusion README](../src/OpenTail.Stingray.Diffusion/README.md) |
| Choose settings for my hardware | [reference/recommended-configurations.md](reference/recommended-configurations.md), [profiles/](profiles) |
| Look up an environment variable | [reference/env-var-inventory.md](reference/env-var-inventory.md) |
| Query capabilities from code | [reference/capabilities.md](reference/capabilities.md) |
| Run a new GGUF from Hugging Face | [reference/061-coverage-tooling.md](reference/061-coverage-tooling.md) (`pull`, `admit-arch`, `gen-vision-scaffold`) |
| Compare Stingray with other engines | [reference/POSITIONING.md](reference/POSITIONING.md) |
| See performance numbers | [../PerformanceLeague.md](../PerformanceLeague.md) (full, measured, hardware-specific) |

## Understanding the design

- [reference/OpenTail.Stingray-Design.md](reference/OpenTail.Stingray-Design.md): subsystem architecture
- [reference/inference-state-architecture.md](reference/inference-state-architecture.md) and
  [reference/adr-0001-session-cache-lifecycle.md](reference/adr-0001-session-cache-lifecycle.md): sessions and state
- [reference/023-native-skills-architecture-and-design-principles.md](reference/023-native-skills-architecture-and-design-principles.md): skills, instructions and tools
- Everything else in [reference/](reference): inventories, release gates, troubleshooting guides,
  the benchmark prompt, the performance/metrics guide

## Engineering backlog

For maintainers and coding agents. **[00-current-work.md](00-current-work.md) is the single
ordered list of open work**; the folders hold the detail docs behind its items. The folder number
is the priority.

| Folder | What is in it |
|---|---|
| [1-correctness/](1-correctness) | Bugs in what is already claimed ([bugstofix.md](1-correctness/bugstofix.md)). Most correctness items are short enough to live in 00-current-work.md itself. |
| [2-coverage/](2-coverage) | Model families not yet supported or half built: Qwen3.5 MoE/GDN (02), Gemma 4 E4B vision (03), GGML op kernels (050), DeepSeek V3.2/V4 (058), ACE-Step (064) |
| [3-product-and-runtime/](3-product-and-runtime) | Front door (103), release checklist, configuration ownership (04), multi-model runtime (032), sessions (010, 051) |
| [4-performance/](4-performance) | [perf-sweep-plan.md](4-performance/perf-sweep-plan.md) across models, then [cpu/](4-performance/cpu) (05), [gpu/](4-performance/gpu) (vulkan-gemm), [diffusion/](4-performance/diffusion) (069, 093, 094), [audio/](4-performance/audio) (066, 079, 080) |
| [9-external-hardware/](9-external-hardware) | Work that needs hardware this machine lacks: CUDA, ARM64, a discrete GPU |

File numbers inside folders are the order the documents were written, kept so that code comments
citing "docs/094" and similar still find them.

Also here: [tts-benchmark-log.txt](tts-benchmark-log.txt) (appended to by the audio perf-baseline
debug tests), [profiles/](profiles) (read by tests), [diffusion-samples/](diffusion-samples) and
`audio-samples/` (local-only sample output, git-ignored), [research/](research) (papers).

## Archive rule

[done/](done) keeps completed plans, working logs, measurements, negative results and superseded
plans. Nothing there is deleted; each archived document has a banner saying what closed and what,
if anything, carried forward.

- When a backlog item closes: delete its line from 00-current-work.md, put the dated evidence in
  the STATUS.md row (or the detail doc), and move a finished detail doc to `done/` with a banner.
- A partly finished doc is split: the closed parts move verbatim to `done/`, the open parts stay.
- 00-current-work.md is not a log. Progress notes go in the item's detail doc; the full backlog
  history up to 2026-09-27 is [done/00-current-work-log-to-2026-09-27.md](done/00-current-work-log-to-2026-09-27.md).
- Audio evidence log: [done/audio-review-new-progress.md](done/audio-review-new-progress.md).
