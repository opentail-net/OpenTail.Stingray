# Performance measurement rules

**Owner:** this file. These rules apply to every timing claim in the performance docs. Other docs link here instead of repeating them.

## Rules

1. **Run one thing at a time, alone.** Before any benchmark, check free memory with `Get-CimInstance Win32_OperatingSystem | Select FreePhysicalMemory` (PowerShell). Do not run other heavy processes concurrently. A real incident (2026-10) came from an unrelated concurrent `dotnet test` run: it caused about 24 GB of memory pressure, which both slowed and contaminated the timing.
2. **Interleave control and candidate runs.** Run A, B, A, B, with at least 2-3 runs each, on the same machine state. Compare medians and spread, not the best run. Compare within one session only: machine load has moved llama.cpp's own numbers by 15-20% between sessions.
3. **Keep a control build.** Copy the control build aside before rebuilding the candidate, so an A/B can be re-run.
4. **Use named models, end to end.** Name the checkpoint, quant, prompt, token counts and thread count. Kernel-only numbers are a lead, not a result, until an end-to-end run confirms them.
5. **Show dispatch.** Confirm the path under test ran (a profiling flag, a dispatch counter or a log line), so a speed change is not from a different path.
6. **Check output.** For a change that should not change numerics, compare greedy output (`--repeat-penalty 1.0`) between control and candidate. If numerics change, record the numerics delta and measure perplexity token by token (`stingray perplexity -c 1024 --dump-nll`) against `llama-perplexity` on the same text, comparing paired per-token NLL, not only totals.
7. **No single-run result counts.** A single run is a lead. Repeat before acting. A single-digit-percent difference is noise unless the samples say otherwise.
8. **Record the number.** Put the dated measurement in `PerformanceLeague.md` (the row, with its command and session), not only in a plan.
9. **An iGPU result is not a discrete-GPU result.** A GPU path that loses on the integrated Radeon describes this machine, not the code (CLAUDE.md rule 13). Say which hardware a result came from.

## Where the rules came from

- Rule 1: the `dotnet test` contamination incident, recorded in the FLUX handoff (`4-performance/diffusion/069-flux-vulkan-gemm-perf-handoff.md`) and the TTS/ASR residency docs.
- Rules 2 and 7: [../../docs/4-performance/2026-10-30-performance-pickup.md](../4-performance/2026-10-30-performance-pickup.md) ("How to measure") and the CPU quantised GEMM survey.
- Rule 6: the numerics method, [numerics-investigation-method.md](numerics-investigation-method.md).
- Rule 9: CLAUDE.md rule 13.
