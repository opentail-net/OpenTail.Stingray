# CPU MoE expert residency: measurement plan (pre-registered)

Status 2026-11-09. Written before any run. The thresholds in section 5 are fixed here and are not to be changed after the numbers are in. Owner: the SharpMind review, item R2 ("investigate, don't immediately implement"). Measurement rules: [performance-measurement-rules](../reference/performance-measurement-rules.md).

## 1. Question

Under memory pressure, how much does CPU MoE decode throughput depend on the OS keeping the memory-mapped expert pages resident?

The most a process-managed expert residency cache could recover on this machine is the gap between the no-pressure and the pressure conditions below. If that gap is small, a second expert cache is not worth its memory and complexity. The review says the same ("measure before introducing another caching subsystem").

## 2. What the code does today (verified)

- CPU MoE kernels read expert weights through `TensorRef.DataPtr`, which points into the memory-mapped GGUF. `ForwardPass.Moe.cs` lines 236-258, 660-672, 756-764, 819-831.
- `ExpertSlotManager` (the SLRU expert cache) is **GPU-only**: it takes a `VulkanBackend`. `MoEPrefetcher` drives only `ExpertSlotManager` (Vulkan) and `CudaExpertSlotManager` (CUDA). The CPU path has no residency cache and no expert prefetch.
- `MmapPrefault` warms the mapped pages before the first forward pass. Its auto mode **skips** prefaulting when the mapped set exceeds about 80% of available RAM, because it would thrash (`MmapPrefault.cs`, `RamGate`).
- `calibrate` times the expert kernels on each device and prints where expert work should run (`ExpertCalibration.cs`). It measures kernels, not paging.

## 3. Model and machine

- **Model:** `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf`, 17.28 GB, architecture `qwen3moe`, on `F:\_models`. Chosen because it is the largest supported MoE checkpoint near the RAM size, so pressure is reachable. OLMoE (3.9 GB, the perf plan's A3 checkpoint) fits in RAM with room to spare and cannot be put under pressure on this machine.
- **Architecture check:** `qwen3moe` is registered (`QwenArchitectures.cs:53`). The model's expert count and experts-used must be read from its metadata before the first run (`stingray inspect`), not assumed.
- **Machine:** Ryzen 7 5700G, 8 physical cores, 64 GB RAM. Model file on `F:`, a Samsung 870 QVO SATA SSD (not NVMe). The paging cost is therefore that of a SATA SSD.
- **Build:** CLI published from the current `HEAD` (commit `5d6e2044`, TensorShape change included), NativeAOT, `win-x64`.

## 4. Protocol

- **Command:** `stingray.exe -m <model> -f <prompt file> -n 128 --temp 0 --repeat-penalty 1.0 -t 8 -g 0 --single-turn --no-display-prompt`. The prompt is about 140 tokens, fixed in the job temp folder. Greedy and `--repeat-penalty 1.0` so that output can be compared (rule 6).
- **Conditions:**
  - **A, control:** no ballast. At least 35 GB available.
  - **B, pressure:** a ballast process that holds memory until about 8 GB is available before the run starts. The available figure is read from `Win32_OperatingSystem.FreePhysicalMemory` before each run and recorded.
- **Order:** A, B, A, B, A, B. Three runs of each (rule 2). The first A run may include cold file reads; it is reported separately, not dropped.
- **Isolation:** one run at a time. No other heavy process (rule 1). MSBuild and the test host are shut down before the runs.
- **Recorded per run:**
  - decode tokens per second, as printed by the CLI (confirm the exact line on the first run);
  - time to first token and total wall time;
  - `\Process(stingray)\Page Faults/sec` (mean and total over the run);
  - `\Memory\Pages Input/sec` (system-wide hard-fault reads, mean and total);
  - the minimum `\Memory\Available MBytes` during the run;
  - the SHA-256 of the generated text.
- **Dispatch check (rule 5):** the log must show that the CPU MoE path ran (no GPU layers, `-g 0`), and the MmapPrefault decision (ran or skipped) must be recorded for each condition.

## 5. Decision rules (fixed in advance)

Let A and B be the medians of decode tokens per second over the three runs of each condition.

- **Output:** all six SHA-256 hashes must match. If they differ, the run is invalid for the comparison, and the cause is investigated before any conclusion.
- **H0, residency not needed:** B ≥ 0.90 × A. The OS page cache keeps up at this pressure. Record R2 as closed for this model and pressure, with the numbers.
- **H1, residency worth a prototype:** B ≤ 0.80 × A **and** the pressure-run hard-fault reads are the dominant cost (the wall-time gap is covered by `Pages Input/sec` × time). Then R2 moves to a prototype, which is a separate plan.
- **Between 0.80 and 0.90:** record the result. Repeat once with a second MoE checkpoint (`gpt-oss-20b-MXFP4.gguf` on F:) before any decision.
- Single-digit-percent differences are noise (rule 7). The ratio is compared only within this session.

## 6. What this does not test

- A residency policy. Only the baseline pressure cost is measured. A cache that keeps the same bytes resident as the OS would only win if its eviction policy is better than the OS's. That needs a prototype and is out of scope here.
- GPU paths (`ExpertSlotManager`, `CudaExpertSlotManager`). Rule 13: the iGPU is not evidence about a discrete GPU.
- Other models, other pressure levels, other thread counts.

## 7. Results

Filled in after the runs, with dated rows in `docs/4-performance/PerformanceLeague-LOG.md` (rule 8).

### 7.1 Attempt 1: invalid for condition B (2026-11-09)

Six runs were made in the order A, B, A, B, A, B. **Condition B did not create memory pressure, so the B runs cannot answer the question.** The attempt is kept in full in the job temp folder (`moe-runs-attempt1-ballast-invalid`) and is not used for the decision. Three defects:

1. **The ballast was compressed away.** The ballast filled memory with one constant byte. Windows memory compression stores such pages in almost no RAM. A probe of 20 GiB showed private bytes of 20.16 GB but working set of 0.08 GB, and free memory did not move. The ballast now fills each 4 KiB page from a random block, which is incompressible. With the new ballast, 20 GiB drops free memory from 40 GB to 20 GB, and release restores 40 GB.
2. **The sampler was unreliable.** `Get-Counter` returned `NA` for most of each run, so the paging figures are not usable. The sampler is now `typeperf`, one sample a second.
3. **The output hash was wrong.** The hash included the `Model loaded in …` line, which carries a timing value, so identical text produced different hashes. The hash now excludes that line and the timing line.

Control (A) decode rates from attempt 1, which ran with no ballast: 15.3, 15.0 and 13.9 tokens/s (128 tokens, 8 threads). These are controls only; the attempt is not evidence about residency.

### 7.2 Attempt 2: the valid run (2026-11-09)

The same protocol, with the three defects of attempt 1 fixed. Full table: `PerformanceLeague.md`, section "MoE expert residency under memory pressure, CPU (2026-11-09)".

| Condition | Decode t/s: median (runs) | Prefill t/s | Wall s | Hard-fault reads per run | Free RAM during run |
|---|---|---:|---:|---|---|
| A, control | 14.1 (15.3, 14.1, 13.4) | 56.5 | 24.9 | 0.0 (warm), 4.6, 5.2 GiB | 40-48 GB |
| B, pressure | 5.6 (5.6, 4.8, 7.1) | 6.4 | 54.8 | 17.3, 20.2, 18.5 GiB | 8-9 GB |

- Decode ratio **0.40**; prefill ratio **0.11**. The generated text is identical across all six runs.
- **Verdict (fixed rules).** H0 rejected: 0.40 is far below 0.90. H1 needs the hard-fault reads to be the dominant cost. The read volume is consistent with that, but the drive's random-fault throughput was not measured, so dominance is not shown. **No residency prototype is justified by this measurement.**
- **Open question, not measured:** whether a process-managed expert residency policy would read fewer bytes than the OS's LRU under the same budget. That needs routing traces (hit rate of a pinned set against LRU at about 8 GB). It is the next measurement if the owner wants the question pursued; it is not started.
- Model-specific: Qwen3-Coder-30B-A3B on one SATA SSD. The review's point (OS page cache is adequate) holds only while the working set fits in RAM, which it does not here.
