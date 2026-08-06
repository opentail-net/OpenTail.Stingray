
---

# Repack A/B run — 2026-08-02 13:20:17

Runner: `scripts/repack-ab.ps1`.  Interpretation: `docs/repack-gemm/README.md`.

| control | value |
|---|---|
| exe | `C:\git\opentail\extensions\OpenTail.Stingray\tools\llama.cpp\llama-cli.exe` |
| model | `SmolLM2-1.7B-Instruct-Q4_K_M.gguf` |
| ctx | 8192 |
| threads | 6 (physical cores) |
| runs per arm | 3, interleaved on/off |
| warmup | enabled (timed prefill must not pay first-touch page faults) |

## Phase 1 — gate: did the flag take effect?


---

# Repack A/B run - 2026-08-02 13:48:06

Runner: `scripts/repack-ab.ps1`.  Interpretation: `docs/repack-gemm/README.md`.

| control | value |
|---|---|
| exe | `llama-completion.exe` (llama-cli is interactive-only in b8585) |
| model | `SmolLM2-1.7B-Instruct-Q4_K_M.gguf` (1006.7 MiB) |
| ctx | 8192 |
| threads | 6 (physical cores) |
| runs per arm | 3, interleaved on/off |
| warmup | enabled (timed prefill must not pay first-touch page faults) |

## Phase 1 - gate: did the flag take effect?


---

# Repack A/B run - 2026-08-02 13:56:55

Runner: `scripts/repack-ab.ps1`.  Interpretation: `docs/repack-gemm/README.md`.

| control | value |
|---|---|
| exe | `llama-completion.exe` (llama-cli is interactive-only in b8585) |
| model | `SmolLM2-1.7B-Instruct-Q4_K_M.gguf` (1006.7 MiB) |
| ctx | 8192 |
| threads | 6 (physical cores) |
| runs per arm | 3, interleaved on/off |
| warmup | enabled (timed prefill must not pay first-touch page faults) |

## Phase 1 - gate: did the flag take effect?


---

# Repack A/B run - 2026-08-02 14:10:20

Runner: `scripts/repack-ab.ps1`.  Interpretation: `docs/repack-gemm/README.md`.

| control | value |
|---|---|
| exe | `llama-completion.exe` (llama-cli is interactive-only in b8585) |
| model | `SmolLM2-1.7B-Instruct-Q4_K_M.gguf` (1006.7 MiB) |
| ctx | 8192 |
| threads | 6 (physical cores) |
| runs per arm | 3, interleaved on/off |
| warmup | enabled (timed prefill must not pay first-touch page faults) |

## Phase 1 - gate: did the flag take effect?


---

# Repack A/B run - 2026-08-02 14:22:23

Runner: `scripts/repack-ab.ps1`.  Interpretation: `docs/repack-gemm/README.md`.

| control | value |
|---|---|
| exe | `llama-completion.exe` (llama-cli is interactive-only in b8585) |
| model | `SmolLM2-1.7B-Instruct-Q4_K_M.gguf` (1006.7 MiB) |
| ctx | 8192 |
| threads | 6 (physical cores) |
| runs per arm | 3, interleaved on/off |
| warmup | enabled (timed prefill must not pay first-touch page faults) |

## Phase 1 - gate: did the flag take effect?

- **repack on**: CPU_REPACK model = 729 MiB; Host model = 1005 MiB

Tensor type census (load-time):

```
  f32         49 tensors
  q4_K       144 tensors
  q6_K        25 tensors
```

- **repack off**: CPU_REPACK model = *absent*; Host model = 1005 MiB

**Gate passed.** Repacked tensors are 729 MiB of 1005 MiB total weights = **72.5%** of weight bytes. The remaining 27.5% (Q6_K / F32 - neither repacks on AVX2) takes the identical path in both arms, so the raw ratio below is a **lower bound**.

## Phase 2 - prefill A/B

Only `prompt eval time` is used: the repacked GEMM is selected only for M > 3
(`repack.cpp:4241`), so decode (M=1) runs GEMV and is not informative here.

| tokens | arm | run | prefill t/s |
|---:|---|---:|---:|
| 909 | on | 0 | 145.02 |
| 909 | off | 0 | 96.18 |
| 909 | on | 1 | 159.63 |
| 909 | off | 1 | 98.64 |
| 909 | on | 2 | 163.72 |
| 909 | off | 2 | 98.61 |
| 4709 | on | 0 | 115.45 |
| 4709 | off | 0 | 79.63 |
| 4709 | on | 1 | 124.41 |
| 4709 | off | 1 | 83.6 |
| 4709 | on | 2 | 126.37 |
| 4709 | off | 2 | 83.9 |

## Phase 3 - verdict

| tokens | repack ON (best) | repack OFF (best) | ratio | un-diluted est. |
|---:|---:|---:|---:|---:|
| 900 | 163.72 | 98.64 | **1.66x** | ~2.21x |
| 4700 | 126.37 | 83.90 | **1.51x** | ~1.86x |

Reading (README section 6): >=2x means the repacked 2D GEMM is the boss tower and the
phase-2 premise holds; <=1.2x relocates it to ordinary `ggml_vec_dot_q4_K_q8_K` codegen
plus threading, a far cheaper target than a 1450-line kernel port.

