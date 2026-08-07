# CPU prefill/decode vs llama.cpp — measured baseline

**Measured:** 2026-08-07. **Status:** one machine, one model, one quantisation, CPU only. Treat the
numbers as a reproducible data point, not a portable claim. The reproduction commands are below;
re-run them rather than citing these figures on different hardware.

## Environment

| | |
|---|---|
| CPU | 12 logical CPUs, **AVX2/FMA only** — no AVX-512, no AVX-VNNI |
| Model | `SmolLM2-1.7B-Instruct-Q4_K_M.gguf` (1.71 B, 1005 MiB) |
| llama.cpp | `b8585-cpu`, build `cad2d3884`, backend `ggml-cpu-haswell.dll` |
| Stingray | commit `72935f4`, Release, `-g 0` (CPU), default settings |
| Threads | 12 for both |

The CPU has no VNNI, so llama.cpp selected its Haswell backend and Stingray's AVX-VNNI path never
engaged. **Both engines are on AVX2 code paths**, which is what makes this a fair comparison — and
also what makes it silent about VNNI-capable hardware.

## Results

llama.cpp is `llama-bench` mean ± stddev over 3 repetitions. Stingray is best-of-3 from separate
CLI processes (see "Why the methodologies differ").

| Prompt | llama.cpp t/s | Stingray t/s | Ratio |
|---:|---:|---:|---|
| 512 | 184.4 ± 1.1 | 145.6 | llama.cpp **+27%** |
| 1024 | 171.2 ± 0.6 | 156.1 | llama.cpp **+10%** |
| 2048 | 152.7 ± 0.5 | 149.3 | llama.cpp **+2%** |
| ~3100 | 139.2 ± 1.6 | 145.6 | **Stingray +5%** |
| decode (128 tok) | 26.5 ± 0.1 | 26.4 | **parity** |

Token counts differ by ~2% between engines (Stingray tokenised 524 / 1018 / 2044 / 3070 against
llama-bench's exact 512 / 1024 / 2048 / 3100), because the Stingray side is driven by a real prompt
through the real tokeniser rather than a synthetic token count.

## Reading it

**The gap is a fixed cost, not kernel quality.** It closes monotonically with prompt length and
crosses over around ~2500 tokens. The signature is in the shapes: llama.cpp decreases
monotonically (184 → 171 → 153 → 139) as quadratic attention takes over, whereas Stingray *rises*
from 512 to 1024 (145.6 → 156.1) before turning over. A throughput curve that improves with more
work is the classic fingerprint of a constant per-run overhead being amortised — here, the .NET JIT
compiling the hot SIMD kernels during the first (and only) measured prefill of each process.

That matters for how to read the short-prompt column: a server or any long-lived process pays that
cost once, not per request. The single-shot CLI is the worst case for it.

## Why the methodologies differ, and who it favours

`llama-bench` runs warm-up iterations and then repeats the workload inside one process against an
AOT-compiled binary. The Stingray column comes from a fresh process per sample, so every sample
includes JIT. **The comparison is therefore structurally unfair to Stingray at short prompts**, and
approximately fair by ~3000 tokens where the fixed cost is a small share of total work. It is
recorded this way deliberately — it is the honest shape of the measurement rather than a tuned
harness — but a like-for-like in-process comparison would be the better next measurement.

Best-of-N is used for Stingray because the machine was in light use during measurement; contention
can only slow a run down, so the maximum is the least-contaminated estimator. Medians run about
2-3% below the best-of values and do not change any conclusion except the ~3100 row, which becomes
parity rather than a small Stingray lead.

## Reproduce

```bash
# llama.cpp side (sweep + decode in one invocation)
./tools/llama.cpp/llama-bench.exe -m models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf \
  -p 512,1024,2048,3100 -n 128 -t 12 -r 3

# Stingray side — prefill. Build Release first; --no-build keeps the sample clean.
dotnet run --project src/OpenTail.Stingray.Cli -c Release -- \
  -m models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf -p "<prompt>" -n 1 --temp 0 -g 0 --no-display-prompt
# reports: "Prefill: <N> tokens, <t/s>"

# Stingray side — decode.
dotnet run --project src/OpenTail.Stingray.Cli -c Release -- \
  -m models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf -p "Write a short story about the sea." \
  -n 128 --temp 0 -g 0 --no-display-prompt
# reports: "Decode: 128 tokens, <t/s>"
```

Interleave the arms rather than running each in a block, and discard a sweep taken while the
machine is doing anything else.

## What this does not measure

- **VNNI and AVX-512 paths.** This CPU has neither. Stingray's AVX-VNNI Q4_K dot products are still
  entirely unmeasured, and llama.cpp would likewise use different kernels there.
- **GPU backends, MoE, other architectures, other quantisations, batching, speculation.**
- **Long-context behaviour past ~3100 tokens**, where the curves may separate again.
- **Memory footprint**, which was not compared at all.

## Related

The `+4.0%` prefill improvement recorded in `CHANGELOG.md` for this commit (SIMD K-pack transpose
plus the KV-outer schedule) is a Stingray-versus-Stingray measurement at ~1550 tokens. It is a
different quantity from this document's llama.cpp comparison and the two should not be added
together. The 2×2 that produced it is documented on
`ForwardPass.ComputePrefillFlashAttention64KvOuterHead`.
