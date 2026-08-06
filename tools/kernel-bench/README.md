# Quantized dot kernel bench — llama.cpp vs OpenTail.Stingray

## Current status

The default C++ and C# paths are Q6_K x Q8_K. Q6_K uses Q8_K activations on both sides, making it
the required apples-to-apples checksum and timing baseline. Q4_K is deliberately not in this
initial comparison: OpenTail uses Q8_KS activations for Q4_K while llama.cpp uses Q8_K, so their
algorithms and numerics differ.

Build the C# side in Release and only with the required JIT setting:

```powershell
$env:DOTNET_TC_QuickJitForLoops = '0'
dotnet run --project ../kernel-bench-cs/OpenTail.Stingray.KernelBench.csproj -c Release -- 8192 512 8
```

It refuses to run if that setting is absent, prints the Q6_K checksum before timing, then reports
best, mean, and population standard deviation. Compare its checksum to the C++ Q6_K path before
comparing its timings.

## The question

Is OpenTail.Stingray's ~4.6x CPU prefill gap **codegen** (RyuJIT cannot match clang on AVX2 intrinsics)
or **structural** (threading, memory layout, dispatch)? Those imply completely different roadmaps.
This harness isolates one kernel — `ggml_vec_dot_q6_K_q8_K` vs `SimdKernels.DotQ6K_Q8K` — on
identical data.

## Why an isolated benchmark is legitimate *here*

`docs/perf-loop-progress.md` records three cases where an isolated microbenchmark disagreed with
end-to-end (iteration 24: a reproduced 2.4-2.6x isolated win that was an 11.9% e2e **loss**;
`ACT_SOA_CPA`: +10-15% kernel, e2e-neutral). So isolation is untrustworthy for *"will this make the
app faster?"*.

That is not this question. We are comparing two implementations of the **same algorithm on the same
data**, so whatever confound distorts one side distorts the other. **The ratio transfers; the
absolute magnitude does not.** Do not quote these milliseconds as application performance.

## Build

Requires CMake 3.20+ and a C/C++ toolchain. `cmake` was NOT on PATH when this was written — install
it or use the VS Developer Prompt.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -- /m:1
./build/kernel-bench            # defaults: k=8192, rows=512, reps=8
./build/kernel-bench 4096 1024 12
```

`LLAMA_CPP_DIR` defaults to the checkout at `extensions/LLamaSharp/llama.cpp`. Override with
`-DLLAMA_CPP_DIR=/path/to/llama.cpp`.

On this Windows machine, use CMake's full path until a new terminal picks up the installer PATH:
`& 'C:\Program Files\CMake\bin\cmake.exe'`. The serial build avoids intermittent MSVC generated-
object locks observed while compiling ggml.

## The trap this guards against

Modern llama.cpp compiles `ggml-cpu` for several ISA levels and dispatches at runtime — the
prebuilt `tools/llama.cpp/` folder shows this (`ggml-cpu-haswell.dll`, `ggml-cpu-alderlake.dll`, ...).
**If the harness links the generic variant it benchmarks the scalar reference**, making llama.cpp
look slow and inverting the conclusion.

Two defences:
1. `GGML_CPU_ALL_VARIANTS=OFF` + `GGML_NATIVE=ON` pins one native build.
2. `main.cpp` calls both `ggml_vec_dot_q6_K_q8_K` and `..._generic`, and **exits non-zero** if the
   vectorised path is under 1.15x the scalar one.

## Procedure — do not skip step 3

1. Run the C++ harness. Note `checksum arch`.
2. Run the C# harness with the same `k` and `rows`.
3. **Confirm the checksums agree before comparing any timing.** Three times in OpenTail's perf log
   an ablation produced plausible numbers while computing the wrong thing; comparing outputs first
   is the only reliable detector.
4. Only then compare `best_ms`. Report n and sd, not a single number.

## C# side

Set `DOTNET_TC_QuickJitForLoops=0` (tiered JIT otherwise invalidates the numbers — perf-loop
iterations 10-11), warm up at least 3 iterations, and allocate the block buffers with
`NativeMemory.AlignedAlloc`, not a GC array — llama.cpp assumes aligned blocks.

Feed byte-identical data. The C++ side uses:

```c
float synth(int i) { return sin(i * 0.017) * 2.0 + cos(i * 0.0031); }
```

with weight row `r` built from `synth(i + r * 7919)`. Reproduce that exactly in C# using `double`
intermediates — "same seed, different RNG" is how two harnesses end up measuring different work.

## Reading the result

- **Ratio near 1.0** — the kernels are fine; the gap is structural. That is the most valuable
  outcome this project could get, because structural gaps are fixable.
- **C# 2-3x slower** — codegen ceiling. Design around it (wider tiles, fewer live vectors — see
  perf-loop iteration 57 on RyuJIT register spilling) rather than chasing the kernel.
