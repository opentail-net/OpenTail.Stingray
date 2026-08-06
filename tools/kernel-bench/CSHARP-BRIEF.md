# Brief: build the C# counterpart to the kernel bench

Hand this to whoever writes the C# side. The C++ harness already exists in this directory
(`main.cpp`, `CMakeLists.txt`, `README.md`) and links llama.cpp's real `ggml`.

---

## 1. The question being answered

Does RyuJIT emit competitive code for quantized dot kernels, or not? OpenTail.Stingray's CPU prefill is
~4.6x behind llama.cpp. That gap is either **codegen** (a ceiling to design around) or
**structural** (threading, layout, dispatch — fixable). Nobody has separated the two, and the
answer changes the roadmap.

**Scope discipline:** this is an isolated microbenchmark. `docs/perf-loop-progress.md` records
three cases where isolated numbers disagreed with end-to-end (iteration 24: a reproduced 2.4-2.6x
isolated win that was an 11.9% e2e **loss**). So it is *not* evidence that any change will make the
app faster. It is only valid because both sides run the same algorithm on the same data under the
same conditions — **the ratio transfers, the magnitude does not.** Do not quote the milliseconds as
application performance.

---

## 2. READ THIS FIRST — the formats do not match for Q4_K

llama.cpp's `ggml_vec_dot_q4_K_q8_K` takes **Q8_K** activations: one float scale per 256-element
super-block.

OpenTail's Q4_K path (`DotQ4K_Q8KS`) takes **Q8_KS**: *eight* scales per super-block. See
`TryResolveQ8Dispatch` in `SimdKernels.cs` and perf-loop iteration 55.

These are different algorithms with different numerics. A naive comparison will produce mismatched
checksums and waste a day.

**Therefore do the work in this order:**

### Step A — Q6_K first (apples to apples)

Both sides use Q8_K activations here:

| | llama.cpp | OpenTail |
|---|---|---|
| dot | `ggml_vec_dot_q6_K_q8_K` | `SimdKernels.DotQ6K_Q8K` |
| activation quantizer | `quantize_row_q8_K` | `SimdKernels.QuantizeRowToQ8K` |

Same activation format, same weight format, same super-block size. Checksums should agree to
within float rounding. **This is the clean measurement — get it working before touching Q4_K.**

### Step B — Q4_K second, with the caveat stated

Compare `ggml_vec_dot_q4_K_q8_K` against `DotQ4K_Q8KS`. Checksums will **not** match closely,
because the activation quantization differs. That is expected, not a bug. Report it as a
timing comparison with an explicit note that the activation formats differ, or skip it — the Q6_K
number already answers the codegen question.

Do **not** "fix" the mismatch by changing OpenTail's activation format to make the numbers line up.
Q8_KS is a deliberate accuracy choice.

---

## 3. What to build

A console project (not a unit test — you need a real process with JIT settings) at
`tools/kernel-bench-cs/`, referencing `OpenTail.Stingray.Cpu`.

`SimdKernels` members are `internal`; add `<InternalsVisibleTo Include="OpenTail.Stingray.KernelBench" />`
to `OpenTail.Stingray.Cpu.csproj` rather than making anything public.

### Requirements

1. **Byte-identical input to the C++ side.** The C++ harness uses:
   ```c
   float synth(int i) { return sin(i * 0.017) * 2.0 + cos(i * 0.0031); }
   ```
   with weight row `r` built from `synth(i + r * 7919)`. Reproduce exactly, using `double`
   intermediates then casting to `float`. Do not substitute an RNG.

2. **Same shapes.** Default `k = 8192`, `rows = 512`, `reps = 8`, accept them as argv like the C++
   harness does.

3. **Aligned native buffers.** `NativeMemory.AlignedAlloc`, not GC arrays — llama.cpp assumes
   aligned blocks and OpenTail's kernels take raw pointers. Free in a `finally`.

4. **Print the checksum before any timing.** Sum the per-row dot results and print with the same
   `%.6f` precision the C++ side uses.

5. **Timing methodology, matching the C++ harness:** >=3 warmup iterations, then `reps` timed
   iterations; report **best, mean and sd**. Not a single number.

6. **`DOTNET_TC_QuickJitForLoops=0` must be set** or the numbers are invalid (perf-loop iterations
   10-11). Assert it at startup and refuse to run if unset — do not merely document it:
   ```csharp
   if (Environment.GetEnvironmentVariable("DOTNET_TC_QuickJitForLoops") != "0")
   {
       Console.Error.WriteLine("Refusing to run: set DOTNET_TC_QuickJitForLoops=0 (tiered JIT invalidates these numbers).");
       return 1;
   }
   ```

7. **Build and run Release.** A Debug number is meaningless here.

---

## 4. Procedure — step 3 is not optional

1. Build and run the C++ harness. Note `checksum arch`.
2. Run the C# harness with the same `k` and `rows`.
3. **Confirm the checksums agree (Q6_K) before comparing any timing.** Three separate times in
   OpenTail's perf log an ablation produced plausible numbers while computing the wrong thing, and
   comparing outputs first was the only reliable detector. If they disagree, you are measuring
   different work and the timings are meaningless.
4. Only then compare `best_ms`, and report n and sd.

The C++ harness **exits non-zero** if its vectorised path is under 1.15x its own scalar reference —
that guards against linking the generic `ggml-cpu` variant and benchmarking the scalar fallback,
which would make llama.cpp look slow and invert the conclusion. If it exits 3, fix the build before
proceeding; do not compare against that number.

---

## 5. How to read the result

- **Ratio near 1.0** — RyuJIT is competitive; the ~4.6x gap is structural (threading, memory
  layout, dispatch). This is the most valuable outcome available, because structural gaps are
  fixable and this would redirect the entire CPU roadmap.
- **C# 2-3x slower** — codegen ceiling. Design around it rather than chasing the kernel: fewer live
  `Vector256` values, wider tiles. See perf-loop iteration 57, where 8 extra vector accumulators
  cost **43%** to RyuJIT register spilling — that is the shape of the constraint.
- **C# faster** — treat as a bug in the harness until proven otherwise. Check the C++ build is
  Release with `GGML_NATIVE=ON`, and re-check the checksums.

Whatever the outcome, record it in `docs/perf-loop-progress.md` as a numbered iteration with n, sd,
and the exact shapes. A negative result with its reason is as valuable as a win — roughly half that
log is documented negatives, and that is deliberate.
