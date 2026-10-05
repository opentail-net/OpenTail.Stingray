# End-to-end CPU comparison: OpenTail.Stingray vs TensorSharp vs llama.cpp (Gemma 4 E4B Q4_K_M, 2026-10-05)

Measured after the Half-conversion / IQ-kernel work in [2026-10-04-moe-decode-gap-investigation.md](2026-10-04-moe-decode-gap-investigation.md) (commit `aac583fe`). Machine: Ryzen 7 5700G (8 cores / 16 threads, AVX2), 64 GB DDR4, no discrete GPU, Windows 11. Everything below is CPU, 8 threads, greedy.

## What was compared
| | build | backend | notes |
|---|---|---|---|
| **ours** | `stingray.exe` Release (JIT, not NativeAOT) built from this tree | managed C# AVX2 kernels | `-g 0 -t 8 --temp 0 --repeat-penalty 1.0` (see "trap" below) |
| **ts-cpu** | `examples/TensorSharp` CLI, `--backend cpu` | pure C# (the like-for-like managed counterpart) | `TS_CPU_THREADS=8` |
| **ts-ggml_cpu** | same CLI, `--backend ggml_cpu` | native ggml CPU kernels (GgmlOps.dll built from source here) | `TS_GGML_CPU_THREADS=8` |
| **llama.cpp** | vendored `tools/llama.cpp/llama-bench.exe` (build 10306) | native, haswell backend | `-t 8 -ngl 0 -r 1`; synthetic tokens, decode starts from an empty context, so not the same prompt as the others |

Checkpoint: `models/_models/gemma-4-E4B-it-Q4_K_M.gguf` (4.6 GB, dense, Q4_K/Q6_K). TensorSharp has no plain Qwen3/Llama-style families; of its supported ones that we also have on disk, Gemma 4 E4B is the smallest one verified to run on both engines (the others are Qwen 3.6/3.8-27B at 13 GB and gpt-oss-20b at 11 GB; a 0.5B Hunyuan GGUF is on disk too and was not tried). All three engines tokenise the templated prompt to the same 142 tokens.

Prompt (`tools/engine-compare/prompt.txt`, ~100 words, deliberately small): a 4-sentence note plus "write a long, detailed essay of at least 400 words", so the model does not stop before 128 tokens. A first prompt with a short answer stopped at about 28 tokens on every engine; those runs are kept as `...-short-answer.csv` (same ordering of engines).

## Method
- One fresh process per run, as a user would launch it (cold JIT, model load included in wall time). Prefill and decode times come from each engine's own report; wall time and peak working set (sampled every 100 ms) from the harness.
- 3 rounds, engine order rotated each round, generation lengths 32 and 128 tokens per engine per round. "Marginal decode" = 96 tokens / (decode time at 128 - decode time at 32), which cancels JIT/warm-up.
- **Memory:** before every run the harness asserts no `stingray`, `TensorSharp`, `llama-*` or test-host process exists, and logs free RAM before/after: 47.5-47.8 GB free (of 63.3) throughout, i.e. nothing else resident. Idle MSBuild/VBCSCompiler servers from the builds were stopped first. Model files are mmapped by all engines and sit in the OS page cache after the first run of the session (all engines had been run once before timing).
- Medians of the 3 rounds; spread was tight (prefill within ~7%, decode@128 within ~3% per engine).

## Results (median of 3, 142-token prompt)
| engine | prefill ms | prefill t/s | decode t/s (128 tok) | marginal decode t/s | wall s (32 tok) | startup s | peak RSS GB |
|---|---|---|---|---|---|---|---|
| **ours** | 2395 | **59.3** | 11.8 | 12.0 | **7.1** | 1.9 | 5.2 |
| ts-cpu (pure C#) | 3199 | 44.4 | 8.7 | 8.5 | 10.6 | 3.9 | 6.3 |
| ts-ggml_cpu | 2853 | 49.8 | 12.5 | 12.4 | 47.6 | 42.2 | 8.9 |
| llama.cpp (llama-bench) | 1727 | **82.2** | 12.3 | 12.2 | 9.3 | 5.0 | 5.1 |

- **Against the like-for-like managed engine (ts-cpu): ours is 1.34x faster at prefill, 1.36x at decode, 1.5x end to end.**
- Against TensorSharp's native ggml CPU path: ours prefills 1.20x faster and decodes 0.94x as fast (6% slower); end to end it is 6.7x quicker only because `ggml_cpu` spends ~33 s in a "Kernel warmup" at every process start (its log: "Kernel warmup completed in 33516.6 ms"), plus a 9 GB resident set.
- Against llama.cpp: decode is within 4% (11.8 vs 12.3), prefill is 1.39x slower (59 vs 82 t/s). Our prefill figure includes tier-0 JIT time in a single cold 142-token prefill (it measured 44-60 t/s across separate sessions); a steady-state prefill number needs a warm-up pass and was not taken here.
- Decode is probably memory-bandwidth-bound (a few GB of weights per token at 12 t/s is on the order of 50 GB/s, near this DDR4 system's peak; not measured with counters here), which would explain why three different implementations land within 6% of each other.

## Output agreement
Greedy text is deterministic per engine across rounds. With `--repeat-penalty 1.0` ours agrees with ts-ggml_cpu for 219 characters, ts-cpu with ts-ggml_cpu for 150 characters, and all of them eventually diverge inside 128 tokens (near-tie flips from different float summation orders); no engine produced degenerate text.

**Trap found: `stingray --temp 0` still applies `--repeat-penalty 1.1` by default** (RunCommand.cs, "default: 1.1"). Greedy output then differs from ggml-family engines after ~85 characters here (at step 23 the raw top-1 token was `,` at 27.38 but ` is` at 24.08 was emitted). The rows above use `--repeat-penalty 1.0`; the same runs with the default penalty are in `results-gemma4-e4b-q4km-2026-10-05.csv` and are within noise of them (decode@128 10.76-10.94 s either way). Any parity comparison against another engine has to pass `--repeat-penalty 1.0`.

## Caveats
One model, one machine, one prompt, 8 threads (ours defaults to the logical processor count, TensorSharp's managed pool to 8, ggml to its own default, so threads were pinned for all). JIT build of ours; NativeAOT not measured. Batch size 1 only. llama-bench is a different harness (synthetic tokens, empty-context decode). TensorSharp was built with Vulkan disabled (`TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=OFF`; its Vulkan shader-gen sub-build failed on a long path) and no CUDA, which is irrelevant to CPU numbers.

## Reproduce
`tools/engine-compare/bench.ps1` (parameters `-Rounds`, `-Engines`, `-ExtraOursArgs`, `-Tag`); build ours with `dotnet build src/OpenTail.Stingray.Cli -c Release` and TensorSharp with `TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=OFF dotnet build TensorSharp.Cli -c Release -p:TensorSharpSkipMlxNative=true` in `examples/TensorSharp/TensorSharp`. The script has the output directory and engine paths of this session hard-coded at the top.
