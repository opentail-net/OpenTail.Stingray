# scripts/

Helper scripts for building, verifying, benchmarking and investigating OpenTail.Stingray. Most are
PowerShell 7 for Windows; the vision/diffusion reference oracles are Python.

The files sit flat in this directory. This page groups them by purpose so you can tell the
supported workflows from the one-off investigations. Nothing has moved yet: CI, docs, tests and
other scripts refer to these paths, so any later move has to update those references in the same
change.

**Status** in the tables below:

- **CI**: a workflow runs it. Breaking it breaks a release.
- **Maintained**: a supported workflow that docs, tests or other scripts refer to. Keep it working.
- **Historical**: written for one issue or investigation. It records how a published measurement
  was taken and is kept for reproducibility. It isn't kept in step with the code, so model paths,
  env vars and flags may have drifted. Read it before you run it.

Most benchmark and investigation scripts expect real checkpoints under `models/` (git-ignored) and
some expect a local llama.cpp build under `tools/` (see CLAUDE.md rule 8). They don't download
anything unless their name says so.

## CI and release

| Script | Status | Purpose | Used by |
|---|---|---|---|
| `check-test-model-coverage.ps1` | CI | Reports which GGUF models the test suite gates on and which are missing locally. | `release.yml`, release checklist |
| `verify-nuget-package.ps1` | CI | Packs and verifies the NuGet package for a given version. | `release.yml`, release checklist |
| `verify-package-consumers.ps1` | Maintained | Proves the documented NuGet usage compiles against the packed package, not against the source projects. | docs plans |
| `check-aot-telemetry.ps1` | Maintained | Publishes the NativeAOT CLI and checks that engine metrics and spans reach an in-process listener. No model needed. | observability guide |

## Code generation and repository maintenance

| Script | Status | Purpose | Used by |
|---|---|---|---|
| `gen-spirv.ps1` | Maintained | Regenerates `src/OpenTail.Stingray.Vulkan/Shaders.Precompiled.g.cs` from the GLSL shaders. Needs the Vulkan SDK. Tests fail on table drift if you skip it. | CLAUDE.md rule 5 |
| `gen-cli-option-inventory.ps1` | Maintained | Regenerates the option tables in `docs/reference/cli-option-inventory.md` from source. | DEVELOPMENT.md |
| `gen-web-scout-data.ps1` | Maintained | Builds the website's feasibility-table data from real `stingray scout` runs (headers only, no weights loaded). | website |
| `Export-UncommittedChanges.ps1` | Maintained | Bundles every uncommitted file into one log for an external code review. | — |

## Environment setup and models

| Script | Status | Purpose | Used by |
|---|---|---|---|
| `download-model.ps1` | Maintained | Downloads GGUF models for development. `stingray pull` is the CLI alternative. | bench scripts, ImageBench, coverage check |
| `list-checkpoints.ps1` | Maintained | Inventories every checkpoint under the known model roots as CSV. | league scripts, sanity check |
| `setup-llamacpp.ps1` | Maintained | Downloads prebuilt llama.cpp binaries for reference runs. | `bench-vs-llamacpp.ps1`, `generate-reference-logits.ps1` |
| `setup-openblas.ps1` | Maintained | Downloads OpenBLAS for Windows x64. | OpenBLAS troubleshooting guide, `BlasInterop.cs` |

## Verification and admission

| Script | Status | Purpose | Used by |
|---|---|---|---|
| `scout-pretest.ps1` | Maintained | Opt-in, non-destructive wrapper around `stingray scout`: static report, memory gate, and with `-Run` a bounded smoke run plus an optional golden check. | CLAUDE.md rule 15, scout plan |
| `scout-leave-one-out.ps1` | Maintained | Leave-one-architecture-out check of the scout similarity ranking. | scout plan |
| `sanity-llm-replies.ps1` | Maintained | Checks that every installed LLM checkpoint still gives a coherent reply (no timing). Use it after broad engine changes. | LLM reply sanity-check doc |
| `sweep-tests.ps1` | Maintained | Resumable real-weight test sweep, one test class per process. | test-sweep docs |
| `generate-reference-logits.ps1` | Historical | Produces llama.cpp reference logits for forward-pass validation. | — |
| `xcheck-llamacpp.ps1` | Historical | Side-by-side greedy output against llama.cpp for one prompt. | design doc |
| `smoke-gemma4-server.ps1` | Historical | Starts Server.Host with Gemma 4 E4B Q8 and calls the endpoints. | — |
| `test-snapshot-reuse.ps1` | Historical | End-to-end check of chat-completion snapshot reuse on MTP runs (issue #106). | — |
| `kvarn-gate/` | Historical | KVarN accuracy gate for issue #180: wikitext-2 fetch and a 20-problem math micro-eval. | — |

## Reference oracles (Python)

Independent reference implementations that produce goldens for the parity tests. They are frozen:
CLAUDE.md says not to add new `scripts/*_ref.py`. New vision work uses the vendored llama.cpp
binaries as the oracle instead.

| Script | Status | Oracle for | Used by |
|---|---|---|---|
| `gemma4uv_ref.py` | Maintained | Gemma 4 unified vision projector | Vision parity tests |
| `llava_ref.py` | Maintained | LLaVA-1.5 CLIP encoder + MLP projector | Vision parity tests |
| `pixtral_ref.py` | Maintained | Pixtral ViT + 2D RoPE projector | Vision parity tests |
| `qwen25vl_ref.py` | Maintained | Qwen2.5-VL ViT + M-RoPE merger | Vision parity tests |
| `glm4v_ref.py` | Maintained | GLM-4.6V ViT + patch merger | Vision parity tests |
| `hunyuanvl_ref.py` | Maintained | HunyuanVL / HunyuanOCR projector | Vision parity tests |
| `exaone4_ref.py` | Maintained | EXAONE 4.5 Vision projector | Vision parity tests |
| `mimovl_ref.py` | Maintained | MiMo-VL mmproj checkpoints | Vision parity tests |
| `flux1_vec_embed_ref.py` | Maintained | FLUX.1 conditioning vector | Diffusion golden tests |
| `sd3_timestep_embed_ref.py` | Maintained | SD3/3.5 timestep + pooled embedding | Diffusion parity tests |
| `compare_tokens.py` | Historical | Early llama-cpp-python token comparison | — (also listed in `.gitignore`) |
| `extract_reference.py` | Historical | Early llama-cpp-python logits extraction | — (also listed in `.gitignore`) |

## Benchmarks: repeatable workflows

These produce or refresh the numbers in `PerformanceLeague.md` and `docs/STATUS.md`. Prefer them
over the historical A/B scripts below.

| Script | Status | Purpose |
|---|---|---|
| `bench-textgen.ps1` | Maintained | Shared text-generation bench runner. Most other bench scripts call it. |
| `bench-league-recheck.ps1` | Maintained | Re-measures our side of the PerformanceLeague LLM rows, one model at a time, into one results file. |
| `league-compare.ps1` | Maintained | Compares a recheck results file with the existing league rows. |
| `league-update-section.ps1` | Maintained | Regenerates the "all checkpoints" section of PerformanceLeague.md from a recheck results file. |
| `league-unsupported-doc.ps1` | Maintained | Regenerates `docs/2-coverage/2026-10-11-checkpoints-not-supported-yet.md` from a recheck results file. |
| `bench-allrows-1k.ps1` | Maintained | Re-benchmarks every on-disk league row at a uniform working context. |
| `bench-allrows-decode.ps1` | Maintained | Companion to the above for the decode t/s column. |
| `bench-allrows-mtp.ps1` | Maintained | Re-measures the MTP rows. |
| `bench-vs-llamacpp.ps1` | Maintained | CPU prefill/decode comparison: Stingray vs llama.cpp vs the pre-rebrand SharpInference snapshot. |
| `bench-cpp.ps1` | Maintained | C++ reference benchmarks (audio.cpp, whisper.cpp) against our figures. |
| `bench-audio.ps1` | Maintained | Audio engine benchmarks. |
| `bench-image.ps1` | Maintained | Image generation benchmark. |
| `run-cpu-baseline.ps1` | Maintained | Reproduces the CPU performance-baseline method: best of three interleaved rounds, greedy, `-g 0`. |
| `Get-StingrayRunStats.ps1` | Maintained | Reads the result file a CLI run writes when `STINGRAY_STATS_JSON` is set. |

## Benchmarks and profiling: historical investigations

Each one answered a single question, usually an A/B for one issue. The issue number in the name
leads to the write-up. Reuse the approach, but check the paths and env vars first.

| Script | Question it answered |
|---|---|
| `bench-129-ab.ps1` | #129: GPU-SLRU MoE prefill, branch vs master |
| `bench-210-ab.ps1` | #210: grouped-by-expert MoE verify batching for MTP on CUDA hybrid |
| `bench-27b-mtp.ps1` | #28: Qwen3.6-27B MTP on vs off |
| `bench-35b-a3b-mtp.ps1` | #44: Qwen3.6-35B-A3B MTP |
| `bench-388-router.ps1` | #388: CPU-MoE prefill router moved to the GPU |
| `bench-390-ab.ps1` | #390: op-offload MoE prefill, pinned-buffer A/B |
| `bench-390-models.ps1` | #390: re-measure of the CUDA GDN-hybrid CPU-MoE rows |
| `bench-390-sweep.ps1` | #390: prefill crossover sweep for register-mode op-offload |
| `bench-all.ps1` | Early multi-model sweep (SmolLM2, Qwen3, MoE, Llama 4 Scout, Gemma 4); some paths are hard-coded to `E:models` |
| `bench-moe-rerun.ps1` | MoE re-run with `bench-all.ps1`'s model paths |
| `bench-carnice.ps1` | Carnice (Qwen3.6-35B-A3B-MTP) CUDA hybrid |
| `bench-q8k-validation.ps1` | #103: multi-prompt validation on Carnice |
| `bench-gemma4-12b-124.ps1` | #124: Gemma 4 12B QAT all-GPU prefill/decode |
| `bench-gemma4-12b-actsoa.ps1` | #124/#173: activation-SoA A/B |
| `bench-gemma4-12b-mmq.ps1` | #124/#173: prefill MMQ A/B |
| `bench-gemma4-12b-soa.ps1` | #124/#173: SoA A/B |
| `bench-gemma4-136.ps1` | #136: all-GPU Gemma 4 row after batched trunk |
| `bench-gemma4-flashtc-146.ps1` | #146: half2 vs tensor-core flash prefill |
| `bench-gemma4-mmq-soa-149.ps1` | #149: Q8_0 SoA weight layout |
| `bench-gemma4-rebaseline.ps1` | Gemma 4 all-GPU re-baseline on merged master |
| `bench-q4k-mmq-156c2.ps1` | #156 C2: Qwen3-8B Q4_K CUDA prefill paths |
| `bench-q4k-soa-156.ps1` | #156: Q4_K interleaved vs SoA decode |
| `bench-prefill.ps1` | Early prefill timing loop |
| `bench-prefill-cli.ps1` | CPU prefill A/B through the real CLI (perf loop iteration 33) |
| `bench-llama-server-prefix.ps1` | Session runtime plan milestone 0: llama-server prefix-cache comparison |
| `repack-ab.ps1` | Whether llama.cpp's repacked 2D Q4_K GEMM explains the remaining CPU gap |
| `ncu-actsoa-prof.ps1` | Nsight Compute: AoS vs SoA activations in the Q8_0 MMQ kernel (elevated) |
| `ncu-mmq-prof.ps1` | Nsight Compute: SoA int8 MMQ at the FFN prefill shape (elevated) |

## Adding a script

- Put a `.SYNOPSIS` (PowerShell) or module docstring (Python) at the top saying what it answers.
- Add a row here in the right section. A one-off investigation goes under **historical**, named
  after its issue.
- Don't write output into the repo root (CLAUDE.md rule 9). Use the temp directory or a git-ignored
  path.
