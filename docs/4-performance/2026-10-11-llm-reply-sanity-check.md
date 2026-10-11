# LLM reply sanity check, 2026-10-11

A safety check after many engine changes: does every installed LLM checkpoint still produce a **coherent reply**? No timing was taken. Script: [`scripts/sanity-llm-replies.ps1`](../../scripts/sanity-llm-replies.ps1) (CPU, `-g 0`, greedy, prompt "What is the capital of France? Answer in one short sentence.", 96 tokens, smallest checkpoint first, one process at a time, files larger than free RAM minus 6 GB skipped, the CLI's memory preflight left on). Coherent means a reply containing "Paris". Anything else is listed for a human to judge. Build: HEAD at the time (`88ea058d`+). The raw per-file results are in the session scratchpad, not the repo.

## Result

Every checkpoint under every drive's `_models` folder (de-duplicated): **91 passed**, 74 are not text-generation models (embeddings, ASR/TTS, diffusion, vision projectors; not run), **2 skipped for memory** (GLM-4.5-Air 42 GB and Llama-4-Scout 46 GB, over the guard; documented in the League's oversized-checkpoint note), **6 produced no reply**, **8 replies need a human judgement**.

## Findings

| File | Verdict after a closer look |
|---|---|
| `gemma-2-2b-it-Q4_K_M` (arch `gemma2`) | **Real defect, not recent.** Output is garbage on CPU and on Vulkan (whitespace and punctuation). Same prompt in llama.cpp (b8585): "The capital of France is: a) Paris". It was also garbage in builds from 2026-10-03 and 2026-09-13, so it is not a regression from the recent changes; there is no `gemma2` golden or STATUS row, so it was probably never correct. `docs/models/gemma.md` listed Gemma-2 as a recommended checkpoint; that row now carries a warning. Needs an owner: diff the Gemma-2 path against llama.cpp (post-norms, soft-capping, sliding-window layers, query scaling) per CLAUDE.md rule 8. |
| `tiiuae-falcon-7b-instruct-Q4_K_M` | One `PlanNotExecutableException: Planned Vulkan backend failed to initialize` during the sweep on a `-g 0` run. **Not reproduced** on a re-run (it loaded and ran). Treat as intermittent; the CPU run should not plan a Vulkan backend at all, so worth a look if it recurs. |
| `granite-vision-3.2-2b-Q3_K_S` | Degenerate text ("bigquery bigquery ..."). llama.cpp is also degenerate on the same file and prompt ("France France France ..."), so this looks like Q3_K_S quality on a vision model, not an engine fault. Inconclusive; do not read it as a regression. |
| `pythia-160m`, `gpt2` x2, `OLMo-1B`, `starcoder2-3b` | Base (non-chat) models: they continue or echo the prompt and never answer the question. Expected; the "Paris" test is not meaningful for them. |
| `orpheus-3b` | A TTS model ("I'm sorry, I can't remember"); not a chat LLM. |
| `stable-diffusion-xl`, `z_image_turbo` (x3), `GLM-5.3` | Not text models (diffusion) or deliberately not admitted / far over memory. Reported as "no reply" by the harness only because it treats every `.gguf` as a candidate. |

## What this does and does not show

- It shows the **91 admitted-and-runnable chat LLMs still answer coherently** after the recent engine changes, on the CPU path. It does not cover Vulkan or CUDA, quality (only "mentions Paris"), long prompts, or the four oversized checkpoints.
- A passing line means the model produced the expected word once, greedy; it is not a golden or a benchmark.
- The harness needs a smarter notion of "chat model" (it currently runs every GGUF whose architecture is not on its non-LLM list). Follow-up: use the same admitted-architecture list as `stingray scout`, and run base models with a completion-style prompt.
