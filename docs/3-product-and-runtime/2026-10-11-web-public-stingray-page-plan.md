# Plan: a Stingray page on OpenTail.Web.Public

Status: proposed, 2026-10-11. Written before seeing the other AI's draft; to be reconciled with it. Site: `C:\Git\OpenTail\extensions\OpenTail.Web.Public` (static HTML, `styles.css`, `site.js`).

## What I found on the site first
- No page mentions Stingray. The engine appears only as **`llm-inference.html` ("OpenTail.LLM")**.
- That page does not match this repository. It shows an `OpenTail.LLM.*` namespace, `InferenceEngine.LoadAsync`, `GenerateStructuredAsync`, an `opentail-llm run --auto --explain` CLI, a "live trace" with an RTX 4090 at 42.8 tok/s, and a calculator with fixed "340 tok/s GPU / 110 tok/s CPU" estimates. None of those APIs, commands or numbers exist or were measured here (the real ones: package `OpenTail.Stingray`, `Model.Load`, `ChatSession`, CLI `stingray`, `JsonConstraint.FromSchemaJson`). The new page must not inherit them.
- Owner decision (2026-10-11): `llm-inference.html` is the early, now fully out-of-date edition of this engine page. It is **superseded by the Stingray page**: replace it (redirect or remove the page and every nav link to it), and do not carry any of its content over.
- Housekeeping: `index - Copy.html` and `llm-inference - Copy.html` are stray copies; Google Fonts are loaded from a third party (a privacy and "local-first" mismatch for a site selling local AI; self-host the font).

## Rules for the page (the same ones the repo README follows)
1. Every number carries its source and date: STATUS.md rows, PerformanceLeague.md, the measured NativeAOT section in `docs/guides/native-aot-deployment.md`. No illustrative or "typical" figures presented as measurements.
2. Every code snippet is one of the README snippets already compiled against the packed package (`scripts/verify-package-consumers.ps1`), copied verbatim. No invented API.
3. Claims are the README's wording, not stronger: "no Python, no llama.cpp, no separate server" (not "zero native", not "zero supply-chain"); the GPU driver and ONNX Runtime are named as native pieces; model licences are the model's own.
4. Hardware statements say what was measured: the dev machine is an iGPU Ryzen 5700G; discrete-GPU behaviour is not measured here and must not be implied (CLAUDE.md rule 13).
5. Terminal "demo" output is a real captured run (command, model, date), not a mock. Prefer a static block over a fake animated trace.
6. Nothing from a "ported, not verified" family (rule 14).

## Page outline (`stingray.html`)
1. **Hero.** The README's one-line description (chat, vision, speech, music, sound effects, images; in your .NET process). Install line `dotnet add package OpenTail.Stingray`; secondary link to GitHub.
2. **Why Stingray** (4 bullets, plain): runs in your process with no sidecar; readable C# you can step into; catalogue models pinned to a commit and SHA-256, verification tiers in STATUS.md; one API across modalities. (Edited from the Gemini suggestion: no "sovereignty/sprawl" language.)
3. **Five lines to a reply.** The verified chat snippet; then structured extraction (the new `Sample.Extract`, with its grounding-check point: constrained shape is not true values).
4. **What runs today.** A table generated from STATUS.md, with its tier symbols and a link to it. Include the gaps, not only the greens.
5. **Bring the models you already have.** GGUF anywhere, Ollama blobs, SafeTensors (narrow, CPU), ONNX voices, whisper.cpp; "used in place, never copied". Links the guide.
6. **Measured, not claimed.** (a) NativeAOT CLI: 62 ms vs 111 ms `--help`, 27 MiB exe, labelled as CLI start-up only; (b) a short League excerpt vs llama.cpp including the rows where we are slower; both with date, machine and the iGPU caveat.
7. **See and hear it.** The curated `docs/audio-samples` and `docs/diffusion-samples` files with their commands (these are in the repo as proof samples; copy a few MB at most).
8. **Honest limits.** Short list: SafeTensors profiles, CPU-only claims, which backends are verified on what hardware, Ollama manifest route not working, and the model-licence note.
9. **Where to go next.** NuGet, GitHub, docs/STATUS.md, samples, and the OpenTail MCP product (one line, only if the commercial positioning wants the cross-link).

## Build approach
- Reuse `styles.css`, nav and footer exactly; add only the styles the page needs. Static, no framework; works without JavaScript.
- Drop the calculator. If an estimator is wanted later, base it on `stingray scout --budget` output and label it an estimate.
- Add a short `SOURCES` comment block (or a section at the bottom) mapping each number to its repo document and date, so the next refresh is mechanical.
- A tiny check script (run before publishing): every `dotnet`/`stingray` command on the page exists in the CLI, every number matches its source file. Same idea as `ProductSurfaceContractTests`.
- Accessibility and link checks on the final page (as the site README already requires).

## Open questions for the owner
- (Answered) `llm-inference.html` is superseded; retire it.
- Is the page for the open-source engine only, or does it also sell something (a hosted/commercial tier)? That decides the call to action.
- GitHub/NuGet URLs and the repo's public status: confirm what may be linked.
- Naming: confirm "OpenTail Stingray" everywhere, with `OpenTail.LLM` retired.

## The "wow" layer (added 2026-10-11)
Principle: Stingray is an engine, so the visuals show **real engine facts**, never a mock UI. Each effect is driven by data captured from the repo (JSON generated by a script from STATUS.md, League, scout output, recorded runs), labelled with date and machine, with vanilla JS/canvas only (no CDN, no library), a no-JS fallback (the same facts as a table) and `prefers-reduced-motion` respected.

Ranked by wow per effort, all honest:
1. **Drop a model on the page.** A drop zone reads the first MB of a user's `.gguf` in the browser (`File.slice`; nothing is uploaded, and the page says so) with a small JS GGUF header parser, then shows architecture, layers, context, quantisation mix, tokenizer, and "admitted by Stingray: yes/no/not yet" from a generated admission list. It is the local-first story demonstrated, and useful.
2. **Replay race.** Token streams from real recorded runs (per-token timestamps) played side by side: Stingray vs llama.cpp on the same model, prompt and machine, from the League. Labelled "replay of a recorded run"; shown even where we are slower. Needs a small capture step (`STINGRAY_EVENTS` has no per-token trace today, so capture with a harness).
3. **Support matrix that breathes.** Architectures x backends (CPU / Vulkan / CUDA) as a heat grid from STATUS.md; hover shows the evidence and tier; filter by modality. Gaps are visible, which is the differentiator.
4. **Will it fit?** Sliders for RAM budget and context; precomputed `stingray scout --budget` results for the catalogue models light up allowed / blocked. Not a formula invented in JS.
5. **Memory-map reveal.** A tensor map of a real GGUF (from `list-tensors`) whose blocks light up as the file is paged in, next to "used in place, never copied". Illustrative of the mechanism, so labelled as a visualisation, with real tensor sizes.
6. **Hear it, see it.** The curated audio and image samples in a strip with the exact command under each; play button, waveform drawn from the actual file.
7. **Start-up bar.** 62 ms vs 111 ms (`--help`), animated, with the "CLI start-up, not time to first token" caption.
8. **Constrained decoding view (later).** Candidate tokens greyed out by the JSON-schema constraint during a real extraction; needs a captured trace from the engine, so only after a capture hook exists.

Not doing: in-browser inference (no WASM build exists), fake "live" terminals, invented tok/s counters, discrete-GPU numbers.
Recommended first three to build: 1, 3 and 2.
