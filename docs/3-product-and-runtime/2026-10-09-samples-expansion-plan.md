# OpenTail.Stingray Samples Expansion Plan

> **Status:** Proposed — inventory and scope first, then add verified examples in small batches  
> **Date:** October 9, 2026  
> **Scope:** Improve the repository's runnable C# samples across the library's real capabilities, especially diffusion and audio, without turning `samples/` into an unmaintained copy of the docs.

## 1. Purpose

The `samples/` directory should let a developer answer two questions quickly:

1. **Can I see a working example of the task I need?**
2. **Does that example use the public NuGet contract I can use in my own application, or internal engine APIs intended for contributors?**

Today, those two roles are mixed. Some samples are useful low-level engineering demonstrations, but use project references to internal source projects. They therefore do not, by themselves, prove that the same code works against the published `OpenTail.Stingray` package. The visible sample set also does not provide dedicated projects for diffusion image generation, vision, or music/sound generation.

This is a separate pass from the documentation architecture plan. It should link to the established docs and verification records, not recreate the model status matrix, model-download catalogue, or technical guides.

## 2. Initial inventory — verify again at execution time

The current `main` sample tree contains:

| Existing sample | Current apparent purpose | Initial observation to verify |
|---|---|---|
| `samples/QuickStart` | One console program with `chat`, `speak`, and `hear` paths | Uses `GgufModel`, `InferenceEngine`, `PiperPipeline` and `WhisperPipeline`; its project references `Core`, `Cpu`, `Engine` and `Audio` source projects rather than consuming the packed NuGet package. Preserve its low-level value, but do not treat it as a package-consumer compile test. |
| `samples/OpenTail.Stingray.Sample.Chat` | Streaming, multi-turn chat | References `OpenTail.Stingray.Engine` directly, so it currently demonstrates source-level APIs rather than proving package-only consumption. |
| `samples/OpenTail.Stingray.Sample.ToolCall` | Tool-calling loop and backend selection | References internal `Core`, `Cpu`, `Cuda` and `Engine` projects. This is a useful advanced example, but should be explicitly classified as an engine-level sample unless deliberately ported to the public API. |
| `samples/OpenTail.Stingray.Sample.HotRouting` | Hot session/model routing demonstration | References internal/session projects and contains simulation-oriented paths. Document precisely which path is simulated and which path exercises the real runtime; avoid implying that simulated results are measured inference evidence. |
| `samples/ChatServer` | ASP.NET Core chat endpoint | References the server source project directly. Keep it useful for developers working from source, and add a separate package-consumer validation if the intent is to demonstrate the NuGet package. |

There is no `samples/README.md` in the current tree. The root README links directly to `samples/QuickStart` and `samples/ChatServer`, but the full set is not presented through one sample catalogue. The checked-in sample projects currently have no dedicated diffusion-generation or audio-generation project; QuickStart's speech paths cover Piper TTS and Whisper STT, not every audio capability.

These are inventory observations, not a claim that each sample is broken. Phase 0 must re-check them against the precise commit being worked on and look for new samples or API changes before acting.

## 3. Principles and boundaries

1. **Separate consumer samples from contributor samples.** Consumer samples should reference the actual published NuGet package (preferably the exact locally packed package during CI), not internal source projects that make inaccessible types appear available. Internal samples may use project references, but must say so clearly.
2. **Every sample has one main job.** Keep the first working path small. Explain optional extensions in the README or a separate sample rather than building a large demo framework.
3. **Code is the executable recipe.** Each sample should compile; its README should state exact prerequisites, asset locations, invocation, expected output, limitations and links to the current canonical guide/status evidence.
4. **Do not commit large model weights or output binaries.** Provide trusted source links and a clear asset checklist. Honour model, voice and dataset licences.
5. **Don't confuse a smoke test with a quality claim.** State whether a sample has been compiled, run with real weights, numerically checked against a reference, or only structurally inspected. Do not present a demo or mocked path as model-validation evidence.
6. **Use verified tasks first.** Sample selection must consult `docs/STATUS.md`, `docs/RUNNING.md`, and the relevant pipeline implementation before deciding which model/checkpoint to demonstrate. Status notes are evidence to re-check, not infallible labels.
7. **Avoid duplicate documentation.** `samples/README.md` is a directory index. The main user guides remain the conceptual/task docs; sample READMEs explain how to build and run their exact code.

## 4. Phase 0 — Audit and choose, before adding code

1. Pin the branch and exact commit SHA. Inventory every sample directory, `.csproj`, README and project reference.
2. Build all existing sample projects with the repository's prescribed .NET 10 commands and review any warnings/errors under the repository's warnings-as-errors policy.
3. For each sample, record its audience (`package consumer`, `source-level contributor`, or `simulation/test harness`), dependencies, task, exact entry point, model assets and whether its current README/run instructions match the code.
4. Inspect the public NuGet API and the actual pipelines before selecting candidate examples. Do not infer that an internal pipeline can be consumed from NuGet just because the implementation class exists.
5. Check the canonical verification record and latest evidence for the specific checkpoint/backends that a sample would use. Exclude or clearly label a candidate whose user-visible result is currently partial or unresolved.
6. Decide what should be retained as-is, renamed/relabelled, migrated to the public API, or replaced. Avoid a wholesale rewrite simply to normalize names.

**Phase 0 output:** a concise inventory table with one decision per existing sample, plus a priority-ordered candidate list based on public API readiness, verification status, user value, model/setup cost and maintenance burden.

## 5. Proposed sample catalogue

The exact sample names and implementation order are subject to Phase 0 evidence. These are candidate priorities, not assumptions that every API is ready today.

### Priority 1 — public package and everyday tasks

#### A. Package-consumer chat sample

- Demonstrate the supported public `Model` / `Context` / executor / `ChatSession` route.
- Compile against the packed `OpenTail.Stingray` NuGet package, not a source project reference.
- Support a model path argument and explain the exact model prerequisite. Only show a `ModelHome` convenience call if that API exists in the package version being demonstrated.
- Include cancellation/disposal and a short streaming interaction; keep advanced options out of the first example.

The existing low-level chat sample may remain as a separate contributor example if it still demonstrates useful engine concepts.

#### B. Speech sample — TTS and/or ASR

- Provide a deliberately small public-library sample for speech generation and a separate or clearly isolated transcription path; their model assets and API dependencies differ.
- The already-used Piper and Whisper paths are initial candidates because they are present in `QuickStart` and the root README. Verify them against the package before reuse.
- Give exact model/config filenames, download sources, output WAV/transcript expectations, input sample-rate requirements and licence notes.

#### C. Text-to-image diffusion sample

- Add one minimal C# library sample for a specific, end-to-end-verified image-generation path, not a showcase of every supported architecture.
- Stable Diffusion 1.5 is a candidate because the current status record describes a coherent end-to-end result; re-check current pipeline/API and evidence before choosing it.
- Explain all required model, text-encoder/tokenizer and VAE files; exact checkpoint source/revision; low-memory preview settings; backend fallback; expected PNG; and known runtime.
- Keep image-to-image, inpainting, ControlNet, LoRA and other extensions out of the first text-to-image sample unless they can be shown cleanly and verified separately.

### Priority 2 — important capabilities with extra setup

#### D. Text-to-music or text-to-sound sample

- Evaluate the current MusicGen and Stable Audio pipelines before selecting a first example. Compare package exposure, required files, generation runtime, licence/gating constraints, and strength of end-to-end or golden-parity evidence.
- Choose one practical, documented path for the first version. Include exact companion files (for example text encoder/tokenizer/codecs where required), duration/sample-rate, output WAV, expected runtime and memory.
- Do not combine multiple audio-generation families into one overly generic sample API merely to make the catalogue look comprehensive.

#### E. Image question-answering / vision sample

- First verify whether a complete public C# path from image + question to a text response is currently exposed and documented. The existing `UnifiedVisionPipeline` example may only demonstrate image-token embedding, which is not the same as a complete image-QA workflow.
- If the complete path is public and verified, provide an image-QA example with a compatible model/projector pair and explain their compatibility requirement.
- If only the image embedding stage is stable/public, either demonstrate that narrower stage honestly or create a separate implementation/API task before promising end-to-end image Q&A.

#### F. Embeddings and reranking (optional follow-on)

- Consider a compact sample that embeds a few texts and reranks candidate passages only after reviewing which pipeline combination has the cleanest public API.
- Keep the first version local and dependency-light; do not introduce a database/vector-store dependency just to demonstrate embedding generation.

### Priority 3 — defer until utility and verification justify the cost

#### G. Video generation

Video diffusion requires multiple large assets and often long runtimes. Do not make it a first-run sample just to check a capability box. Add one only when there is a specific model variant with current end-to-end evidence, clear setup instructions, and a runnable minimal request. Treat variants (for example, Wan 2.1 vs Wan 2.2 A14B) separately.

#### H. Advanced engine/session demonstrations

Keep tool-calling internals and hot routing if they still add value, but identify them as advanced/source-level examples and state precisely whether each run uses actual weights, a simulated forward pass, or a mixture. Do not use these samples as substitutes for first-party public-package task examples.

## 6. Sample structure and README contract

Create `samples/README.md` as the canonical index. For each sample, list its audience, task, project path, package/source dependency mode, model size/setup burden and verification status.

Each sample README should contain:

1. **What it demonstrates** and what it explicitly does not demonstrate.
2. **Prerequisites:** .NET SDK, OS/backend assumptions, RAM/disk guidance where grounded, exact model/companion files and licences.
3. **Prepare assets:** links to official or otherwise justified model sources; no invented filenames or vague "download the model" instructions.
4. **Run:** copyable commands from repository root and from the sample directory if useful.
5. **Expected result:** output file, short console result or API endpoint; don't use fabricated model output as proof of a live run.
6. **Verification level and known limits:** compiled; real-weights run; reference-checked if applicable; exact checkpoint/backend/device and evidence link.
7. **Related docs:** link to the task guide, `STATUS.md`, and `RUNNING.md` where relevant.

Avoid placing a full architecture tutorial in each README. Link out to the existing docs.

## 7. Automated verification

### Required for every sample

- `dotnet build` succeeds under the repository's normal configuration and warnings-as-errors policy.
- Sample README commands, project path and argument names match the current code.
- The sample is indexed from `samples/README.md`; the root and docs navigation point to the index rather than maintaining a second list of all samples.
- No model weights, tokens, licences requiring acceptance, or generated outputs are silently bundled.

### Required for public package-consumer samples

- CI packs the intended NuGet package(s), then builds the consumer sample against those package files from a clean project setup.
- The test must not use `ProjectReference` to internal engine projects or accidentally inherit internal-only namespaces.
- Pin and record which package version the user is meant to use. If testing a not-yet-published package, describe it as a package-artifact test, not as validation against an already-published release.

### Runtime checks

- Keep a fast compile/smoke tier that doesn't require downloading large gated assets.
- Add opt-in real-weight runtime checks only when the assets and runtime cost are acceptable for the CI runner. Otherwise document a reproducible manual verification receipt with commit, checkpoint hash, backend/device, command and result.
- Never claim a diffusion or audio sample was runtime-tested based only on compilation.

## 8. Rollout

1. **Inventory and decide** — complete Phase 0; no new projects yet.
2. **Index and classify** — add `samples/README.md`; label public-consumer vs engine-level examples; correct stale commands/docs discovered by the audit.
3. **Package API baseline** — ensure one small chat example compiles against packed NuGet; fix the root README snippet if its stated API does not match the actual package.
4. **Representative modality samples** — implement and verify the small speech sample and one text-to-image diffusion sample.
5. **Second wave** — choose audio generation and vision/image-QA based on public API readiness and verified end-to-end paths; consider embeddings/reranking.
6. **Video and specialized demos** — add only if the model's end-to-end status, real asset setup, runtime and maintenance cost make it a useful developer example.
7. **Add CI gates and finalize navigation** — ensure all samples build, public package samples consume packed artifacts, and each sample index entry resolves.

## 9. Definition of done

- `samples/README.md` inventories all maintained samples and accurately describes their audience and dependency mode.
- Existing useful samples are retained or deliberately replaced; none is silently dropped merely for naming consistency.
- There is at least one clean public-package chat sample, one speech task sample, and one C# diffusion image-generation sample, each compiled and run at the level claimed.
- A further audio-generation sample and a complete vision sample are either delivered with accurate evidence or explicitly deferred for a named API/verification prerequisite.
- All public-package samples compile against packed package artifacts, not just internal source project references.
- Every sample has exact asset/setup/run instructions and links to the canonical evidence/docs.
- CI distinguishes build validation from real-model inference validation.
- No stale sample claims, mock-vs-real ambiguity, or unsupported model/backend claims remain in the sample catalogue.

## 10. Non-goals

This plan does not require one sample for every model architecture, every backend, every CLI command or every modality variant. It does not replace task guides or model status records. It does not make a large video model the first-run demo. The goal is a small, trustworthy set of examples that demonstrate the breadth of Stingray without pretending all features have the same setup cost or verification strength.
