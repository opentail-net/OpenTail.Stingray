# Front door: README and first-run experience (design, 2026-09-26)

## Problem

The repository and the NuGet package present capability lists, not a way in.

- `README.md` (97 KB) opens with a verification matrix written for maintainers. A newcomer cannot
  tell what to install, which model to download, or what a first success looks like.
- The NuGet package README (`src/OpenTail.Stingray/README.md`) shows C# that does not compile:
  - `new InferenceEngine(model, cpu)` has no such constructor (the real one takes an
    `IForwardPass`, an `ITokenizer` and a model id);
  - `StreamChatAsync` does not exist (it is `GenerateAsync`);
  - it pins `--version 1.0.6` while the source is at 1.0.7;
  - it lists capabilities the status record grades 🔴 or 🟡 (e.g. Kimi, Nemotron vision).
- Every task command assumes the user already has the right checkpoint at a repo-relative path
  (`models/kokoro-82m-q8_0.gguf`, …). Several defaults have no recorded public download source:
  the local Kokoro GGUF's `general.name` is a hash. So a new user cannot obtain them at all.
- Image generation needs four separately sourced files (DiT, VAE, T5, CLIP) passed by path.
- For a .NET developer, getting a chat reply in code means assembling a forward pass, tokenizer
  and engine by hand. The only convenient loader lives in the Server package.

## Audience (decided with the user)

.NET developers first: the in-process library is what nobody else offers. The CLI is the
two-minute demo and the tool for non-developers.

## What the README must do

1. Answer in the first screen: what it is, why a .NET developer would pick it, and a visible
   result.
2. Give a first success in minutes. It uses only commands and code that have been run, from a
   clean directory, against publicly downloadable models.
3. State costs up front: download size, RAM, GPU optional, rough speed on ordinary hardware.
4. Organise by what people want to do: chat, describe an image, speak text, transcribe audio,
   generate an image, embed and rerank, serve an OpenAI-compatible API. Each recipe has the same
   shape: what you get, what to download, the command or C#, what you will see.
5. Use progressive disclosure: quick start, then recipes, then library/server, then links to depth
   (docs/STATUS.md for the per-model verification record, benchmarks, design docs).
6. Stay honest: say which recipes are verified and link the evidence. The old README becomes
   `docs/STATUS.md` (git mv, history kept); the new one links to it.
7. Use one name (`stingray`) everywhere. The CLI help currently says `opentail-llm-cli`.

## The front door (tooling)

1. **Starter manifest**: one verified bundle per task, versioned in the repo, with public source
   (HF repo + file + sha256), size, licence, hardware notes, and the status-record row that
   verifies it. Only 🟢 rows with a public source qualify. Candidates to confirm:

   | Task | Candidate | Why |
   |---|---|---|
   | chat | Qwen2.5 / SmolLM2 GGUF (official or bartowski) | small, verified against llama.cpp, permissive |
   | describe image | dots.ocr GGUF + mmproj (fixed 2026-09-26) or Gemma 3/4 | verified vs llama-mtmd-cli |
   | speak | Piper `en_US-lessac-medium` (rhasspy/piper-voices) | fastest engine (RTF 0.13), public, small |
   | transcribe | Whisper `ggml-base.bin` (ggerganov/whisper.cpp) | public, small |
   | image | FLUX.1-schnell GGUF (city96) + VAE + encoders (comfyanonymous), or Z-Image-Turbo | Apache-2.0; needs a size warning |
   | embed / rerank | bge-small / ms-marco MiniLM (HF) | already used in the CLI examples |

2. **A model home outside the repo**: `%LOCALAPPDATA%\stingray\models` / `~/.cache/stingray/models`,
   overridable, searched after explicit paths. The repo-relative `models/` stays a developer
   convenience.
3. **`stingray setup <task>`**: shows size, licence and expected speed, asks, then downloads the
   bundle through the existing `pull` machinery (resumable, sha-checked).
4. **Task commands that need no paths once set up**: `stingray chat`, `speak`, `transcribe`,
   `describe`, `image`. A missing model gives a one-line fix (`run: stingray setup speak (63 MB)`),
   never a stack trace.
5. **`stingray models`**: installed bundles, which tasks are ready, what to run to fix the rest.
6. **A small library facade** so the README's C# is three lines per task, and the snippets are
   compiled by a test so they cannot rot:
   `await using var chat = await StingrayChat.OpenAsync("qwen2.5-0.5b");` /
   `Speech.SynthesizeAsync(...)` / `Transcriber.TranscribeAsync(...)`. The facade is built on the
   existing engines, not new math.
7. **Docs-as-tests**: every README command and snippet is exercised by a test (downloads cached),
   so the README fails CI before it lies.

## Model catalog (refined with the user, 2026-09-26)

"An actual reference in our code to fetch the models": the starter manifest above, made concrete.

- **What it is**: one versioned catalog file in the repo, embedded and source-generated so it
  works under NativeAOT. It is the single source of truth for the CLI, the library, the README
  and the tests. Each entry is one ready-to-use bundle:
  - a short id (`qwen2.5-0.5b`, `piper-lessac`), the task, and a one-line "why pick this";
  - files: HF repo, file, sha256, size. Multi-file bundles are one entry, e.g. FLUX = DiT + VAE
    + two text encoders;
  - licence, RAM/VRAM needs, a measured speed note, and a link to the test / docs/STATUS.md row
    that proves it works.
- **Keeping it small is a rule**:
  - one default per task, so `stingray setup speak` needs no choice;
  - at most three alternatives per task, each justified by a user-level reason ("faster",
    "better quality", "clones a voice", "multilingual", "fits in 8 GB"), never by architecture;
  - admission requires a 🟢 status plus an automated test on those exact files (sha-checked);
  - everything else still works via `pull -r <repo>` / `-m <path>`. The catalog is a
    recommendation list, not the coverage list; docs/STATUS.md stays the coverage record.
- **Presentation**:
  - `stingray models` shows tasks, what is installed, and the one command that fixes each gap;
  - `stingray models <task>` shows that task's 2–4 options: size, speed on this machine,
    quality note, licence. Options that do not fit in RAM are hidden or flagged;
  - non-permissive licences need an explicit yes before download.
- **One catalog, four consumers**:
  - the CLI (`setup`, `models`, task commands resolving by id);
  - the library (`await Models.EnsureAsync("speak")` returns local paths);
  - the README (recipes name catalog ids, not hand-pasted URLs);
  - the tests (download, verify sha, run), so an upstream file that moves fails CI, not a user.

## Order of work

1. Move `README.md` to `docs/STATUS.md`; write the new README around what works today: explicit
   model downloads, commands and C# that have actually been run. Fix the NuGet README's
   non-compiling snippets in the same pass.
2. Starter manifest + model home + `setup` + `models`.
3. Task commands and the library facade; rewrite the README's recipes around them.
4. Docs-as-tests.

Steps 2–4 are the substance. Step 1 makes the front page truthful now.
