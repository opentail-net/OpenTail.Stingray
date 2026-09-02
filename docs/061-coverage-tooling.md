# 061 — Coverage tooling: `pull`, `admit-arch`, `gen-vision-scaffold`

Three CLI commands added 2026-09-02 to speed up the recurring manual work behind this project's
"run any GGUF from Hugging Face" goal (`docs/00-current-work.md`). None of them change engine
behavior — they're operator/developer tooling, same tier as `doctor`/`list-tensors`/`plan`.

All three are pure C#/.NET, consistent with this project's own "no Python, no P/Invoke" design
(README.md/CLAUDE.md) — this replaced an earlier draft of the third tool that generated Python
reference scripts, which conflicted with that rule.

---

## `stingray pull -r <repo>`

Downloads a GGUF model straight from a Hugging Face repo id, closing the gap between "a GGUF
exists on HF" and "it's a file this engine can load" — every prior session fetched checkpoints by
hand outside the tool before running them.

```
stingray pull -r bartowski/Qwen2.5-7B-Instruct-GGUF                 # auto-picks Q4_K_M (or nearest)
stingray pull -r bartowski/Qwen2.5-7B-Instruct-GGUF -q Q8_0         # explicit quant substring match
stingray pull -r bartowski/Qwen2.5-7B-Instruct-GGUF --list          # list files, don't download
stingray pull -r bartowski/Qwen2.5-7B-Instruct-GGUF -o D:\models    # destination directory
```

How it works:
1. Accepts either a bare `owner/name` repo id or a full `https://huggingface.co/...` URL.
2. `GET https://huggingface.co/api/models/{repo}` lists the repo's file tree (`siblings`); this
   is filtered down to `*.gguf`. `HF_TOKEN`, if set, is sent as a bearer token — needed for
   gated repos the account has accepted terms for.
3. Quant selection: `-q <substring>` filters by a case-insensitive substring; with no `-q` and
   multiple `.gguf` files present, it prefers `Q4_K_M`, then `Q4_K_S`/`Q5_K_M`/`Q4_0`/`Q8_0` in
   that order, else the first file alphabetically.
4. Sharded checkpoints (`model-00001-of-00005.gguf`) are detected by filename pattern — picking
   any one shard pulls every shard in the set.
5. Download is a streamed `HttpClient` GET against `.../resolve/main/<file>?download=true`, with
   `Range`-header resume: a partial file present on disk restarts from its length (falling back
   to a full restart if the server doesn't honor the Range request), and a file whose size
   already matches the expected size is skipped entirely.

Deliberately NOT built: a manifest/alias/model-store layer (same scope line `ListModelsCommand`
already draws), tokenizer/config sibling-file fetching, or checksum verification (HF's `resolve`
CDN doesn't reliably expose a stable content hash in the plain siblings listing).

## `stingray admit-arch -m <path>`

Automates the mechanical half of the architecture-admission workflow that
`ModelCompatibility.cs`'s long allowlist-comment history (`minicpm`/`xverse`/`orion`/`internlm2`/
`ernie4_5`/...) shows repeating: download an unsupported-architecture checkpoint, run it under a
diagnostic bypass, and compare its greedy output token-for-token against an independent
reference (llama.cpp). Most of those turned out to need **zero new forward-pass code** — the real
blocker was almost always the tokenizer axis (SPM merges-vs-scores, byte-fallback, etc.) — but
that was only ever discovered by hand each time.

```
stingray admit-arch -m models/new-arch-model.gguf
stingray admit-arch -m models/new-arch-model.gguf -p "The capital of France is" -n 8
stingray admit-arch -m models/new-arch-model.gguf --reference-tokens 700,9689,315,10298,357,11855,93937,2
```

What it does, in order:
1. Reports whether the architecture is already in `ModelCompatibility`'s allowlist (and exits
   immediately if so — nothing to admit).
2. **Tokenizer triage**: reads `tokenizer.ggml.model`/`.merges`/`.scores` and flags the two known
   recurring shapes — "scores-only SPM" (already handled by `GgufTokenizer.
   SpmMergePiecesByScore`) and genuine Unigram-LM (`tokenizer.ggml.model=t5`).
3. **Tensor triage**: dumps the layer-0 tensor inventory (name/dtype/shape) for a reviewer to
   eyeball against a known-working architecture before spending time on a real run.
4. **Real run**: constructs a CPU `ForwardPass` directly (bypassing `ModelCompatibility.
   ValidateForTextGeneration`, the same bypass `--allow-unverified-arch` uses), tokenizes the
   given prompt, prefills, and greedy-decodes `-n` tokens — rejecting immediately on empty/NaN
   logits (a structural failure, not worth comparing further).
5. **Verdict**: with `--reference-tokens` (a comma-separated id list captured from `llama-server
   .../completion` with `return_tokens:true`, or `llama-tokenize`/`llama-cli --temp 0 --top-k 1`),
   compares token-for-token and prints either a full-match `ADMIT` block — including a
   paste-ready allowlist comment for `ModelCompatibility.cs` — or the exact divergence position.

What it does **not** do: it cannot manufacture the reference token sequence itself (no independent
oracle lives in this repo — that's what makes the comparison trustworthy) or evaluate license
bucket (bucket-1 permissive vs. bucket-2, see `docs/01-gguf-model-coverage-plan.md`'s "License
policy: code vs. checkpoint" — that's still a human judgment call before committing a permanent
parity test).

## `stingray gen-vision-scaffold -m <mmproj> -a <arch>`

Cuts the boilerplate cost of starting a new vision-architecture golden-parity test. Every existing
one (`Llava`/`Pixtral`/`GLM-4.6V`/`HunyuanVL`/`Exaone4`/`MiMoVl`/`Qwen2.5-VL`/`Gemma4UV`) was
hand-written from scratch against the same shape of setup: read the mmproj's real tensor names/
shapes/`clip.vision.*` metadata, then write a parity test comparing the C# encoder against an
independent reference.

```
stingray gen-vision-scaffold -m models/mmproj-step3-vl.gguf -a step3vl
```

Output:
- A printed report of every `clip.vision.*` metadata key this project's encoders read, plus the
  full tensor inventory grouped by suffix (`v.blk.N.attn_q.weight x27 Float16 [1152,1152]`, ...) —
  real values for the checkpoint given, not guessed.
- `tests/OpenTail.Stingray.Tests.Vision/<Arch>VisionEmbedderParityTests.cs` — a `[Fact(Skip=...)]`
  skeleton wired to `VisionTestPaths.FindFixtureDir`, with the tensor inventory embedded as a doc
  comment and explicit TODOs for the parts that need real per-architecture work.
- Refuses to overwrite an existing file of the same name (prints `SKIP` instead).

**The oracle step is intentionally left manual and pointed at real C++, not Python.** Earlier
vision parity tests in this project used a hand-written numpy reimplementation of llama.cpp's mtmd
code (`scripts/*_ref.py`) as the independent reference. That pattern is retired going forward —
this project ships no Python — in favor of running the real, already-vendored
`tools/llama.cpp/llama-mtmd-cli.exe` (or `llama-mtmd-debug.exe`) directly against the same
checkpoint to capture golden embeddings, the same "run the real external reference binary"
pattern already used for text-generation parity receipts (`llama-tokenize`/`llama-server`), just
extended to vision. The `*_ref.py` scripts already in `scripts/` predate this decision and are not
retroactively removed by it, but no new ones should be added.

Per CLAUDE.md rule 8, the generated scaffold's TODOs explicitly say to read the real
`tools/mtmd/models/<arch>.cpp` reference before writing any encoder math — this tool only removes
the boilerplate, not the need to check the reference.
