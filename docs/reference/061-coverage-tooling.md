# 061 — Coverage tooling: `pull`, `admit-arch`, `gen-vision-scaffold`, `verify-goldens`

CLI commands added to speed up the recurring manual work behind this project's
"run any GGUF from Hugging Face" goal (`docs/00-current-work.md`). None of them change engine
behavior — they're operator/developer tooling, same tier as `doctor`/`list-tensors`/`plan`.

All three are pure C#/.NET, consistent with this project's own "no Python, no P/Invoke" design
(README.md/CLAUDE.md) — this replaced an earlier draft of the third tool that generated Python
reference scripts, which conflicted with that rule.

---

## `stingray scout -m <gguf>`

Read-only evidence dossier; run it **first** when triaging or admitting an architecture (agent loop:
[architecture-admission-agent-playbook.md](architecture-admission-agent-playbook.md); plan:
[checkpoint scout plan](../3-product-and-runtime/2026-10-09-checkpoint-scout-and-ai-admission-plan.md)).

```text
stingray scout -m model.gguf                                   # terminal summary
stingray scout -m model.gguf --format json -o scout.json        # versioned JSON (schema_version 1), deterministic
stingray scout -m model.gguf --budget 64G [--reserve 8G]        # also report the execution-feasibility decision
```

* Opens the GGUF header, metadata and tensor index only. It reads no tensor values (the architecture probe is given a source that throws
  on a value read), constructs no forward pass, downloads nothing and does not hash the file (`sha256_state` is `not_computed`).
* Report: artifact identity (bare file name, no paths), metadata and tokenizer facts, tensor signature (bytes by dtype, layer-normalized
  patterns, partial-layer / dtype / shape irregularities), architecture resolution via the registry, blockers, resource preflight,
  ordered next commands, and per-stage receipts (`Passed` / `Failed` / `Blocked` / `NotRun`).
* **Blockers.** `confirmed` means the engine's own gate would refuse (unregistered or non-admitted architecture, a dtype
  `ModelCompatibility.IsSupportedWeightDType` rejects, a malformed tensor size); `suspected` is a lead (tokenizer shapes, undeclared architecture).
* **Unknown stays unknown.** The host working-set and KV estimates are `Unknown` with a reason and null bytes; file size is never reported as
  peak RAM. With `--budget` the execution decision is therefore `blocked` until a real estimator exists, and without it `not_assessed`. It is never `allowed` today.
* Exit codes: 0 report produced (blockers do not change it), 1 file unreadable as GGUF, 64 bad option, 66 file missing.
* Not yet implemented (see plan): feature hypotheses and architecture relatives (Phase 2), `scripts/scout-pretest.ps1`.
* Advisory only: nothing in the report admits or promotes an architecture.

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
bucket (bucket-1 permissive vs. bucket-2, see `docs/done/01-gguf-model-coverage-plan.md`'s "License
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

---

## `stingray verify-goldens [--dir <dir>] [--golden <pattern>] [--baseline <file>] [--diff <file>] [--strict]`

Runs forward-pass parity verification across checked-in `.golden.json` references against locally
present weights, validating model architectures, pinned hashes, and hyperparameter invariants,
measuring pure decode throughput, and recording or diffing against identity-aware per-host baselines.

```
stingray verify-goldens                                         # verify all goldens whose models exist
stingray verify-goldens -g smollm                               # filter goldens by pattern
stingray verify-goldens --baseline host-baseline.json           # record baseline of current run (Schema 2)
stingray verify-goldens --diff host-baseline.json               # diff current results against baseline
stingray verify-goldens --strict                                # strict regression gate (fails on unpinned, near-ties, etc.)
```

### Verification & Timing
1. **Architecture validation**: Compares `general.architecture` from model GGUF metadata against `golden.Architecture`. Mismatches immediately fail verification as `ArchMismatch`.
2. **Model discovery & Pin status**: Matches models using `ModelLocator` (respecting `STINGRAY_MODEL_DIRS`). Fingerprints checkpoints with `ModelFingerprinter.CheckPin()`:
   - `Verified`: checkpoint SHA-256 matches the golden's pinned hash.
   - `Mismatch` / `NotRecorded`: labeled `UnpinnedFile` (runnable for diagnostics in normal mode, rejected in strict mode).
3. **Hyperparameter guards**: Validates `expectedHyperparameters` (e.g. `ropeDim`, `numExperts`) against model metadata before execution.
4. **Pure decode timing**: `GoldenParityRunner` measures `Prefill` duration and subsequent timed `Forward` decode steps separately. Model loading, fingerprinting, hyperparameter checks, baseline I/O, and the stepwise-vs-prefill consistency check are strictly excluded from decode timing.
5. **Summary table**: Color-coded Spectre.Console output showing verdicts (`Exact`, `NearTie`, `Diverged`, `GuardFailed`, `ArchMismatch`, `UnpinnedFile`, `Skipped`) with pure decode tok/s.

### Baselines & Diffing (Schema 2)
- **Baseline export (`--baseline <path>`)**: Records `GoldenBaselineFile` (Schema 2) with host environment metadata (`OsDescription`, `ProcessArchitecture`, `ProcessorCount`, `RuntimeDescription`), model SHA-256, golden evidence SHA-256, prompt tok/s, decode tok/s, and decode step counts.
- **Identity-aware diffing (`--diff <path>`)**:
  - Compares model and golden evidence hashes between current and baseline runs.
  - If either model or golden identity changed, reports `IdentityChanged` and suppresses speed regression/improvement classifications across different artifacts.
  - Detects legacy Schema 1 baselines (setup-inclusive throughput) and informs the operator to regenerate the baseline.
  - Checks host environment comparability.
- **Robustness**: Handles duplicate entries in baseline files with clear errors and returns non-zero on missing diff files, malformed JSON, or baseline write failures.

### Normal vs Strict Modes
- **Normal mode**: Exit 0 on exact or near-tie parity (pinned or unpinned). Exit 1 on genuine divergence, architecture mismatch, failed hyperparameter guard, invalid arguments, or diff/baseline errors.
- **Strict mode (`--strict`)**: Regression gate returning non-zero if ANY golden:
  - Is missing its checkpoint (`Skipped`).
  - Is unpinned (`Mismatch` or `NotRecorded` SHA-256).
  - Yields a `NearTie` rather than an exact match.
  - Diverges, crashes, or fails an architecture or hyperparameter guard.
  - Matches no files (empty filter selection), or is cancelled/incomplete.

