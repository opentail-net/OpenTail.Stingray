# 061 — Coverage tooling: `pull`, `admit-arch`, `gen-vision-scaffold`, `verify-goldens`

CLI commands added to speed up the recurring manual work behind this project's
"run any GGUF from Hugging Face" goal (`docs/00-current-work.md`). None of them change engine
behavior — they're operator/developer tooling, same tier as `doctor`/`list-tensors`/`plan`.

All three are pure C#/.NET, consistent with this project's own "no Python, no P/Invoke" design
(README.md/CLAUDE.md) — this replaced an earlier draft of the third tool that generated Python
reference scripts, which conflicted with that rule.

---

## External access (one policy for everything that touches the network)

Everything that leaves the machine asks `ExternalAccess.Evaluate()` (`src/OpenTail.Stingray.Core/Net/`). **The default is allowed. Deny wins whenever it is asked for:**

| Setting | Effect |
|---|---|
| unset, or `STINGRAY_ALLOW_EXTERNAL=1` (`true`/`on`/`yes`/`allow`) | allowed |
| `STINGRAY_ALLOW_EXTERNAL=0` (`false`/`off`/`no`/`deny`) | **denied** |
| `STINGRAY_OFFLINE=1` or `HF_HUB_OFFLINE=1` | **denied** (these predate this policy; they beat an explicit allow) |
| any other value of `STINGRAY_ALLOW_EXTERNAL` | **denied** (a typo must not silently open the network) |

* The check runs before **every** request, not once at startup, and a refusal says which setting caused it and how to change it. `pull`, catalog installs (`setup`) and all remote features honour it.
* New network code goes through `ExternalHttpClient` only. It contacts Hugging Face hosts and nothing else (`huggingface.co`, `hf.co` and their subdomains, which covers the CDN and storage hosts a download redirects to), requires https, and follows redirects itself so that every hop is host-checked.
* `HF_TOKEN` is sent to `huggingface.co` itself and **never** to a CDN or storage host. Tokens, query strings and URLs are never logged; the log keeps only request count, host names and bytes received.
* Nothing is cached or written to disk by the gateway, and nothing is downloaded unless a command whose purpose is downloading (`pull`, `setup`) is run.
* Still on its own client, not yet routed through the gateway: `pull`'s file download (it does honour the policy). Moving it is a follow-up.
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
* **Host working-set estimate** (`HostMemoryEstimator`, CPU run only; says nothing about GPU placement). An upper bound, not a prediction, as the sum of: every tensor byte
  (the CPU path touches all of them); the **Q4_K repack copy** (an additional anonymous 1216/1152 of the dense Q4_K bytes, capped at a quarter of the budget, which is why a
  Q4_K_M file costs about 1.8x its size; routed-expert stacks are not repacked); 192 MiB base; fp32 KV for `--ctx-size` tokens (default 4096, capped at the model's limit);
  and batched-prefill scratch. Each term is listed in the JSON with its certainty. `file size` is never reported as peak RAM.
  * **Unknown stays Unknown.** Hyperparameters that cannot be resolved or are zero (incomplete metadata), an unregistered architecture, or a family whose state layout is not
    modelled (MLA, hybrid/GDN, RWKV) make the estimate `Unknown` with null bytes and a reason, and an unknown estimate is `blocked` under any budget.
  * **Gate:** with `--budget`, `allowed` only if `estimate + reserve <= budget` (reserve default 8G, `--reserve`); otherwise `blocked`. Without `--budget`: `not_assessed` (the estimate
    is still shown). It is a safety gate for choosing what to run, not proof that a run will fit.
  * **Calibration (2026-10-09, real CPU runs, `STINGRAY_GC_STATS` peakWorkingSet; estimate made with the same `-c`):**

    | Model (file) | Run | Measured peak | Estimate | Estimate / measured |
    |---|---|---|---|---|
    | SmolLM2-135M Q4_K_M | ctx 512, 8-token prompt | 263 MiB | 342 MiB | 1.30 |
    | SmolLM2-360M Q4_K_M | ctx 2048, short | 439 MiB | 766 MiB | 1.74 |
    | SmolLM2-1.7B Q4_K_M | ctx 4096, ~600-token prompt | 2256 MiB | 4271 MiB | 1.89 |
    | SmolLM2-1.7B Q4_K_M | ctx 2048, context filled | 2902 MiB | 3119 MiB | 1.07 |
    | Mistral-7B Q4_K_M | ctx 2048, short | 7689 MiB | 9028 MiB | 1.17 |
    | Mistral-7B Q4_K_M | ctx 2048, context filled (~1900 tokens) | 8597 MiB | 9028 MiB | 1.05 |
    | Phi-3.5-MoE Q3_K_M (blind: predicted before measuring) | ctx 2048, 4 new tokens | 19,622 MiB | 27,950 MiB, then 20,825 MiB after the `_exps` fix | 1.42, then ~1.05 |

    The estimate was at or above the measurement in every run, and tightest (1.05-1.07) where the run actually filled the context, which is the case the gate has to protect. The
    MoE result is one run: it showed the routed-expert stacks are **not** repacked, and the estimator was changed on that evidence. Other MoE families and any model with a different
    kernel path are not measured, so a MoE estimate carries that caveat. An earlier Mistral run reported 4289 MiB in 1.6 s because the prompt exceeded the context and the run
    failed; it was discarded (CLAUDE.md rule 12).
* Exit codes: 0 report produced (blockers do not change it), 1 file unreadable as GGUF, 64 bad option, 66 file missing.
* **Feature findings** (`ScoutFeatures`): each has a stable id, a one-line interpretation, evidence (metadata keys and tensor names/dtypes/shapes) and a caveat.
  `Known` = the structure was observed (e.g. `nextn_predict_layers` metadata plus several `blk.N.nextn.*` tensors); `Hypothesis` = one-sided or name-only
  evidence. Neither means the semantics are supported. Ids: `ffn.expert_routed`, `ffn.shared_expert`, `attn.{fused_qkv,separate_qkv,qkv_layout_mixed,qk_norm,gqa,mha,mqa,head_count_per_layer,kv_heads_per_layer,mla_low_rank,sparse_indexer}`,
  `recurrent.{ssm,hybrid_ssm,rwkv_time_mix}`, `mtp.nextn_head`, `rope.{multi_axis,scaling,partial}`, `multimodal.{projector_file,vision_tensors_in_text_file,audio_tensors}`,
  `quant.low_bit_or_block_fp`, plus the tokenizer findings. The names are standard GGUF conventions; they were cross-checked by hand against upstream llama.cpp on 2026-10-09 (a local checkout that is not part of this repo), and nothing in scout reads or needs llama.cpp.
  A rule fires only on evidence it can cite; one isolated MTP-like tensor, a projector alone, or `ssm_*` names alone never reach `Known` semantics.
* **Remote scout: `scout -r owner/repo [-f <file>] [--revision <rev>] [--max-index-mb N]`.** Inspects a model hosted on Hugging Face **without downloading it**: it resolves the repo to an
  immutable commit, then reads only the start of the file (header, metadata and tensor index, typically 2-16 MiB of a 20-60 GiB model) with HTTP Range requests, and runs the same analysis as a
  local file. Subject to the external-access policy above (`STINGRAY_ALLOW_EXTERNAL`, default allowed). Verified 2026-10-09 on the live Hub:
  * the report for `bartowski/SmolLM2-135M-Instruct-GGUF` Q4_K_M, scouted remotely, is **identical** to scouting the same file locally in all 11 sections (only `source`/`network` differ); 3 requests, 2.0 MiB received;
    the Hub-published SHA-256 equals the one computed locally;
  * a 20 GiB community MoE (`ornith-ai/Ornith-1.5-35B-A3B-GGUF`, 3.7 M downloads/30 days): 3.3 s, 9 requests, 16 MiB; it is `qwen35moe`, whose nearest admitted parent differs by exactly the extra next-n (MTP) head;
  * a 56.9 GiB split BF16 model (`unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF`, selected by stem): both shard indexes, 3.4 s, 10 MiB.
  * **Guardrails (each has a test against a fake Hub with a misbehaving variant):** all requests pinned to the commit SHA, never `main`; a ranged answer must be `206` with a `Content-Range` that starts where asked and
    agrees on the file size every time; a server that ignores Range and offers a large file is refused without reading its body; never more bytes than asked, never more than `--max-index-mb` (default 128) per shard;
    past the cap the report says the inspection is **incomplete** (`scout.remote.index_over_cap`) and nothing more is fetched, it never falls back to a download.
  * **Provenance, labelled by source:** `artifact.source` carries repo, full commit, path(s), licence, gated/private, the repo's **30-day** download count (per repository, not per file), the Hub's own declared architecture
    (a disagreement with the file is reported, and the file wins), and the Hub-published SHA-256 (`published_by_huggingface`: the Hub's claim, not hashed here; absent for split models, where two hashes are not one). `network` lists
    request count, hosts and bytes only.
  * A repo with several models needs `-f` (the choices and sizes are listed, exit 64, and nothing is fetched until you choose); projector files (`mmproj*`) are not candidates; split models are chosen by stem; a split model with a missing shard is refused.
  * `--emit-signature` works on a hosted admitted file: the origin records repo, commit and the Hub's published hash with `sha256_source: published_by_huggingface`, and nothing is hashed locally.
  * Limits: it needs the Hub API to keep its current shape (`sha`, `siblings[].lfs.sha256`, `gguf`, `downloads`); a gated or private repo is reported as access-restricted (the Hub answers 401 for a repo that does not exist, too); remote metadata can never establish numerical parity.* **Nearest admitted structural parent** (`--signatures <dir>` adds your own): scout ranks the file against reference *signatures* of admitted architectures and lists the
  closest three with exact differences (`missing/extra tensor pattern`, `layer coverage`, `rank`, `feature ...`). A signature is **structure only**: tensor-name patterns
  (layer index as `*`), tensor rank, layer coverage, structural feature ids, metadata key names. It deliberately ignores quantization, model size, head counts, file name
  and tensor order, so fine-tunes, merges and re-quantizations of a family land on it. It does **not** see metadata values (rope, norm epsilon, activation, sliding window) or
  the tokenizer, so a match is a lead for a golden run, never proof of equivalent maths. Findings: `arch.structural_parent` (Known: identical to the family it declares),
  `arch.relabel_candidate` (Hypothesis: identical structure under an unregistered name), `arch.nearest_parent` (Hypothesis: closest, with N differences).
  The ranking is a count of differences, not a probability; it has not been calibrated (plan Phase 5).
  * **Structure and origin are separate.** A signature file holds a `structure` (what the family looks like, plus a `structure_id`, the SHA-256 of its canonical
    patterns and features) and a list of `origins` (file name, size, SHA-256, optional repo/revision, scout build). Origins never affect ranking. Several files that
    share a structure are one signature with several origins.
  * **Shipped set:** `src/OpenTail.Stingray.Cli/Scout/Signatures/*.signature.json`, embedded in the binary; only **Admitted** architectures qualify, and a guard test fails if
    an architecture is demoted or a file is stale or hand-edited. Independent of any llama.cpp version or of what is on the user's disk.
  * **Adding a reference** (maintainers or contributors): `stingray scout -m <admitted.gguf> --emit-signature <name>.signature.json [--origin-repo owner/repo --origin-revision <rev>]`.
    It refuses a file that does not resolve to an Admitted architecture, **hashes the file** (cached beside it as `<file>.sha256`, like `stingray hash`), and if the target file
    already holds the same structure it adds the file to its origins; a different structure is refused (use a new name). Load with `--signatures <dir>` without rebuilding.
* Not yet implemented (see plan): the quant picker, demand-ranked backlog and batch triage that build on remote scout; working-set estimates for MLA, hybrid and recurrent families; calibration of the ranking beyond the leave-one-out check.
* Advisory only: nothing in the report admits or promotes an architecture.

### `scripts/scout-pretest.ps1 -Model <gguf> [-Budget 64G] [-Reserve 8G] [-ContextSize 2048] [-Run] [-Golden <golden.json>]`

Opt-in wrapper that chains existing commands and writes one receipt (JSON, file names only, no paths). It implements no inference and decides nothing about admission.

* **Default is a plan:** `scout`, then the memory gate, then what it *would* run. Nothing executes without `-Run`.
* **Stages (same names as scout's):** 0 artifact and 1 static contract (from scout), 2 feasibility (scout's gate; anything but `allowed` stops here as **Blocked**), 3 smoke (`-Run`: a real
  CPU generation for an admitted architecture, otherwise `admit-arch`), 4 internal consistency (always NotRun: no command exposes it), 5 independent reference (`-Run -Golden`: `admit-arch --golden`),
  6 admission readiness (always NotRun: a person decides). `admission` in the receipt is always `not_decided`.
* **A pass needs evidence, not just exit 0:** the smoke stage counts only if the output shows weights were loaded (`Pre-faulted` / `Ran cleanly`), per CLAUDE.md rule 12. A golden failure is split into
  a parity verdict (`NOT YET ADMISSIBLE`, `UNPINNED`...) versus a tool error with no verdict line, because `admit-arch` returns 1 for both.
* **Feedback into the estimator:** the smoke stage records measured peak working set, the estimate, and `estimate_exceeded`. A `true` means the estimator under-counted and needs recalibrating.
* **Non-destructive:** writes only the receipt (`-OutDir`, default temp). It never touches the checkpoint, downloads, edits the repo or changes status. `-ComputeHash` additionally lets `stingray hash` write its usual `<file>.sha256`
  cache beside the model; otherwise an existing cache is only read.
* **One heavy run at a time:** takes the same named mutex as the heavy test suites and `capture-golden` (`Global\OpenTailStingray.HeavyTests`), waiting `-GateWaitSeconds` (default 600) before reporting Blocked.
* **Bounded:** each stage has `-TimeoutSeconds` (default 900); on timeout the whole process tree is killed.
* **Exit codes:** 0 passed or planned only, 1 a stage Failed, 2 Blocked, 64 bad arguments, 66 file missing.
* **Verified 2026-10-09 (real runs, SmolLM2-135M unless noted):** plan only; `-Run` smoke Passed with peak 263 MiB vs estimate 470 MiB; tiny budget on the 1.7B file Blocked with nothing started; golden captured into scratch and
  Passed (`GOLDEN MATCH`); a golden for another architecture Failed as a tool error; Mistral-7B with a 3 s limit Failed as a timeout and left no `stingray` process; with another process genuinely holding the gate it reported Blocked and started nothing, then
  ran normally once released. **Not verified:** Ctrl+C mid-run (PowerShell's `finally` should kill the child and release the gate), and a hard kill of PowerShell itself, which cannot run cleanup and would leave the child running.
* `capture-golden` needs a llama.cpp you provide (`tools/llama.cpp` is git-ignored and not part of this repo); *checking* a recorded golden needs none.
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

**Exit codes (as of 2026-10-09; scripts such as the planned `scout-pretest.ps1` rely on these):** `0` = already allowlisted, ran
cleanly with no reference supplied, full `ADMIT`, or a passing golden (exact / near-tie, pinned); `1` = everything else that is not
a pass (REJECT, divergence, unpinned or mismatched golden, unreadable golden, architecture mismatch, bad `--reference-tokens`);
`66` = model file missing. `verify-goldens` is the same shape: `0` pass, `1` any divergence / guard failure / crash / architecture
mismatch (or, with `--strict`, any skipped / unpinned / near-tie). Neither separates "failed the check" from "the tool errored";
read the printed verdict (or `scout`'s JSON) to tell them apart. Splitting them is deliberately not done here: other callers rely on `1`.

The tokenizer-shape classification and layer-0 inventory it prints live in `ArchitectureTriage` (`src/OpenTail.Stingray.Cli/ArchitectureTriage.cs`),
shared with `scout` so the two cannot drift.

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

