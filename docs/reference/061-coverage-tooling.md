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
    modelled (MLA, RWKV) make the estimate `Unknown` with null bytes and a reason, and an unknown estimate is `blocked` under any budget.
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
  * **Hybrid recurrent family (qwen35 / qwen35moe; Gated-DeltaNet layers + full attention), calibrated 2026-10-09** with a model-specific formula (class remarks in `HostMemoryEstimator`). Measured on real CPU runs (`ctx 2048`, `STINGRAY_GC_STATS` peak working set); "filled" = a prompt of about 1,700 tokens, "full" = about 1,950-2,000:

    | Model | File | Short run | Filled | Full | Estimate | Estimate / measured (filled; full) |
    |---|---|---|---|---|---|---|
    | Qwen3.5-0.8B Q4_K_M | 508 MiB | 857 | 1,053 | not run | 1,219 | 1.16 |
    | Qwen3.5-4B Q4_K_M | 2,614 | 3,001 | 3,430 | 3,493 | 3,709 | 1.08; 1.06 |
    | Qwen3.5-9B Q4_K_M | 5,417 | 5,810 | 6,285 | not run | 6,632 | 1.06 |
    | Qwen3.6-35B-A3B Q6_K (MoE) | 27,951 | 28,370 | 28,857 | 28,915 | 29,090 | 1.008; 1.006 |

    What the measurements showed: **no Q4_K repack copy** on this path (peak is the file plus a roughly constant 350-420 MiB, where the dense formula would over-count by nearly 2x); KV only for the full-attention layers (GDN layers allocate no pages: `c4cea419`);
    GDN state is exact and eager (`GdnStateCache`); batched-prefill scratch is `ctx x (3 x hidden + 5 x value + 3 x ffn)` floats plus, for a MoE, `topk x (2 x expertFfn + hidden)` per token, which accounted for the 35B's extra ~160 MiB.
  * **MLA (`deepseek2`), added 2026-10-10.** KV is the expanded fp32 K (key_length) + V (value_length) per head, `layers x heads x (K+V) x ctx x 4`; scratch is the dense formula x1.25; fixed overhead is the 192 MiB base + 96 MiB. Absorbed layouts (split `attn_k_b`/`attn_v_b`, e.g. Kimi-VL) add `layers x kv_lora x heads x (K+V) x 4` (283 MiB on a 27-layer model). Peak working set (`STINGRAY_GC_STATS`, CPU, Q2_K files, one run each): DeepSeek-V2-Lite (file 6132 MiB; the fit) 15 tokens 6399 / 813 tokens 7101 / 1623 tokens 7875 MiB against estimates 6451 / 7168 / 7987 (1.008 / 1.009 / 1.014). Kimi-VL-A3B (file 6270 MiB; slope blind, the fixed absorbed term fitted to its 24-token run) 24 / 785 / 1550 tokens: 6813 / 7550 / 8256 MiB measured against about 6860 / 7578 / 8284. Two checkpoints, one family, Q2_K only, short repetitive prompts: treat it as an upper bound with a 1-2% margin, not a precise figure; a second absorbed-MLA model (DeepSeek-V3-class) would test the absorbed term properly.
* **RWKV7, added 2026-10-10.** No KV and no Q4_K repack copy; terms are weights, 128 MiB base, the fixed-size wkv state (`layers x (embd x head size + 2 x embd) x 4`) and 32 MiB of prefill buffers, flat in context. Peak working set (CPU, Q4_K_M, G1h 1.5B file 1039 MiB / 2.9B file 1959 MiB; 17 / 734 / 1454-token prompts): 1092 / 1152 / 1143 and 2005 / 2060 / 2041 MiB against estimates 1210 and 2138 MiB (1.05x and 1.04x). RWKV GGUFs declare `head_count` 0, so the family is handled before the attention-shaped completeness check. Two models of one generation (RWKV7); RWKV6 is not measured.
    **The MoE margin is thin (0.6% at a full context, 175 MiB); the 8 GiB default reserve is the real slack.** Note the first Qwen3.5-4B/35B "full" attempt and the first Phi attempt below failed (prompt 2,050 tokens against a 2,048 context) and were discarded.
  * **Dense-family MoE re-check:** Phi-3.5-MoE with ~1,800 prompt tokens measured 20,495 MiB against 20,825 estimated (1.016x, the earlier estimate without the expert buffers); the expert prefill buffers (shown to exist on the hybrid MoE) are now counted, giving 21,089 (1.029x). One MoE only; other MoE families remain unmeasured.
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
  * Limits: it needs the Hub API to keep its current shape (`sha`, `siblings[].lfs.sha256`, `gguf`, `downloads`); a gated or private repo is reported as access-restricted (the Hub answers 401 for a repo that does not exist, too); remote metadata can never establish numerical parity.* **Quant picker: `scout -r owner/repo --quants [--max-quants N] [--budget SIZE] [--reserve SIZE] [-c N]`.** Answers "which quantisations of this model can this machine run": it asks the Hub about the repo
  **once**, reads each model file's index remotely (about 2 MiB each, no weights), applies the same analysis and memory gate as `scout`, and prints one row per file, largest first, with a verdict and a `pull` command
  **pinned to the inspected commit and exact file**. The budget defaults to this machine's detected RAM (stated in the output, `budget_source: detected_ram`) and the reserve to the smaller of 8 GiB and a quarter of it; `--budget`/`--reserve` override.
  * Verdicts: `fits` (upper-bound CPU estimate + reserve is within budget), `too big` (a known estimate over budget), `unknown` (no estimate for this family, so nothing is promised either way),
    `unsupported` (a confirmed blocker: an architecture this engine does not run, a storage type with no kernel, or a retired GGML type the loader rejects), `not inspected` (index over `--max-index-mb`, access refused).
  * "Largest that fits" is recommended because within one model a larger quant is usually closer to the original. That is a heuristic, not a measured quality ranking, and the output says so.
  * Verified 2026-10-09 on the live Hub: the 23-quant `bartowski/SmolLM2-135M-Instruct-GGUF` in 14 s, 47 requests, 46.0 MiB received (20 fit, 3 unsupported); it found that the three repacked `Q4_0_4_4/4_8/8_8` files use GGML type IDs 31-33, which this build
    rejects (reported as `unsupported`, with the reason, rather than as a read failure). `ornith-ai/Ornith-1.5-35B-A3B-GGUF` (5 quants, 20-66 GiB) in 12 s, 41 requests, 80.0 MiB received: all 5 `unknown`, because the hybrid recurrent family has no memory model yet; the footer
    says "cannot tell", not "too big".
  * **Limits:** CPU run only (GPU placement is not estimated); MLA (deepseek2) and RWKV families are `unknown` (todo.md). The hybrid recurrent family (qwen35 / qwen35moe, the most downloaded right now) IS modelled since 2026-10-09: on the 35B repo Q8_0 fits (37.0 GiB estimated) and BF16 is too big (67.9 GiB) on this 63 GiB machine; the 31-quant `unsloth/Qwen3.5-9B-GGUF` cost 352 MiB and 177 requests (each index is ~11 MiB because of the 248k-token vocabulary).
    Memory is an upper bound calibrated on one machine. Past `--max-quants` (default 40, largest first) files are skipped with a note.
* **`pull --revision <rev>`.** `pull` now resolves the repo's head to its commit SHA once and fetches every file from that commit (previously it listed at one moment and downloaded from `main` at another). `--revision` fetches a specific commit,
  branch or tag, for example the commit `scout -r` printed, so the file you pull is the file it inspected.* **Load preflight** (`LoadPreflight`, plan P1). The commands that load or install a model ask the same analyzer `scout` runs (so they cannot disagree), against this machine's RAM (reserve = the smaller of 8 GiB and a quarter):
  * `chat` on a CPU run (`--backend cpu` or `--gpu-layers 0`) checks the GGUF's index before loading. A **known** estimate over memory stops it with the numbers and what to do; `--ignore-preflight` loads anyway and prints a warning. An **unknown** estimate (MLA, RWKV) only notes that fit was not checked, so nothing that loads today is broken. GPU runs are not checked: GPU placement is not estimated.
  * `setup` (and a task command offering to install a missing model) reads the **pinned remote index** before asking to download and prints "Fits this machine: about X GiB". A model that will not fit is not downloaded unattended (`--yes` and non-interactive refuse; interactive asks "Download anyway?" defaulting to no).
  * Deliberate deviation from the plan's wording: Unknown warns instead of blocking. Moving the estimator into a library project (so `serve` can share it) is not done yet.* **Nearest admitted structural parent** (`--signatures <dir>` adds your own): scout ranks the file against reference *signatures* of admitted architectures and lists the
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
* **Demand-ranked backlog: `scout --backlog [--limit N]`.** Queries the Hugging Face model list API in one call (`filter=gguf&sort=downloads&direction=-1&limit=N&expand[]=gguf&expand[]=downloads&expand[]=gated`) to identify which GGUF architectures are not yet admitted here, ranked by 30-day downloads.
  * Subject to the external access policy (`STINGRAY_ALLOW_EXTERNAL`, default allowed; `STINGRAY_OFFLINE` / `HF_HUB_OFFLINE` deny).
  * `--limit N`: number of repositories to inspect (default 50, hard cap 200).
  * Downloads are per repository over 30 days, not per file or inference use (stated in output).
  * List-only: never downloads weights or parses full model files.
  * Architectures are grouped and marked `admitted`, `not admitted`, or `unregistered` via `ArchitectureRegistry.Find`. Ported-but-unverified architectures (`NotAdmitted`) stay internal per CLAUDE.md rule 14 and are displayed as `unregistered`.
* Added 2026-10-10: `scout --backlog --triage N` remote-scouts (index only; the smallest quant when a repo has several) the most downloaded repo of each of the top N unregistered families and tabulates the nearest admitted relative and the CPU peak estimate; advisory only (rule 15). Not yet implemented (see plan): working-set estimate for RWKV (MLA done 2026-10-10, below); calibration of the ranking beyond the leave-one-out check.
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

---

## Task favourites: `stingray models use <task> <id>`

Configures persistent per-user default model preferences for tasks (`chat`, `speak`, `transcribe`), avoiding the need to pass `--model <id>` on every command or rely solely on catalogue defaults.

```
stingray models use chat qwen2.5-1.5b          # configure favourite for chat
stingray models use chat --clear               # remove favourite for chat
stingray models                                # lists tasks and marks active favourites
stingray models chat                           # lists chat options and badges (favourite)
stingray models --local                        # shows catalogue models with (favourite) badge
```

### Storage and Configuration
- **File location**: `favourites.json` in the per-user configuration directory:
  - Windows: `%APPDATA%\stingray\favourites.json`
  - Linux / macOS: `$XDG_CONFIG_HOME/stingray/favourites.json` (fallback: `~/.config/stingray/favourites.json`)
- **Override**: `STINGRAY_CONFIG_DIR` environment variable overrides the directory path.
- **Format**: JSON object mapping task name to catalogue model id (e.g. `{"chat": "qwen2.5-1.5b"}`).
- **Atomic updates**: Writes are atomic (writes to temporary file in the same directory and renames). Corrupt files are never silently overwritten; updates throw a named error.

### Resolution Order
When running task-driven commands (e.g. `stingray chat`, `stingray speak`, `stingray transcribe` via `CatalogTaskResolver`):
1. **Explicit file flag**: `--model-file <path>` (highest precedence; points directly to weights on disk).
2. **Explicit model flag**: `--model <id>` (validates against catalogue).
3. **Favourite**: `favourites.json` entry for the task.
4. **Catalogue default**: `ModelCatalog.DefaultFor(task)` (built-in fallback).

### Validation and Named Errors
- Favourites map to catalogue ids only. Setting a favourite validates that `<id>` exists in the catalogue and serves `<task>`.
- **Stale favourite**: If a favourite points to a model id that no longer exists or does not match the task, resolution fails with a named error directing the user to fix or clear it (`stingray models use <task> <id>` or `stingray models use <task> --clear`). It never silently falls back to the catalogue default.
- **Corrupt file**: If `favourites.json` exists but cannot be parsed as valid JSON, resolution fails with a named error. It never silently ignores the file or falls back to the default.

---

## Local Inventory: `stingray models --local [--verify]`

Inspects all locally available models without loading weights or constructing forward passes.

```
stingray models --local                        # scans model home and STINGRAY_MODEL_DIRS
stingray models --local --verify               # re-hashes installed catalogue files against pinned SHA-256
```

### Features & Output
- **Catalogue State**: Lists catalogue entries with their disk status (`Installed`, `Partial`, or `Missing`).
- **Disk Discovery**: Scans the model cache directory and any paths configured in `STINGRAY_MODEL_DIRS`, reading GGUF header/tensor metadata index (`GgufModel.Open`, weights unread).
- **Split-GGUF Deduplication**: Multi-shard GGUFs matching the shard naming pattern (`<stem>-00001-of-00002.gguf`) are grouped into a single logical model entry. The inventory reports the canonical first shard path, combines the file sizes across all shards, and verifies shard completeness. If any sibling shard is missing or damaged, an `InvalidEntry` row names the missing shard rather than silently showing a partial model.
- **Safety Preflight**: Evaluates each discovered model against host RAM via `LoadPreflight.EvaluateFile` to report whether the model fits this machine.
- **Opt-in Hash Verification (`--verify`)**: Re-computes SHA-256 hashes for installed catalogue entries against their pinned catalogue hashes, using `.sha256` sidecars (`ModelFingerprinter`) to avoid redundant work.


