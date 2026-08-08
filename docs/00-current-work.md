# Current work

The active engineering backlog: work that is not yet proven or product-complete. Completed work,
measurements, negative results, and superseded plans live in [done](done).

## The goal that orders this list

**Run any GGUF from Hugging Face.** Breadth of models is the objective; throughput is subordinate
to it. A model that runs slowly is a worse outcome than a fast one, but a model that cannot be
loaded at all is not an outcome. Where a performance item and a coverage item compete, coverage
wins — that is the change in priority as of 2026-08-08, and it is why the ordering below differs
from the previous release-hardening ordering.

Two consequences worth stating plainly, because they cut against the previous roadmap:

- CPU/CUDA kernel performance work is now the **last** local priority, not the third.
- DSpark speculative decoding and SafeTensors Phases 4-6 are **parked**, not scheduled. Both are
  implemented far enough to be useful and neither moves the goal.

## Ordered runway

1. [01 — GGUF model coverage](01-gguf-model-coverage-plan.md) — architectures, IQ quant formats,
   tokenizer pre-types, chat templates. **The goal, restated as work.**
2. [02 — Qwen3.5 MoE / Gated DeltaNet](02-qwen35moe-plan.md) — a large, popular GGUF family whose
   hybrid path exists but is not fully evidenced.
3. [03 — Gemma 4 E4B vision](03-gemma4-e4b-vision-plan.md) — multimodal coverage; blocked on a
   usable reference implementation, see the doc.
4. [04 — configuration and operator quality](04-quality-of-life-improvements-plan.md).
5. [05 — CPU architecture kernel coverage](05-cpu-architecture-kernel-opportunities.md) —
   performance only, except its scalar-fallback-format item, which §2 of plan 01 supersedes.

Work needing hardware this machine does not have is in
[90 — external hardware work](90-external-hardware-work.md). It is not part of the local runway.

---

## Priority 1 — model coverage

See [01-gguf-model-coverage-plan.md](01-gguf-model-coverage-plan.md) for the audit and the ordered
work. The three findings that justify its position at the top:

1. **The architecture gate does not run on the CLI inference path.** `ValidateForTextGeneration` is
   called by `doctor`, `static-plan`, and the server loader — not by `RunCommand`. The CLI will
   attempt architectures the server refuses. `docs/cpu-performance-baseline.md` contains a measured
   CPU baseline for OLMoE, an architecture the gate rejects.
2. **The whole IQ quant family is unimplemented.** `DType` declares eight IQ formats;
   `Dequantize.ToFloat32` implements one (`IQ4_NL`). Large models on Hugging Face are distributed
   predominantly as IQ quants, so this excludes more repos than the architecture list does.
3. **DEFECT FOUND AND FIXED (2026-08-08) — Qwen3 tokenized differently from llama.cpp.** Only
   `tekken` had an explicit pre-tokenizer regex; every other byte-BPE model silently got GPT-2's,
   whatever `tokenizer.ggml.pre` declared. Measured on Qwen3-0.6B against `llama-tokenize` b8585,
   three of five probes diverged: `IT'S`, `(hello)`, `a  b`. Nothing errored —
   `Decode(Encode(s)) == s` still holds when the split is wrong, which is why the existing suites
   never caught it. Fixed by `PreTokenizerPatterns`, a ported pre-type → regex-cascade table; all
   parity rows now pass. Digits look like the obvious discriminator but are not — see the plan.

`CLAUDE.md` overstates architecture support — it lists `deepseek2` and OLMoE, neither of which the
gate admits. Correct it in the same change that resolves the gate.

## Priority 2 — model families already part-built

1. **Qwen3.5 MoE / GDN.** Ornith-1.0 9B exercises the hybrid Gated-DeltaNet path end to end, which
   covers items 1 and 2 of that plan in practice. What remains is GDN state-lifecycle conformance
   coverage and a benchmark once correctness is settled. Use
   [qwen35moe-tensor-layout.md](qwen35moe-tensor-layout.md) as the authoritative layout, never the
   superseded SSM plan.
2. **Gemma 4 E4B vision.** The `gemma4v` mmproj load boundary and fixed-grid preprocessing are
   implemented; the ViT encoder forward, projector, embedding splice, and image parity are not.
   Blocked on an oracle: the local llama.cpp build rejects the paired `gemma4` text GGUF, so it
   cannot serve as a reference. Do not guess the encoder from tensor names.
   Note the 12B is a different, working path (encoder-free `gemma4uv`) and is not blocked by this.

## Priority 3 — operator quality

[04-quality-of-life-improvements-plan.md](04-quality-of-life-improvements-plan.md). Configuration
ownership is partly closed as of 2026-08-08:

- **Done — inventories regenerated from source.** `docs/cli-option-inventory.md` went from 96
  hand-maintained rows to all **149** declared options, produced by the checked-in
  `scripts/gen-cli-option-inventory.ps1` (`-Check` fails when stale). The count guard now also
  asserts the ROW count: the declared count had tracked source correctly the whole time while the
  table silently fell 53 rows behind — it was measuring the one thing not drifting.
- **Done — stale registry entries retired.** Three `KnownEnvironmentVariables` entries were never
  environment variables: `STINGRAY_ARGMAX_NEG_INF` (a CUDA `#define` in an NVRTC kernel string) and
  the glob patterns `STINGRAY_MOE_` / `STINGRAY_SNAPKV`. Registry 159 → 156. A new test requires
  every entry to appear as a **quoted string literal** in `src/`.
- **Still open:** Class/Notes classification of both inventories, and extending source-tracked
  effective configuration beyond static planning knobs. The generator preserves hand-written Class
  values across regeneration, so classification is now safe to start.

## Priority 4 — performance

[05-cpu-architecture-kernel-opportunities.md](05-cpu-architecture-kernel-opportunities.md). All of
it is performance-only; none of it unlocks a model. Its item 3 (native IQ4_NL/MXFP4 kernels) is now
downstream of plan 01 §2 — correctness first, kernels after.

Do not reopen the closed Q4_K repacked-GEMM investigation. Every performance item requires dispatch
proof, interleaved control/candidate samples, named-model end-to-end measurement, and numerical
validation. No single-run result is sufficient.

**Q6_K baseline (2026-08-07).** The checksum-guarded `kernel-bench-cs` harness at `k=8192`,
`rows=512`, `reps=12`, with `DOTNET_TC_QuickJitForLoops=0`, produced independent best times of
0.1676, 0.1760, and 0.1845 ms (checksum `2363.599609`). This replaces the stale 0.2063 ms figure but
is not a new performance claim: a candidate must be interleaved against this implementation in the
same process and beat the observed run-to-run range.

---

## Standing state, not plans

**Behaviour change — tokenization of byte-BPE models (2026-08-08).** Encoding no longer goes through
`CodeGenTokenizer` for any model with a declared `tokenizer.ggml.pre`; `PreTokenizerPatterns` now
supplies the model's own split cascade and our merge loop does the encode. Token IDs for Qwen and
the StarCoder/SmolLM family **change** as a result — they were wrong before, and are now
llama.cpp-identical on every probe measured. Anything that cached token IDs or a prompt-hash across
this change should be invalidated. See [01-gguf-model-coverage-plan.md](01-gguf-model-coverage-plan.md) §3.

**Open question — greedy decode is not bit-deterministic under CPU contention.**
`HotSessionGreedyReplayTests.HotSession_MultiTurn_MatchesFullGreedyReplay_OnRealModel` failed once on
2026-08-08 — turn 2, token 2: the session produced `284 (" and")` where full greedy replay of the
identical 13-token prefix produced `2 ("<|im_end|>")`. It did **not** reproduce: 5/5 clean repeats
plus a full-suite run all pass. The failing run coincided with concurrent CPU load (a
`llama-tokenize` oracle run against a 1.7B model in another process).

This is the **second** sighting of the same thing; `done/upstream-port-status.md` records the first
as "a once-seen flake under CPU contention that never recurred" and set it aside. Two sightings in
the same suite make it worth naming rather than re-forgetting: under greedy sampling the argmax
should not depend on machine load, so something in the CPU path is accumulation-order dependent
(a parallel reduction is the obvious suspect). Until that is understood, a divergence in this suite
is **not** automatically a state-reuse bug, and it is also not automatically noise.

Do not chase it by re-running until green. The useful next step is to determine whether two runs of
the same prefix on a quiet machine produce bit-identical logits, and whether thread count changes
the answer.

**ARCHITECTURE ADMITTED — `olmoe`, 2026-08-08, after fixing a real QK-norm defect.**

- **Bug fixed:** OLMoE-shaped QK-norm took its RMS per head (128 elements) where the model takes it
  over the whole `n_embd` projection (2048) before reshaping into heads. Before the fix, output was
  degenerate. This affects any model whose `attn_q_norm` weight is full-width, since the flag is
  detected from element count. **CUDA and Vulkan still carry the same bug** —
  `HeadNorm(..., perChannel)` was written to the same wrong assumption and needs the matching fix
  plus hardware validation. That is open work, tracked in
  [90-external-hardware-work.md](90-external-hardware-work.md).
- **Admitted on perplexity parity, not greedy parity.** wikitext at a matched 2048-token context:
  llama.cpp 7.4868 vs 7.3889 here (1.3%). Greedy token-for-token parity is NOT achieved and is not
  expected to be — the divergence is at a position whose top five candidates span 1.55 logits.
  A "second structural defect" was claimed here earlier and has been **withdrawn**; see the plan for
  what that claim got wrong.
- `olmo2` remains OUT — no fixture, no receipt.

See [01-gguf-model-coverage-plan.md](01-gguf-model-coverage-plan.md) §1b.

**Known defect — one deliberate red test.**
`ContinuousBatchingTests.PrefillWithCache_Chunked_MatchesFull` fails deterministically across five
runs with byte-identical values (1.86363602 vs 1.5929451). Cause and the declined fix are analysed
in [done/03-cpu-prefill-plan.md](done/03-cpu-prefill-plan.md): the int8 batched prefill path's
microkernel dispatch and accumulation order depend on batch shape, so chunked and single-shot
prefill diverge. Making it batch-shape-invariant would cost the 3.48x measured on that path. It is
left red on purpose — it is the only automated detector for this behaviour.

**Test suite.** ForwardPass discovers 1,358 tests, runs in ~18 min, and has exactly the one failure
above; ~369 skips, of which 327 are "no CUDA device" and 35 are missing model fixtures. Core 488,
Server 261, Cli 367, Sessions 79, TurboQuant 78, Vision 73, Pipeline 52 — all green, zero warnings
under `TreatWarningsAsErrors`. CI floors are set from that measured discovery count. Full record in
[done/upstream-port-status.md](done/upstream-port-status.md).

Never pipe a long test run through `tail`; an 18-minute run whose failure output is discarded costs
another 18 minutes to redo.

**Release.** `<Version>` is 1.0.3 and the release candidate is committed and pushed, but no
`stingray-v1.0.3` tag exists, so nothing is published — nuget.org still has 1.0.2. Publication is
tag-triggered and CI fails if the tag and `<Version>` disagree. The `CHANGELOG.md` Unreleased
section (Mistral/Tekken tokenizer, Jinja fixes, prefix-cache fixes) is **not** in the 1.0.3
packages, which were built from the earlier candidate commit. See
[nuget-release-checklist.md](nuget-release-checklist.md).

**Parked, with working implementations.** Not scheduled; revisit only if the goal changes.
- DSpark speculative decoding — [done/07-dspark-plan.md](done/07-dspark-plan.md). Remaining:
  continuous-batching integration, load-aware verify length.
- SafeTensors Phases 4-6 — [done/08-safetensors-support-plan.md](done/08-safetensors-support-plan.md).
  Remaining: GPU offload, quantized packages, SentencePiece.
- Session-native runtime CLI surface — [done/01-session-native-inference-runtime-plan.md](done/01-session-native-inference-runtime-plan.md).
  The CPU-dense restart lane is proven and server-exposed; there is no CLI named-session command.

## Archive rule

Move a document or section to [done](done) when its outcome is implemented and verified, or when a
measured negative result closes that line of investigation. Add a banner saying what closed and what
carried forward. Keep active documents short: decision, remaining work, acceptance evidence, links.
