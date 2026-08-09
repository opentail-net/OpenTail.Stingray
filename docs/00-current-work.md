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

**Related but distinct — a scheduling-timing flake, seen once, 2026-08-08.** Post-`gptneox` full
`OpenTail.Stingray.Tests.ForwardPass` run (1368 tests): 2 failures, one the known deliberate red
test (below), the other `ContinuousBatchingConstraintTests.ConstrainedAndUnconstrained_Coexist_PerSequenceMasking`
— `Assert.Equal(2, fake.MaxBatchWidth)` got `1`. Not a model-math bug: this test drives
`ContinuousBatchingEngine` against a `FakeBatchedForwardPass`, so it cannot be affected by the
`RunTrunk`/`PrefillCore`/`Dispose` changes gptneox made — the assertion is purely about whether two
async requests' decode steps land in the same batching tick, which is scheduler-timing-sensitive by
construction. Re-ran in isolation 4/4 clean (0.13s each). Read as thread-pool contention from
running inside a 1368-test parallel suite, not a regression — recorded rather than chased, per the
policy above for the same reason.

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

**ARCHITECTURE ADMITTED — `granite`, 2026-08-08, full 24-token exact greedy match.**

- Needs a "scale trio" (embedding/residual/logit scale) + an attention-scale override, all read
  from GGUF metadata (`ModelHyperparams.ResidualScale`/`AttentionScaleOverride`/`LogitScale` new,
  `EmbeddingScale` generalized). `MiniCPM` (not `MiniCPM3` — that's MLA, unrelated) shares the same
  llama.cpp graph and reuses this implementation once a permissively-licensed checkpoint exists.
- **Found and fixed two real bugs while building the receipt, neither specific to Granite's own
  math:** (1) `PrefillCore` never applied `EmbeddingScale` — latent since Gemma 4 was the only prior
  architecture to set one, and Gemma 4 never reaches `PrefillCore`. (2) The core Jinja engine's
  `ExprParser.ParseArgList` hung/leaked unboundedly (measured to 47 GB RAM) on `key=value` filter
  arguments like `tojson(indent=4)` — Granite's chat template was the first one exercised this
  session complex enough to trigger it. Both fixed; `JinjaChatTemplate` construction is now lazy too
  (built on first access, not at every model load) as defense in depth.
- **NOT wired for Granite/MiniCPM:** TurboQuant prefill, continuous-batching admission
  (`PrefillWithCache`), multi-sequence batched decode, CUDA/Vulkan. A model run through those paths
  today silently skips the scale trio and produces wrong output — same category of gap as OLMoE's
  CUDA/Vulkan QK-norm fix, tracked the same way.

See [01-gguf-model-coverage-plan.md](01-gguf-model-coverage-plan.md) §1d.

**ARCHITECTURE ADMITTED — `smollm3`, 2026-08-08, full 24-token exact greedy match.** One twist over
the plain llama trunk (NoPE every 4th layer, reusing Llama-4's existing gate expression). See
[01-gguf-model-coverage-plan.md](01-gguf-model-coverage-plan.md) §1e.

**ARCHITECTURE ADMITTED — `apertus`, 2026-08-08, 11-token exact prefix (one full sentence), first
new-kernel architecture built this session.** Non-gated FFN (no `ffn_gate` tensor) with xIELU
activation, detected from tensor inventory not architecture string. Two real bugs found and fixed:
xIELU's GGUF-stored parameters are pre-softplus and need a transform that lives in llama.cpp's
`ggml_xielu()` wrapper, not its compute kernel (missing it produced fluent-looking garbage, not an
error); and `PrefillCore`/`DenseFfn`'s non-gated branches disagreed by 3.3 logits at default Q8
settings (same known int8-prefill approximation as OLMoE, amplified by xIELU's quadratic term).
See [01-gguf-model-coverage-plan.md](01-gguf-model-coverage-plan.md) §1g.

**ARCHITECTURE ADMITTED — `gptneox`, 2026-08-09, 22-of-24-token exact match, second new-kernel
architecture built this session.** LayerNorm-with-bias (not RMSNorm), a non-gated biased-GELU FFN,
fused `attn_qkv.weight`/`bias` (contiguous Q/K/V rows — cross-checked against llama.cpp's actual
converter, which contradicted an earlier third-party claim that the layout was interleaved
per-head), and true parallel residual (`x + attn(ln1(x)) + ffn(ln2(x))`, both norms reading the same
pre-layer input) — implemented as an isolated branch in `RunTrunk`/`PrefillCore` so every other
architecture's code path is unchanged. One real defect, caught by the stepwise oracle-free
consistency test rather than by inspection: `PrefillCore`'s final output-norm wasn't updated to the
new bias-aware dispatcher, so prefill-in-one-call and prefill-then-decode agreed on every argmax but
disagreed by 261.6 on raw logit magnitude — fixed, now agrees near-exactly. This was rebuilt from a
from-scratch review after an earlier attempt's work was lost from `src/` mid-session by an unrelated
out-of-band edit; the rebuild independently re-verified every claim against the real llama.cpp
source rather than trusting the lost attempt's own (also since found to be inaccurate) writeup. See
[01-gguf-model-coverage-plan.md](01-gguf-model-coverage-plan.md) §1h.

**ARCHITECTURE ADMITTED — `falcon`, 2026-08-09, full 10-of-10-token exact match including EOS.**
Reuses every `gptneox` mechanism unchanged (LayerNorm-with-bias, non-gated GELU FFN, fused
`attn_qkv.weight`, `UseParallelResidual`'s 3-way sum), plus one new wrinkle: Falcon-7B has no
separate `ffn_norm` tensor at all — attention and FFN read the SAME LayerNorm output (confirmed
against `falcon.cpp`: `build_ffn(attn_norm, ... // !! use the attn norm, not the result`).
`ForwardPass` now falls `_ffnNorm`/`_bFfnNorm` back to `_attnNorm`/`_bAttnNorm` when the tensor is
absent. Caught and fixed one self-inflicted defect before it could ever crash: that aliasing means
`_bFfnNorm[i] == _bAttnNorm[i]`, so `Dispose()`'s unconditional free-both loop would have
double-freed the same allocation the same way the GPT-NeoX receipt's aliased-QKV-bias defect did —
fixed with a pointer-equality guard before ever running the test, not after a crash. Also the first
checkpoint on this profile with MQA (71 query heads, 1 shared KV head) — worked first try on the
existing GQA-parametrized fused-QKV split. Full 1368-test suite regression not re-run for this
smaller, narrower-scope change (targeted OLMoE/PrefillAttentionParityTests/Repro_Pos13Parity/
Granite/GptNeox classes instead, all clean) — see
[01-gguf-model-coverage-plan.md](01-gguf-model-coverage-plan.md) §1i.

**ARCHITECTURE ADMITTED — `olmo2`, 2026-08-09, full 24-of-24-token exact match.** §1c's premise for
this one was wrong — it was listed "gate-only, code exists" alongside `olmoe`; checked directly
against `olmo2.cpp` and found a genuinely different third residual pattern (**post-norm
sandwiching**: no `attn_norm`/`ffn_norm` tensor exists at all, attention and FFN read the raw
residual, norm is applied to each sublayer's OUTPUT before the residual add). Built from three
already-existing mechanisms recombined, not new kernels: the `_wGate[i].DataPtr is null`-style
tensor-presence sentinel (Apertus/GPT-NeoX precedent) now also gates `_attnNorm`/`_ffnNorm`'s
absence; Gemma 4's existing `_postAttnNorm`/`_postFfwNorm` post-norm-before-residual mechanism,
generalized in `ModelGraph.cs` from Gemma-4-only detection to plain tensor presence; and OLMoE's
already-fixed whole-vector QK-norm. One real gap found by reasoning forward before writing the
test, not by a failing assertion: `PrefillCore`'s batched loop has no post-norm step at all
(previously invisible because Gemma 4's own per-layer-head-dim check already routes it away from
`PrefillCore`) — fixed by widening `PrefillDispatch`'s existing per-layer-head-dim sequential
fallback to cover any post-norm model. Targeted regression included `Gemma4CpuForwardPassTests`
specifically (the code path most directly touched by the `ModelGraph.cs` refactor) — clean. See
[01-gguf-model-coverage-plan.md](01-gguf-model-coverage-plan.md) §1j.

**Architecture queue swept for remaining candidates, 2026-08-09 — license-checked four, scope-
corrected two; nothing left immediately buildable.** `glm4`/`glm4moe` and `exaone`: SKIPPED,
restrictive custom licenses (glm-4: registration-gated, PRC jurisdiction; EXAONE: explicitly
non-commercial) — same bucket as `internlm2`/`hunyuan`. `deepseek2`/`minicpm3` (MLA): SKIPPED, both
license-blocked too (DeepSeek's custom use-restricted license; MiniCPM3's weights need a separate
registration-gated license despite Apache-2.0 code) — the MLA lift is now moot regardless of its
size. `gpt-oss`/`bitnet`: NOT license-blocked, but checked against their real llama.cpp sources and
found to need substantially more than the plan's brief note suggested — gpt-oss needs attention
sinks (a real numerical addition to the softmax), alternating sliding-window attention, biased MoE
expert tensors, and an OpenAI-specific SwiGLU/gating variant (five real additions, not the "MXFP4
already dequantizes" framing implied); bitnet needs Sub-LN (an extra norm INSIDE each sublayer) AND
a genuinely new ternary packed-weight format, two independent blockers. See
[01-gguf-model-coverage-plan.md](01-gguf-model-coverage-plan.md) §1c item 6 for the full findings.
Remaining queue: `mamba`/`jamba`/`rwkv` (recurrent, a different forward-pass family entirely —
biggest lift of anything in the plan) and re-checking `bitnet`'s own license (not yet done).

**New-kernel plan drafted, 2026-08-08 — design only except items 3, 1, and now falcon
(Apertus/xIELU, GPT-NeoX, Falcon), all built.** Assessed the six
previously-"unassessed" architectures (Nemotron, Seed-OSS, Hunyuan, Dots1, LFM2, Apertus): none
were buildable-and-testable today (3 restrictive-licensed, 2 have no small checkpoint, only
Apertus is clean on both — Apache-2.0, but only 8B/70B exist). Full plan, grouped by shared
kernel/mechanism so the highest-leverage item is obvious: [01-gguf-model-coverage-plan.md](01-gguf-model-coverage-plan.md) §1f.

**`minicpm` tried and NOT admitted, 2026-08-08 — tokenizer gap, not a forward-pass bug.** The only
Apache-2.0 checkpoint available (`MiniCPM4-0.5B`) uses Unigram-LM SentencePiece (`scores` array, no
`merges` array), a different algorithm from the BPE-order SPM this engine implements. Forward-pass
scale trio is implemented (reuses Granite's) and presumed correct once a compatible checkpoint or a
Unigram-SPM implementation exists. See §1d.

**Known defect — one deliberate red test.**
`ContinuousBatchingTests.PrefillWithCache_Chunked_MatchesFull` fails deterministically across five
runs with byte-identical values (1.86363602 vs 1.5929451). Cause and the declined fix are analysed
in [done/03-cpu-prefill-plan.md](done/03-cpu-prefill-plan.md): the int8 batched prefill path's
microkernel dispatch and accumulation order depend on batch shape, so chunked and single-shot
prefill diverge. Making it batch-shape-invariant would cost the 3.48x measured on that path. It is
left red on purpose — it is the only automated detector for this behaviour.

**Test suite.** ForwardPass discovers 1,368 tests as of the `gptneox` receipt (2026-08-08; was
1,358 — the net growth is Granite/SmolLM3/Apertus/GptNeox parity tests added this session), runs in
~17.5 min (1051s measured directly), and has exactly the two failures above (the one deliberate red
test, plus the scheduling flake, non-reproducing); 375 skips, most "no CUDA device" or missing model
fixtures (fixtures are deleted after each receipt by design — a skip there is expected, not a
coverage gap). Re-confirmed via a full run on 2026-08-09 after the `gptneox` rebuild (RunTrunk/
PrefillCore/Dispose all touched, shared by every dense CPU architecture): same 1368/2-failed/375-
skipped, zero new failures, ruling out a regression from that change. Core 488,
Server 261, Cli 367, Sessions 79, TurboQuant 78, Vision 73, Pipeline 52 — figures not re-verified
this session, zero warnings under `TreatWarningsAsErrors` as of the last full check. CI floors are
set from measured discovery counts. Full record in
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
