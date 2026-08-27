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

## Also active, separate from the GGUF-coverage goal — session architecture migration

Not ordered by "run any GGUF" — a parallel, session/runtime-layer thread.
[028 — InferenceSession → HotSession migration plan](028-inference-session-to-hotsession-migration-plan.md)
is fully done: Phases 1 (KV memory governance), 2 (cross-session prefix sharing), and 3
(fork/branching) are all implemented and verified against real models, each with its own
`HotSession`-native test.

**Remaining**: [030 — delete InferenceSession/InferenceRuntime](030-delete-inferencesession-todo.md).
The plan's own stated end state — once all three phases are done, the superseded
`InferenceSession`/`InferenceRuntime` architecture and its ~20+ tests get deleted, not archived —
has not been executed yet. Doc 030's two gating decisions are now both resolved (2026-08-27, ported
both):
`ISessionMetadata`/`ISessionMetrics` are ported onto `HotSession` (`HotSession.Metadata`/`.Metrics`,
covered by `HotSessionMetricsMetadataTests.cs`), and `HotSessionTurnResult` now carries
`FinishReason`/`ToolCalls` derived from its chunk stream the way `GenerationResult` used to. The
actual deletion pass (doc 030's "How to execute" steps 1–5) has not been run yet.

**Also found (and fixed) while stress-testing this path**: a severe, unrelated prefill-packing
defect affecting real concurrent `HotSession` traffic at 5–15 simultaneous requests, since
resolved — see the entry below and [031](031-concurrent-decode-batch-tier-divergence-bug.md). Was
never a migration-plan defect and never blocked Phases 1-3 (all three are done/verified
independent of this).

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
   [reference/qwen35moe-tensor-layout.md](reference/qwen35moe-tensor-layout.md) as the authoritative layout, never the
   superseded SSM plan.
2. **Gemma 4 E4B vision.** **Updated 2026-08-15** — the `gemma4v` mmproj load boundary, fixed-grid
   preprocessing, ViT encoder forward pass, token-reduction pool, and projector are now all
   implemented (`Gemma4VVisionEncoder.cs`), architecture fully reverse-engineered against the real
   mmproj + local llama.cpp source (not guessed from tensor names — see
   [03-gemma4-e4b-vision-plan.md](03-gemma4-e4b-vision-plan.md)'s current implementation contract).
   Passes a real-mmproj structural sanity test but is **not numerically parity-verified**: the
   local llama.cpp build still rejects the paired `gemma4` text GGUF, so no oracle exists yet for
   end-to-end comparison — that is a parity blocker, not an implementation blocker, and does not
   block the remaining work. Still open: embedding splice into the text decoder, decoder-side
   image-token mask semantics (explicitly NOT assumed to match Gemma 3 — needs its own
   investigation before touching `PagedKvCache`), and CLI/API surface.
   Note the 12B is a different, working path (encoder-free `gemma4uv`) and was never blocked by
   any of this.
3. **Gemma 3 vision (SigLIP).** **New, 2026-08-15** — a separate, simpler ViT family
   (`clip.projector_type=gemma3`, `Gemma3VisionModel.cs`/`Gemma3VisionEncoder.cs`) paired with the
   Gemma 3 4B text model rather than E4B. Both `gemma3` and `gemma4` text architectures are already
   admitted, so once the splice/mask work above lands (shared infrastructure, not
   architecture-specific) this path could reach genuine end-to-end multimodal inference sooner
   than E4B, which needs the same work regardless. Loader and encoder implemented and verified
   against the real `models/mmproj-gemma-3-4b-it-f16.gguf`; structural sanity test passes
   (1/1, 604.9s — attention parallelized across heads since it's ~21x gemma4v's compute at 4096
   patches). Two real, non-obvious findings from this checkpoint's export, documented in
   [03-gemma4-e4b-vision-plan.md](03-gemma4-e4b-vision-plan.md)'s addendum: a different metadata
   key convention (`clip.projector_type`, not `clip.vision.projector_type`) and a NAME-vs-FUNCTION
   swap on the `ffn_up`/`ffn_down` tensors (proven via bias-length evidence, not a storage
   transpose). Same caveat as `gemma4v`: not numerically parity-verified, no oracle available.
4. **Llama 4 vision (E4B Scout/Maverick).** **New, 2026-08-15** — a third ViT family
   (`clip.projector_type=llama4`, `Llama4VisionModel.cs`/`Llama4VisionEncoder.cs`), the only one of
   four researched candidates (`llama4`/`qwen2vl`/`qwen3vl`/`glm4v`) whose paired text decoder
   (`llama4`) is already admitted AND needs no new engine-wide RoPE machinery — the other three all
   require genuine multi-axis M-RoPE, which doesn't exist anywhere in this engine yet (see
   [06-llama4-vision-plan.md](06-llama4-vision-plan.md)'s Context section for the full comparison).
   Structural sanity test passes against the real
   `models/mmproj-llama-4-scout-17b-16e-instruct-f16.gguf` (1/1, 132.1s — one 336x336 tile, 34
   blocks, 577 tokens including a real [CLS] token this checkpoint has that neither `gemma4v` nor
   `gemma3` do). Real findings this time: a flat F16 (not 4D F32) patch-embed tensor, no FFN gate
   tensor at all (plain, not gated, FFN), real pre- AND post-layernorm both present, and — the one
   that would have been a silent wrong-answer bug if missed — a NORM/interleaved 2D-RoPE pairing
   convention, genuinely different from `gemma4v`'s NEOX split-half convention, confirmed only by
   reading `clip.cpp`'s shared `build_rope_2d` helper directly rather than assuming the existing
   `ApplyRope2DHalf` helper would transfer. Multi-tile ("llava-uhd") preprocessing and decoder
   splice are both explicitly out of scope, same precedent as the other two encoders — this
   processes one fixed-square tile per call. llama.cpp's own code separately flags this exact
   projector as known to have degraded quality (ggml-org/llama.cpp#13282), independent of whether
   the port itself is correct. Same caveat as the other two: not numerically parity-verified.

## Priority 3 — operator quality

[04-quality-of-life-improvements-plan.md](04-quality-of-life-improvements-plan.md). Configuration
ownership Phase 0 deliverable 1 (both inventories) is now **DONE, 2026-08-15**:

- **Done — inventories regenerated from source.** `docs/cli-option-inventory.md` went from 96
  hand-maintained rows to all **149** (then, as of the 2026-08-15 pass below, **153**) declared
  options, produced by the checked-in `scripts/gen-cli-option-inventory.ps1` (`-Check` fails when
  stale). The count guard now also asserts the ROW count: the declared count had tracked source
  correctly the whole time while the table silently fell 53 rows behind — it was measuring the one
  thing not drifting.
- **Done — stale registry entries retired.** Three `KnownEnvironmentVariables` entries were never
  environment variables: `STINGRAY_ARGMAX_NEG_INF` (a CUDA `#define` in an NVRTC kernel string) and
  the glob patterns `STINGRAY_MOE_` / `STINGRAY_SNAPKV`. Registry 159 → 156. A new test requires
  every entry to appear as a **quoted string literal** in `src/`.
- **Done, 2026-08-15 — Class classification complete for every row of both inventories.** The
  registry had drifted again since 2026-08-08 (grew back to **162** names across later sessions'
  architecture/kernel work; the doc's own reconciliation prose said 156, which was stale) and the
  table had separately drifted from the registry in both directions (5 rows missing, the same 3
  ghost names from the paragraph above still sitting in the table despite being removed from
  source) — re-diffed to zero drift, then every one of the 162 env-var rows and 153 CLI-option rows
  given an explicit Class, not just the ~half that had one via a summary "ownership register" that
  was never propagated into the actual per-row table. Also found and fixed a real
  `gen-cli-option-inventory.ps1` bug: its description parser only matched single-line
  `[Description("...")]`, so multi-line concatenated descriptions (`"..." + "..."`) silently
  produced blank cells — 3 `RunCommand` rows affected, now populated. Full detail in
  [04-quality-of-life-improvements-plan.md](04-quality-of-life-improvements-plan.md) item 1.
- **Still open:** extending source-tracked effective configuration beyond static planning knobs
  (item 2 of the plan), and removing the obsolete/dead-looking switches classification surfaced —
  that needs a per-variable owner call, not a drive-by deletion, so it wasn't done as part of the
  classification pass itself.

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

**License policy changed, 2026-08-09 (operator decision): architecture code and checkpoint license
are now separate concerns.** The gate itself is a string check against a GGUF's self-declared
architecture — it doesn't distribute or vendor anyone's weights, so the CODE can be built and
admitted regardless of whether the best available checkpoint is permissively licensed. What
changes: for a "bucket 2" architecture (only a restrictively-licensed checkpoint exists), verify
once against a transient, never-vendored local download, then delete BOTH the GGUF and the test
file — no permanently-skipping test referencing a restricted model stays in the tree. The
verification evidence instead lives as a comment on the `ModelCompatibility` allowlist entry plus
a plan-doc section, explicitly flagged "no automated test, licence reason" with a "don't modify
without good reason" warning (there's no regression net for that specific profile — a shared-code
change could silently break it and nothing in CI would notice). Full policy:
[01-gguf-model-coverage-plan.md](01-gguf-model-coverage-plan.md), "License policy: code vs.
checkpoint" (below the standing evidence rule).

**ARCHITECTURE ADMITTED — `exaone`, 2026-08-09, full 24-of-24-token exact match — first bucket-2
admission under the new policy.** Genuinely gate-only (unlike `olmo2`'s false-positive premise) —
confirmed against `exaone.cpp` before downloading anything: ordinary pre-norm llama-style trunk,
already in `isNeoxRope`. The one thing worth checking: this checkpoint's top-level `rope_freqs.weight`
(llama3.1-style per-dimension frequency correction) is already read generically by `ForwardPass`
(built for Gemma 4, detected by tensor name/shape, not architecture-gated) — zero new code needed.
No test persists in the tree; the verification evidence (prompt tokens, reference continuation) is
recorded as a comment on the `"exaone"` allowlist entry in `ModelCompatibility.cs` and in
[01-gguf-model-coverage-plan.md](01-gguf-model-coverage-plan.md) §1k.

**`internlm2` re-checked 2026-08-09, still NOT admitted — turned out to be blocked on the TOKENIZER
axis, not architecture or license.** The 2026-08-08 assessment said "already fully supported, zero
tokenizer work" based on `tokenizer.ggml.model: llama`, but didn't check for a merges array — this
checkpoint has `tokenizer.ggml.scores` with NO merges at all, the same Unigram-LM SentencePiece gap
already blocking `minicpm`. Measured directly: `Encode("The capital of France is")` fell through to
one-token-per-character output instead of the reference ids. Architecture code is still presumed
correct (confirmed against `internlm2.cpp`: identical shape to `exaone`) but unverifiable without a
working Unigram-LM tokenizer — fixing that shared gap would unlock both architectures at once.

**ARCHITECTURE ADMITTED — `starcoder2`, 2026-08-09, full 24-of-24-token exact match — second
bucket-2 admission, and it found a real bug.** Reuses `gptneox`/`falcon`'s LayerNorm-with-bias +
non-gated GELU FFN infrastructure exactly, but with the ORDINARY sequential residual, not their
parallel 3-way sum. Confirmed a latent bug flagged (but not yet triggered) back in the `gptneox`
receipt: `RunTrunk`'s sequential FFN pre-norm still called `FastRmsNorm` instead of the bias-aware
`FastNorm`, because no prior architecture combined `HasNormBias=true` with sequential (non-parallel)
residual — `starcoder2` was the first. Symptom was unambiguous: the engine's own prefill and decode
disagreed with EACH OTHER (maxDiff 30.4, argmax mismatch), not just with llama.cpp — a structural-bug
signature, not numerical noise. Fixed, regression-checked, clean. No test persists (BigCode
OpenRAIL-M license); evidence recorded on the `"starcoder2"` allowlist entry and in
[01-gguf-model-coverage-plan.md](01-gguf-model-coverage-plan.md) §1l.

**`mistral` (classic Mistral-7B/Mixtral family) confirmed already fully covered, 2026-08-09 — zero
new work.** These checkpoints declare `general.architecture: llama`, already admitted. The newer
`mistral3`/`mistral4` arch strings are separate (likely vision-multimodal Pixtral-family variants),
not assessed.

**ARCHITECTURE ADMITTED — `cohere2` (Command-R7B), 2026-08-09, 1-token exact match plus a
documented 0.0655-logit near-tie — the tightest accepted this session, and three genuinely new
mechanisms built.** Confirmed against `cohere2.cpp` before writing code: (1) LayerNorm WITHOUT a
learned bias (`SimdKernels.LayerNorm`'s bias param is now null-safe; new
`ModelHyperparams.UsesLayerNorm` decouples "use LayerNorm math" from `HasNormBias`); (2) generic
(non-Gemma4-gated) sliding-window attention, 3 local + 1 global layers, added to `ModelGraph.cs`
using the exact formula from `llama-hparams.cpp`'s `set_swa_pattern`; (3) RoPE applied ONLY on SWA
layers — global layers get none at all, confirmed by `cohere2.cpp` having no `else` branch on its
`if (is_swa)` rope call, the opposite rule from Llama-4/SmolLM3's period-based NoPE. Also found:
`PrefillCoreAttention` has no sliding-window-masking parameter at all (only ever needed by Gemma 4,
which is always routed away before reaching it) — fixed by widening `PrefillDispatch`'s sequential
fallback to cover any SWA model without per-layer head dims, reusing `RunTrunk`'s already-correct
`Attention()` windowing instead of teaching `PrefillCore` SWA masking. `logit_scale` also uses the
OPPOSITE convention from Granite's (direct multiply, not reciprocal). No test persists (CC-BY-NC-4.0
license); evidence recorded on the `"cohere2"` allowlist entry and in
[01-gguf-model-coverage-plan.md](01-gguf-model-coverage-plan.md) §1m.

**`ernie4_5` (dense) assessed 2026-08-09 — architecturally trivial AND genuinely Apache-2.0
(bucket-1, not even bucket-2!), but blocked on the SAME Unigram-LM tokenizer gap as `minicpm`/
`internlm2`.** Confirmed against `ernie4-5.cpp`: identical shape to `exaone`, zero new code needed.
`baidu/ERNIE-4.5-0.3B-PT` (386 MB, the smallest checkpoint used all session) has
`tokenizer.ggml.scores` with no merges array — confirmed the practical symptom directly
(character-fragment tokenization, same failure signature as `internlm2`). **Unigram-LM
SentencePiece now confirmed blocking THREE architectures** — the highest-leverage remaining
tokenizer-axis gap in the plan, one implementation away from three architecture receipts. Not
attempted (tokenizer axis, not architecture — outside the standing priority), but flagged clearly
for whenever that priority shifts.

**ARCHITECTURE ADMITTED — `glm4` (non-multimodal), 2026-08-09, 14-of-24-token exact prefix then a
0.0214-logit near-tie — the deepest/tightest near-tie accepted this session, bucket-1 (genuinely
MIT), two real defects found and fixed.** Much smaller in scope than §1c's "conditional RoPE"
estimate: that MRoPE path only applies to the multimodal branch (confirmed against `glm4.cpp`); a
text-only checkpoint needs ordinary RoPE plus a fused gate+up FFN tensor (`ffn_up` at double
width, no separate `ffn_gate` — confirmed via `ggml_vec_swiglu_f32`'s actual math: first half=gate,
second half=up), split by byte offset the same way GPT-NeoX's fused QKV already is. Two real
defects: (1) the fused-tensor split gave each slice the FULL tensor's `GgufTensorInfo` instead of a
correctly-sized one, and `PrefaultWeights` sizes its read range from that Info — the second slice's
prefault range ran off the end of the mmap for the last layer, measured directly as
`AccessViolationException`, and fixed retroactively for the pre-existing (never-triggered but
identical) GPT-NeoX/Falcon fused-QKV case too; (2) partial RoPE had no implementation at all for the
"normal"/non-NEOX rotation convention this checkpoint uses (`rope.dimension_count=64` of
headDim=128) — only the NEOX variant existed — fixed with a new `SimdKernels.ApplyRoPECachedPartial`
kernel. Also caught a wrong checkpoint first try (`glm-4-9b-chat-GGUF` declares the legacy `chatglm`
architecture, not `glm4`) before it wasted the receipt. Unlike every other new-kernel receipt this
session, the parity test PERSISTS (`Glm4GreedyParityTests.cs`) since the checkpoint used
(`THUDM/GLM-4-9B-0414`) is genuinely MIT, not registration-gated like the original GLM-4-9B-Chat
family. See [01-gguf-model-coverage-plan.md](01-gguf-model-coverage-plan.md) §1n.

**Architecture queue swept for remaining candidates, 2026-08-09 — checkpoint license findings from
before the policy change still stand as fact for the checkpoints they were about, just no longer as
a hard blocker; `glm4` itself turned out to have a genuinely MIT checkpoint anyway (see above).**
`glm4moe`'s checkpoint status is unassessed, `deepseek2`'s (custom
use-restricted license), and `minicpm3`'s (registration-gated weight license despite Apache-2.0
code) are all still not MIT/Apache-2.0/BSD/MPL — but under the new policy that no longer rules the
architectures out, only means any future receipt for them would be bucket-2 (verify once, no
persisted test), same as `exaone`. Not built yet: `glm4`/`glm4moe` needs real, contained new kernel
work (conditional/multi-section RoPE); MLA (`deepseek2`/`minicpm3`) is the biggest single lift in
the plan, deliberately deprioritized behind smaller wins. `gpt-oss`/`bitnet`: NOT license-blocked
(gpt-oss's checkpoint is Apache-2.0 already), but checked against their real llama.cpp sources and
found to need substantially more than the plan's brief note suggested — gpt-oss needs attention
sinks (a real numerical addition to the softmax), alternating sliding-window attention, biased MoE
expert tensors, and an OpenAI-specific SwiGLU/gating variant (five real additions, not the "MXFP4
already dequantizes" framing implied); bitnet needs Sub-LN (an extra norm INSIDE each sublayer) AND
a genuinely new ternary packed-weight format, two independent blockers. See
[01-gguf-model-coverage-plan.md](01-gguf-model-coverage-plan.md) §1c item 6 for the full findings.
Remaining queue: `mamba`/`jamba`/`rwkv` (recurrent, a different forward-pass family entirely —
biggest lift of anything in the plan) and re-checking `bitnet`'s own license (not yet done).

**MLA "biggest single lift" claim VERIFIED against real source, 2026-08-09 — confirmed, not
revised.** Read `examples/llama.cpp/llama.cpp/src/models/deepseek2.cpp` (721 lines) in full,
checking whether this session's repeated pattern of "initial estimate was too pessimistic" also
applied here. It does not: five genuinely new mechanisms confirmed (query nope/rope split with an
absorption matmul into the compressed latent space, post-attention output decompression via a
`wv_b` weight the attention op has no parameter for today, a compressed-latent KV cache page
layout, a DeepSeek-specific YaRN `mscale` correction, and the leading-dense-block MoE toggle also
needed independently by `dots1`). Still deprioritized behind smaller wins; see §1f item 4 for the
full breakdown.

**ARCHITECTURE ADMITTED — `stablelm`, 2026-08-09, 4-of-24-token exact prefix then a 0.0207-logit
near-tie on a near-lossless Q8_0 checkpoint, bucket-2, smallest code change of any new-kernel
architecture this session.** LayerNorm-with-bias, non-gated-FFN plumbing, and NEOX partial rope
were all already generic (built for `gptneox`/`falcon`/`glm4`), so `stablelm-2-zephyr-1_6b` needed
none of them as new work. The one real finding: this checkpoint's GGUF carries a stale
`stablelm.use_parallel_residual=true` metadata key that `stablelm.cpp`'s graph builder never
actually reads — the real sequential-vs-parallel choice is made by branching on whether the
per-layer `ffn_norm` TENSOR exists, and this checkpoint has real `ffn_norm` tensors on every layer
(genuinely sequential, despite the metadata saying true). Reusing the pre-existing
`GetBool(metadata, "{arch}.use_parallel_residual")` fallback (written for `gptneox`, where the key
genuinely is consulted) would have silently taken the wrong branch — fixed by deriving
`UseParallelResidual` from `blk.0.ffn_norm.weight` tensor presence for `stablelm` specifically.
Bucket-2 (Stability AI "other" license, non-commercial/gated) — no persisted test; evidence
recorded on the `"stablelm"` allowlist entry and in
[01-gguf-model-coverage-plan.md](01-gguf-model-coverage-plan.md) §1o.

**ARCHITECTURE ADMITTED — `hunyuan-dense`, 2026-08-09, FULL 24-of-24-token exact greedy match
(deterministic, on a degenerate un-templated-prompt reference), bucket-2.** The only family from the
user's flagship-family list with no prior attempt. `llama_model_hunyuan_dense` turned out to be a
5-line stub that inherits its entire `load_arch_hparams`/`load_arch_tensors`/graph wholesale from
`llama_model_hunyuan_vl` (confirmed in `models.h` first) — an ordinary RMSNorm pre-norm trunk,
standard GQA, SiLU-gated FFN, no MoE, and `use_mrope()` never triggers for a text-only dense
checkpoint (no `rope.dimension_sections` key). One genuinely new mechanism: weighted QK-norm applied
AFTER RoPE, not before — this engine had Qwen3's "weighted before RoPE" and Llama-4's "unweighted L2
after RoPE" but no "weighted after RoPE" combination; added `ModelHyperparams.QkNormAfterRope`,
wired into `PrefillCore`/`RunTrunk` only (the paths a plain receipt exercises — `PrefillCoreTq`/
`BatchVerify`/`BatchForwardMulti` still apply the Qwen3 timing unconditionally and would need the
same fix before running this architecture). Also needed a new tokenizer pre-type:
`tokenizer.ggml.pre=hunyuan-dense` is a DISTINCT llama.cpp pre-type from the plain `"hunyuan"`
already in this engine's table (which is actually the Qwen-2 cascade) — added a 3-stage cascade
(`PreTokenizerPatterns.DigitRun3`/`Cjk`/`HunyuanDenseTail`) shared with `deepseek3-llm`/`joyai-llm`,
verified against `llama-tokenize` before writing any forward-pass code. The reference completion
itself degenerates (raw, un-templated prompt against an Instruct-tuned checkpoint) to a repeated
token — this engine reproduces the identical repeated sequence for all 24 positions, still valid
token-for-token parity evidence. Regression-checked against the full `Tests.ForwardPass` suite (995
passes, 2 failures, both confirmed unrelated: `PrefillWithCache_Chunked_MatchesFull` reproduces
identically on unmodified `HEAD` via `git stash` [FP-accumulation-order sensitivity, unrelated
SmolLM2 fixture], `ConstrainedAndUnconstrained_Coexist_PerSequenceMasking` passed 3/3 in isolation
[load-sensitive concurrency flake]) and `Tests.Core` (480 passes, 0 failures — the new pre-tokenizer
cascade doesn't disturb any existing pre-type). Bucket-2 (Tencent Hunyuan Community License) — no persisted test; evidence
recorded on the `"hunyuan-dense"` allowlist entry and in
[01-gguf-model-coverage-plan.md](01-gguf-model-coverage-plan.md) §1p.

**ARCHITECTURE ADMITTED — `gpt2`, 2026-08-09, FULL 22-of-22-token exact greedy match, bucket-1
(genuinely MIT).** With the flagship-family list fully worked through, picked by scanning every
`examples/llama.cpp/llama.cpp/src/models/*.cpp` file by line count and taking the smallest genuinely
unassessed candidate (`gpt2.cpp`, 148 lines). The first architecture this session — this whole
plan's entire scope until now — with NO RoPE at all: GPT-2 encodes position via a learned absolute
position-embedding table (`position_embd.weight`) added to the token embedding once, before the
trunk starts, not via rotary embeddings inside attention. This engine had zero prior support for
that. Added `ForwardPass._posEmbdTensor` (loaded only when the tensor exists) and threaded a
`position` parameter through `EmbedTokenInto`/`EmbedToken` and all 8 of their call sites (every
prefill/decode dispatch path) — mechanical plumbing, since every call site already had the position
value in scope. Disabling RoPE needed NO new field at all: `ModelHyperparams.NoRopeLayerStep = 1`
makes the EXISTING Llama-4/SmolLM3 periodic-skip formula (`(layer+1) % step != 0`) evaluate to
"never" for every layer, reusing dispatch every call site already had. Everything else
(LayerNorm-with-bias, fused `attn_qkv.weight`/`.bias`, non-gated biased-GELU FFN) was already
generic from `gptneox`/`falcon` — zero new code. A Q6_K quant tried first diverged at position 5 on
only a 0.106-logit gap; rather than accept a near-tie on the very first receipt for a brand-new
mechanism, re-ran against a near-lossless F16 checkpoint of the same base model instead, which
matched FULLY with no near-tie at all — confirming the quantization explanation rather than assuming
it. Checkpoint: `openai-community/gpt2` (124M), genuinely MIT (confirmed via the HF API's
`cardData.license`). Permanent test (bucket-1) — `Gpt2GreedyParityTests.cs` — and evidence also
recorded on the `"gpt2"` allowlist entry and in
[01-gguf-model-coverage-plan.md](01-gguf-model-coverage-plan.md) §1q.

**ARCHITECTURE ADMITTED — `granitemoe`, 2026-08-09, FULL 24-of-24-token exact greedy match,
bucket-1, essentially a free admission.** `llama_model_granite_moe::graph` is a type alias for
`llama_model_granite::graph` (confirmed in `models.h` before writing any code) — the SAME graph as
dense Granite (already admitted 2026-08-08). This engine's generic MoE dispatch and the
Granite-family scale block (`isGraniteFamily` in `ModelGraph.cs`) already explicitly checked
`arch=="granitemoe"` from when the dense receipt was built, but had never been exercised until now.
Standard softmax gating, no shared expert, standard GQA — every mechanism this checkpoint exercises
was already correct; the only failure along the way was a wrong test assertion (`LogitScale`
already carries the reciprocal of the raw metadata value, momentarily forgotten), not an engine
defect. Checkpoint: `ibm-granite/granite-3.0-1b-a400m-instruct` (1B total/400M active MoE),
genuinely Apache-2.0. Permanent test (bucket-1) — `GraniteMoeGreedyParityTests.cs` — and evidence
also recorded on the `"granitemoe"` allowlist entry and in
[01-gguf-model-coverage-plan.md](01-gguf-model-coverage-plan.md) §1r.

**ARCHITECTURE ADMITTED — `olmo` (v1), 2026-08-09, FULL 24-of-24-token exact greedy match on the
first real attempt, bucket-1 (genuinely Apache-2.0, AI2).** Predecessor to the already-admitted
`olmo2`/`olmoe`. One genuinely new mechanism: LayerNorm with NEITHER a learned scale NOR a bias at
all (confirmed against `olmo.cpp`: every `build_norm` call passes both weight and bias as `NULL`,
no `attn_norm`/`ffn_norm`/`output_norm` tensor exists in the GGUF at all) — a THIRD norm shape
distinct from weighted LayerNorm-with-bias (`gptneox`/`falcon`/`gpt2`) and bias-less-but-weighted
LayerNorm (`cohere2`). A missing norm tensor already meant something else here (`olmo2`'s "skip
normalizing entirely, sandwich-normed on the output instead"), so this needed a genuine arch-string
check (`ModelHyperparams.UsesUnweightedNorm`) rather than a generalized tensor-presence rule. Added
`SimdKernels.PureLayerNorm` and wired it into `RunTrunk`'s three norm points ahead of the existing
null-DataPtr-means-skip check; `PrefillCore`'s batched norm steps were not taught this third mode,
routed to the sequential path instead via a new `unweightedNormUnsupported` flag in
`PrefillDispatch`'s fallback gate (same established pattern `olmo2`/`cohere2`/Gemma-4 already use).
Everything else (plain MHA, standard RoPE, SiLU-gated FFN, tied embeddings) was already generic.
Checkpoint: `allenai/OLMo-1B-hf` (1.25 GB Q8_0). Permanent test (bucket-1) —
`OlmoGreedyParityTests.cs` — and evidence also recorded on the `"olmo"` allowlist entry and in
[01-gguf-model-coverage-plan.md](01-gguf-model-coverage-plan.md) §1s.

**ARCHITECTURE ADMITTED — `starcoder` (v1), 2026-08-09, FULL 23-of-23-token exact greedy match on
the first real attempt, bucket-2, near-zero code change.** Noticed while building `gpt2` that
`starcoder.cpp`'s graph is structurally identical — same absolute position embeddings, LayerNorm-
with-bias, fused `attn_qkv.weight`/`.bias`, non-gated biased-GELU FFN. The only change needed:
widening `gpt2`'s `NoRopeLayerStep=1` gate from a single-arch check to `arch is "gpt2" or
"starcoder"`. Also exercises MQA (`head_count_kv=1`) through the already-generic fused-QKV split
(first proven on `falcon`). Checkpoint: `bigcode/starcoderbase-1b` (BigCode OpenRAIL-M — restricted
use, not permissive) — bucket-2, no persisted test; evidence recorded on the `"starcoder"`
allowlist entry and in [01-gguf-model-coverage-plan.md](01-gguf-model-coverage-plan.md) §1t.

**`xverse` TRIED and BLOCKED, 2026-08-09 — a genuine tokenizer-axis defect, NOT an architecture
problem.** `xverse.cpp` is a literal plain-llama clone (zero new code needed), but the checkpoint's
GGUF has NEITHER `tokenizer.ggml.merges` NOR `tokenizer.ggml.scores`, despite declaring
`tokenizer.ggml.model=llama`. Root-caused: llama.cpp's own SPM loader defaults every score to
`0.0f` when the array is absent and still tokenizes correctly (measured: a sensible 7-token
result); this engine's SPM path produces dramatically fragmented (near byte-level) output on the
same input. A THIRD distinct tokenizer-axis gap, separate from the Unigram-LM issue blocking
`minicpm`/`internlm2`/`ernie4_5` — likely a smaller, more contained fix (an existing code path
mishandling absent scores, not a whole unimplemented algorithm). Not investigated further per the
standing architecture-first priority; GGUF deleted, allowlist entry reverted. See
[01-gguf-model-coverage-plan.md](01-gguf-model-coverage-plan.md) for the full finding.

**ARCHITECTURE ADMITTED — `codeshell`, 2026-08-09, FULL 24-of-24-token exact greedy match,
bucket-2, genuinely zero new production code.** Same LayerNorm-with-bias/fused-QKV/non-gated-GELU-
FFN shape as `gptneox`/`falcon`/`starcoder`, but uses REAL RoPE (NEOX convention, already in this
engine's `isNeoxRope` list from an earlier session pass) rather than `gpt2`/`starcoder`'s absolute
position embeddings — needed neither the `NoRopeLayerStep` widening nor any new RoPE work. The only
failure along the way was a wrong test assertion (assumed NORM rope, codeshell is genuinely NEOX),
not an engine defect — same "test bug, not engine bug" pattern as `granitemoe`. Checkpoint:
`WisdomShell/CodeShell-7B` (custom license, not permissive) — bucket-2, no persisted test; evidence
recorded on the `"codeshell"` allowlist entry and in
[01-gguf-model-coverage-plan.md](01-gguf-model-coverage-plan.md) §1u.

**`baichuan`/`orion` CHECKED and BLOCKED, 2026-08-09 — same Unigram-LM tokenizer gap as
`minicpm`/`internlm2`/`ernie4_5`.** Checked via an HTTP range-request partial download (first 8 MB
only, enough for the metadata section) instead of pulling either multi-GB checkpoint in full — both
declare `tokenizer.ggml.scores` with no `merges` key. Architecture side was never the blocker
(`baichuan`'s 7B variant is a literal zero-code plain-llama clone; `orion` needed only a small new
LayerNorm-with-bias + gated-SiLU-FFN combination). Neither downloaded in full.

**`arcee` CHECKED and BLOCKED, 2026-08-09 — YaRN RoPE scaling, a substantial unimplemented
mechanism, not the "easiest" item it first looked like.** Built and then FULLY REVERTED a ReLU²
activation kernel + `ModelHyperparams.UsesReluSquared` + both FFN call-site wirings once
`arcee-ai/AFM-4.5B`'s real metadata showed `rope.scaling.type=yarn` (factor 20, confirmed on both
the Instruct and Base releases via the same partial-download trick) — this engine has no YaRN
implementation at all (NTK-by-parts frequency interpolation + an attention/mscale factor,
comparable in complexity to the YaRN piece already flagged in MLA's scope), and it affects every
position, not just long-context generation, so even a short greedy-parity probe would exercise the
wrong math. Reverted completely (confirmed via grep) rather than leave inert code for an
architecture never added to the allowlist. Both checkpoints deleted. See
[01-gguf-model-coverage-plan.md](01-gguf-model-coverage-plan.md) for the full finding and the
ReLU² design notes, kept so a future YaRN implementation doesn't need to re-derive them.

**ARCHITECTURE ADMITTED — `jais2`, 2026-08-09, FULL 3-of-3-token exact greedy match (including a
natural EOS stop), bucket-2 — the `arcee` ReLU² work found a real home.** LayerNorm-with-bias,
separate biased Q/K/V/output projections, and standard NEOX RoPE were all already generic. The one
new piece — non-gated FFN with ReLU-squared activation — is the EXACT mechanism reverted for
`arcee` earlier this session (`arcee` also needed YaRN, `jais2` needs nothing else new), so the
kernel design carried over cleanly: `SimdKernels.ReluSqrInPlace` + `ModelHyperparams.
UsesReluSquared`, wired into both `DenseFfn` and `PrefillCore`'s non-gated-FFN branches. Also
needed a new pre-tokenizer regex (`PreTokenizerPatterns.Jais2`, registered under `"jais-2"`) —
Llama-3's pattern with the trailing whitespace alternative replaced by a cascading fixed-length run
(512, 256, ..., 1), ported directly from `llama-vocab.cpp`. Checkpoint: `yoriis/JAIS2-IT-0.3` (a
third-party fine-tune of the gated `inceptionai/Jais-2-8B-Chat`) — bucket-2, no persisted test;
evidence recorded on the `"jais2"` allowlist entry and in
[01-gguf-model-coverage-plan.md](01-gguf-model-coverage-plan.md) §1v.

**`plamo3` CHECKED and BLOCKED, 2026-08-09 — a FOURTH distinct tokenizer-axis gap, not an
architecture problem.** `plamo3.cpp` needed only one real architecture-side piece (cohere2-style
scalar-period SWA combined with Gemma-4-style dual-frequency RoPE, since PLaMo-3 runs RoPE on
every layer at two different bases rather than skipping it on global layers like cohere2) — built,
then reverted once `tokenizer.ggml.model=plamo2` was found: a genuinely distinct llama.cpp vocab
TYPE (`LLAMA_VOCAB_TYPE_PLAMO2`), not a pre-type variant within SPM or BPE. This engine has no
support for it at all — falls through to byte-BPE with no merges array, the same fragmentation
signature as `xverse`/`baichuan`/`orion` but from a different root cause. Reverted the
`ModelGraph.cs` SWA/RoPE work completely (confirmed via grep) rather than leave it unverified and
unreachable, matching `arcee`'s precedent. Checkpoint deleted without ever being tested. See
[01-gguf-model-coverage-plan.md](01-gguf-model-coverage-plan.md) for the full finding and the
architecture-side design, kept so a future PLaMo2-tokenizer implementation doesn't need to
re-derive it.

**ARCHITECTURE ADMITTED — `maincoder`, 2026-08-09, FULL 24-of-24-token exact greedy match,
bucket-1, genuinely zero new code.** A literal Qwen3-shaped architecture — RMSNorm, biasless GQA
with weighted per-head QK-norm before RoPE, standard SiLU-gated FFN, standard interleaved RoPE —
confirmed against `maincoder.cpp` before writing any code. `tokenizer.ggml.pre=qwen2` already
covered. Picked after `talkie.cpp` (checked first) turned out to need an asymmetric QK-norm,
unweighted RMSNorm everywhere, and a novel embedding-skip residual topology, on top of only a 13B
checkpoint existing — deprioritized in favor of `maincoder`'s much smaller lift and genuinely small
1B checkpoint. Checkpoint: `Maincode/Maincoder-1B` (1.1 GB Q8_0), genuinely Apache-2.0. Permanent
test (bucket-1) — `MaincoderGreedyParityTests.cs` — and evidence also recorded on the
`"maincoder"` allowlist entry and in
[01-gguf-model-coverage-plan.md](01-gguf-model-coverage-plan.md) §1w.

**`nanbeige` CHECKED and BLOCKED, 2026-08-09 — two independent problems, found only after the
architecture work was already built and compiled clean.** Needed a genuinely new mechanism —
Universal-Transformer-style "weight looping" (`num_loops=2` on the real checkpoint, physical
layers reused with a periodic `output_norm` insertion) — built via a clean `LoopedTensorSource :
IModelTensorSource` decorator that transparently rewrites `blk.{i}...` lookups for looped layers,
letting `ForwardPass`'s huge per-layer constructor loop stay completely unchanged (avoided a much
more invasive mechanical rewrite of dozens of call sites). Blocked on: (1) the SAME Unigram-LM
tokenizer gap already blocking `minicpm`/`internlm2`/`ernie4_5`/`baichuan`/`orion` (now six
architectures), and (2) independently, this session's local `tools/llama.cpp` reference binary
doesn't even recognize `nanbeige` as an architecture — the source tree read from is newer than the
compiled binary, so no reference exists at all right now regardless of the tokenizer. Reverted
completely (confirmed via grep), matching `arcee`/`plamo3`'s precedent. Checkpoint deleted without
ever being verified. See [01-gguf-model-coverage-plan.md](01-gguf-model-coverage-plan.md) for the
full finding and the `LoopedTensorSource` design, kept as the pattern for any future
weight-looping architecture once both blockers clear.

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

**FIXED, 2026-08-14 (was: known defect, SEVERE, core path).** A real-model concurrency stress test
(N identical, deterministic greedy `HotSession` requests, all must produce the same output) found
5 ≤ N ≤ 15 diverging reproducibly. The original hypothesis (`SimdKernels.MatMulBatched`'s
small-batch tiered dispatch, batch-composition-dependent in the exact-F32 DECODE path) was
disproven by a controlled raw-API diagnostic — decode alone at N=8 showed zero divergence. The
actual mechanism was in **prefill**: `ContinuousBatchingEngine.RunPrefillStep` packs multiple
unrelated sessions' prompts into one combined batch via `ForwardPass.PrefillPackedMulti`, and
`SimdKernels.MatMulBatched`'s OpenBLAS-vs-tiered kernel choice was gated on that *combined* batch
size (`N >= MinBatchForBlas`, default 16) with no awareness that `N` spans multiple independent
prompts — so a short prompt's own prefill numerics silently depended on how many other sessions
happened to be packed alongside it. Fixed with a new `allowBlas` parameter (default `true`) on
`SimdKernels.MatMulBatched`/`ForwardPass.MatMulBatchedCached`/`MatMulBatchedDualCached`;
`PrefillPackedMulti`'s six matmul call sites now pass `allowBlas: false`. All five tests in
`HotSessionConcurrencyStressTests.cs` are green; full regression (`Tests.ForwardPass.Fast`,
`Tests.Sessions.Fast`, `Tests.Server.Fast`, `Tests.TurboQuant`, `Tests.Cli`, `Tests.Vision`,
real-model `ContinuousBatchingTests`) passes except three pre-existing, unrelated failures
confirmed (via `git stash` bisection against the pre-fix baseline) to be unaffected by this
change. Full investigation and resolution:
[031-concurrent-decode-batch-tier-divergence-bug.md](031-concurrent-decode-batch-tier-divergence-bug.md).

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
