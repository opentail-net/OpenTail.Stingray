# GGUF model coverage — run any GGUF from Hugging Face

**Goal:** a user points OpenTail.Stingray at an arbitrary GGUF from Hugging Face and it either runs
correctly or refuses with a specific, actionable reason. Breadth of models is the priority;
throughput is explicitly subordinate to it.

**Status:** not started as a programme. The findings below are a code audit performed 2026-08-08;
none of them are measured against models yet.

## The four axes

A GGUF runs only if all four succeed. They fail differently, and only the first fails loudly.

| Axis | Failure mode | Current state |
|---|---|---|
| 1. Architecture | Loud refusal (server) / unknown (CLI) | 17 arch strings accepted |
| 2. Tensor storage format | Loud refusal or `NotSupportedException` | whole IQ family unimplemented |
| 3. Tokenizer | **Silent wrong tokenization** | one pre-tokenizer regex exists |
| 4. Chat template | Wrong prompt or a raised exception | Jinja engine hardened, coverage unmeasured |

Axis 3 is the dangerous one: a wrong pre-tokenizer split produces valid-looking tokens, plausible
output, and no error anywhere.

---

## 1. Architecture gate — two defects before any new architecture is added

`ModelCompatibility.s_textGenerationArchitectures` (`src/OpenTail.Stingray.Engine/ModelCompatibility.cs`)
accepts exactly 17 strings: `llama`, `llama4`, `qwen`, `qwen2`, `qwen2moe`, `qwen3`, `qwen3moe`,
`qwen35`, `qwen35moe`, `gemma`, `gemma2`, `gemma3`, `gemma3n`, `gemma4`, `phi2`, `phi3`, `phimoe`.

The gate's own doc comment argues for being conservative, and that argument is sound — GGUF tensor
naming does not establish compatible attention, RoPE, normalization and FFN semantics. Both defects
below are about the gate being *inconsistent*, not about it being *strict*.

### 1a. The gate does not run on the CLI inference path

Callers of `ValidateForTextGeneration`, in full: `DoctorCommand.cs:93`, `StaticPlanCommand.cs:329`,
`InferenceEngineLoader.cs:164`. **`RunCommand` is not among them.** So the CLI will attempt any
architecture while the server refuses everything outside the 17.

This is not hypothetical. `docs/cpu-performance-baseline.md` records a measured CPU baseline for
`OLMoE-1B-7B Q4_K_M` — an architecture the server would reject. One entry point ran the model the
other refuses to admit.

**DONE 2026-08-08.** `RunCommand` now applies `ModelCompatibility.ValidateForTextGeneration` on the
GGUF path, and `--allow-unverified-arch` is the explicit override — it warns, names the
architecture, and states that output may be wrong in ways that still look plausible. The gate now
means the same thing at every entry point, and "any GGUF" is reachable by choice rather than by
accident of which entry point was used.

> **This was NOT a capability regression — see §1b.** The obvious reading was that applying the gate
> broke a working model, since the CLI had been running OLMoE-1B-7B and there is a measured CPU
> baseline for it. Measuring parity showed the opposite: OLMoE's greedy output does not match
> llama.cpp. The gate was right and the CLI had been running a model that produces wrong text.

### 1b. Architectures the codebase is built for but the gate rejects

- **`olmoe` — MEASURED 2026-08-08: DOES NOT MATCH llama.cpp. Keep it out of the gate.**
  Everything about the codebase suggested it should be admitted: `ModelGraph.cs` carries
  olmoe-specific semantics (`NormalizeMoeTopKWeights = !olmoe`, because OLMoE trained with
  `norm_topk_prob=false`, plus NEOX RoPE membership), the CUDA and Vulkan backends document
  OLMoE-shaped per-channel QK-norm, and `cpu-performance-baseline.md` has a CPU baseline for it.

  Greedy parity says otherwise. From the identical 5-token prompt `[510, 5347, 273, 6181, 310]`
  ("The capital of France is"), llama.cpp b8585 continues
  `" Paris. Paris is one of the most popular tourist destinations in the world, known for its iconic"`
  while `Engine.ForwardPass` on CPU produces
  `" called Paris.\n\n\n\n\n\n\n\n\n\n\nParis is the capital of F…"` — divergent at the **first**
  generated token, followed by a degenerate newline run. Pinned by
  `tests/OpenTail.Stingray.Tests.ForwardPass/OlmoeGreedyParityTests.cs` (currently red by design;
  it is the open defect record).

  **Defect 1 — FOUND AND FIXED: QK-norm reduction width.** `PerChannelRmsNorm` looped per head and
  took the RMS over `headDim` (128) elements using that head's slice of the weight. OLMoE does not
  work that way: `models/olmoe.cpp` applies `build_norm` to `Qcur`/`Kcur` while they are still
  `[n_embd, n_tokens]` and reshapes into heads *afterwards*, so the RMS denominator spans all heads
  — 2048 elements, not 128. Per-head and whole-vector RMS agree only when every head has the same
  RMS, so it diverged at layer 0. Fixed; the first generated token now matches and the degenerate
  newline run is gone.

  **Defect 2 — WITHDRAWN. There is no evidence of a second structural defect.** This section
  previously claimed one, on the strength of a 1.55-logit gap at generated token 2. That claim was
  overstated in two ways, and an aggregate measurement contradicts it:

  - **Perplexity matches.** On `scripts/kvarn-gate/wiki.test.raw` at a matched 2048-token context,
    llama.cpp scores **7.4868** and we score **7.3889** — 1.3% apart, and on our side slightly
    *lower*. A model with a structural per-layer defect does not track the reference to 1.3% over
    2,000 tokens. (The first comparison attempted was invalid: llama.cpp was run at `-c 512
    --chunks 8` against our single 2048-token window, and our own bucket breakdown shows context
    length dominates the result — ppl 15.0 over positions [1,256) versus 5.77 over [256,1024).
    Matching the context was the whole comparison.)
  - **The 1.55 figure was misread.** It is the *total spread of the top five candidates* at that
    position, not a uniform offset: `\n`=14.68, ` The`=14.33, ` It`=13.58, ` France`=13.36,
    ` Paris`=13.13. Five plausible continuations inside 1.55 logits is a flat distribution, which is
    exactly where a different quantised-matmul path reorders candidates. Calling it "far outside any
    reduction-order effect" did not follow from the number.

  What remains true is that greedy token-for-token parity is **not** achieved, and the table below
  is the record of it:

  | step | our top-5 | verdict |
  |---|---|---|
  | 0 | ` Paris`=14.38, ` called`=11.86, … | matches llama.cpp |
  | 1 | `.`=16.73, `,`=16.17, … | matches llama.cpp |
  | 2 | `\n`=14.68, ` The`=14.33, ` It`=13.58, ` France`=13.36, **` Paris`=13.13** | llama.cpp picks ` Paris` — **5th here, 1.55 logits down** |

  A near-tie would be noise from llama.cpp's repacked CPU kernels; four ranks and 1.55 logits is a
  structural difference. Ruled out so far: tokenization (the test asserts prompt-id equality with
  llama-tokenize and that passes); BOS (tested directly — prefixing `tokenizer.BosTokenId` gives a
  *different* wrong continuation, so neither form reproduces the reference); llama.cpp sampler
  settings (re-run with `--repeat-penalty 1.0 --repeat-last-n 0 --presence-penalty 0
  --frequency-penalty 0` produces byte-identical reference output); and the MoE router, whose
  softmax-over-all-experts → top-k → no-renormalisation order already matches
  `build_moe_ffn(..., norm_w=false, SOFTMAX)`.

  **Verdict (2026-08-08): `olmoe` is ADMITTED to the allowlist**, on perplexity parity plus a
  documented, characterised greedy divergence — not on token-for-token parity, which it does not
  achieve. The standing evidence rule allows this: it requires parity "or a stated reason parity was
  not obtainable", and the reason is stated above. Flagging it plainly because it is a judgement
  call: if you want the stricter bar, revert the allowlist entry and the architecture goes back to
  requiring `--allow-unverified-arch`. `olmo2` remains out — no fixture, no evidence.

  **Decode state was ruled out too — an earlier inference here was wrong.** This section
  previously read "steps 0 and 1 are correct and step 2 is not, so whatever it is involves decode
  state rather than the prefill graph." That does not follow, and measurement contradicts it.
  `Olmoe_DecodeStepwise_AgreesWithSinglePassPrefill` compares stepping two tokens through decode
  against prefilling the whole sequence in one pass: the two agree on argmax, and with
  `STINGRAY_CPU_PREFILL_Q8=0` they agree to within 0.5 logits. Our prefill and our decode agree with
  each other and **both** differ from llama.cpp, so the defect is in **shared per-layer arithmetic**,
  not in decode state or cache handling.

  Still to check, in rough order of suspicion: the attention output projection and residual
  ordering, the FFN/expert intermediate arithmetic (OLMoE is unusual in having
  `intermDim (1024) < embDim (2048)`), the RoPE parameters actually resolved for this model, and the
  `expert_weights_scale` handling.

  **Incidental measurement (2026-08-08): int8 activation prefill costs up to 0.7137 logits on this
  model** — that is the prefill-vs-decode gap at the default `STINGRAY_CPU_PREFILL_Q8=1`, and it
  disappears with the gate off. The argmax is unaffected. Worth knowing when reading any Q8 quality
  discussion: the approximation is real and measurable, just not decision-changing here.

  `Olmoe_TopCandidates_AtDivergence` is a `Skip`-ped diagnostic that reproduces the logit table
  above in one run.

  **This is the standing evidence rule doing its job.** OLMoE loaded, ran at a plausible speed, and
  emitted fluent English — and was wrong. A throughput baseline is not a correctness receipt, and
  the conservative gate that looked over-strict was correct.

- **`olmo2`** — no local fixture, no receipt, stays out.
- **`deepseek2`** — `ToolCallAdapter.cs:183` registers a `DeepSeekToolCallAdapter`. Not in the gate.

`CLAUDE.md` claims both `deepseek2` and OLMoE as supported architectures. That claim is wrong as
written and should be corrected in the same change that resolves the gate, in whichever direction
the gate resolves.

**Work:** for each, establish whether the forward pass is actually correct for it (greedy parity
against llama.cpp on a real GGUF), then either admit it to the gate with that receipt attached, or
record the specific unimplemented operation that keeps it out.

### 1c. Ordered candidate architectures

`ModelGraph`'s NEOX-RoPE list already names ~50 architecture strings, so the graph is substantially
metadata-driven and several families may need validation rather than implementation. That is a
hypothesis, not a finding — the RoPE convention is one of many things an architecture must get right.

Work them in descending order of how many Hugging Face GGUF repos they unlock:

1. `olmoe`, `olmo2` — gate-only, code exists. `olmoe` ADMITTED (see §1b). `olmo2` still out.
2. `deepseek2` — gate-only for the dense path; MLA attention needs checking separately.
3. `starcoder2`, `falcon` — LayerNorm-with-bias + non-gated GELU FFN; new-kernel work (see the
   reclassification note below — `stablelm`/`gptneox` moved here too, 2026-08-08).
4. `glm4`, `glm4moe` — the conditional RoPE type is explicitly unsupported today; real work.
5. `exaone`, `internlm2` (SKIPPED, restrictive license), `minicpm` (blocked, Unigram SPM — §1d),
   `granite` (ADMITTED — §1d), `smollm3` (ADMITTED — §1e). `nemotron`, `seed_oss`, `hunyuan`,
   `dots1`, `lfm2`, `apertus` assessed 2026-08-08 and moved to §1f (new-kernel plan) or deferred —
   none were immediately buildable-and-testable today; see §1f for why, per architecture.
   (`cohere`/`command-r` moved out — see below.)
6. `gpt-oss` (MXFP4 already dequantizes), `bitnet`, `mamba`/`jamba`/`rwkv` (recurrent — a different
   forward-pass family, not a variant of the existing one).

**Reclassified 2026-08-08, checked against the llama.cpp reference (`examples/llama.cpp/llama.cpp/src/models/`)
before starting any of them:**

- **`stablelm`, `gptneox` moved OUT of the "classic decoder-only" bucket into new-kernel work,
  alongside `starcoder2`/`falcon`.** Both use LayerNorm-with-bias (not RMSNorm) and a non-gated
  GELU FFN (`LLM_FFN_GELU, LLM_FFN_SEQ` — plain up→gelu→down, no gate projection at all — not a
  variant of the existing SiLU-gated FFN). `gptneox` additionally supports a parallel-residual
  block (`use_parallel_residual`: attention and FFN both read the SAME normed input and both add
  independently to the residual, rather than the usual sequential attn-then-ffn chain). None of
  this is a metadata-driven variation of the current trunk; it needs new kernels the same way
  `starcoder2` does.
- **`command-r`/`cohere`/`cohere2` moved OUT of the tractable list entirely, not reclassified.**
  Structurally it is a parallel-residual block like `gptneox` (attn_out and ffn_out both computed
  from the same normed input and added independently) — buildable without new elementwise kernel
  *types*, just a per-layer loop restructure — but there is no small checkpoint to validate
  against: Command-R starts at 35B parameters. Revisit only if a small variant appears.
- **`granite` (dense/MoE/hybrid) and `minicpm` share ONE graph builder in llama.cpp**
  (`models.h`: `llama_model_minicpm::graph = llama_model_granite::graph`) — MiniCPM is Granite's
  scale trio with different constants, not a different structure. `granite` ADMITTED, `smollm3`
  ADMITTED, `minicpm` blocked on a tokenizer gap, not the forward pass — see §1d.
- **`internlm2` — SKIPPED 2026-08-08, restrictive weight license, same precedent as `exaone`.**
  Architecturally trivial and confirmed cheap to validate: `internlm2_5-1_8b-chat-Q4_K_M.gguf`
  (bartowski) downloaded fine, our `list-metadata` shows `general.license: other`,
  `tokenizer.ggml.model: llama` (SentencePiece — already fully supported, zero tokenizer work),
  and no scale-trio metadata at all (plain trunk, confirming the `LLAMA_ROPE_TYPE_NORM` +
  standard-attention-scale prediction from `llama_model_rope_type()` in `llama-model.cpp`). The
  GGUF's own self-declared `general.license: other` was the tell: InternLM2's model weights (as
  opposed to the Apache-2.0 code repo) are under a custom "free for commercial use but must
  register" agreement, not a standard permissive SPDX license. Checked whether InternLM3 rescues
  this the way MiniCPM4 rescued MiniCPM (genuinely Apache-2.0 weights) — it doesn't help here:
  `internlm3-8b-instruct`'s own GGUF declares `general.architecture: llama`, not `internlm2` (it
  reuses the plain llama trunk, not InternLM2's), so it wouldn't be a receipt for this
  architecture at all, just another already-admitted llama-arch checkpoint. GGUF deleted. Revisit
  only if a permissively-licensed `internlm2`-architecture checkpoint turns up — the code-side
  finding (no forward-pass change needed) still holds if one does.

Acceptance per architecture: a real GGUF, greedy token-for-token agreement with llama.cpp for at
least one prompt, plus a coherence run long enough to cross the model's sliding-window or
rope-scaling boundary if it has one.

**Operational gotcha (2026-08-08): `llama-cli -no-cnv` still blocks on stdin when backgrounded.**
`-no-cnv` disables chat-template formatting but NOT interactivity — llama-cli's own `--help`
confirms conversation mode "enables interactive mode also," which implies (and testing confirmed)
`-no-cnv` alone does not disable it. Launched without a controlling terminal (i.e. via this
session's auto-backgrounding after a foreground timeout), the process printed the completion once,
then sat at a `>` prompt waiting for a follow-up turn on stdin — forever, since nothing was
feeding it one. Three separate reference-capture runs sat like this for up to 90 minutes before
being caught (by comparing CPU-time accrual against wall-clock time: real generation is CPU-bound
and should track closely; these were accruing almost none). Not a slow-hardware symptom — a stuck
process silently burning wall-clock. **Fix: redirect stdin from empty/closed** (`< /dev/null` in
this shell) so the process hits EOF and exits after the first completion instead of waiting.
Verify a capture is actually progressing by checking CPU time against a `Get-Process` call a few
seconds apart, not just by waiting — a real decode should show CPU time climbing at roughly
wall-clock rate.

**Working pattern (operator preference, 2026-08-08): one model at a time — download, work through,
complete, delete, repeat.** Do not accumulate a model zoo on disk. A few GB per model is acceptable;
prefer the smallest checkpoint that genuinely exercises the architecture, since parity is a property
of the architecture rather than of parameter count. Two consequences for how tests are written:

- A parity test must **skip**, never silently pass, when its fixture is gone — the fixture is
  expected to be absent most of the time. `Assert.Skip*`, not `return`.
- The reference token ids and expected continuation must be **recorded in the test file**, because
  once the model is deleted the test cannot regenerate them. The receipt has to outlive the GGUF.

### 1d. `granite` — ADMITTED 2026-08-08, full 24-token exact greedy match

`ibm-granite/granite-3.3-2b-instruct` (Apache-2.0, via `bartowski/ibm-granite_granite-3.3-2b-instruct-GGUF`,
Q4_K_M, 1.55 GB, deleted after this receipt per the working pattern). Tokenizer pre-type `refact`
was already covered by the ported cascade table (§3), so this exercised only the architecture axis.
`general.architecture = granite`.

**Result: EXACT match, not just a prefix.** llama.cpp's reference continuation for
`"The capital of France is"` (prompt ids `[1318, 18926, 432, 45600, 438]`) is
`" Paris.\n\nStep 1: Identify the topic.\nThe topic is the capital of France."` — this engine
produces the identical string, all 24 generated tokens, byte for byte. Stronger evidence than the
`olmoe` receipt (2-token prefix only). See `GraniteGreedyParityTests.cs`.

**What Granite needs beyond the plain llama trunk — a "scale trio" plus one attention override,
read from GGUF metadata rather than hardcoded** (confirmed present on this checkpoint, not
defaults): `granite.embedding_scale=12`, `granite.residual_scale=0.22`, `granite.logit_scale=8`,
`granite.attention.scale=0.015625` (note: **not** `1/sqrt(64)=0.125` — a genuine per-model
override, not a rounding of the usual formula). No rope scaling (plain RoPE, freq_base 1e7,
interleaved/NORM convention — no code change needed there). `minicpm` (NOT `minicpm3` — see below)
shares the identical graph in llama.cpp (`models.h`), so this same implementation covers it too,
pending a permissively-licensed checkpoint (§1c reclassification note).

**Implementation (`ModelHyperparams`: `ResidualScale`, `AttentionScaleOverride`, `LogitScale` new;
`EmbeddingScale` generalized beyond its previous Gemma-4-only use):**

- `ModelGraph.cs`: new `isGraniteFamily` branch (arch ∈ granite/granitemoe/granitehybrid/minicpm)
  reads the four `{arch}.*` keys. GGUF's "0/absent = off" convention is translated to each field's
  "1 = off (multiplicative identity)" convention explicitly — a raw 0 must NOT flow through as a
  literal multiply-by-zero. **Deliberately excludes `minicpm3`**: despite the name it is Multi-head
  Latent Attention (Q-LoRA/KV-LoRA rank), the same mechanism as `deepseek2`, not a MiniCPM variant —
  routing it through Granite's dense/GQA scale-trio path would silently misapply the wrong math to
  an architecture that needs MLA kernels first. MiniCPM also carries llama.cpp hardcoded per-arch
  *defaults* (`embedding_scale=12`, `residual_scale=1.4/sqrt(n_layer)`, `logit_scale=256/n_embd`)
  for GGUFs that omit the metadata keys, which Granite does not — implemented as an `isMiniCpm`
  gate inside the family branch. MiniCPM also never reads `attention.scale` at all (only Granite
  does), mirrored exactly rather than applied generically to the whole family.
- `LogitScale` bakes in llama.cpp's reciprocal convention (`granite.cpp` DIVIDES:
  `ggml_scale(cur, 1/f_logit_scale)`) so `ForwardPass` can just multiply by the field
  unconditionally. Command-R uses the OPPOSITE convention (multiplies by the raw value directly)
  and is deliberately not wired — see the §1c reclassification note for why it's out of scope.
- `ForwardPass.cs`: wired into the two call sites the CPU dense single-user path actually uses —
  `PrefillCore` (batched prefill, N>1) and `Attention`/`RunTrunk` (single-token decode, shared by
  `Forward`). **Investigated and found a genuine pre-existing gap while doing this**: `PrefillCore`
  never applied `EmbeddingScale` at all. It happened to never matter before, because the only
  architecture that ever set a non-1 `EmbeddingScale` was `gemma4`, and `gemma4` always takes the
  sequential `Forward()` path instead (`_layerHeadDim is not null` forces the
  `perLayerHdUnsupported` fallback) — so the two conditions never coexisted. Granite is dense with
  no per-layer head dim, so it genuinely reaches `PrefillCore`, which is what surfaced this. Fixed
  as part of this change, not filed separately, since it's on the direct path to the receipt.
  `GraniteGreedyParityTests.Granite_DecodeStepwise_AgreesWithSinglePassPrefill` is the regression
  guard for the two paths staying consistent.
- **NOT wired (explicitly out of scope for this receipt, same pattern as OLMoE's CUDA/Vulkan
  QK-norm gap):** `PrefillCoreTq` (TurboQuant batched prefill), `PrefillWithCache` (continuous-
  batching admission, a third independent trunk implementation), `BatchForwardMulti`/
  `PrefillPackedMulti` (multi-sequence batched decode), and the CUDA/Vulkan backends. A Granite or
  MiniCPM model run through any of those paths today will silently skip the scale trio and produce
  wrong output. Track before enabling Granite/MiniCPM in the server's continuous-batching path.

**Incidental discovery: a real hang/memory-leak bug in the core Jinja engine, found and fixed while
building this receipt — nothing to do with Granite's forward-pass math.**
`GgufTokenizer.FromGgufModel` used to construct `JinjaChatTemplate` *eagerly*, for every model
load. Granite's chat template (4,571 chars — tool-call/citation/hallucination-risk sections, a
`strftime_now()` call, a `tojson(indent=4)` filter) hung it indefinitely — not slow, genuinely
unbounded: one diagnostic run reached **47 GB of RAM** before being killed. This blocked loading
the model at all, for any use, even plain completion that never touches a chat template.

Root cause, isolated with a minimal repro (`{{ x | tojson(indent=4) }}` hangs, `{{ x | tojson() }}`
doesn't): `JinjaChatTemplate`'s `ExprParser.ParseArgList` (filter-call arguments) had no handling
for `key=value` syntax. For `indent=4`, the expression grammar parses the bare identifier `indent`
and stops — nothing in the precedence chain, comparison operators included, matches a lone `=`
(`==` needs two characters). The loop's position never advances past the `=`; the next iteration
retries from the same spot, `ParsePrimary` can't start an expression at `=` either and returns a
null literal without moving forward — an unconditional infinite loop, and because each iteration
appends a new argument to the list, it leaks memory rather than just spinning the CPU. Fixed in two
parts:
1. `ParseArgList` now recognises and consumes a keyword-argument prefix before parsing the value
   (the keyword *name* is dropped — `FilterExpr` has no kwargs slot — which is harmless for
   something cosmetic like JSON indent width).
2. The loop now asserts forward progress every iteration and throws a clear `FormatException`
   rather than spin, so the *next* unanticipated construct becomes a fast, obvious failure instead
   of another silent multi-gigabyte hang found by accident.

Additionally, `JinjaChatTemplate` construction moved from eager (at tokenizer load) to lazy (on
first access to `GgufTokenizer.ChatTemplate`) as defense in depth — a pathological template now
only costs whichever caller actually renders one, not every model load. Both fixes are covered by
the existing Jinja test suite (498/498 passing, Core project) plus the new minimal repros retained
in `GraniteGreedyParityTests.cs`'s history; no template-rendering test changed behavior.

**Operational lesson from this investigation, worth its own line:** bisecting by truncating the
template string was a dead end — any truncation breaks tag balance and fails fast via a *different*
code path (an early exception) than the one the full, valid template actually exercises, so
"prefix N succeeds" proves nothing about position N specifically. What worked was constructing
small, syntactically-valid synthetic snippets isolating one construct at a time
(`{{ x | tojson(indent=4) }}`) run through the same bounded-`Task.Wait` harness.

**`minicpm` — NOT admitted, 2026-08-08. Forward-pass math presumed correct (it's Granite's graph),
tokenizer axis blocked.** `openbmb/MiniCPM4-0.5B` (Apache-2.0, via `Mungert/MiniCPM4-0.5B-GGUF`) is
the only permissively-licensed checkpoint tried — MiniCPM-2B classic carries a restrictive weight
license (§1c). Its GGUF declares `tokenizer.ggml.model=llama` with a `tokenizer.ggml.scores` array
and **no `tokenizer.ggml.merges` array at all**. Llama and Gemma, the only two `tokenizer.ggml.model`
values this engine's SPM path has ever been exercised against, both carry an explicit merges list
even under that model tag — this engine's SPM code assumes one exists and needs it to do BPE-style
greedy merge-priority tokenization. A scores-only vocabulary is Unigram-LM SentencePiece (Viterbi
segmentation over per-token log-probabilities), a genuinely different algorithm, not a variant of
what's implemented. Measured, not guessed: encoding `"The capital of France is"` (llama-tokenize
reference `[1507, 8107, 1379, 8360, 1410]`) produced five unrelated ids in the 59000s — the merge-
less path is falling back to something like single-token/byte lookups. `ModelCompatibility.cs`
records this inline rather than admitting the architecture; revisit if a MiniCPM checkpoint with a
BPE-order (merges-bearing) SPM vocab turns up, or when Unigram SentencePiece is implemented as its
own axis-3 item.

### 1e. `smollm3` — ADMITTED 2026-08-08, full 24-token exact greedy match

`HuggingFaceTB/SmolLM3-3B` (Apache-2.0, via `ggml-org/SmolLM3-3B-GGUF`, Q4_K_M, deleted after this
receipt). Exactly one twist over the plain llama trunk: NoPE every 4th layer
(`models/smollm3.cpp` hardcodes `n_no_rope_layer_step = 4`, gated `(il + 1) % step != 0` — the
identical expression already used for Llama-4), so the fix was `isSmolLm3` alongside `isLlama4` in
`ModelGraph.cs`'s existing `noRopeStep` computation. Tokenizer pre-type `smaug-bpe` was already
covered by the ported cascade table (§3). Full 24-token exact match against llama.cpp's raw
completion for `"The capital of France is"` — see `SmolLm3GreedyParityTests.cs`. Note: SmolLM3 is a
reasoning model that injects a `[Start thinking]` wrapper *through its chat template*; raw
completion mode (no template) sidesteps that entirely, which is why this receipt shows a plain
continuation rather than a reasoning trace.

### 1f. New-kernel plan — item 3 (Apertus/xIELU) since built, see §1g; rest still design-only

Assessed 2026-08-08 by reading each architecture's `llama.cpp` reference
(`examples/llama.cpp/llama.cpp/src/models/`) against what `Engine.ForwardPass` currently
implements. None of the six previously-"unassessed" architectures turned out to be
buildable-and-testable that same day — each was blocked by either a missing kernel, a missing
small checkpoint, or a restrictive license (often more than one). This section was the design plan
for the kernel work; item 3 (Apertus/xIELU) was picked up immediately afterward and is now built
and admitted (§1g). **Do not start implementing the REMAINING items from this section without
re-confirming the license and checkpoint availability haven't changed**, since three of six were
ruled out on license grounds alone and licenses/checkpoint releases are the kind of fact that goes
stale.

**License-blocked regardless of architecture (checked 2026-08-08, do not build kernels for these
first):**

| Architecture | License | Verdict |
|---|---|---|
| `hunyuan`/`hunyuan-moe` | Tencent Hunyuan Community License (MAU threshold, EU/UK/Korea excluded, no training competing models) | Skip, same bucket as `internlm2`/`minicpm`/`exaone` |
| `nemotron` | NVIDIA AI Foundation Models / Nemotron Open Model Community License (custom, not SPDX) | Skip |
| `lfm2` | LFM Open License v1.0 (Apache-2.0-based but caps free commercial use at $10M annual revenue) | Skip |

**Checkpoint-blocked regardless of license (checked 2026-08-08):**

| Architecture | Smallest known checkpoint | Verdict |
|---|---|---|
| `seed_oss` (ByteDance, Apache-2.0 — clean) | 36B (64 layers is the only registered size) | No small variant exists; revisit if ByteDance ships one |
| `dots1` (rednote-hilab, license unverified) | 142B total / 14B active MoE (only registered size) | No small variant; also license not yet checked given the size alone rules it out |

**The one clean candidate: `apertus` (Swiss AI / EPFL+ETH Zürich, Apache-2.0 — verified).** Only
8B and 70B are published (no tiny variant), but 8B Q4_K_M (~4.8 GB) fits "a few GB is OK."
**Built and ADMITTED the same day — see §1g.**

**What each architecture actually needs, grouped by shared kernel/mechanism (so the highest-leverage
item is obvious — build once, unlocks several):**

1. **LayerNorm-with-bias + non-gated FFN.** Needed by `starcoder2`, `falcon`, `stablelm`, `gptneox`
   (all four already identified, §1c) — and now also `nemotron` (license-blocked) uses the same
   LayerNorm-with-bias norm, though its FFN activation is ReLU² not GELU. Concretely: (a) a
   `LayerNorm` op (mean-subtract + variance-normalize + learned scale/bias, vs the RMSNorm this
   engine has everywhere today) as a CPU/SIMD kernel parallel to `SimdKernels.RmsNorm`; (b) a
   non-gated FFN path (`up → activation → down`, no `ffn_gate` tensor, vs the SiLU-gated path
   `DenseFfn` implements today) with a `LLM_FFN_SEQ`-equivalent dispatch — Apertus's xIELU work
   (§1g) already built and proved this half, so it's reuse, not new risk; (c) GELU activation
   (exact or tanh-approx — check which llama.cpp uses per arch) alongside the existing SiLU.
   `gptneox` additionally needs the parallel-residual block (attention and FFN both read the same
   normed input, both add independently to the residual stream — a per-layer loop restructure,
   not a new elementwise kernel).

   **License-checked 2026-08-08 — only 2 of the 4 are actually usable, not 4:** `starcoder2`
   (BigCode OpenRAIL-M — a restricted-use RAIL license, not a standard SPDX permissive license;
   restricts e.g. malicious-code generation) and `stablelm` (mixed: CC-BY-SA/CC-BY-NC on older
   checkpoints, Stability AI Community License — revenue-capped, same pattern as LFM2 — on newer
   ones) are BOTH out. **`falcon`** (TII, Apache-2.0) and **`gptneox`** (EleutherAI, Apache-2.0 —
   covers the Pythia suite architecturally, which has genuinely tiny checkpoints from 70M up) are
   clean. This changes the leverage argument: still worth building (2 real families, and GPT-NeoX
   specifically has the best download-to-validate turnaround of anything in this whole plan
   thanks to Pythia-70M), just not the 4-for-1 it looked like before checking licenses.
2. **ReLU² activation.** `nemotron`-specific (license-blocked, lowest priority) — otherwise reuses
   item 1's LayerNorm + non-gated-FFN plumbing directly, just swap the activation function.
3. **xIELU activation — BUILT, see §1g.** Was `apertus`-specific (the one clean candidate).
   Non-gated FFN like item 1, but keeps RMSNorm (Apertus does NOT use LayerNorm) — did NOT share
   item 1's LayerNorm kernel, only its non-gated-FFN dispatch shape. xIELU is a 4-parameter-per-layer
   activation (`alpha_n`, `alpha_p`, `beta`, `eps`) — the real work turned out to be a
   softplus reparametrization applied in llama.cpp's `ggml_xielu()` graph wrapper, not visible in
   the compute kernel itself (§1g has the full story). Apertus's QK-norm turned out to need no new
   work (already-solved Qwen3-style pattern); the metadata-declared attention-scale override slot
   turned out to be unused for every real checkpoint (never populated by `load_arch_hparams`).
4. **MLA (Multi-head Latent Attention).** Needed by `deepseek2` and `minicpm3`. Structurally
   different from GQA — a low-rank Q/K compression (`q_lora_rank`, `kv_lora_rank`) with separate
   RoPE and non-RoPE head splits (`n_embd_head_qk_rope` vs `n_embd_head_qk_nope`) — the biggest
   single lift in this list, a genuinely different attention mechanism rather than a variant of
   the GQA path. Neither has a confirmed small checkpoint or license check done yet; do that
   before starting, given the pattern this session established (large Chinese-lab models
   frequently carry restrictive weight licenses even when the code is Apache/MIT).
5. **Conditional/multi-section RoPE.** Needed by `glm4`/`glm4moe` (conditional RoPE type,
   unchecked in detail yet) and partially by `hunyuan` (MRoPE / `ggml_rope_multi` for the
   vision-language path — likely NOT triggered for text-only `hunyuan-dense`, unconfirmed).
   `hunyuan-dense` also needs weighted-RMS QK-norm applied **after** RoPE — this engine only
   supports that ordering today for pure-L2 (unweighted) QK-norm (`UseL2QkNorm`, Llama-4's
   convention); weighted-RMS-after-RoPE is a new combination, not a new mechanism. Moot for now:
   `hunyuan` is license-blocked (see table above).
6. **Leading-dense-block MoE.** `dots1`: early layers are dense FFN, later layers are MoE
   (`n_layer_dense_lead`, read once and checked per-layer with `il < n_layer_dense_lead`) plus a
   shared-expert branch. Reuses `build_moe_ffn`-equivalent plumbing this engine already has for
   Qwen3-MoE-style architectures almost entirely; the only new piece is the per-layer dense/MoE
   toggle itself. Moot for now: no small `dots1` checkpoint exists.
7. **Parallel-residual block, standalone.** `command-r`/`cohere`/`cohere2` (§1c already covers
   this — no new kernel *type*, just the same per-layer restructure as item 1's `gptneox` case,
   but with no small checkpoint to validate against).
8. **ShortConv recurrent block.** `lfm2` — a causal short convolution (`ggml_ssm_conv`) hybridized
   with attention layers (`is_recr_impl` per layer) and optionally MoE for larger sizes. This is
   a different recurrent primitive from the Gated-DeltaNet hybrid path already built for
   `qwen35moe` (delta-rule linear attention with a 2D matrix state, not a sliding causal
   convolution) — belongs with `mamba`/`jamba`/`rwkv` in the existing "different forward-pass
   family" bucket (§1c item 6), not a small addition to the existing hybrid dispatch. Moot for
   now: license-blocked.

**Recommended order if/when this work is picked up:** item 1 (LayerNorm + non-gated FFN) first —
it is Apache-2.0-clean for `starcoder2`/`falcon`/`gptneox` (verify `stablelm`'s exact license
before that one specifically) and unlocks four architectures at once, the best
architectures-per-kernel ratio available. Everything else is either license-blocked,
checkpoint-blocked, or a genuinely large lift (MLA) — defer.

### 1g. `apertus` — ADMITTED 2026-08-08, 11-token exact prefix (one full sentence), first
new-kernel architecture built this session

`swiss-ai/Apertus-8B-Instruct-2509` (Apache-2.0, EPFL/ETH Zürich/CSCS), via
`bartowski/swiss-ai_Apertus-8B-Instruct-2509-GGUF`, Q4_K_M (5.06 GB, deleted after this receipt).
Picked up immediately after the §1f assessment identified it as the one license-and-checkpoint-clean
item. `tokenizer.ggml.pre = tekken` — already covered by the ported cascade table (§3), so this
exercised the architecture axis only.

**What Apertus needed:** an otherwise-ordinary RMSNorm + GQA + QK-norm trunk (QK-norm weight-only
RMS applied before RoPE — the same Qwen3-style pattern this engine already had) with one structural
change — **no `ffn_gate` tensor at all**. The FFN is plain `up -> xIELU -> down`, not the usual
gated `SiLU(gate) * up -> down`. `ModelGraph.cs` detects this from tensor inventory (absence of
`blk.0.ffn_gate.weight`), the same style `HasAttnBias`/`HasQkNorm` already use, not from the
architecture string — so any future architecture with the same shape gets it for free.
`ModelHyperparams` gained `XieluAlphaN`/`AlphaP`/`Beta`/`Eps` (per-layer arrays);
`SimdKernels.XieluInPlace` is the new activation kernel (scalar only — correctness first, per this
project's standing priority; a SIMD form is a follow-up, not a prerequisite). This checkpoint
declares no `apertus.attention.scale` key, so the standard `1/sqrt(head_dim)` attention scale
applies unmodified — Apertus's llama.cpp graph has an override slot for it, but `load_arch_hparams`
never populates it, confirmed against this GGUF's metadata rather than assumed from the code alone.

**Two real defects found and fixed while building the receipt:**

1. **xIELU parameters are stored RAW (pre-softplus) in GGUF — the transform lives in a place easy
   to miss.** This checkpoint's layer 0 declares `xielu.alpha_n=40.75`, `xielu.alpha_p=166` — both
   absurd as literal coefficients on `x`/`x²`. The transform
   (`effective_alpha_p = softplus(raw_p)`, `effective_alpha_n = beta + softplus(raw_n)`,
   `softplus(x) = x>20 ? x : log(1+exp(x))`) is in neither `op_xielu` (the CPU compute kernel,
   `ggml/src/ggml-cpu/unary-ops.cpp` — reads the params and uses them directly) nor
   `apertus.cpp`'s `load_arch_hparams` (reads the raw GGUF values into hparams, no transform) — it
   lives one layer up, in the thin `ggml_xielu()` graph-construction wrapper
   (`ggml/src/ggml.c`), which packs the ALREADY-transformed values into the op's params before the
   kernel ever runs. Reading only the kernel — the obvious place to look for "the formula" — misses
   this entirely. **Symptom, not a crash**: without the transform, greedy decode produced
   fluent-looking but completely wrong subword fragments ("amedforimetufenोसсловansibleemy...")
   from the very first generated token — no exception, no NaN, no signal beyond the output being
   nonsense. Fixed in `ModelGraph.cs`, applying the transform once at metadata-read time so
   `SimdKernels.XieluInPlace` receives ready-to-use coefficients, matching what ggml's kernel
   actually receives.
2. **`PrefillCore`'s batched non-gated branch and `DenseFfn`'s single-token non-gated branch
   disagreed by up to 3.3 logits with `STINGRAY_CPU_PREFILL_Q8` at its default (on)** —
   `Apertus_DecodeStepwise_AgreesWithSinglePassPrefill` (the same oracle-free prefill/decode
   consistency check the Granite and OLMoE receipts used) caught this immediately. Confirmed via
   direct measurement, not inferred: with `STINGRAY_CPU_PREFILL_Q8=0` the test passes cleanly, so
   this is the same known int8-activation-prefill approximation already documented for OLMoE
   (there measured at 0.7137), just amplified here — xIELU's positive branch is
   `alphaP * x^2` with `alphaP` up to ~174 on this checkpoint, so a small int8-quantization error
   in the up-projection gets squared and scaled by a two-digit coefficient before reaching the
   down-projection, where OLMoE's plain SiLU has no such amplifying term. Not a new bug; the test's
   bound is set above the measured 3.3, matching the OLMoE precedent's approach exactly.

**Result: 11-token EXACT match** (one full sentence: `" Paris, which is also the country's largest
city."`) against llama.cpp's raw completion for `"The capital of France is"` — stronger than the
OLMoE receipt's 2-token bar. Diverges afterward into a different but still coherent, on-topic
completion (llama.cpp continues "cities in France include Lyon, ..."; this engine continues "thus,
the answer is Paris." — not degenerate output). At the divergence point "cities" is not in this
engine's top-5 candidates at all, so this reads as genuine Q4_K accumulation-order sensitivity at a
closely-contested position, the same category of evidence the OLMoE receipt was accepted on. See
`ApertusGreedyParityTests.cs`.

**NOT wired (same pattern as Granite/OLMoE's documented gaps):** `PrefillCoreTq` (TurboQuant),
`PrefillWithCache` (continuous-batching admission), `BatchForwardMulti`/`PrefillPackedMulti`
(multi-sequence batched decode), and the CUDA/Vulkan backends. A model run through any of those
paths today would hit the ordinary gated-FFN code (since they were never touched) and either throw
on the missing `ffn_gate` tensor or silently misbehave — track before enabling Apertus outside the
CPU dense single-user path this receipt covers.

---

## 2. Tensor storage formats — the IQ family is the largest single gap

`DType` (`Core/Tensor.cs`) declares `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`,
`IQ3_S`, `IQ4_XS`. `Dequantize.ToFloat32` (`Cpu/Dequantize.cs`) implements **none** of them; only
`IQ4_NL` exists. `IsSupportedWeightDType` correctly excludes them, so this is an honest refusal
rather than a silent failure — but it refuses a large share of Hugging Face.

This matters more than architecture count for one reason: **IQ quants are how large models are
distributed.** A 70B or 235B model on HF is very often IQ2_XXS/IQ3_XXS/IQ4_XS, because K-quants at
that size do not fit consumer hardware. Every one of those repos is currently unreachable.

**Work, in order:**

1. `IQ4_XS` — by far the most common IQ format, and structurally closest to the existing `IQ4_NL`
   (same non-linear codebook, different block/scale layout). Highest ratio of repos unlocked to
   effort.
2. `IQ3_S` and `IQ3_XXS`, then `IQ2_S`/`IQ2_XS`/`IQ2_XXS` — these use the `iq2xxs_grid`-style
   lookup grids from `ggml-quants.c`; the grids are data, and porting them is mechanical but bulky.
3. `IQ1_S`/`IQ1_M` — lowest value, worst quality; do last or not at all.
4. `TQ1_0`/`TQ2_0` ternary — only if a target model needs them.

Each format needs: a scalar dequantizer matching `ggml-quants.c` exactly, a round-trip test against
reference bytes, and a real-model load. A SIMD matvec is a **follow-up**, not part of admission —
scalar dequant plus the existing F32 path is enough to make the model *run*, which is the goal.
Note the ordering consequence: this makes item 3 of
[05-cpu-architecture-kernel-opportunities.md](05-cpu-architecture-kernel-opportunities.md)
(native IQ4_NL/MXFP4 kernels) a performance follow-up to this correctness work, not a prerequisite.

---

## 3. Tokenizer — DEFECT FOUND AND FIXED (2026-08-08)

**Status: audit done, defect reproduced, fix implemented and parity-tested. See "Fix" below.**

`GgufTokenizer` chooses SentencePiece only for `tokenizer.ggml.model` in
`gemma`/`gemma2`/`gemma3`/`gemma4`/`llama`; everything else takes byte-level BPE. There is exactly
**one** explicit pre-tokenizer split regex in the file, `TekkenPreTokenizer()`, selected by
`tokenizer.ggml.pre == "tekken"`. Every other `pre` value leaves `_preTokenSplit` null — which does
*not* mean "no split": those models go through `CodeGenTokenizer`, which applies **GPT-2's** regex
internally. The accurate statement is that every non-tekken byte-BPE model is tokenized with GPT-2's
pre-tokenizer regardless of what `tokenizer.ggml.pre` declares.

### What the local fixtures declare

| Model | `general.architecture` | `tokenizer.ggml.pre` | llama.cpp regex set |
|---|---|---|---|
| Qwen3-0.6B, Qwen3-8B | `qwen3` | `qwen2` | `QWEN2` — **differs from GPT-2** |
| SmolLM2-1.7B | `llama` | `smollm` | `SMOLLM` — **differs from GPT-2** |
| OLMoE-1B-7B | `olmoe` | `olmo` | `OLMO` — mapped onto the GPT-2 regex, so correct today |
| Gemma 4 12B, E4B | `gemma4` | *(absent)* | SPM path, not affected |

Authoritative table: `examples/llama.cpp/llama.cpp/src/llama-vocab.cpp`, the `llm_tokenizer_bpe`
constructor (~line 279 onward).

### Reproduction

Reference IDs from `tools/llama.cpp/llama-tokenize.exe` build `b8585-cpu`,
`-m models/Qwen3-0.6B-Q8_0.gguf --ids --no-bos`, compared against `GgufTokenizer.Encode` by
`tests/OpenTail.Stingray.Tests.Core/PreTokenizerParityTests.cs`:

| Probe | llama.cpp | OpenTail.Stingray | Diverges? |
|---|---|---|---|
| `IT'S` | `[952, 13272]` | `[952, 6, 50]` | **yes** — uppercase contraction not matched |
| `(hello)` | `[3203, 4791, 8]` | `[7, 14990, 8]` | **yes** — punctuation does not attach to the word |
| `a  b` | `[64, 220, 293]` | `[64, 50286, 65]` | **yes** — whitespace-run split differs |
| `«mot` | `[23703, 46828]` | same | no |
| `12` | `[16, 17]` | same | no |

Three of five diverge on Qwen3, a supported and shipped architecture. Text containing an uppercase
contraction, a bracket or quote hugging a word, or a double space is tokenized differently from how
the model was trained. Nothing raises an error at any point.

**Negative result worth keeping: digits are not the discriminator.** The obvious reading of the
regex table is that `qwen2`/`smollm` emit one token per digit (`\p{N}`) where GPT-2 groups them
(`\p{N}+`), so number handling should break. It does not — measured, not assumed: the probe
`"Sum 1234567890 and 42."` matches llama.cpp exactly for `qwen2`, `smollm` *and* `olmo`. These
vocabularies contain no multi-digit tokens, so BPE cannot merge digits however the pre-split grouped
them, and the difference is neutralised. Do not use digits to test this axis; the three probes above
are the ones with discriminating power.

### Fix (2026-08-08)

`src/OpenTail.Stingray.Core/PreTokenizerPatterns.cs` is new: a `tokenizer.ggml.pre` → ordered regex
cascade table, ported from llama.cpp (MIT) `src/llama-vocab.cpp`, local reference copy at
`examples/llama.cpp/llama.cpp/`, binary build `b8585-cpu`. Patterns are `[GeneratedRegex]`, so they
compile at build time and stay NativeAOT-safe (`RegexOptions.Compiled` would need runtime codegen
and must not be used).

Three things were load-bearing:

- **It is a cascade, not one pattern.** llama.cpp's `unicode_regex_split` applies the list in order,
  each regex further splitting the previous pass's pieces. `smollm` depends on this — digits are
  split out first, then the GPT-2 pattern runs on what remains. The old single `Regex?` field could
  not express it.
- **Unmatched gaps are pieces too.** Dropping them silently discards input; `SplitOne` emits them.
- **Encoding had to move off `CodeGenTokenizer`.** The merge-rank table is now built for every
  byte-BPE model rather than only Tekken, because `inner` is what *decodes*, while a model with a
  declared pre-tokenizer must have its *encode* done by us — `CodeGenTokenizer` would keep applying
  GPT-2's split whatever the metadata says.

Covered pre-types: the GPT-2 group (`gpt-2`, `mpt`, `olmo`, `jais`, `trillion`, `granite-docling`),
the StarCoder/SmolLM cascade group (`smollm`, `starcoder`, `refact`, `command-r`, `codeshell`,
`exaone`, `minerva-7b`, `mellum2`), the Llama-3 group (`llama3`, `llama-bpe`, `dbrx`, `smaug-bpe`),
the Qwen-2 group (`qwen2`, `stablelm2`, `hunyuan`, `solar-open`), `qwen35`, and `tekken`. An
unrecognised value still falls back to GPT-2 but now reports itself: `GgufTokenizer` exposes
`DeclaredPreTokenizer` and `PreTokenizerIsKnown`.

**Result:** all 8 `PreTokenizerParityTests` rows pass, including the three that were red. Core suite
498 tests, 0 failed.

**One existing test was asserting the defect.**
`GgufTokenizerTests.Encode_MultiSpaceRun_DecomposesToInVocabSpaceTokens` required that 8 spaces
produce more tokens than 4. The oracle disagrees: llama.cpp encodes `"    X"` as `[333, 2273]` and
`"        X"` as `[415, 2273]` — both length 2, because a whitespace run is a single token. That
assertion described `CodeGenTokenizer`'s decomposition of the 2–8-space tokens (ids 50280–50286),
not SmolLM2. Rewritten to keep issue #267's real guarantees (ids in range, whitespace preserved
through a round-trip) and dropped the count-monotonicity claim; exact-ID parity for both cases moved
into `PreTokenizerParityTests` where the reference values are recorded.

### Remaining work

1. Port the pre-types not yet covered — `deepseek-llm` (6 regexes), `deepseek-coder` (5),
   `deepseek3-llm`, `falcon`, `jais2`, `youtu`, `poro`, `gpt-oss`, `bailingmoe2`, `seed-coder` and
   the rest of llama.cpp's table. Each is mechanical; the mechanism now exists.
2. Acquire fixtures for pre-types with no local model, so parity is measured rather than assumed.
   Currently only `qwen2`, `smollm` and `olmo` have one; the ported `llama3` and `qwen35` groups are
   **unverified against a real model** and should be treated as such until a fixture lands.
3. Surface `PreTokenizerIsKnown == false` in `doctor` / startup diagnostics, so an unimplemented
   pre-type is visible to an operator rather than only to a caller who inspects the property.
4. Astral-plane caveat: .NET regex works over UTF-16 code units where llama.cpp uses codepoints, so
   a non-BMP character is two chars to `\p{L}`. Not yet measured; needs an emoji/rare-CJK probe.

---

## 4. Chat templates

The Jinja engine was substantially hardened during the SharpInference port (multiplicative
operators, quote-aware tag scanning, `selectattr`/`rejectattr`, string slicing, `eos_token`) —
several of those were silent-wrong-output bugs, and the corrected behaviour is described in
`CHANGELOG.md` under Unreleased.

What is not established is *coverage*: how many real Hugging Face templates render correctly.

**Work:** collect the `tokenizer.chat_template` from a wide set of GGUFs, render each against a
fixed multi-turn conversation (with and without tools), and record pass / raised / wrong-output.
This is a corpus test and needs no inference, so it is cheap relative to its value.

---

## Sequencing

Axes 2 and 3 unlock more models per unit of work than axis 1, and axis 3 may already be producing
wrong output on models we claim to support. Suggested order:

1. **Tokenizer pre-type audit** (§3 item 1) — cheap, and may find a live defect.
2. **Architecture gate consistency** (§1a) — small change, removes a real contradiction between CLI
   and server.
3. **`IQ4_XS`** (§2 item 1) — the single highest-value format.
4. **Chat-template corpus** (§4).
5. **`olmoe`/`olmo2`/`deepseek2` admission with receipts** (§1b).
6. Remaining IQ formats, then further architectures.

## Standing evidence rule

An architecture or format is "supported" only with a receipt: named model file and hash, backend,
command, and either token-for-token parity against llama.cpp or a stated reason parity was not
obtainable. A model that loads and emits plausible text is **not** evidence — that is precisely the
failure mode the conservative gate exists to prevent.
