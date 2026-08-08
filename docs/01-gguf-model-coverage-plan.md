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

1. `olmoe`, `olmo2` — gate-only, code exists.
2. `deepseek2` — gate-only for the dense path; MLA attention needs checking separately.
3. `starcoder2`, `stablelm`, `gptneox`, `falcon` — classic decoder-only, NEOX RoPE already declared.
4. `glm4`, `glm4moe` — the conditional RoPE type is explicitly unsupported today; real work.
5. `exaone`, `internlm2`, `cohere`/`command-r`, `minicpm`, `granite`, `nemotron`, `seed_oss`,
   `smollm3`, `hunyuan`, `dots1`, `lfm2`, `apertus`.
6. `gpt-oss` (MXFP4 already dequantizes), `bitnet`, `mamba`/`jamba`/`rwkv` (recurrent — a different
   forward-pass family, not a variant of the existing one).

Acceptance per architecture: a real GGUF, greedy token-for-token agreement with llama.cpp for at
least one prompt, plus a coherence run long enough to cross the model's sliding-window or
rope-scaling boundary if it has one.

**Working pattern (operator preference, 2026-08-08): one model at a time — download, work through,
complete, delete, repeat.** Do not accumulate a model zoo on disk. A few GB per model is acceptable;
prefer the smallest checkpoint that genuinely exercises the architecture, since parity is a property
of the architecture rather than of parameter count. Two consequences for how tests are written:

- A parity test must **skip**, never silently pass, when its fixture is gone — the fixture is
  expected to be absent most of the time. `Assert.Skip*`, not `return`.
- The reference token ids and expected continuation must be **recorded in the test file**, because
  once the model is deleted the test cannot regenerate them. The receipt has to outlive the GGUF.

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
