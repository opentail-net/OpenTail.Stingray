# GPT-OSS (`gpt-oss` GGUF architecture, llama.cpp class `openai_moe`) implementation plan

Status: PROPOSED, not started. Written 2026-09-02 from a direct reading of the vendored
reference (`examples/llama.cpp/llama.cpp/src/models/openai-moe.cpp`, all 177 lines, plus the
`ggml_swiglu_oai`/attention-sinks/`set_swa_pattern` primitives it calls into) — not from memory
or the user's pasted external plan alone. The user shared an external (ChatGPT-authored) planning
outline for this work; its phase structure is broadly reused below, but every specific technical
claim in it was checked against this codebase and the real reference before being kept, and one
claim was found to be **wrong** (see "Correction to the external plan" below) — flagged rather
than carried over silently, per this project's standing practice for external input.

## Correction to the external plan: MXFP4 is NOT missing

The external plan's Phase 5 treats MXFP4 (OpenAI's packed-4-bit MoE weight format) as "the main
new low-level feature," proposing a multi-stage implementation (loader → CPU dequant → CPU fused
GEMM → Vulkan). **This is already fully implemented and wired in this codebase**, checked directly
rather than assumed:

- `DType.MXFP4` is a registered dtype (`Tensor.cs:90`, block size 32 / 17 bytes-per-block —
  matches ggml's real MXFP4 layout: 32 packed 4-bit values + 1 scale byte).
- `Dequantize.cs` and `SimdKernels.cs` both have real MXFP4 code (`SimdKernels.cs:811`:
  `MatVecMxfp4` is wired into the main `MatVec` dispatch switch, not a dead/unused function).
  a — `MXFP4` is in `ModelCompatibility.cs`'s admitted weight-dtype list (`ModelCompatibility.cs:709`),
  alongside `NVFP4`, meaning any *already-admitted* architecture can load MXFP4 tensors today.

**What this changes about the plan**: gpt-oss's real remaining gap is entirely the *architecture
graph* — attention sinks, alternating sliding/full attention, biased QKV/output/MoE tensors, the
OpenAI-specific SwiGLU activation, and select-then-softmax MoE gating — not the weight format.
This makes the overall scope meaningfully smaller than the external plan's phase list implies;
Phase 5 below is correspondingly much shorter than that plan's "Phase 5" (MXFP4).

## What's genuinely new vs. what already exists

Verified directly against this codebase, not assumed from the external plan's own "likely
existing" column:

| Mechanism | Status in this codebase | Evidence |
|---|---|---|
| MXFP4 weight format | **Already implemented** | `SimdKernels.MatVecMxfp4`, admitted dtype list |
| RMSNorm, standard RoPE, GQA | Already implemented (used by dozens of architectures) | — |
| Per-layer alternating SWA/full attention | Partial — `IsSwaLayer`/`LayerHeadDim`-style per-layer flags exist for Gemma 4's 5:1 SWA pattern (`ModelGraph.cs`) | Needs a *2:1* (`swa_period=2`) variant, not a new mechanism class |
| Standard MoE top-k routing (softmax-then-select) | Already implemented (many MoE architectures) | — |
| **GPT-OSS's route-then-softmax gating** (select top-k by RAW router logit, THEN softmax only the selected subset) | **NOT implemented** — every existing MoE architecture here softmaxes (or sigmoids) the FULL expert set first, then selects | Confirmed against ggml's own gating-function switch (see below) |
| SwiGLU (standard, gate*up) | Already implemented | — |
| **`ggml_swiglu_oai`** (GPT-OSS's specific clamped, additive-not-multiplicative SwiGLU variant) | **NOT implemented** | Exact formula extracted below |
| **Attention sinks** | **NOT implemented anywhere in the live engine** — only referenced in this session's own DeepSeek4 alpha scaffolding (`DeepSeek4TensorSet.cs`), itself unwired/dead | grep confirms no other match |
| **Biased QKV/output/MoE-router/MoE-expert tensors** | Partial — several architectures already have Q/K/V/output biases (e.g. Qwen2); **per-expert MoE biases** (`ffn_gate_inp_b`, `ffn_gate_exps_b`, `ffn_down_exps_b`, `ffn_up_exps_b`) are a new shape not seen elsewhere in this codebase's MoE path | Not independently re-verified this session — flag to confirm during Phase 2 |

So the real net-new work is: attention sinks (one primitive), the OAI SwiGLU variant (one
primitive), select-then-softmax MoE gating (one primitive), per-expert MoE bias plumbing (a
wiring extension), and the 2:1 alternating-SWA layer pattern (a config variant of an existing
mechanism, not a new one). Four small, well-bounded primitives plus one wiring extension — this
is a substantially smaller lift than "MXFP4 + everything else," which is why this doc puts it
ahead of harder-to-bound work in the backlog (`00-current-work.md` had it listed as "needs
multiple substantial new mechanisms," written before this reference reading — that assessment is
now more precise, not overturned: the mechanisms are real, but fewer/smaller than it implied).

## Exact reference formulas, extracted directly (not paraphrased from the external plan)

### Attention sinks (`ggml_soft_max_add_sinks`, `examples/ggml/src/ggml-cpu/ops.cpp:5541-5551`)

A per-head learned scalar (`attn_sinks`, shape `[n_head]`) acts as one extra "virtual key" whose
score participates in the softmax denominator but contributes NOTHING to the weighted-V sum
(it has no corresponding V row). Given real per-key scores `s[0..n)` for a head and that head's
sink value `sink`:

```
max' = max(max(s), sink)
softmax_numerator[i] = exp(s[i] - max')          // for each real key i, same as always
denom = sum(exp(s[i] - max')) + exp(sink - max')  // the ONLY change: sink adds to the denominator
output[i] = softmax_numerator[i] / denom
```

This uniformly shrinks every real key's attention weight by a learned, per-head, per-token-
independent amount — the "sink" absorbs attention mass without ever being attended TO. Cheap to
implement (one extra scalar per head per attention call) and cheap to unit-test (assert weights
still sum to `< 1` when a sink is present, `== 1` when absent).

### GPT-OSS SwiGLU (`ggml_cuda_op_swiglu_oai_single`, `examples/ggml/src/ggml-cuda/unary.cuh:107-114`
— the CUDA reference was read since it's a plain scalar formula, easiest to extract cleanly;
the CPU implementation in `examples/ggml/src/ggml-cpu/ops.cpp:3325` implements the identical
formula and should be the actual port source, not the CUDA file, since this is a CPU-first port)

```
alpha = 1.702, limit = 7.0   (compile-time constants in the reference, deepseek4.cpp:2187-2188 —
                               "TODO: move to hparams?" in the reference itself, i.e. even upstream
                               treats these as provisional constants, not config-driven — port
                               them the same way, as constants, not as new hyperparameter fields)

x_clamped = min(gate, limit)
g_clamped = clamp(up, -limit, limit)
swish = x_clamped / (1 + exp(-alpha * x_clamped))     // SiLU with an alpha-scaled sigmoid, NOT plain SiLU
output = swish * (1 + g_clamped)                       // ADDITIVE combine, NOT gate*up like standard SwiGLU
```

The `(1 + g_clamped)` combine (not `gate * up`) is the detail most likely to get silently
mis-ported by someone pattern-matching against this codebase's existing `SiLuMul`-style kernels —
worth its own explicit unit test comparing against hand-computed values, not just "looks like
SiLU so reuse the existing kernel."

### Select-then-softmax MoE gating (`llama-graph.cpp:1970-1973`, `2048-2053`)

Every other MoE architecture in this codebase's existing code softmaxes (or sigmoids) the FULL
`n_expert`-wide logit vector, THEN selects the top-k. GPT-OSS's `SOFTMAX_WEIGHT` gating function
does the opposite:

```
probs = router_logits                          // NO activation applied yet -- raw logits
selected = top_k(probs, k)                      // select by RAW LOGIT value
selected_weights = softmax(selected_logits)     // softmax ONLY over the k selected logits
```

This is a real, order-of-operations difference from every existing MoE path here, not a cosmetic
one — get this backwards (softmax-then-select, this codebase's existing default) and expert
selection will differ from the reference whenever the top-k boundary sits somewhere a full-set
softmax and a top-k-only softmax would rank differently (which, per the `deepseek2` investigation's
own finding about routing-margin sensitivity, is not a rare edge case for MoE routers generally —
worth treating this ordering as load-bearing, not incidental).

### Alternating sliding/full attention (`llama-hparams.cpp:8-22`, `set_swa_pattern`)

`openai-moe.cpp:9-11` calls `set_swa_pattern(2)` (2 = alternate every layer) with the default
`dense_first=false`. The reference's own formula: `is_swa[il] = (il % n_pattern) < (n_pattern - 1)`.
For `n_pattern=2`: `is_swa[il] = (il % 2) < 1`, i.e. **even layers (0, 2, 4, …) are sliding-window,
odd layers (1, 3, 5, …) are full/global** — SWA-first, alternating strictly 1:1. Distinct from
Gemma 4's existing 5-SWA:1-global repeating pattern already supported by this codebase's
`IsSwaLayer`/per-layer-flag machinery (`ModelGraph.cs`) — same underlying mechanism (a per-layer
boolean), different period/phase, so this needs a new hyperparameter-driven pattern generator, not
a new mechanism class. Also note per-layer RoPE frequency base: `openai-moe.cpp:80-81` calls
`model.get_rope_freq_base(cparams, il)`/`get_rope_freq_scale(...)` per layer, and
`load_arch_hparams` (`openai-moe.cpp:13-15`) separately reads `{arch}.rope.freq_base_swa` — SWA
layers may use a DIFFERENT RoPE base than global layers. Confirm this is read correctly wherever
this codebase resolves per-layer RoPE base (needs checking against `ModelGraph.cs`'s existing
per-layer RoPE handling — not done this session).

## Model shape (from the reference's `load_arch_hparams`, `openai-moe.cpp:17-21` — the ONLY
tensor-shape facts confirmed from the reference this session; every OTHER specific number below
— hidden size, head counts, expert count, vocab size, YaRN parameters — comes from the user's
pasted external plan, NOT independently verified against a real GGUF this session, since no
gpt-oss checkpoint is on disk yet)

The reference recognizes exactly two sizes by layer count: `n_layer==24` → 20B, `n_layer==36` →
120B. The external plan's specific numbers for the 20B variant (32 local experts, top-4 routing,
hidden 2880, 64/8 Q/KV heads, 128-token sliding window, YaRN factor 32/orig-ctx 4096/beta_fast
32/beta_slow 1/theta 150000) are plausible and match public knowledge of `openai/gpt-oss-20b`, but
should be re-confirmed via `list-tensors`/`list-metadata` against a real downloaded GGUF before
being hardcoded anywhere in an implementation — treat them as a planning estimate, not a verified
spec, until that check happens.

**Real download sizes, checked 2026-09-02** (correcting the external plan's lack of a concrete
number): `ggml-org/gpt-oss-20b-GGUF` offers native MXFP4 at **12.1 GB** (the format the model was
actually trained/released in — no requantization needed, unlike every DeepSeek checkpoint this
project has dealt with so far) up to Q8_0 at 22.3 GB.
[ggml-org/gpt-oss-20b-GGUF](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF) — a **dramatically**
smaller and more practical download than any DeepSeek checkpoint tackled in
`docs/058-deepseek-full-lineage-implementation-plan.md` (99GB+ minimum there vs. 12.1GB here),
and this codebase's MXFP4 support means the checkpoint can be used AS-IS in its native format,
not requantized.

## Phased plan

Reusing the external plan's phase structure (it's sound), corrected for what's actually
missing/already-present per the audit above, and merged with this project's own conventions
(golden-verification ladder, `--allow-unverified-arch` gate discipline, `ModelCompatibility.cs`
documentation pattern established across the DeepSeek work).

### Phase 0 — Architecture mapping (this document)

Done. Table above.

### Phase 1 — Gate registration, deliberately blocked

Add `"gpt-oss"` to `ModelCompatibility.cs`'s known-architectures tracking with a comment
documenting the plan/status (matching the `deepseek4`/`deepseek32` comment-block pattern) —
**not** added to the admitted set. This lets the loader recognize and report on the architecture
(useful diagnostics, GGUF metadata inspection) without claiming support.

### Phase 2 — Forward graph on real weights directly (no separate "reference/BF16 first" stage
needed, unlike the external plan's Phase 2)

**Correction to the external plan's sequencing**: it recommends building against
dequantized/reference (BF16/F32) weights first and deferring MXFP4 to a later phase, reasoning
that MXFP4 is new/risky. Since MXFP4 is already implemented and already routes through the same
`SimdKernels.MatVec` dispatch every other dtype uses, there's no reason to build a separate
non-MXFP4 path first — the native 12.1GB MXFP4 GGUF can be the primary target from the start,
same as how every other architecture in this codebase is developed directly against its native
quantization. This removes an entire phase from the external plan.

Build the graph: RMSNorm → QKV projection (with bias) → RoPE (per-layer freq base for SWA vs.
global, see above) → attention (sinks + alternating SWA/full mask) → output projection (with
bias) → residual → RMSNorm → MoE (select-then-softmax gating, biased router + biased experts,
OAI SwiGLU) → residual. Four new primitives (attention sinks, OAI SwiGLU, select-then-softmax
gating, per-expert MoE bias) plus the 2:1 SWA pattern generator, wired into a single new
`GptOssForwardPass`-style class or extension — decide during implementation whether this fits
cleanly into the existing generic `ForwardPass.cs` (many of its pieces — GQA, biased QKV, standard
MoE with a *different* gating function passed as a parameter — already have precedent there) or
warrants its own class the way the DeepSeek work did (recommend checking `ForwardPass.cs`'s
existing MoE gating dispatch first — if it's already parameterized by gating function, this may
be a much smaller diff than a whole new forward-pass class).

Exit criterion: loads the real 12.1GB MXFP4 GGUF, runs one token through every layer with no
crash.

### Phase 3 — Attention isolated verification

Unit-test attention sinks and the OAI SwiGLU formula against hand-computed values BEFORE
running full generation (same discipline as the DeepSeek `DeepSeek4AlphaTests.cs` synthetic
tests) — cheap, catches formula-shape bugs (e.g. the `(1+g)` additive combine vs. a
naively-ported `gate*up`) before they're buried under 24+ layers of compounding error, the same
lesson the `deepseek2` investigation learned the hard way. Confirm the 2:1 SWA pattern produces
`is_swa = [true, false, true, false, ...]` for layers 0-23.

### Phase 4 — MoE routing parity

Per the external plan's own strongest point, worth keeping verbatim: **do not accept "numerically
close" router logits without checking expert-selection identity**. A tiny numeric difference can
flip which experts get selected even when the logits themselves look fine — exactly the failure
mode `docs/done/032-deepseek2-mla-yarn-moe-routing-investigation.md` spent multiple rounds
diagnosing. Build a router-logit/top-k-index trace (mirroring `STINGRAY_TRACE_ROUTERS`, already
built for the DeepSeek investigation) and check expert-index identity, not just weight magnitude,
against ground truth once a reference trace is available.

### Phase 5 — Golden parity harness

Reuse this project's established ladder (tensor inventory → single-layer trace → prefill logits →
greedy decode token-for-token parity → long-context/window-boundary checks at 127/128/129 tokens,
the sliding-window boundary specifically, and any YaRN-extended-context boundary once those
parameters are confirmed). Only admit `gpt-oss` to `ModelCompatibility.cs` after this passes, per
this codebase's standing policy for every other architecture gate.

### Phase 6 — Harmony chat format (separate deliverable, not blocking Phase 0-5)

The external plan's point stands and is worth keeping: model correctness (tokens → logits) and
chat-format correctness (Harmony response format, reasoning-effort levels, tool calls) are
separable. Treat Harmony as a chat-template/protocol adapter (this codebase already has a
`ChatTemplate.cs`/tool-call adapter architecture — `ToolCallAdapter.cs`,
`JsonToolArgumentConstraint.cs` — Harmony likely fits that existing extension point rather than
needing new infrastructure) — scope as its own follow-up once Phase 5 closes, not a Phase 0
blocker.

### Phase 7 — Vulkan/perf

Deliberately last, per the external plan's own reasoning (premature GPU work before CPU
correctness is a debugging trap) — and this codebase's own standing convention (CLAUDE.md rule 7:
performance pass only after correctness is verified and the architecture is admitted).

## Progress, 2026-09-02 — Phases 1-2 written, download running in parallel

Per user direction: started the native MXFP4 20B download
(`ggml-org/gpt-oss-20b-GGUF`, actual filename `gpt-oss-20b-MXFP4.gguf` — the first download
attempt guessed a lowercase filename and failed with a clean 404; the HF API's file listing gave
the real, uppercase-MXFP4 name) in the background via `hf download`, and wrote Phases 1-2 without
waiting for it to finish, exactly as instructed.

- `src/OpenTail.Stingray.Engine/GptOssAlpha.cs` — `GptOssHyperparams` (every GGUF key
  `load_arch_hparams` reads, openai-moe.cpp:3-22, including `IsSwaLayer` reproducing
  `set_swa_pattern`'s exact alternation formula) and `GptOssGraph` (the three new primitives:
  `SoftmaxWithSink`, `SwigluOai`, `SelectThenSoftmaxGate` — formulas exactly as extracted in this
  doc's own reference section above). 14 synthetic tests, all passing — including one
  specifically pinning that `SwigluOai` is additive (`up=0` does NOT zero the output, unlike
  standard multiplicative SwiGLU) and one pinning that `SelectThenSoftmaxGate`'s softmax runs
  ONLY over the selected subset (an excluded low logit provably doesn't affect the selected
  weights), the two details this doc flagged as most likely to be silently mis-ported.
- `src/OpenTail.Stingray.Engine/GptOssTensorSet.cs` — tensor resolution for every gpt-oss tensor
  (openai-moe.cpp:24-58), reusing `DeepSeek4TensorRef` as the resolved-tensor wrapper (now shared
  across three architecture ports, not just two).
- `src/OpenTail.Stingray.Engine/GptOssForwardPass.cs` — full `IForwardPass`: GQA attention with
  biased QKVO, per-head attention sinks, alternating sliding/full-window causal masking (window
  truncation implemented as a masking rule over the same growing cache, not a physically
  separate SWA cache — flagged as a simplification of the reference's real paged-cache-policy
  distinction, Phase 6's still-open KV-cache-policy question), and MoE with per-expert biases +
  `SwigluOai` + `SelectThenSoftmaxGate`.
- **Real bug caught before it shipped**: this file was originally written using interleaved RoPE
  (this codebase's default assumption for most architectures, copied by habit from the DeepSeek
  ports without checking) — WRONG for gpt-oss. Checked `llama-model.cpp`'s rope-type switch
  directly: `LLM_ARCH_OPENAI_MOE` groups with `LLAMA_ROPE_TYPE_NEOX` (alongside qwen3next/mimo2/
  mellum), not the `LLAMA_ROPE_TYPE_NORM`/interleaved group llama/deepseek fall into. Fixed to
  `ApplyRopeNeox` before any test ran against it — a concrete example of why this project's
  practice of checking each architecture's specific convention against the reference, rather than
  assuming the previous port's convention carries over, matters even for "boring" mechanics like
  RoPE pairing.
- `ModelCompatibility.cs` updated with a `gpt-oss` comment block matching the established
  DeepSeek pattern (alpha code exists, unwired, not admitted).
- Builds clean, 0 warnings; all 14 new tests pass.
- **Known, deliberate gap carried into the forward pass**: RoPE is plain (non-YaRN). The
  external plan's specific YaRN parameters (factor 32, orig-ctx 4096, beta_fast 32, beta_slow 1,
  theta 150000) were never independently confirmed against a real GGUF this session — rather than
  bake unverified numbers into the constructor, this is left as an explicit, documented gap,
  parallel to (and reusing the exact same fix pattern as) deepseek32's now-closed YaRN gap.
- **Not done yet**: real-weight verification (the download was still in progress, ~134MB of
  12.1GB, when this update was written) — Phases 3-5 (attention/MoE-routing isolated verification,
  golden parity) all depend on it. Per-layer RoPE frequency base (global vs. SWA) IS implemented,
  independent of the YaRN gap above.

## Open question for the user (superseded above for the download; still open for the rest)

The download decision is resolved (in progress). Once it completes: confirm whether to proceed
straight to Phase 3 verification against it, and whether the external plan's specific YaRN
parameters should be trusted as-is or re-derived from the real GGUF's own metadata once available
(`list-metadata` against the downloaded file will settle this cheaply).
