# Task: add GPT-NeoX (Pythia) support to OpenTail.Stingray

## Read this whole file before touching anything. Do not skip sections.

You are adding support for one GGUF architecture (`gptneox`, which covers EleutherAI's Pythia
model family) to a C# LLM inference engine. This document is meant to be **complete** — you should
not need to explore the wider codebase to do this task. Where you do need to look at something
outside this folder, the exact file and line range is given.

## Hard rule: where you may write

**You may ONLY create or edit files inside this folder
(`scratch/gptneox-layernorm/`), specifically inside `scratch/gptneox-layernorm/output/`.**

Do NOT edit anything under `src/` or `tests/` directly. The files you need are already copied into
`scratch/gptneox-layernorm/output/`:

- `SimdKernels.cs` — copy of `src/OpenTail.Stingray.Cpu/SimdKernels.cs`
- `ModelGraph.cs` — copy of `src/OpenTail.Stingray.Core/ModelGraph.cs`
- `ForwardPass.cs` — copy of `src/OpenTail.Stingray.Engine/ForwardPass.cs`
- `ModelCompatibility.cs` — copy of `src/OpenTail.Stingray.Engine/ModelCompatibility.cs`
- `ApertusGreedyParityTests.reference.cs` — a **finished, working example** of the exact kind of
  test you need to write for this task, for a different architecture (Apertus) that was added the
  same way. Use it as your template for the new test file, which you should create as
  `scratch/gptneox-layernorm/output/GptNeoxGreedyParityTests.cs`.

Edit the copies in `output/`. When you are done, write a file
`scratch/gptneox-layernorm/output/DONE.md` summarizing exactly what you changed (which files, what
you added, what you tested, what passed/failed) and stop. A human will review your diff against the
real `src/` files and merge it themselves. **Do not attempt to copy your changes into `src/`
yourself.**

You will still need to run `dotnet build` / run tests to verify your work — that's fine, the build
reads from the real `src/` tree. To actually test your changes, after you've finished editing the
copies in `output/`, copy them OVER the real files (`cp output/X.cs ../../src/.../X.cs`), build,
test, and if anything fails, fix it in `output/` and re-copy. This is fine — the point of the
restriction is that your FINAL merged state lives in `output/` for review, not that you can never
touch the real tree while iterating. Just make sure the last thing you do before finishing is
re-copy your final `output/` files over the real ones so build/test reflects your actual final
answer, confirm it passes, then leave the real tree in that state (a human is going to review with
`git diff` anyway, so a real-tree diff existing is fine and expected — what's not fine is if
`output/` and the real tree disagree at the end).

## Why this document is this detailed

A previous architecture (Apertus, added earlier the same way) had a bug that was very easy to miss:
a required mathematical transform lived in a totally different function than the one that looked
like "the formula" at first glance, so a plausible-looking implementation produced fluent-looking
**wrong** output with no error, no crash, no NaN — just wrong text that still looked like language.
That is the standing risk on this whole codebase: a model that loads and produces fluent text is
NOT evidence it's correct. Every number in this document was checked against the actual reference
source, not assumed. Do the same for anything you're unsure about — check
`examples/llama.cpp/llama.cpp/` (a local copy of the reference C++ implementation, MIT licensed)
rather than guessing or relying on general knowledge of these algorithms. A "GELU" or "LayerNorm"
you remember from training data may not be byte-identical to the exact variant this specific
codebase needs.

---

## 1. What GPT-NeoX / Pythia is, and what it needs

The engine already implements RMSNorm-based, gated-SiLU-FFN transformers (Llama/Qwen/Mistral
family) and has recently added two more architectures (Granite, Apertus) that needed small
additions. GPT-NeoX needs three things none of those did:

1. **LayerNorm instead of RMSNorm**, with a learned bias in addition to the learned scale. Every
   norm in the model (attention norm, FFN norm, final output norm) uses this, not RMSNorm.
2. **A non-gated FFN**: `down(gelu(up(x)))`, not `down(silu(gate(x)) * up(x))`. There is no
   `ffn_gate` tensor. (This part is low-risk: Apertus already added a non-gated FFN path for its
   own activation, xIELU. You are adding a second non-gated activation, GELU, reusing the same
   dispatch shape.)
3. **Parallel residual** (for Pythia specifically — Pythia sets `use_parallel_residual=true`; other
   GPT-NeoX-family checkpoints might set it false, and you should support both, branching on the
   metadata flag, matching the reference implementation exactly): attention and FFN both read from
   the SAME normalized input (not sequentially chained), and the layer output is
   `input + attn_out + ffn_out` (a three-way sum), not the usual two sequential
   normalize-then-residual-add steps.

Also needed, all mechanical, all bias terms that don't exist anywhere in the engine yet:
- Bias on the attention norm, FFN norm, and final output norm (LayerNorm always has these; the
  engine's RMSNorm never does, so this is new plumbing, not a reuse of anything).
- Bias on the FFN up and down projections (`ffn_up.bias`, `ffn_down.bias`). Q/K/V and attention
  output bias (`attn_output.bias`) ALREADY EXIST in the engine (`_hasAttnBias` /
  `_hasAttnOutputBias`, used by Qwen models) — reuse those, do not build new plumbing for them.

## 2. The reference checkpoint and its confirmed facts

Already downloaded and inspected: `models/pythia-160m-Q8_0.gguf` (174.6 MB,
`mradermacher/pythia-160m-GGUF`, `pythia-160m.Q8_0.gguf`, from `EleutherAI/pythia-160m`,
**Apache-2.0**, confirmed clean — this is important, do not substitute a different checkpoint
without checking its license against MIT/Apache-2.0/BSD/MPL first).

Confirmed metadata (from `dotnet run --project src/OpenTail.Stingray.Cli -c Release --
list-metadata -m models/pythia-160m-Q8_0.gguf`), do not re-derive these, they are already measured:

```
general.architecture                  gptneox
gptneox.attention.head_count          12
gptneox.attention.layer_norm_epsilon  1E-05
gptneox.block_count                   12
gptneox.context_length                2048
gptneox.embedding_length              768
gptneox.feed_forward_length           3072
gptneox.rope.dimension_count          16
gptneox.use_parallel_residual         true
gptneox.vocab_size                    50304
tokenizer.ggml.model                  gpt2
tokenizer.ggml.pre                    olmo
```

What this tells you:

- `head_dim = embedding_length / head_count = 768 / 12 = 64`, but
  `rope.dimension_count = 16` — this is **partial RoPE**: only the first 16 of each head's 64 dims
  get rotated, the rest pass through unrotated. The engine already fully supports this
  (`ModelHyperparams.RopeDim`, used by `qwen35moe`) — you do NOT need to build partial-RoPE
  support, just make sure `RopeDim` gets read from `gptneox.rope.dimension_count` for this
  architecture. Check `ModelGraph.cs` for how `RopeDim` is currently populated (search for
  `RopeDim = ` in the returned `ModelHyperparams` and for `ropeDim` earlier in the method) and
  confirm/extend so it picks up this key for `gptneox`. **Verify this by checking the actual value
  the engine resolves at runtime is 16, not 64** — do not assume the existing code already handles
  it for this specific arch string just because the mechanism exists.
- `tokenizer.ggml.pre = olmo` — already fully implemented (folds onto the GPT-2 pretokenizer
  cascade in `PreTokenizerPatterns.cs`). You do not need to touch tokenizer code at all for this
  checkpoint. If `tokenizer.Encode("The capital of France is")` doesn't match the reference ids
  below, the bug is NOT in the tokenizer — look at whatever you just changed instead.
- RoPE type: check `llama_model_rope_type()` in
  `examples/llama.cpp/llama.cpp/src/llama-model.cpp` for `LLM_ARCH_GPT_NEOX` (or `GPTNEOX`) to
  confirm NEOX vs NORM convention before assuming — the plan doc for this repo
  (`docs/01-gguf-model-coverage-plan.md`) already asserts NEOX for this family but you should
  re-confirm against the actual source, not trust a doc summary, same standard as everything else
  in this file.

## 3. The reference C++ implementation (already read for you, verbatim)

This is the file the real inference engine (llama.cpp) uses for this architecture:
`examples/llama.cpp/llama.cpp/src/models/gptneox.cpp`. Full contents at the time this plan was
written:

```cpp
#include "models.h"

void llama_model_gptneox::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
    ml.get_key(LLM_KV_USE_PARALLEL_RESIDUAL,   hparams.use_par_res);
    // (switch on n_layer()/n_ff() for a human-readable model size label — not relevant to you)
}

void llama_model_gptneox::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm   = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd}, 0);
    output        = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "bias", i),   {n_embd}, 0);

        layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa}, 0);
        layer.wqkv_b = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "bias", i), {n_embd + 2*n_embd_gqa}, 0);

        layer.wo   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);
        layer.wo_b = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i),   {n_embd}, 0);

        layer.ffn_norm   = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_norm_b = create_tensor(tn(LLM_TENSOR_FFN_NORM, "bias", i),   {n_embd}, 0);

        layer.ffn_down   = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
        layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i),   {n_embd}, 0);

        layer.ffn_up     = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "bias", i),   {n_ff}, 0);
    }
}

// graph construction, per layer (paraphrased from build_arch_graph — the actual control flow):

for each layer:
    cur = LayerNorm(inpL, attn_norm, attn_norm_b)          // norm reads inpL, NOT any prior residual buffer

    // self-attention
    Qcur, Kcur, Vcur = split(wqkv * cur + wqkv_b)          // NOTE: fused QKV weight+bias here in
                                                            //   llama.cpp; the engine already keeps
                                                            //   Q/K/V as separate weights (_wq/_wk/_wv)
                                                            //   with separate bias (_bq/_bk/_bv) — you
                                                            //   do NOT need to fuse them, this is just
                                                            //   how the GGUF stores it; the engine's
                                                            //   GGUF loader already splits fused qkv
                                                            //   tensors into separate _wq/_wk/_wv
                                                            //   automatically for every other arch
                                                            //   that ships them fused (check by
                                                            //   grepping ForwardPass.cs / GgufModel for
                                                            //   "wqkv" or "attn_qkv" handling before
                                                            //   assuming you need to add this — if it's
                                                            //   already handled, you have nothing to do
                                                            //   here beyond the usual _wq/_wk/_wv/_bq/_bk/_bv
                                                            //   resolution every other arch already uses)
    Qcur = RoPE(Qcur)   // NEOX convention, partial: only first rope.dimension_count=16 dims rotated
    Kcur = RoPE(Kcur)
    cur = attn_output_projection(attention(Qcur, Kcur, Vcur))   // wo * attn_out + wo_b
                                                                  // scale = 1/sqrt(head_dim) = 1/sqrt(64), no override

    if use_parallel_residual:
        attn_out = cur                                      // stash the attention branch's output
        cur = LayerNorm(inpL, ffn_norm, ffn_norm_b)          // *** reads inpL again, the SAME
                                                              //     un-modified input attention read,
                                                              //     NOT cur+inpL, NOT any post-attention
                                                              //     residual buffer *** — this is the
                                                              //     part that's easy to get wrong, see
                                                              //     the explicit warning in section 5.
        cur = down(gelu(up(cur)))                            // FFN, no gate, plain GELU (see section 4)
        cur = cur + inpL                                     // ffn_out + original input
        cur = cur + attn_out                                 // + attention branch's output
        inpL = cur                                           // this is the WHOLE layer's residual
                                                              //   update — there is exactly one
                                                              //   addition step per layer here, not two
    else:
        ffn_inp = cur + inpL                                 // ordinary sequential residual after attention
        cur = LayerNorm(ffn_inp, ffn_norm, ffn_norm_b)        // reads ffn_inp, i.e. the ALREADY-updated
                                                              //   residual — this is the SAME shape the
                                                              //   engine's existing RunTrunk loop already
                                                              //   does for every other architecture, so
                                                              //   the sequential branch needs almost no
                                                              //   new code beyond swapping RmsNorm for
                                                              //   LayerNorm and the FFN dispatch
        cur = down(gelu(up(cur)))
        cur = cur + ffn_inp                                  // ordinary second residual add
        inpL = cur

final: cur = LayerNorm(inpL, output_norm, output_norm_b)
       logits = output_weight * cur   // no bias on the final output projection
```

## 4. Exact formulas

**LayerNorm** (mean/variance normalize + learned scale + learned bias — this does NOT exist in the
engine anywhere; every existing norm is RMSNorm, which has no mean-subtraction and no bias):

```
mean = sum(x[i] for i in 0..n) / n
var  = sum((x[i] - mean)^2 for i in 0..n) / n
y[i] = (x[i] - mean) / sqrt(var + eps) * weight[i] + bias[i]
```

`eps` comes from `gptneox.attention.layer_norm_epsilon` (already read into
`ModelHyperparams.RmsNormEps` generically for every arch via
`GetFloat(metadata, $"{arch}.attention.layer_norm_rms_epsilon", 1e-5f)` — **check whether this
generic read also covers the non-"_rms_" key name `layer_norm_epsilon` that GPT-NeoX actually
declares; it may not, since the key string is different, not just missing a fallback.** Confirm by
checking what `hp.RmsNormEps` actually resolves to for this checkpoint before assuming it's right —
it is used only as a fallback constant `1e-5f` in `GetFloat` if the key lookup fails, and Pythia's
value happens to also be `1e-5`, so a wrong key lookup could accidentally look like it worked. Do
not let that fool you — fix the actual key lookup, don't rely on the fallback matching by
coincidence for this one checkpoint.

**GELU** — already implemented in this codebase, just needs a non-multiplying variant. The exact
formula (confirmed against `examples/llama.cpp/llama.cpp/ggml/src/ggml-cpu/vec.h`,
`ggml_gelu_f32`):

```
gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * x * (1 + 0.044715 * x^2)))
```

This is EXACTLY the formula already implemented in
`scratch/gptneox-layernorm/output/SimdKernels.cs` as `GeluTanhMul_Scalar` / `GeluTanhMul` (search
for `kAlpha = 0.7978845608028654f` — that's `sqrt(2/π)` — and `kBeta = 0.044715f`). The existing
kernel computes `gelu(gate[i]) * up[i]` (it was built for Gemma 4, which IS gated). You need a
variant that just computes `gelu(x[i])` in place, with no second array to multiply against —
follow the pattern of `XieluInPlace` in the same file (search for it — it's a simple in-place
unary kernel, added for Apertus, a good template for "new unary activation, in place, scalar is
fine, no AVX required for a first correctness pass").

## 5. Explicit warnings — things that will look right and be wrong

1. **The parallel-residual FFN-norm input.** Read section 3's pseudocode again. In the
   `use_parallel_residual` branch, the FFN's LayerNorm reads `inpL` — the layer's ORIGINAL input,
   before attention ran — not the attention output, not any post-attention residual sum. If you
   reuse the engine's existing per-layer buffer flow without checking this carefully, it is very
   easy to accidentally feed the FFN norm the wrong buffer (e.g. `_hidden` after it's already been
   overwritten by the attention output projection) and get a plausible-looking but wrong result.
   Concretely, in `RunTrunk` (`ForwardPass.cs`, the single-token decode path — search for `private
   ReadOnlySpan<float> RunTrunk`), the existing code does:
   ```
   Copy(_residual, _hidden, _embDim);              // save residual  (this IS the "inpL" for this layer)
   FastRmsNorm(_normBuf, _hidden, attnNormW, ...);  // norm for attention
   ... attention ...
   FusedMatVec(_hidden, wo, attnOut, ...);          // _hidden now holds the attention OUTPUT, not inpL anymore
   AddInPlace(_hidden, _residual, _embDim);         // _hidden = attn_out + residual (sequential path only!)
   Copy(_residual, _hidden, _embDim);               // residual updated to include attention
   FastRmsNorm(_normBuf, _hidden, ffnNormW, ...);   // FFN norm reads the ALREADY-residual-added _hidden
   ```
   For parallel residual you must NOT let `_hidden` get overwritten with the residual-added value
   before the FFN norm runs — the FFN norm needs the buffer that was saved into `_residual` BEFORE
   attention (i.e. skip the `AddInPlace(_hidden, _residual, ...)` step entirely at that point, keep
   `_residual` untouched from its very first save, run the FFN norm against `_residual` — not
   `_hidden` — and only do the three-way sum (`_residual + attn_out + ffn_out`) once, at the very
   end of the layer, writing the result into `_hidden` for the next layer). Write this as a
   **separate code branch** (`if (_hp.UseParallelResidual) { ... } else { <existing sequential code,
   untouched> }`), not as a modification of the existing sequential path with extra conditionals
   sprinkled through it — this keeps every other architecture's code path byte-identical to before
   and makes your new branch easy to review in isolation.
2. **Do this same restructuring in BOTH places, and make them agree.** The engine has two separate
   implementations of "run the transformer trunk": `RunTrunk` (single-token decode, called by
   `Forward`) and `PrefillCore` (batched prefill for prompts with more than 1 token — search
   `ForwardPass.cs` for `private ReadOnlySpan<float> PrefillCore`). Both need the parallel-residual
   branch, independently, because they are separate code (this is not a bug in the existing code,
   it's just how the batched-vs-single-token paths are structured here). **After you're done, write
   a test that prefills a short prompt in one call and ALSO steps the same tokens through decode one
   at a time, and asserts the two produce the same argmax logits at the end** — this exact pattern
   already exists for Apertus and Granite; copy it. Search
   `ApertusGreedyParityTests.reference.cs` for `Apertus_DecodeStepwise_AgreesWithSinglePassPrefill`
   and copy that test's structure, renamed for GPT-NeoX. If you skip this test, you have no way to
   know whether the two paths agree, and past architectures have gotten this wrong silently (a
   0-to-few-token prompt happens to route through the single-token path with the batched path never
   exercised, so tests can pass while the batched path is actually broken).
3. **Don't guess formulas from memory. Check the actual reference.** The Apertus work earlier found
   a bug where the "obvious" formula (read directly from the compute kernel) was subtly wrong
   because a required transform lived in a different function one layer up in the call stack (a
   thin wrapper function that packs already-transformed parameters before the kernel ever sees
   them). LayerNorm and GELU are common enough that you may feel confident you already know the
   formula — check them against `examples/llama.cpp/llama.cpp/` anyway, the same way this document
   did (see section 4's citations). If GPT-NeoX turns out to need something not described in this
   document, that means this document's author (a previous agent) missed it — go find the actual
   answer in the reference source, do not guess.
4. **`ffn_up.bias` has shape `{n_ff}` (the intermediate dim, 3072 for this checkpoint), `ffn_down.bias`
   has shape `{n_embd}` (768).** Check you're adding the right bias to the right buffer at the right
   point — up-projection output is `n_ff`-wide (add its bias there, BEFORE the GELU activation, not
   after), down-projection output is `n_embd`-wide (add its bias there, before it joins the
   residual). Get this backwards and you'll add a 768-wide bias array to a 3072-wide buffer or vice
   versa, which either throws an out-of-bounds native memory access (visible, but ugly) or reads
   garbage past the buffer (worse — silent, plausible-looking wrong numbers).
5. **Do not run the full test suite while iterating.** It takes about 18 minutes and produces a huge
   amount of output. Use targeted single-method runs instead — see section 7.

## 6. Step-by-step implementation order

Work in this order. Each step should build cleanly before you move to the next.

### Step A — `SimdKernels.cs`: add two kernels

1. `LayerNormInPlace` (or `LayerNorm(output, input, weight, bias, size, eps)` — non-in-place is
   probably easier to reason about, follow the `RmsNorm(output, input, weight, size, eps)`
   signature shape as your template, just adding a `bias` parameter and the mean-subtraction step).
   Scalar implementation is fine for correctness — do not attempt AVX vectorization on a first pass;
   note in a comment that a SIMD form is a follow-up, matching how `XieluInPlace` and the IQ-quant
   dequantizers in this codebase are already documented as "correctness first, SIMD later."
2. `GeluInPlace(float* x, int n)` — the non-gated GELU, following `XieluInPlace`'s shape (search
   for it in the same file — it's short). Use the exact formula from section 4. Consider whether to
   share the `kAlpha`/`kBeta` constants with the existing `GeluTanhMul_Scalar` (they're currently
   local `const` inside that method) rather than redefining them — your call, either is fine as
   long as the numeric values match exactly.

### Step B — `ModelGraph.cs`: read the new metadata and detect the shape

Add to `ModelHyperparams`:
- `bool UseParallelResidual` (default `false`)
- Bias arrays / flags for norm bias and FFN bias. Follow the existing pattern: `HasAttnBias` /
  `HasAttnOutputBias` are plain `bool` fields, detected once (not per-layer, since GPT-NeoX has
  bias on every layer uniformly) by probing tensor presence — search `ModelGraph.cs`/`ForwardPass.cs`
  for how `hasAttnBias`/`_hasAttnBias` gets set (tensor-inventory probe, e.g.
  `tensorSource?.FindTensor("blk.0.attn_q.bias") is not null`) and add equivalent
  `HasNormBias` and `HasFfnBias` flags the same way, probing for `blk.0.attn_norm.bias` and
  `blk.0.ffn_up.bias` respectively.
- A `LayerNormEps` mechanism, OR fix the existing `RmsNormEps` key lookup to also read
  `{arch}.attention.layer_norm_epsilon` (GPT-NeoX's actual key name, confirmed in section 2) as a
  fallback/alternative to `{arch}.attention.layer_norm_rms_epsilon` (see the warning in section 4 —
  do not let the coincidental default value hide a real key-lookup bug).
- `UseParallelResidual` should be read from `{arch}.use_parallel_residual` as a plain bool
  (`GetBool`-style helper — check if one already exists in `ModelGraph.cs`; if not, it's a two-line
  addition following `GetFloat`'s shape).
- **A signal that this whole non-RMSNorm path is active at all** — you need SOME way for
  `ForwardPass.cs` to know "this model uses LayerNorm, not RMSNorm" so it doesn't have to guess.
  Consider whether the tensor-inventory-based `HasNormBias` flag above is sufficient as that signal
  (a model with bias on its norm tensors is, in every architecture this engine will ever load, a
  LayerNorm model — RMSNorm-with-bias is not a thing any of these architectures do), or whether you
  want a separate explicit flag for clarity. Either is fine; document your choice.
- Also confirm/wire `RopeDim` picks up `gptneox.rope.dimension_count` (see section 2's warning about
  this — check don't assume).

### Step C — `ForwardPass.cs`: the norm/bias tensor loading, the FFN dispatch, and the two trunk
implementations

1. Wherever `_attnNorm[i] = ResolveTensor(...)`, `_ffnNorm[i] = ResolveTensor(...)` etc. currently
   happen (constructor, search for `_attnNorm[i] = ResolveTensor`), conditionally also resolve
   `_attnNormBias[i]`, `_ffnNormBias[i]`, `_outputNormBias` when `HasNormBias` is true. New arrays
   need declaring alongside the existing `_attnNorm`/`_ffnNorm` field declarations.
2. Wherever `_wUp[i]`/`_wDown[i]` get resolved, conditionally also resolve `_bFfnUp[i]`/
   `_bFfnDown[i]` when `HasFfnBias` is true. (Q/K/V/output-projection bias already has this exact
   pattern via `_hasAttnBias`/`_bq`/`_bk`/`_bv`/`_bo` — copy that shape, don't invent a new one.)
3. In `DenseFfn` (search for `private void DenseFfn`), add a third branch alongside the existing
   "no gate → xIELU" (Apertus) and "gated → SiLU" (everyone else) branches: "no gate AND
   `HasFfnBias`/whatever signal you chose → GELU with bias". Note both Apertus's non-gated path and
   this new one share "no `ffn_gate` tensor" as their trigger — you need a way to distinguish which
   NON-gated activation to apply (xIELU vs GELU). The cleanest signal is probably "does
   `hp.XieluAlphaN` exist" vs "does this model have FFN bias / use LayerNorm" — pick something that
   can't be true for both at once, and assert that invariant somewhere (e.g. in `ModelGraph.cs`,
   right after computing both, `Debug.Assert` or just a defensive check that a model isn't
   simultaneously flagged for both non-gated variants — this should never happen for any real
   checkpoint, but a bug that made it happen should fail loudly, not silently pick one).
4. Do the same in `PrefillCore`'s dense-FFN dispatch (search for `MatMulBatchedDualCached(batchFfnGate`
   — the surrounding `if (_wGate[layer].DataPtr is null)` block is Apertus's branch, add a sibling
   branch the same way, using the batched kernels `MatMulBatchedCached` the same way Apertus's
   branch does, just swapping `XieluInPlace` for your new `GeluInPlace` and adding the bias adds).
5. Implement the parallel-residual restructuring in `RunTrunk` AND `PrefillCore`, per section 5's
   explicit warning. This is the part most likely to need care — reread section 5 point 1 before
   writing this code, and write it as a clearly separated `if (_hp.UseParallelResidual) { ... }
   else { <untouched existing code> }` branch in each function, not threaded through the existing
   sequential logic.
6. Wire `LayerNormInPlace` in everywhere `FastRmsNorm`/`SimdKernels.RmsNorm` is currently called for
   attn_norm/ffn_norm/output_norm — but ONLY when this model uses LayerNorm (your Step B signal),
   leaving the RMSNorm calls completely untouched for every other architecture. Same "add a branch,
   don't rewrite the existing path" principle as everywhere else in this task.

### Step D — `ModelCompatibility.cs`: admit `gptneox`

Add `"gptneox"` to the allowlist `HashSet<string>`, with a comment following the exact style of the
existing entries (`granite`, `apertus`, `smollm3` — read them for the tone/level of detail expected:
what the architecture needed, what receipt backs the admission, one or two sentences).

## 7. How to build and test (exact commands — do not deviate)

Build only the projects you touched, not the whole solution, while iterating:

```
dotnet build tests/OpenTail.Stingray.Tests.ForwardPass -c Release
```

Run ONLY your new test(s) by name, never the whole suite while iterating (the whole suite takes
about 18 minutes and its output is not useful for debugging a single new architecture):

```
tests/OpenTail.Stingray.Tests.ForwardPass/bin/Release/net10.0/OpenTail.Stingray.Tests.ForwardPass.exe -method "*GptNeox*"
```

**Never pipe a long-running test command through `tail`** — if it hangs or produces more output
than expected, you want to see all of it, not have it silently truncated.

### Getting the reference (llama.cpp) output to compare against

Two binaries matter, both already built and present at `tools/llama.cpp/`:

```
tools/llama.cpp/llama-tokenize.exe -m models/pythia-160m-Q8_0.gguf -p "The capital of France is" --ids --no-bos
```

gives you the reference PROMPT token ids.

```
tools/llama.cpp/llama-completion.exe -m models/pythia-160m-Q8_0.gguf -p "The capital of France is" -n 24 --temp 0 --top-k 1 --seed 0 -no-cnv --override-kv tokenizer.ggml.add_bos_token=bool:false
```

gives you the reference GREEDY CONTINUATION. **Use `llama-completion.exe`, not `llama-cli.exe`** —
this specific build of `llama-cli.exe` does not support `-no-cnv` properly (it prints an error to
stderr but then silently falls back to an interactive chat mode instead of raising, which looks
like the process is hanging forever if you don't know to expect it — it isn't hanging, it's waiting
for interactive input that will never come. If a `llama-cli.exe` run seems to hang, kill it and use
`llama-completion.exe` instead; don't wait for it or increase the timeout, it will never finish).

**Also run these with `run_in_background` or generous timeouts and CHECK that they actually
completed rather than assuming** — model loading takes a few seconds, generation of 24 tokens on a
160M model should take well under 30 seconds total on typical hardware. If a run is taking much
longer than that, something is wrong (check CPU usage of the process — genuine work shows CPU time
climbing at roughly wall-clock rate; a stuck process shows almost none).

The generated continuation string's leading space belongs to the first generated token, not the
prompt — see `ApertusGreedyParityTests.reference.cs`'s `ReferencePrefix`/`ReferenceContinuationFull`
constants for the exact convention to follow.

### Writing the test

Copy the STRUCTURE of `ApertusGreedyParityTests.reference.cs` closely:
- `Assert.SkipWhen(path is null, ...)` when the model file is missing — **never** a silent `return`.
  This is a hard requirement in this codebase: a parity test must skip, not silently pass, when its
  fixture is absent.
- Record the prompt token ids and reference continuation as `const`/`static readonly` fields with a
  doc comment citing the exact `llama-tokenize`/`llama-completion` commands used, so the receipt
  survives after the GGUF is deleted (see section 8 — you WILL delete the GGUF when done).
- Include a `..._DecodeStepwise_AgreesWithSinglePassPrefill` test (see section 5, point 2) — this is
  not optional, it is the only thing that would have caught several real bugs in past architectures.
- If the full 24-token continuation doesn't match exactly, that is not automatically a failure to
  fix — some past architectures (OLMoE, Apertus) were admitted on a partial-prefix match plus
  evidence the divergence is a plausible near-tie (check the top-5 logit candidates at the
  divergence point — if the reference token isn't even close to being in the top-5, that is NOT a
  near-tie and probably IS a real remaining bug; if it's a near-tie, or the divergence produces
  still-coherent continued text rather than degenerate garbage, that is acceptable evidence,
  following the precedent in `docs/01-gguf-model-coverage-plan.md` §1b, §1g). Use your judgement
  the same way those sections describe, and write down your reasoning in the test's doc comment and
  in `DONE.md`, the same way those sections did.

## 8. When you're done

1. All new/changed code lives in `scratch/gptneox-layernorm/output/` (mirrored into the real `src/`
   tree so you could build/test it — see the note in the "Hard rule" section above).
2. Delete `models/pythia-160m-Q8_0.gguf` once your test passes (or once you've recorded why it
   doesn't and are stopping) — this repo's working pattern is
   download → work through → complete → delete, one model at a time, never accumulate a model zoo.
3. Write `scratch/gptneox-layernorm/output/DONE.md`: what you changed, what the test results were
   (paste the actual test runner output, not a summary), what you're unsure about (if anything), and
   any bugs you found and fixed along the way worth recording (the way this document records the
   Apertus xIELU bug — future readers benefit from knowing what looked right and wasn't).
4. Do not touch `docs/00-current-work.md`, `docs/01-gguf-model-coverage-plan.md`, or
   `src/OpenTail.Stingray.Engine/ModelCompatibility.cs` in the real tree — write your proposed
   documentation updates into `scratch/gptneox-layernorm/output/docs-changes.md` instead (plain text
   describing what should be added/changed and where) for the human reviewer to apply. (Your
   `ModelCompatibility.cs` COPY in `output/` should still have the real edit in it per Step D — that
   file's copy is meant to be diffed and merged directly like the others; it's specifically the
   actual `docs/*.md` files this step is asking you not to touch, since those also cover
   architectures and history well outside this one task's scope.)
5. Stop. Do not start a different architecture or task.
