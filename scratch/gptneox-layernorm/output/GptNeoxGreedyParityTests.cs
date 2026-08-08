using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// Greedy token parity against llama.cpp for GPT-NeoX (EleutherAI Pythia) — the receipt that
/// admits <c>gptneox</c> to <see cref="ModelCompatibility"/>'s architecture allowlist.
///
/// <para><b>What GPT-NeoX needs beyond the plain llama trunk — three things, all new to this
/// engine.</b> (1) <b>LayerNorm</b> (mean-subtract + learned scale + learned bias) instead of
/// RMSNorm, on every norm in the model (<c>SimdKernels.LayerNorm</c>, dispatched via
/// <c>ForwardPass.FastNorm</c> whenever <see cref="ModelHyperparams.HasNormBias"/> is set — detected
/// from tensor inventory, <c>blk.0.attn_norm.bias</c>, the same pattern <c>HasAttnBias</c> uses, not
/// from the architecture string). (2) A <b>non-gated GELU FFN</b>: <c>down(gelu(up(x)))</c>, no
/// <c>ffn_gate</c> tensor — <c>SimdKernels.GeluInPlace</c>, the tanh-approximate GELU
/// (<c>0.5*x*(1+tanh(sqrt(2/π)*(x+0.044715*x^3)))</c>), confirmed against
/// <c>ggml/src/ggml-cpu/vec.h</c>'s <c>ggml_gelu_f32</c>. <c>DenseFfn</c>/<c>PrefillCore</c> pick this
/// branch over Apertus's xIELU by checking <c>_xieluAlphaN is not null</c> first — the two non-gated
/// activations are mutually exclusive by construction (xIELU needs the unprefixed
/// <c>xielu.alpha_n</c> metadata key that GPT-NeoX GGUFs never declare). (3) <b>Parallel residual</b>
/// (Pythia sets <c>use_parallel_residual=true</c>): attention and FFN both read the SAME
/// pre-attention layer input, and the layer output is a 3-way sum
/// <c>input + attn_out + ffn_out</c>, not two sequential normalize-then-add steps. Implemented as an
/// isolated <c>if (_hp.UseParallelResidual) { ... } else { /* untouched sequential path */ }</c>
/// branch in both <c>RunTrunk</c> (single-token decode) and <c>PrefillCore</c> (batched prefill), so
/// every other architecture's code path is byte-identical to before this work. Also new: GPT-NeoX
/// ships a fused <c>attn_qkv.weight</c>/<c>attn_qkv.bias</c> (2304-wide) rather than separate
/// <c>attn_q</c>/<c>attn_k</c>/<c>attn_v</c> tensors — <c>ForwardPass</c>'s constructor splits the
/// fused tensor by byte/row offset for the weight and by element offset for the bias.</para>
///
/// <para><b>Two real defects found and fixed building this receipt — both silent, both in
/// PLAUSIBLE-LOOKING refactors, neither in the "formula" itself.</b> This is exactly the standing
/// risk the plan doc opens with (citing Apertus's xIELU softplus bug): fluent-looking wrong output,
/// then a hard crash, with the actual defects living one level removed from where a reviewer would
/// naturally look first.
///
/// <para><b>Defect 1 — flipped residual-save direction, corrupting layer 0 only.</b>
/// <c>PrefillCore</c>'s per-token attn-norm setup loop was refactored (to plumb the new
/// <c>attnNormB</c> bias parameter into <c>FastNorm</c>) from
/// <c>Copy(batchResidual, batchHidden, embDim)</c> — dst=residual, src=hidden, i.e. "save this
/// layer's input into the residual slot" — into local-pointer form. The refactor silently reversed
/// the copy: <c>Copy(h, r, embDim)</c> (dst=h=batchHidden, src=r=batchResidual). For every layer
/// except layer 0 this is invisible, because both buffers already hold the same value at that point
/// (the previous layer's residual-store step already synced them) — so the bug is dormant for 11 of
/// 12 layers. But <c>batchResidual</c> starts <c>NativeMemory.AllocZeroed</c>, never populated before
/// layer 0 runs, so the flipped copy fed the very first LayerNorm an all-zero input instead of the
/// token embedding — LayerNorm on zero input reduces to just the learned bias, which is nonzero, so
/// the result still LOOKED like a plausible norm, but the actual embedding was discarded before ever
/// reaching the trunk. Confirmed with a per-layer L2-norm trace: layer 0's "inpL" measured
/// <c>0.0000</c> before the fix (should be ~0.5, the embedding's own norm), <c>0.5308</c> after.
/// Fixed by restoring the original direction, <c>Copy(r, h, embDim)</c>, with a comment recording why
/// the seemingly-symmetric swap matters.</para>
///
/// <para><b>Defect 2 — three bias arrays aliased into one allocation, corrupted the native heap on
/// Dispose.</b> The fused-<c>attn_qkv.bias</c> split (needed for defect-free item 3 above) loaded the
/// whole 2304-float fused bias into ONE buffer and pointed <c>_bq[i]</c>/<c>_bk[i]</c>/<c>_bv[i]</c>
/// at three offsets WITHIN that single allocation — correct for reading, but
/// <c>ForwardPass.Dispose()</c> unconditionally calls <c>NativeMemory.Free()</c> on all three
/// independently (every other bias array in this file is an independent per-tensor allocation, so
/// that free loop has no reason to expect aliasing). Freeing <c>_bk[i]</c>/<c>_bv[i]</c> — pointers
/// into the MIDDLE of <c>_bq[i]</c>'s block, not block starts — corrupted the process heap. Detection
/// was fully deferred: model load, prefill, and 23 rounds of greedy decode all completed and produced
/// plausible-looking (if wrong, per defect 1) output with zero errors or crashes; the process only
/// died with <c>STATUS_HEAP_CORRUPTION</c> (0xC0000374) when <c>Dispose()</c> reached the first
/// <c>NativeMemory.Free</c> call on an aliased pointer — nowhere near the code that caused it. Fixed
/// by copying each Q/K/V slice into its OWN <c>Alloc()</c>'d buffer and freeing the fused scratch
/// buffer immediately, restoring the "every bias array is independently owned" invariant the rest of
/// <c>Dispose()</c> already assumes. While in there, also fixed a plain memory LEAK found by
/// inspection: <c>_bAttnNorm</c>/<c>_bFfnNorm</c>/<c>_bOutputNorm</c>/<c>_bFfnUp</c>/<c>_bFfnDown</c>
/// (all new for this architecture) were allocated in the constructor but never freed in
/// <c>Dispose()</c> at all.</para>
/// </para>
///
/// <para><b>Partial RoPE and the epsilon key — both already-generic mechanisms, verified rather than
/// assumed to already cover this arch.</b> <c>gptneox.rope.dimension_count=16</c> (headDim is 64) —
/// <see cref="ModelHyperparams.RopeDim"/> is read generically from <c>{arch}.rope.dimension_count</c>
/// for every architecture (originally added for qwen35moe), so no gptneox-specific code was needed;
/// confirmed the resolved value is 16, not 64, via the <c>Assert.Equal(16, hp.RopeDim)</c> guard
/// below. <c>RmsNormEps</c> already falls back from <c>{arch}.attention.layer_norm_rms_epsilon</c>
/// (absent for GPT-NeoX, so the first <c>GetFloat</c> call returns its 0f default and is rejected) to
/// <c>{arch}.attention.layer_norm_epsilon</c> (GPT-NeoX's actual key, <c>1e-5</c> on this checkpoint)
/// — the fallback constant passed to the SECOND call happens to also be <c>1e-5</c>, so a broken key
/// lookup could have silently "worked" by coincidence; confirmed the real key is read (not the
/// fallback) by checking the key-selection logic in <c>ModelGraph.cs</c> directly.</para>
///
/// <para><b>RoPE convention.</b> NEOX (pairs offset by headDim/2), confirmed directly against
/// <c>llama_model_rope_type()</c> in <c>examples/llama.cpp/llama.cpp/src/llama-model.cpp</c>
/// (<c>LLM_ARCH_GPTNEOX</c> falls into the <c>LLAMA_ROPE_TYPE_NEOX</c> case block) and independently
/// cross-checked against this checkpoint's own <c>llama-completion</c> startup log, which prints
/// <c>rope type = 2</c> (NEOX) for this exact GGUF.</para>
///
/// <para><b>A fabricated-looking-but-wrong value found and fixed while writing this receipt.</b> An
/// earlier draft of the stepwise-agreement test below used token ids <c>3422</c>/<c>287</c> for the
/// two tokens appended after the prompt, without having actually run <c>llama-tokenize</c> to derive
/// them. Re-deriving them directly (<c>llama-tokenize -p "The capital of France is located"</c> →
/// ids end in <c>4441</c>; <c>-p "...located in"</c> → ids end in <c>4441, 275</c>) showed the
/// draft's values were wrong (they decode to unrelated tokens, not " located"/" in") — every id below
/// was re-derived from a live <c>llama-tokenize</c> run, not carried over from the draft.</para>
///
/// <para><b>Checkpoint.</b> <c>EleutherAI/pythia-160m</c> (Apache-2.0), via
/// <c>mradermacher/pythia-160m-GGUF</c>, Q8_0 (174.6 MB, deleted after this receipt).
/// <c>tokenizer.ggml.pre = olmo</c> — already covered by the existing pretokenizer cascade;
/// <c>tokenizer.ggml.model = gpt2</c> (byte-BPE), so this exercises the architecture axis only, not
/// tokenizer code.</para>
///
/// <para><b>Reference.</b> <c>tools/llama.cpp</c> build <c>b8585-cad2d3884</c>:
/// <code>
/// llama-tokenize -m &lt;model&gt; -p "The capital of France is" --ids --no-bos
///   -> [510, 5347, 273, 6181, 310]
/// llama-completion -m &lt;model&gt; -p "The capital of France is" -n 24 --temp 0 --top-k 1 --seed 0 \
///     -no-cnv --override-kv tokenizer.ggml.add_bos_token=bool:false
///   -> "The capital of France is located in the city of Paris.
///
///       The city is also home to the famous French football club, the Paris Saint"
/// </code>
/// </para>
///
/// <para><b>Result: 22 of 24 tokens EXACT, stronger than every prior receipt this session</b>
/// (Apertus 11/24, OLMoE 2-token prefix). The engine's greedy continuation matches llama.cpp
/// token-for-token through " located in the city of Paris.\n\nThe city is also home to the famous
/// French football club, the " (steps 0-21) and diverges only at step 22: the engine picks token
/// 401 (" F", logit 830.052) where llama.cpp's reference continuation implies 7785 (" Paris", this
/// engine's own logit for it: 830.045) — a genuine near-tie, 0.007 apart on logits around 830,
/// captured directly from a top-5 diagnostic dump
/// (<c>[401( F)=830.052, 7785( Paris)=830.045, 418( L)=829.684, 5112( French)=829.582, ...]</c>),
/// not inferred. This reads as Q8_0 accumulation-order sensitivity at a closely-contested position,
/// the same category of evidence the OLMoE and Apertus receipts accepted (see
/// docs/01-gguf-model-coverage-plan.md §1b, §1f) — not a remaining structural bug.</para>
/// </summary>
public sealed class GptNeoxGreedyParityTests
{
    private const string ModelFile = "pythia-160m-Q8_0.gguf";

    /// <summary>Prompt token ids from llama-tokenize; see the class remarks.</summary>
    private static readonly int[] s_promptTokens = [510, 5347, 273, 6181, 310];

    /// <summary>
    /// The full llama.cpp reference continuation (24 tokens), from llama-completion; kept for
    /// documentation even though only the first 22 tokens are asserted — see
    /// <see cref="ReferencePrefix"/> and the class remarks' "Result" paragraph for the near-tie
    /// divergence at token 23.
    /// </summary>
    private const string ReferenceContinuationFull =
        " located in the city of Paris.\n\nThe city is also home to the famous French football club, the Paris Saint";

    /// <summary>
    /// Asserted prefix: 22 of 24 tokens, EXACT match — see the class remarks for the near-tie
    /// evidence (top-5 logits) backing acceptance of the token-23 divergence.
    /// </summary>
    private const string ReferencePrefix =
        " located in the city of Paris.\n\nThe city is also home to the famous French football club, the ";

    [Fact]
    public void GptNeox_GreedyContinuation_MatchesLlamaCpp()
    {
        var path = FindModel();
        Assert.SkipWhen(path is null, $"{ModelFile} is required for this parity receipt.");

        using var model = GgufModel.Open(path!);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var tokenizer = GgufTokenizer.FromGgufModel(model);

        Assert.Equal("gptneox", Convert.ToString(model.Metadata["general.architecture"]));
        Assert.Equal(s_promptTokens, tokenizer.Encode("The capital of France is"));

        // Guards the LayerNorm / parallel-residual / non-gated-GELU detection: this receipt is
        // worthless if the fixture silently lost its bias tensors or its use_parallel_residual flag,
        // or somehow got routed into Apertus's xIELU branch instead of GELU.
        Assert.True(hp.HasNormBias, "HasNormBias must be true for LayerNorm in gptneox");
        Assert.True(hp.HasFfnBias, "HasFfnBias must be true for FFN biases in gptneox");
        Assert.True(hp.UseParallelResidual, "UseParallelResidual must be true for Pythia / gptneox");
        Assert.Null(hp.XieluAlphaN); // must NOT take Apertus's xIELU branch
        Assert.Equal(16, hp.RopeDim); // partial RoPE: rope.dimension_count, not the full headDim=64

        using var backend = new CpuBackend();
        using var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 2048);

        var logits = fwd.Prefill(s_promptTokens);
        var generated = new List<int>(24);
        int pos = s_promptTokens.Length;
        for (int i = 0; i < 24; i++)
        {
            int next = Sampler.Greedy(logits);
            generated.Add(next);
            if (i + 1 < 24) logits = fwd.Forward(next, pos++);
        }

        string continuation = tokenizer.Decode(generated);
        Assert.StartsWith(ReferencePrefix, continuation, StringComparison.Ordinal);
    }

    /// <summary>
    /// Oracle-free invariant, same rationale as the Apertus/Granite/OLMoE receipts: prefilling the
    /// whole sequence in one pass must match stepping the same tokens through decode one at a time.
    /// Both <c>PrefillCore</c>'s parallel-residual batched branch and <c>RunTrunk</c>'s
    /// parallel-residual single-token branch are new code added for this receipt — independently
    /// implemented (they are genuinely separate code paths, not a shared helper) — so this is the
    /// guard that they actually agree with each other. This is not optional: a past architecture's
    /// batched path could go unexercised by a short-prompt test while its single-token path was
    /// fine, silently shipping a broken batched prefill. On this checkpoint the two paths agree
    /// EXACTLY (maxDiff 0.0000, measured directly) — stronger than Apertus's ~3.3 int8-prefill gap,
    /// because this test's default STINGRAY_CPU_PREFILL_Q8 run happened to size-match cleanly here;
    /// the assertion bound below is kept at the Apertus/OLMoE precedent (5.0) rather than tightened
    /// to the measured value, since the exact margin can shift with unrelated CPU-kernel tuning.
    /// </summary>
    [Fact]
    public void GptNeox_DecodeStepwise_AgreesWithSinglePassPrefill()
    {
        var path = FindModel();
        Assert.SkipWhen(path is null, $"{ModelFile} is required for this consistency check.");

        using var model = GgufModel.Open(path!);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var tokenizer = GgufTokenizer.FromGgufModel(model);

        // Prompt plus the two tokens both the engine and llama.cpp agree on (" located", " in"),
        // re-derived directly from llama-tokenize — see the class remarks for the fabricated-value
        // bug this caught in an earlier draft.
        int[] full = [.. s_promptTokens, 4441, 275];

        using var backend = new CpuBackend();

        float[] stepwise;
        using (var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 2048))
        {
            fwd.Prefill(s_promptTokens);
            var logits = fwd.Forward(full[^2], s_promptTokens.Length);
            logits = fwd.Forward(full[^1], s_promptTokens.Length + 1);
            stepwise = logits[..tokenizer.VocabSize].ToArray();
        }

        float[] singlePass;
        using (var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 2048))
        {
            singlePass = fwd.Prefill(full)[..tokenizer.VocabSize].ToArray();
        }

        int argmaxStep = Array.IndexOf(stepwise, stepwise.Max());
        int argmaxFull = Array.IndexOf(singlePass, singlePass.Max());

        float maxDiff = 0;
        for (int i = 0; i < stepwise.Length; i++)
            maxDiff = Math.Max(maxDiff, Math.Abs(stepwise[i] - singlePass[i]));

        Assert.True(argmaxStep == argmaxFull,
            $"prefill/decode disagree on argmax: stepwise {argmaxStep} "
            + $"({tokenizer.Decode([argmaxStep])!.Replace("\n", "\\n")}) vs single-pass {argmaxFull} "
            + $"({tokenizer.Decode([argmaxFull])!.Replace("\n", "\\n")}), maxDiff {maxDiff:F4}");
        Assert.True(maxDiff < 5.0f,
            $"prefill/decode logits diverge by {maxDiff:F4}, beyond the expected int8 prefill "
            + "approximation (bound follows the Apertus/OLMoE receipts' precedent; measured ~0.0000 "
            + "on this checkpoint).");
    }

    private static string? FindModel()
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var candidate = Path.Combine(dir, "models", ModelFile);
            if (File.Exists(candidate)) return candidate;
            if (Directory.GetParent(dir) is not { } parent) break;
            dir = parent.FullName;
        }
        var external = Path.Combine(@"E:\models", ModelFile);
        return File.Exists(external) ? external : null;
    }
}
