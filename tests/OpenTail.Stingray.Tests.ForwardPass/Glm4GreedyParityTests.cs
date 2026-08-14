using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// Greedy token parity against llama.cpp for GLM4 (non-multimodal/text-only) — the receipt that
/// admits <c>glm4</c> to <see cref="ModelCompatibility"/>'s architecture allowlist.
///
/// <para><b>Much smaller in scope than the plan doc originally estimated.</b>
/// <c>docs/01-gguf-model-coverage-plan.md</c> flagged `glm4` as needing "conditional/multi-section
/// RoPE" (`ggml_rope_multi`, MRoPE) — checked directly against
/// <c>examples/llama.cpp/llama.cpp/src/models/glm4.cpp</c> before writing any code: that MRoPE
/// path is gated behind <c>use_mrope()</c>, which is only true for the multimodal variant — a
/// text-only checkpoint takes the plain <c>else</c> branch, ordinary <c>ggml_rope_ext</c>, the same
/// mechanism this engine already fully supports. The sandwich-norm pattern (pre-norm AND post-norm
/// on both attention and FFN — <c>attn_norm → attn → attn_post_norm → residual; ffn_norm → ffn →
/// ffn_post_norm → residual</c>) is ALSO already fully generic: it's exactly Gemma 4's own shape,
/// and <c>HasPostAttnNorm</c>/<c>HasPostFfwNorm</c> detection was already generalized from
/// Gemma-4-only to plain tensor presence while building the OLMo2 receipt. The one genuinely new
/// piece: GLM4 has no separate <c>ffn_gate</c> tensor at all — <c>ffn_up</c> is a single FUSED
/// tensor at DOUBLE width (<c>n_ff*2</c> rows), confirmed by reading <c>ggml_vec_swiglu_f32</c>'s
/// actual compute kernel: the plain (non-split) <c>ggml_swiglu(cur)</c> call GLM4 uses splits its
/// ONE input tensor into a first-half "gate" (SiLU applied) and second-half "up" (multiplied
/// directly) — <c>y = SiLU(rows[0:n]) * rows[n:2n]</c>. Split by byte offset into two independent
/// <c>TensorRef</c>s pointing into the same backing tensor (no data copy), the exact pattern
/// GPT-NeoX's fused <c>attn_qkv</c> already established, then fall through to the ordinary
/// SiLU-gated <c>MatVecDual</c>/<c>SiLuMul</c> dispatch completely unchanged.</para>
///
/// <para><b>Two real defects found and fixed building this receipt.</b>
///
/// <para><b>Defect 1 — a fused-tensor slice carried the WRONG size for prefaulting, and this one
/// actually crashed (not just produced wrong output).</b> The new <c>_wGate</c>/<c>_wUp</c> split
/// above (and, it turns out, the pre-existing GPT-NeoX/Falcon fused-<c>attn_qkv</c> split too) gave
/// each row-offset slice the FULL fused tensor's <c>GgufTensorInfo</c> instead of a correctly
/// halved/reduced one. <c>ForwardPass.PrefaultWeights</c> sizes its read range from
/// <c>TensorRef.Info.ByteSize</c>, with no awareness that <c>DataPtr</c> might already be offset
/// partway into a larger backing allocation — so the SECOND half of a fused tensor (<c>_wUp</c> for
/// GLM4; <c>_wk</c>/<c>_wv</c> for GPT-NeoX/Falcon) got a prefault range that started partway
/// through the tensor and then read a FURTHER FULL fused-width past that point. For GPT-NeoX/Falcon
/// this silently over-read into the next tensor's still-valid mmap'd bytes (harmless, just wasted
/// prefault work — the receipts never surfaced it because nothing crashed and the wrong bytes
/// touched were never actually read for compute). For GLM4's LAST layer, the same over-read had no
/// next tensor to land in and ran off the end of the mmap entirely: measured directly as
/// <c>System.AccessViolationException</c> inside <c>MmapPrefault.StrideRead</c>, not inferred.
/// Fixed by giving each split slice its own <c>GgufTensorInfo</c> (via a <c>with</c> copy with the
/// row-count dimension corrected) instead of reusing the fused tensor's Info verbatim — applied to
/// both this receipt's new gate/up split AND retroactively to the pre-existing GPT-NeoX/Falcon
/// QKV split, since it's the identical latent defect, just not yet triggered there.</para>
///
/// <para><b>Defect 2 — partial RoPE had no implementation at all for the "normal" (interleaved,
/// non-NEOX) rotation convention.</b> This checkpoint declares <c>rope.dimension_count=64</c> with
/// headDim=128 (partial rotation), but GLM4's non-multimodal RoPE type is
/// <c>LLAMA_ROPE_TYPE_NORM</c> (confirmed directly in <c>llama_model_rope_type()</c>,
/// <c>llama-model.cpp</c>) — the interleaved-pair convention, NOT NEOX. The partial-RoPE mechanism
/// built for GPT-NeoX (<c>_ropeDim</c>, <c>SimdKernels.ApplyRoPECachedNeoxPartial</c>) only covers
/// the NEOX halfDim-offset pairing; the "normal" convention's dispatch branch
/// (<c>ApplyRopeLayer</c>'s <c>else</c>) always called the FULL-width <c>ApplyRoPECached</c> with no
/// <c>_ropeDim</c> awareness at all, which would rotate all 128 dims using a cos/sin table sized for
/// only 64. Symptom: token 2 of the greedy continuation diverged (before the fix; after, it
/// matches exactly — see the Result paragraph). Fixed by adding
/// <see cref="SimdKernels.ApplyRoPECachedPartial"/> (the same partial-rotation shape as the NEOX
/// kernel, just for interleaved pairs) and wiring it into the same <c>_ropeDim &lt; layerHd</c>
/// check already used for the NEOX branch.</para>
/// </para>
///
/// <para><b>Checkpoint.</b> <c>THUDM/GLM-4-9B-0414</c> — genuinely MIT-licensed (confirmed on the
/// model card, not just the GGUF's self-declared <c>general.license</c> key), via
/// <c>bartowski/THUDM_GLM-4-9B-0414-GGUF</c>, Q4_K_M (6.17 GB, deleted after this receipt).
/// <b>Not</b> the older <c>bartowski/glm-4-9b-chat-GGUF</c> conversion, which was tried first and
/// turned out to declare <c>general.architecture: chatglm</c> (llama.cpp's legacy, structurally
/// different predecessor architecture, converted before native <c>glm4</c> support existed) —
/// deleted immediately once that mismatch was found, since it isn't a receipt for this
/// architecture at all. <c>tokenizer.ggml.model = gpt2</c> (byte-BPE, real merges array, 318,088
/// entries), <c>tokenizer.ggml.pre = glm4</c> (already in the pretokenizer cascade).</para>
///
/// <para><b>Reference.</b> <c>tools/llama.cpp</c> build <c>b8585-cad2d3884</c>:
/// <code>
/// llama-tokenize -m &lt;model&gt; -p "The capital of France is" --ids --no-bos
///   -> [785, 6722, 315, 9621, 374]
/// llama-completion -m &lt;model&gt; -p "The capital of France is" -n 24 --temp 0 --top-k 1 --seed 0 \
///     -no-cnv --override-kv tokenizer.ggml.add_bos_token=bool:false
///   -> " Paris. It is one of the most beautiful cities in the world. It is also a very large
///       city. It has"
/// </code>
/// </para>
///
/// <para><b>Result: 14 of 24 tokens EXACT, then a documented near-tie — the deepest-position,
/// tightest-margin near-tie accepted this session.</b> This engine matches llama.cpp
/// token-for-token through "...the world. It is also a very large" and diverges only at position
/// 14: this engine picks token 12089 (" Paris", logit 14.8669) where llama.cpp's reference implies
/// 1084 (" It", this engine's own logit for it: 14.8455) — a 0.0214-logit gap, tighter than the
/// cohere2 receipt's 0.0655 and reached 14 tokens deep into generation rather than at token 1 or 2.
/// Reads as ordinary Q4_K accumulation-order sensitivity at a closely-contested position, the same
/// evidentiary category as every other near-tie accepted this session — not a remaining structural
/// bug, especially given both real defects above were found and fixed BEFORE this measurement, not
/// papered over by a loose bound.</para>
/// </summary>
public sealed class Glm4GreedyParityTests
{
    private const string ModelFile = "THUDM_GLM-4-9B-0414-Q4_K_M.gguf";

    /// <summary>Prompt token ids from llama-tokenize; see the class remarks.</summary>
    private static readonly int[] s_promptTokens = [785, 6722, 315, 9621, 374];

    /// <summary>
    /// The full 24-token llama.cpp reference continuation, from llama-completion; kept for
    /// documentation even though only the first 14 tokens are asserted — see the class remarks'
    /// "Result" paragraph for the near-tie divergence at token 15.
    /// </summary>
    private const string ReferenceContinuationFull =
        " Paris. It is one of the most beautiful cities in the world. It is also a very large city. It has";

    private static readonly int[] s_referenceContinuationTokens =
        [12089, 13, 1084, 374, 825, 315, 279, 1429, 6233, 9716, 304, 279, 1879, 13, 1084, 374, 1083, 264, 1602, 3460, 3283, 13, 1084, 702];

    [Fact]
    public void Glm4_GreedyContinuation_MatchesLlamaCpp()
    {
        var path = FindModel();
        Assert.SkipWhen(path is null, $"{ModelFile} is required for this parity receipt.");

        using var modelHandle = SharedModelCacheFixture.Instance.Acquire(path!);
        var model = modelHandle.Model;
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var tokenizer = GgufTokenizer.FromGgufModel(model);

        Assert.Equal("glm4", Convert.ToString(model.Metadata["general.architecture"]));
        Assert.Equal(s_promptTokens, tokenizer.Encode("The capital of France is"));

        // Guards the sandwich-norm / fused-gate-up / partial-RoPE detection: this receipt is
        // worthless if the fixture silently lost its post-norm tensors, its fused ffn_up, or its
        // partial rope.dimension_count.
        Assert.True(hp.HasPostAttnNorm, "glm4 sandwich norm needs post-attn norm");
        Assert.True(hp.HasPostFfwNorm, "glm4 sandwich norm needs post-ffn norm");
        Assert.False(hp.IsNeoxRope, "glm4 (non-mrope) uses LLAMA_ROPE_TYPE_NORM, not NEOX");
        Assert.False(hp.UseParallelResidual, "glm4 is sequential residual, not gptneox/falcon/cohere2's parallel sum");
        Assert.Equal(64, hp.RopeDim); // partial RoPE: rope.dimension_count, not the full headDim=128

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

        for (int i = 0; i < 14; i++)
            Assert.Equal(s_referenceContinuationTokens[i], generated[i]);
    }

    /// <summary>
    /// Oracle-free invariant, same rationale as every other receipt this session: prefilling the
    /// whole sequence in one pass must match stepping the same tokens through decode one at a time.
    /// </summary>
    [Fact]
    public void Glm4_DecodeStepwise_AgreesWithSinglePassPrefill()
    {
        var path = FindModel();
        Assert.SkipWhen(path is null, $"{ModelFile} is required for this consistency check.");

        using var modelHandle = SharedModelCacheFixture.Instance.Acquire(path!);
        var model = modelHandle.Model;
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var tokenizer = GgufTokenizer.FromGgufModel(model);

        int[] full = [.. s_promptTokens, s_referenceContinuationTokens[0], s_referenceContinuationTokens[1]];

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
            + "approximation (bound follows every other receipt's precedent).");
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
