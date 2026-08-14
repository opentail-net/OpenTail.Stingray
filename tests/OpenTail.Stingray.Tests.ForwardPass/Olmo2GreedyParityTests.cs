using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// Greedy token parity against llama.cpp for OLMo2 — the receipt that admits <c>olmo2</c> to
/// <see cref="ModelCompatibility"/>'s architecture allowlist.
///
/// <para><b>The plan doc's original premise for this architecture was wrong, and this receipt is
/// also the correction.</b> <c>docs/01-gguf-model-coverage-plan.md</c> §1c originally speculated
/// "olmoe, olmo2 — gate-only, code exists" — explicitly caveated there as "a hypothesis, not a
/// finding." Checked directly against
/// <c>examples/llama.cpp/llama.cpp/src/models/olmo2.cpp</c> before writing any code: OLMo2 is a
/// THIRD residual pattern, distinct from both the ordinary pre-norm trunk and
/// <c>gptneox</c>/<c>falcon</c>'s parallel residual (see their own receipts) — <b>post-norm
/// sandwiching</b>. There is no <c>attn_norm</c>/<c>ffn_norm</c> tensor in the GGUF at all;
/// attention and FFN both read the RAW residual directly, and the norm (plain RMSNorm, no bias)
/// is applied to each sublayer's OUTPUT, immediately before the residual add:
/// <c>x1 = x + PostNorm(Attn(x)); x2 = x1 + PostNorm(FFN(x1))</c>.</para>
///
/// <para><b>What actually needed building — small, because it reuses three already-existing
/// mechanisms rather than adding a new one.</b>
/// (1) <c>ForwardPass</c>'s constructor now leaves <c>_attnNorm[i]</c>/<c>_ffnNorm[i]</c> at their
/// default (<c>DataPtr</c> null) when the tensor is absent — the exact sentinel pattern
/// Apertus/GPT-NeoX already use for "no <c>ffn_gate</c> tensor" — and <c>RunTrunk</c>/
/// <c>PrefillCore</c>'s pre-norm steps now copy the raw residual straight through when that
/// sentinel is set, instead of normalizing (previously they unconditionally called
/// <c>GetNormWeight</c> on the tensor, which would have thrown on a real OLMo2 GGUF).
/// (2) The POST-norm application itself is not new at all: <c>_postAttnNorm</c>/
/// <c>_postFfwNorm</c> and the "apply, then residual-add" call sites in <c>RunTrunk</c> already
/// existed for Gemma 4, and — confirmed directly against llama.cpp's tensor-name table
/// (<c>LLM_TENSOR_ATTN_POST_NORM</c>/<c>FFN_POST_NORM</c> both map to
/// <c>blk.%d.post_attention_norm</c>/<c>blk.%d.post_ffw_norm</c>) — OLMo2 uses the exact same
/// tensor names and roles. The only change was generalizing <c>HasPostAttnNorm</c>/
/// <c>HasPostFfwNorm</c> detection in <c>ModelGraph.cs</c> from "gated inside the Gemma-4-only
/// block" to plain tensor-presence, so any architecture can activate it.
/// (3) QK-norm reuses the OLMoE fix unchanged — whole-vector RMS (2048 elements), not per-head —
/// the same convention, confirmed by this checkpoint's <c>attn_q_norm</c>/<c>attn_k_norm</c>
/// tensors both being <c>[2048]</c>, not <c>[headDim]</c>.</para>
///
/// <para><b>One real gap found by reasoning forward from the architecture, before writing the
/// test — not by a failing assertion.</b> The post-attention/post-FFW norm application was
/// documented (in <c>MoeBatchedPrefillSupported</c>'s own doc comment, for the MoE case) as
/// applying "only on <c>RunTrunk</c>" — <c>PrefillCore</c>'s batched loop has no equivalent step
/// at all. Gemma 4 never surfaces this gap because its own per-layer-head-dim check already routes
/// it away from <c>PrefillCore</c> entirely. OLMo2 has no per-layer head dims, so without a fix it
/// would have silently reached <c>PrefillCore</c> and produced wrong output — missing both
/// post-norms — for every prefill. Fixed by widening <c>PrefillDispatch</c>'s existing
/// per-layer-head-dim fallback (sequential per-token <c>Forward()</c> instead of the batched core)
/// to also cover any model with <c>_postAttnNorm</c>/<c>_postFfwNorm</c> set — the same pattern
/// Gemma 4 already uses, just no longer gated to that one architecture. Out of scope, same as
/// every other new-kernel receipt this session: <c>PrefillCoreTq</c>, <c>PrefillWithCache</c>
/// (continuous-batching admission), <c>BatchForwardMulti</c>/<c>PrefillPackedMulti</c>, and
/// CUDA/Vulkan do not know about this fallback and would still silently misbehave.</para>
///
/// <para><b>Checkpoint.</b> <c>allenai/OLMo-2-0425-1B</c> (Apache-2.0, AI2), official first-party
/// GGUF (<c>allenai/OLMo-2-0425-1B-GGUF</c>), Q8_0 (1.58 GB, deleted after this receipt).
/// <c>tokenizer.ggml.model = gpt2</c> (byte-BPE), <c>tokenizer.ggml.pre = dbrx</c>.</para>
///
/// <para><b>Reference.</b> <c>tools/llama.cpp</c> build <c>b8585-cad2d3884</c>:
/// <code>
/// llama-tokenize -m &lt;model&gt; -p "The capital of France is" --ids --no-bos
///   -> [791, 6864, 315, 9822, 374]
/// llama-completion -m &lt;model&gt; -p "The capital of France is" -n 24 --temp 0 --top-k 1 --seed 0 \
///     -no-cnv --override-kv tokenizer.ggml.add_bos_token=bool:false
///   -> " Paris. The French language is spoken in France. The French people are known as the
///       French. The French flag is red"
/// </code>
/// </para>
///
/// <para><b>Result: full 24-of-24-token EXACT match, byte for byte</b> — the same strength as the
/// Granite, SmolLM3, and Falcon receipts, and stronger than OLMoE's own (2-token prefix,
/// perplexity-only — see the <c>olmoe</c> writeup in the plan doc §1b, whose QK-norm defect fix
/// this checkpoint's Q/K-norm reuses unchanged). See <c>Olmo2GreedyParityTests.cs</c> — this file.</para>
/// </summary>
public sealed class Olmo2GreedyParityTests : HeavyTestBase
{
    private const string ModelFile = "OLMo-2-0425-1B-Q8_0.gguf";

    /// <summary>Prompt token ids from llama-tokenize; see the class remarks.</summary>
    private static readonly int[] s_promptTokens = [791, 6864, 315, 9822, 374];

    /// <summary>
    /// The full 24-token llama.cpp reference continuation, from llama-completion. See the class
    /// remarks' "Result" paragraph.
    /// </summary>
    private const string ReferenceContinuation =
        " Paris. The French language is spoken in France. The French people are known as the French. The French flag is red";

    private static readonly int[] s_referenceContinuationTokens =
        [12366, 13, 578, 8753, 4221, 374, 22066, 304, 9822, 13, 578, 8753, 1274, 527, 3967, 439, 279, 8753, 13, 578, 8753, 5292, 374, 2579];

    [Fact]
    public void Olmo2_GreedyContinuation_MatchesLlamaCpp()
    {
        var path = FindModel();
        Assert.SkipWhen(path is null, $"{ModelFile} is required for this parity receipt.");

        using var modelHandle = SharedModelCacheFixture.Instance.Acquire(path!);
        var model = modelHandle.Model;
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var tokenizer = GgufTokenizer.FromGgufModel(model);

        Assert.Equal("olmo2", Convert.ToString(model.Metadata["general.architecture"]));
        Assert.Equal(s_promptTokens, tokenizer.Encode("The capital of France is"));

        // Guards the post-norm-sandwich / no-pre-norm detection: this receipt is worthless if the
        // fixture silently lost its post_attention_norm/post_ffw_norm tensors, or somehow picked
        // up a pre-norm attn_norm/ffn_norm tensor this architecture should never have.
        Assert.True(hp.HasPostAttnNorm, "HasPostAttnNorm must be true for olmo2's post-norm sandwich");
        Assert.True(hp.HasPostFfwNorm, "HasPostFfwNorm must be true for olmo2's post-norm sandwich");
        Assert.False(hp.HasNormBias, "olmo2 uses plain RMSNorm, not GPT-NeoX-style LayerNorm-with-bias");
        Assert.False(hp.UseParallelResidual, "olmo2 is post-norm sandwich, not gptneox/falcon's parallel residual");
        Assert.True(hp.HasQkNorm, "olmo2 has per-projection Q/K RMSNorm, same convention as olmoe");

        using var backend = new CpuBackend();
        using var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 2048);

        var logits = fwd.Prefill(s_promptTokens);
        var generated = new List<int>(s_referenceContinuationTokens.Length);
        int pos = s_promptTokens.Length;
        for (int i = 0; i < s_referenceContinuationTokens.Length; i++)
        {
            int next = Sampler.Greedy(logits);
            generated.Add(next);
            if (i + 1 < s_referenceContinuationTokens.Length) logits = fwd.Forward(next, pos++);
        }

        string continuation = tokenizer.Decode(generated);
        Assert.Equal(ReferenceContinuation, continuation);
        Assert.Equal(s_referenceContinuationTokens, generated);
    }

    /// <summary>
    /// Oracle-free invariant, same rationale as every other receipt this session: prefilling the
    /// whole sequence in one pass (<c>PrefillCore</c>, now routed through the post-norm-aware
    /// sequential fallback in <c>PrefillDispatch</c> — see the class remarks) must match stepping
    /// the same tokens through single-token decode (<c>RunTrunk</c>). Both ultimately call the
    /// same per-token <c>Forward</c> path for this architecture (PrefillDispatch's fallback), so
    /// this is less a test of two INDEPENDENT implementations agreeing (as it is for gptneox/
    /// falcon's genuinely separate batched code) and more a guard against the fallback routing
    /// itself regressing — still worth pinning explicitly.
    /// </summary>
    [Fact]
    public void Olmo2_DecodeStepwise_AgreesWithSinglePassPrefill()
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
            + "approximation (bound follows the prior receipts' precedent).");
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
