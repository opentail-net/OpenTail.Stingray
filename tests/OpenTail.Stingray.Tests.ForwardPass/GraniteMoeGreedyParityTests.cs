
namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// Greedy token parity against llama.cpp for Granite MoE — the receipt that admits
/// <c>granitemoe</c> to <see cref="ModelCompatibility"/>'s architecture allowlist.
///
/// <para><b>Zero new code — the closest thing to a free admission this session.</b> Confirmed
/// against <c>examples/llama.cpp/llama.cpp/src/models/granite-moe.cpp</c> and <c>models.h</c>
/// before writing any code: <c>llama_model_granite_moe::graph</c> is a type alias for
/// <c>llama_model_granite::graph</c> (<c>using graph = llama_model_granite::graph;</c>) — the
/// SAME graph builder as dense Granite (already admitted, see the receipt in
/// <c>GraniteGreedyParityTests</c>), which already branches on <c>n_expert == 0</c> internally to
/// pick the dense-FFN or <c>build_moe_ffn</c> path. This engine's own MoE dispatch
/// (<c>ModelHyperparams.IsMoE</c>/<c>NumExperts</c>/<c>NumActiveExperts</c>) and the Granite-family
/// scale block (<c>ResidualScale</c>/<c>EmbeddingScale</c>/<c>AttentionScaleOverride</c>/
/// <c>LogitScale</c>, gated by <c>isGraniteFamily</c> in <c>ModelGraph.cs</c>) already checks
/// <c>arch.Equals("granitemoe", ...)</c> explicitly — added when the dense Granite receipt was
/// built, evidently in anticipation of this exact admission. This checkpoint uses plain softmax
/// gating with top-k renormalization and no shared expert (no <c>ffn_gate_shexp</c> tensor,
/// <c>granitemoe.expert_feed_forward_length</c> absent so <c>ExpertIntermediateDim</c> falls back
/// to the plain <c>feed_forward_length</c> key, which already matches the expert tensors' actual
/// width) — both already the default path this engine's generic MoE FFN takes for every other
/// admitted MoE architecture. The only genuine work this receipt required was downloading a
/// checkpoint, adding the string to the allowlist, and writing this test.</para>
///
/// <para><b>Checkpoint.</b> <c>ibm-granite/granite-3.0-1b-a400m-instruct</c> (1B total / 400M
/// active MoE, 32 experts / 8 active per token) — genuinely Apache-2.0 (confirmed via the GGUF's
/// own <c>general.license</c> key AND the HF API's <c>cardData.license</c>, not just one source),
/// via <c>bartowski/granite-3.0-1b-a400m-instruct-GGUF</c>, Q8_0 (1.42 GB, deleted after this
/// receipt — the checkpoint is gone, but the parity test stays, since the license is genuinely
/// permissive). <c>tokenizer.ggml.model = gpt2</c> (byte-BPE, real merges array).</para>
///
/// <para><b>Reference.</b> <c>tools/llama.cpp</c> build <c>b8585-cad2d3884</c>:
/// <code>
/// llama-tokenize -m &lt;model&gt; -p "The capital of France is" --ids --no-bos
///   -> [1318, 18926, 432, 45600, 438]
/// llama-completion -m &lt;model&gt; -p "The capital of France is" -n 24 --temp 0 --top-k 1 --seed 0 \
///     -no-cnv --override-kv tokenizer.ggml.add_bos_token=bool:false
///   -> " Paris. Paris is a beautiful city with a rich history and culture. It is home to many
///       famous land"
/// </code>
/// </para>
///
/// <para><b>Result: FULL 24-of-24-token exact match, no near-tie, no divergence.</b> Confirms the
/// zero-new-code prediction: every generic mechanism this checkpoint exercises (Granite-family
/// scale block, standard softmax-gated MoE, standard GQA) was already correct.</para>
/// </summary>
public sealed class GraniteMoeGreedyParityTests : HeavyTestBase
{
    private const string ModelFile = "granite-3.0-1b-a400m-instruct-Q8_0.gguf";

    /// <summary>Prompt token ids from llama-tokenize; see the class remarks.</summary>
    private static readonly int[] s_promptTokens = [1318, 18926, 432, 45600, 438];

    /// <summary>The full llama.cpp reference continuation (24 tokens); see the class remarks.</summary>
    private static readonly int[] s_referenceContinuationTokens =
        [2716, 297, 32, 2716, 297, 438, 312, 36493, 11297, 623, 312, 20815, 8142, 461, 27668, 32, 2030, 438, 6765, 372, 5075, 15863, 3291, 13260];

    [Fact]
    public void GraniteMoe_GreedyContinuation_MatchesLlamaCpp()
    {
        var path = FindModel();
        Assert.SkipWhen(path is null, $"{ModelFile} is required for this parity receipt.");

        using var modelHandle = SharedModelCacheFixture.Instance.Acquire(path!);
        var model = modelHandle.Model;
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var tokenizer = GgufTokenizer.FromGgufModel(model);

        Assert.Equal("granitemoe", Convert.ToString(model.Metadata["general.architecture"]));
        Assert.Equal(s_promptTokens, tokenizer.Encode("The capital of France is"));

        // Guards the Granite-family scale block and MoE detection: this receipt is worthless if
        // the fixture silently lost its expert routing or scale metadata.
        Assert.True(hp.IsMoE);
        Assert.Equal(32, hp.NumExperts);
        Assert.Equal(8, hp.NumActiveExperts);
        Assert.False(hp.HasSharedExpert, "this checkpoint has no ffn_gate_shexp tensor");
        Assert.Equal(0.22f, hp.ResidualScale);
        Assert.Equal(12f, hp.EmbeddingScale);
        Assert.Equal(1f / 6f, hp.LogitScale, precision: 5); // LogitScale already carries the reciprocal of the raw metadata value (6)

        using var backend = new CpuBackend();
        using var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 2048);

        int n = s_referenceContinuationTokens.Length;
        var logits = fwd.Prefill(s_promptTokens);
        var generated = new List<int>(n);
        int pos = s_promptTokens.Length;
        for (int i = 0; i < n; i++)
        {
            int next = Sampler.Greedy(logits);
            generated.Add(next);
            if (i + 1 < n) logits = fwd.Forward(next, pos++);
        }

        for (int i = 0; i < n; i++)
            Assert.Equal(s_referenceContinuationTokens[i], generated[i]);
    }

    /// <summary>
    /// Oracle-free invariant, same rationale as every other receipt this session: prefilling the
    /// whole sequence in one pass must match stepping the same tokens through decode one at a time.
    /// </summary>
    [Fact]
    public void GraniteMoe_DecodeStepwise_AgreesWithSinglePassPrefill()
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

        Assert.True(argmaxStep == argmaxFull,
            $"prefill/decode disagree on argmax: stepwise {argmaxStep} vs single-pass {argmaxFull}");
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
