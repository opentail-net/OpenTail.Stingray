
namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// Greedy token parity against llama.cpp for MainCoder — the receipt that admits
/// <c>maincoder</c> to <see cref="ModelCompatibility"/>'s architecture allowlist.
///
/// <para><b>Zero new code — a literal Qwen3-shaped architecture.</b> Confirmed against
/// <c>examples/llama.cpp/llama.cpp/src/models/maincoder.cpp</c> before writing any code: plain
/// RMSNorm pre-norm trunk, biasless GQA attention with weighted per-head QK-norm applied BEFORE
/// RoPE (shape <c>[headDim]</c>, shared across heads — confirmed via <c>list-tensors</c>, the
/// exact Qwen3 convention this engine already defaults to), standard SiLU-gated FFN, and standard
/// interleaved (non-NEOX) RoPE — confirmed via <c>llama_model_rope_type()</c> returning
/// <c>LLAMA_ROPE_TYPE_NORM</c> for <c>LLM_ARCH_MAINCODER</c>, matching this engine's default (no
/// arch string needed in the <c>isNeoxRope</c> list). <c>tokenizer.ggml.pre = qwen2</c> with a
/// real 151,387-entry merges array — already in this engine's pre-tokenizer cascade table. Every
/// mechanism this checkpoint exercises predates this session entirely.</para>
///
/// <para><b>Checkpoint.</b> <c>Maincode/Maincoder-1B</c> (1B, code-generation focused, trained
/// with RL per its own tags), genuinely Apache-2.0 (confirmed via the HF API's
/// <c>cardData.license</c>, not gated), via <c>Maincode/Maincoder-1B-GGUF</c>, Q8_0 (1.1 GB,
/// deleted after this receipt — the checkpoint is gone, but the parity test stays).</para>
///
/// <para><b>Reference.</b> <c>tools/llama.cpp</c> build <c>b8585-cad2d3884</c>:
/// <code>
/// llama-tokenize -m &lt;model&gt; -p "The capital of France is" --ids --no-bos
///   -> [785, 6722, 315, 9625, 374]
/// llama-completion -m &lt;model&gt; -p "The capital of France is" -n 24 --temp 0 --top-k 1 --seed 0 \
///     -no-cnv --override-kv tokenizer.ggml.add_bos_token=bool:false
///   -> " Paris. The population of Paris is 2,161,556. The population of Paris is "
/// </code>
/// </para>
///
/// <para><b>Result: FULL 24-of-24-token exact match, no near-tie, no divergence anywhere</b> — on
/// the very first real attempt, with zero engine code changes.</para>
/// </summary>
public sealed class MaincoderGreedyParityTests : HeavyTestBase
{
    private const string ModelFile = "maincoder-1b-Q8_0.gguf";

    /// <summary>Prompt token ids from llama-tokenize; see the class remarks.</summary>
    private static readonly int[] s_promptTokens = [785, 6722, 315, 9625, 374];

    /// <summary>The full llama.cpp reference continuation (24 tokens); see the class remarks.</summary>
    private static readonly int[] s_referenceContinuationTokens =
        [12095, 13, 576, 7042, 315, 12095, 374, 220, 17, 11, 16, 21, 16, 11, 20, 20, 21, 13, 576, 7042, 315, 12095, 374, 220];

    [Fact]
    public void Maincoder_GreedyContinuation_MatchesLlamaCpp()
    {
        var path = FindModel();
        Assert.SkipWhen(path is null, $"{ModelFile} is required for this parity receipt.");

        using var modelHandle = SharedModelCacheFixture.Instance.Acquire(path!);
        var model = modelHandle.Model;
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var tokenizer = GgufTokenizer.FromGgufModel(model);

        Assert.Equal("maincoder", Convert.ToString(model.Metadata["general.architecture"]));
        Assert.Equal(s_promptTokens, tokenizer.Encode("The capital of France is"));

        // Guards the QK-norm/RoPE detection: this receipt is worthless if the fixture silently
        // lost its weighted-before-RoPE QK-norm or fell into the NEOX rope convention.
        Assert.True(hp.HasQkNorm, "maincoder has learned attn_q_norm/attn_k_norm");
        Assert.False(hp.UseL2QkNorm, "maincoder's QK-norm is weighted, applied before RoPE (Qwen3 convention)");
        Assert.False(hp.IsNeoxRope, "maincoder uses standard/NORM RoPE (llama_model_rope_type returns NORM)");

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
    public void Maincoder_DecodeStepwise_AgreesWithSinglePassPrefill()
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
