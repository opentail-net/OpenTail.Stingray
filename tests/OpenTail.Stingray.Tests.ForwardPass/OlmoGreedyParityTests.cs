using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// Greedy token parity against llama.cpp for OLMo v1 — the receipt that admits <c>olmo</c> to
/// <see cref="ModelCompatibility"/>'s architecture allowlist.
///
/// <para><b>The one genuinely new mechanism: LayerNorm with NEITHER a learned scale NOR a bias.</b>
/// Confirmed against <c>examples/llama.cpp/llama.cpp/src/models/olmo.cpp</c> before writing any
/// code: every <c>build_norm</c> call in the graph passes BOTH the weight and bias arguments as
/// <c>NULL</c> (<c>build_norm(inpL, NULL, NULL, LLM_NORM, il)</c>), and
/// <c>load_arch_tensors</c> never creates an <c>attn_norm</c>/<c>ffn_norm</c>/<c>output_norm</c>
/// tensor at all — confirmed independently via <c>list-tensors</c> on the real checkpoint, not
/// just read off the source. This is a THIRD norm shape this engine had no mechanism for: not
/// weighted LayerNorm-with-bias (GPT-NeoX/Falcon/GPT-2/StarCoder2's <c>UsesLayerNorm</c>), not
/// bias-less-but-still-weighted LayerNorm (Command-R/cohere2's <c>UsesLayerNorm</c> without
/// <c>HasNormBias</c>), and not RMSNorm at all — genuinely no learned parameters whatsoever.</para>
///
/// <para>The tricky part: a MISSING norm tensor already means something specific in this engine —
/// OLMo2's convention, where a null-DataPtr <c>_attnNorm[layer]</c>/<c>_ffnNorm[layer]</c> means
/// "skip normalizing here entirely, this sublayer reads the raw residual, it gets normed only on
/// its OUTPUT via a post-norm". OLMo v1's missing tensor means something different — "normalize
/// here as usual, just with no weight or bias to apply" — so tensor ABSENCE alone can't
/// disambiguate the two; needed a genuine <c>ModelHyperparams.UsesUnweightedNorm</c> arch-string
/// check (<c>arch == "olmo"</c>), not a generalized tensor-presence rule.</para>
///
/// <para>Added <see cref="SimdKernels.PureLayerNorm"/> (mean-subtract + variance-normalize, no
/// weight or bias parameter at all — a genuinely new kernel, structurally between the existing
/// <c>LayerNorm</c> [weighted, bias-optional] and <c>PureRmsNorm</c> [no mean-subtraction]) and
/// wired it into <c>RunTrunk</c>'s three norm points (pre-attention, pre-FFN, final) ahead of the
/// existing null-DataPtr-means-skip check, so <c>UsesUnweightedNorm</c> takes priority over that
/// sentinel rather than being confused with it. Also made <c>ForwardPass._outputNorm</c>'s
/// resolution conditional on tensor presence (previously an unconditional
/// <c>ResolveTensor("output_norm.weight")</c> that would have thrown for this checkpoint, since
/// OLMo v1 has no such tensor either). <c>PrefillCore</c>'s batched norm steps were NOT taught this
/// third mode — routed to the sequential <c>RunTrunk</c>/<c>Forward</c> path instead via a new
/// <c>unweightedNormUnsupported</c> flag in <c>PrefillDispatch</c>'s fallback gate, the same
/// established pattern OLMo2/cohere2/Gemma-4 already use for their own PrefillCore gaps.</para>
///
/// <para><b>Everything else was already generic or trivial.</b> Plain GQA-shaped attention
/// (actually plain MHA here — <c>head_count == head_count_kv == 16</c>), standard interleaved
/// (non-NEOX) RoPE, standard SiLU-gated FFN, no biases anywhere, tied embeddings (no separate
/// <c>output.weight</c> tensor — already-generic fallback to <c>token_embd.weight</c>). The
/// <c>tokenizer.ggml.pre = olmo</c> pre-type was already in this engine's GPT-2 pre-tokenizer
/// cascade group from earlier session work.</para>
///
/// <para><b>Checkpoint.</b> <c>allenai/OLMo-1B-hf</c> (16 layers on this specific checkpoint,
/// though llama.cpp's own <c>type</c> switch only names 22/32/80-layer sizes — cosmetic, doesn't
/// affect correctness), genuinely Apache-2.0 (AI2), via <c>nopperl/OLMo-1B-GGUF</c>, Q8_0 (1.25 GB,
/// deleted after this receipt — the checkpoint is gone, but the parity test stays).
/// <c>tokenizer.ggml.model = gpt2</c> (byte-BPE, real merges array).</para>
///
/// <para><b>Reference.</b> <c>tools/llama.cpp</c> build <c>b8585-cad2d3884</c>:
/// <code>
/// llama-tokenize -m &lt;model&gt; -p "The capital of France is" --ids --no-bos
///   -> [510, 5347, 273, 6181, 310]
/// llama-completion -m &lt;model&gt; -p "The capital of France is" -n 24 --temp 0 --top-k 1 --seed 0 \
///     -no-cnv --override-kv tokenizer.ggml.add_bos_token=bool:false
///   -> " the city of Paris. The capital of France is the city of Paris.\nThe capital of France is
///       the city of"
/// </code>
/// </para>
///
/// <para><b>Result: FULL 24-of-24-token exact match, no near-tie, no divergence anywhere</b> — on
/// the very first real attempt, including through the entirely new unweighted-norm code path this
/// receipt exists to validate.</para>
/// </summary>
public sealed class OlmoGreedyParityTests : HeavyTestBase
{
    private const string ModelFile = "olmo-1b-Q8_0.gguf";

    /// <summary>Prompt token ids from llama-tokenize; see the class remarks.</summary>
    private static readonly int[] s_promptTokens = [510, 5347, 273, 6181, 310];

    /// <summary>The full llama.cpp reference continuation (24 tokens); see the class remarks.</summary>
    private static readonly int[] s_referenceContinuationTokens =
        [253, 2846, 273, 7785, 15, 380, 5347, 273, 6181, 310, 253, 2846, 273, 7785, 15, 187, 510, 5347, 273, 6181, 310, 253, 2846, 273];

    [Fact]
    public void Olmo_GreedyContinuation_MatchesLlamaCpp()
    {
        var path = FindModel();
        Assert.SkipWhen(path is null, $"{ModelFile} is required for this parity receipt.");

        using var modelHandle = SharedModelCacheFixture.Instance.Acquire(path!);
        var model = modelHandle.Model;
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var tokenizer = GgufTokenizer.FromGgufModel(model);

        Assert.Equal("olmo", Convert.ToString(model.Metadata["general.architecture"]));
        Assert.Equal(s_promptTokens, tokenizer.Encode("The capital of France is"));

        // Guards the unweighted-norm detection: this receipt is worthless if the fixture silently
        // lost its "no attn_norm/ffn_norm/output_norm tensor at all" shape.
        Assert.True(hp.UsesUnweightedNorm, "olmo v1 has no attn_norm/ffn_norm/output_norm tensor at all");

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
    /// whole sequence in one pass must match stepping the same tokens through decode one at a
    /// time — specifically exercises that the unweighted-norm fallback (forced through the
    /// sequential path for BOTH prefill and decode via <c>unweightedNormUnsupported</c>) still
    /// agrees with itself across chunk boundaries.
    /// </summary>
    [Fact]
    public void Olmo_DecodeStepwise_AgreesWithSinglePassPrefill()
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
