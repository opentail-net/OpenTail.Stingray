using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// Greedy token parity against llama.cpp for GPT-2 — the receipt that admits <c>gpt2</c> to
/// <see cref="ModelCompatibility"/>'s architecture allowlist.
///
/// <para><b>The one genuinely new mechanism: learned absolute position embeddings.</b> Every
/// architecture admitted before this one uses RoPE (rotary position embeddings, applied inside
/// attention per layer); GPT-2 has NO rotary embeddings at all — position is encoded once, via a
/// <c>position_embd.weight</c> lookup table ADDED to the token embedding before the trunk starts
/// (<c>src/models/gpt2.cpp</c>: <c>inpL = ggml_add(tok_embd_lookup, pos_embd_lookup)</c>). This
/// engine had no support for this at all. Added <c>ForwardPass._posEmbdTensor</c> (loaded only
/// when a <c>position_embd.weight</c> tensor exists) and threaded a <c>position</c> parameter
/// through <c>EmbedTokenInto</c>/<c>EmbedToken</c> and all 8 of their call sites (every prefill
/// and decode dispatch path), so the position row is added right after the token-embedding copy
/// wherever a token gets embedded. Disabling RoPE entirely needed NO new field: setting
/// <c>ModelHyperparams.NoRopeLayerStep = 1</c> makes the existing periodic-skip formula
/// (<c>(layer+1) % step != 0</c>, built for Llama-4/SmolLM3's every-4th-layer NoPE) evaluate to
/// "skip RoPE on every layer" for free, reusing the same dispatch every other call site already
/// has instead of adding a new flag and a new set of guards.</para>
///
/// <para><b>Everything else was already generic.</b> LayerNorm-with-bias (<c>UsesLayerNorm</c>,
/// tensor-presence detected), the fused <c>attn_qkv.weight</c>/<c>.bias</c> single-tensor QKV
/// split (built for GPT-NeoX/Falcon, arch-agnostic — keyed on tensor name, not architecture
/// string), and the non-gated biased-GELU FFN path (<c>DenseFfn</c>'s <c>_wGate[layer].DataPtr is
/// null</c> branch, also built for GPT-NeoX) all applied to this checkpoint with zero new code.
/// </para>
///
/// <para><b>Checkpoint.</b> <c>openai-community/gpt2</c> (the original 124M base model, genuinely
/// MIT-licensed), via <c>sjfalken/openai-gpt2-124M-F16-gguf</c> (F16, near-lossless, 252 MB).
/// <b>Not</b> a Q6_K quant tried first (<c>RichardErkhov/openai-community_-_gpt2-gguf</c>), which
/// diverged at position 5 with only a 0.106-logit gap — reads as ordinary quantization
/// sensitivity for a genuinely small/weak 124M model (expected to be MORE quantization-sensitive
/// than the larger checkpoints this session's other near-ties came from, not less), confirmed by
/// re-running against this F16 checkpoint and getting a full exact match with no near-tie at all.
/// <c>tokenizer.ggml.model = gpt2</c> (byte-BPE, real merges array), <c>tokenizer.ggml.pre =
/// gpt-2</c> (already in the pre-tokenizer cascade — the original reference group).</para>
///
/// <para><b>Reference.</b> <c>tools/llama.cpp</c> build <c>b8585-cad2d3884</c>:
/// <code>
/// llama-tokenize -m &lt;model&gt; -p "The capital of France is" --ids --no-bos
///   -> [464, 3139, 286, 4881, 318]
/// llama-completion -m &lt;model&gt; -p "The capital of France is" -n 24 --temp 0 --top-k 1 --seed 0 \
///     -no-cnv --override-kv tokenizer.ggml.add_bos_token=bool:false
///   -> " the capital of the French Republic, and the capital of the French Republic is the
///       capital of the French Republic." (22 tokens — the model's own EOS-adjacent stop, short
///       of the 24 requested)
/// </code>
/// </para>
///
/// <para><b>Result: FULL 22-of-22-token exact match.</b> No near-tie, no divergence — every
/// position agrees with llama.cpp's F16 reference exactly, including through the entirely new
/// position-embedding and no-RoPE code paths.</para>
/// </summary>
public sealed class Gpt2GreedyParityTests
{
    private const string ModelFile = "gpt2-f16.gguf";

    /// <summary>Prompt token ids from llama-tokenize; see the class remarks.</summary>
    private static readonly int[] s_promptTokens = [464, 3139, 286, 4881, 318];

    /// <summary>The full llama.cpp reference continuation (22 tokens); see the class remarks.</summary>
    private static readonly int[] s_referenceContinuationTokens =
        [262, 3139, 286, 262, 4141, 2066, 11, 290, 262, 3139, 286, 262, 4141, 2066, 318, 262, 3139, 286, 262, 4141, 2066, 13];

    [Fact]
    public void Gpt2_GreedyContinuation_MatchesLlamaCpp()
    {
        var path = FindModel();
        Assert.SkipWhen(path is null, $"{ModelFile} is required for this parity receipt.");

        using var model = GgufModel.Open(path!);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var tokenizer = GgufTokenizer.FromGgufModel(model);

        Assert.Equal("gpt2", Convert.ToString(model.Metadata["general.architecture"]));
        Assert.Equal(s_promptTokens, tokenizer.Encode("The capital of France is"));

        // Guards the no-RoPE / LayerNorm detection: this receipt is worthless if the fixture
        // silently lost its NoRopeLayerStep=1 override or fell back to RMSNorm.
        Assert.Equal(1, hp.NoRopeLayerStep);
        Assert.True(hp.UsesLayerNorm, "gpt2 uses LayerNorm-with-bias, not RMSNorm");
        Assert.False(hp.UseParallelResidual, "gpt2 is a plain sequential residual trunk");

        using var backend = new CpuBackend();
        using var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 1024);

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
    /// time — specifically exercises that the position embedding lands on the correct absolute
    /// position in BOTH the batched-prefill and sequential-decode dispatch paths.
    /// </summary>
    [Fact]
    public void Gpt2_DecodeStepwise_AgreesWithSinglePassPrefill()
    {
        var path = FindModel();
        Assert.SkipWhen(path is null, $"{ModelFile} is required for this consistency check.");

        using var model = GgufModel.Open(path!);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var tokenizer = GgufTokenizer.FromGgufModel(model);

        int[] full = [.. s_promptTokens, s_referenceContinuationTokens[0], s_referenceContinuationTokens[1]];

        using var backend = new CpuBackend();

        float[] stepwise;
        using (var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 1024))
        {
            fwd.Prefill(s_promptTokens);
            var logits = fwd.Forward(full[^2], s_promptTokens.Length);
            logits = fwd.Forward(full[^1], s_promptTokens.Length + 1);
            stepwise = logits[..tokenizer.VocabSize].ToArray();
        }

        float[] singlePass;
        using (var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 1024))
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
