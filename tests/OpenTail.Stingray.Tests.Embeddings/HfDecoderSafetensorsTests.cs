using System.Diagnostics;
using System.Numerics.Tensors;

namespace OpenTail.Stingray.Tests.Embeddings;

/// <summary>
/// Decoder-only HF SafeTensors packages through the existing <c>ForwardPass</c> (profile
/// <c>dense-qwen-cpu</c>): Qwen/Qwen3-0.6B loaded from its BF16 <c>model.safetensors</c> is compared with the
/// same model's Q8_0 GGUF (already greedy-parity-checked against llama.cpp elsewhere) on prefill logits and a
/// greedy continuation. Agreement is bounded by Q8_0 quantization, so the gate is cosine and top-1, not
/// bit-equality. trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 (random 2-layer, hidden 8) is a loader
/// harness only: it must pass the capability inspector and produce finite logits over the full vocab.
/// </summary>
public sealed class HfDecoderSafetensorsTests
{
    private static readonly string[] Prompts =
    [
        "The capital of France is",
        "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
        "Translate to German: The weather is nice today.",
    ];

    [Fact]
    public void Qwen3_06B_Safetensors_MatchesGgufLogitsAndGreedy()
    {
        string? dir = BertEncoderOnnxParityTests.FindRepoDir("models/_models/hf/Qwen__Qwen3-0.6B");
        string? models = BertEncoderOnnxParityTests.FindRepoDir("models/_models");
        string? gguf = models is null ? null : Path.Combine(models, "Qwen3-0.6B-Q8_0.gguf");
        Assert.SkipUnless(dir != null && gguf != null && File.Exists(gguf), "Qwen3-0.6B safetensors or Q8_0 GGUF not found");

        var report = ModelPackageInspector.Inspect(dir!);
        Assert.True(report.IsSupported, string.Join("; ", report.Rejections.Select(r => r.Subject + ": " + r.Detail)));

        var sw = Stopwatch.StartNew();
        using var st = SafetensorsTensorSource.Open(dir!);
        var stHp = ModelHyperparams.FromGgufMetadata(st.Metadata, st);
        using var ggufModel = GgufModel.Open(gguf!);
        var gHp = ModelHyperparams.FromGgufMetadata(ggufModel.Metadata, ggufModel);
        var tokenizer = GgufTokenizer.FromGgufModel(ggufModel);
        Assert.Equal(gHp.HeadDim, stHp.HeadDim);
        Assert.Equal(128, stHp.HeadDim);

        using var backendA = new CpuBackend();
        using var backendB = new CpuBackend();
        using var fwdSt = new Engine.ForwardPass(st, backendA, stHp, maxContextLength: 256);
        using var fwdG = new Engine.ForwardPass(ggufModel, backendB, gHp, maxContextLength: 256);
        Console.WriteLine($"[Qwen3St] loaded both in {sw.ElapsedMilliseconds} ms");

        int vocab = gHp.VocabSize;
        foreach (var prompt in Prompts)
        {
            var ids = tokenizer.Encode(prompt);
            sw.Restart();
            var a = fwdSt.Prefill(ids)[..vocab].ToArray();
            var b = fwdG.Prefill(ids)[..vocab].ToArray();
            float cos = TensorPrimitives.CosineSimilarity(a, b);
            int topA = TensorPrimitives.IndexOfMax(a), topB = TensorPrimitives.IndexOfMax(b);

            // Greedy continuation, 16 tokens each.
            var genA = new List<int> { topA };
            var genB = new List<int> { topB };
            for (int i = 1; i < 16; i++)
            {
                genA.Add(TensorPrimitives.IndexOfMax(fwdSt.Forward(genA[^1], ids.Count + i - 1)[..vocab]));
                genB.Add(TensorPrimitives.IndexOfMax(fwdG.Forward(genB[^1], ids.Count + i - 1)[..vocab]));
            }
            int agree = 0;
            while (agree < genA.Count && genA[agree] == genB[agree]) agree++;
            Console.WriteLine($"[Qwen3St] \"{prompt.Split('\n')[0]}\": prefill cos {cos:F5}, top1 {topA}/{topB}, greedy agree {agree}/16 ({sw.ElapsedMilliseconds} ms)");
            Console.WriteLine($"[Qwen3St]   st:   {tokenizer.Decode(genA).Replace("\n", "\\n")}");
            Console.WriteLine($"[Qwen3St]   gguf: {tokenizer.Decode(genB).Replace("\n", "\\n")}");
            Assert.True(cos >= 0.99f, $"prefill logits cosine {cos}");
            Assert.Equal(topB, topA);
            Assert.True(agree >= 8, $"greedy diverged at {agree}");
        }
    }

    [Fact]
    public void TinyQwen2_Safetensors_LoadsAndProducesFiniteLogits()
    {
        string? dir = BertEncoderOnnxParityTests.FindRepoDir("models/_models/hf/trl-internal-testing__tiny-Qwen2ForCausalLM-2.5");
        Assert.SkipUnless(dir != null, "tiny-Qwen2ForCausalLM-2.5 not found");

        var report = ModelPackageInspector.Inspect(dir!);
        Assert.True(report.IsSupported, string.Join("; ", report.Rejections.Select(r => r.Subject + ": " + r.Detail)));
        Assert.Equal("dense-qwen-cpu", report.ProfileId);

        using var st = SafetensorsTensorSource.Open(dir!);
        Assert.Equal("qwen2", st.Metadata["general.architecture"]);
        Assert.Contains(st.Tensors, t => t.Name == "blk.0.attn_q.bias");
        var hp = ModelHyperparams.FromGgufMetadata(st.Metadata, st);
        using var backend = new CpuBackend();
        using var fwd = new Engine.ForwardPass(st, backend, hp, maxContextLength: 64);
        var logits = fwd.Prefill([9707, 11, 1879, 0])[..hp.VocabSize].ToArray();
        Assert.Equal(152064, logits.Length);
        Assert.All(logits, v => Assert.True(float.IsFinite(v)));
        Assert.NotEqual(0f, TensorPrimitives.MaxMagnitude(logits)); // signed value of largest magnitude
        Console.WriteLine($"[TinyQwen2] logits finite over {logits.Length} vocab; top1 {TensorPrimitives.IndexOfMax(logits)}");
    }

    /// <summary>llama.cpp greedy receipt for openai-community/gpt2 (F16 GGUF), from <c>Gpt2GreedyParityTests</c>.</summary>
    private static readonly int[] s_gpt2Prompt = [464, 3139, 286, 4881, 318];
    private static readonly int[] s_gpt2Reference =
        [262, 3139, 286, 262, 4141, 2066, 11, 290, 262, 3139, 286, 262, 4141, 2066, 318, 262, 3139, 286, 262, 4141, 2066, 13];

    [Fact]
    public void Gpt2_Safetensors_MatchesLlamaCppGreedyReceipt()
    {
        string? dir = BertEncoderOnnxParityTests.FindRepoDir("models/_models/hf/openai-community__gpt2");
        Assert.SkipUnless(dir != null, "openai-community/gpt2 safetensors not found");

        var report = ModelPackageInspector.Inspect(dir!);
        Assert.True(report.IsSupported, string.Join("; ", report.Rejections.Select(r => r.Subject + ": " + r.Detail)));
        Assert.Equal("dense-gpt2-cpu", report.ProfileId);

        var sw = Stopwatch.StartNew();
        using var st = SafetensorsTensorSource.Open(dir!);
        var hp = ModelHyperparams.FromGgufMetadata(st.Metadata, st);
        Assert.True(hp.UsesLayerNorm);
        Assert.Equal(1, hp.NoRopeLayerStep);
        var tok = HuggingFaceTokenizerSource.Load(dir!);
        Assert.True(tok.IsUsable, string.Join("; ", tok.Rejections.Select(r => r.Detail)));
        var tokenizer = GgufTokenizer.FromSource(tok.Source!);
        Assert.Equal(s_gpt2Prompt, tokenizer.Encode("The capital of France is"));

        using var backend = new CpuBackend();
        using var fwd = new Engine.ForwardPass(st, backend, hp, maxContextLength: 256);
        var logits = fwd.Prefill(s_gpt2Prompt);
        var generated = new List<int>();
        int pos = s_gpt2Prompt.Length;
        for (int i = 0; i < s_gpt2Reference.Length; i++)
        {
            int next = TensorPrimitives.IndexOfMax(logits[..hp.VocabSize]);
            generated.Add(next);
            if (i + 1 < s_gpt2Reference.Length) logits = fwd.Forward(next, pos++);
        }
        Console.WriteLine($"[Gpt2St] {sw.ElapsedMilliseconds} ms: {tokenizer.Decode(generated).Replace("\n", "\n")}");
        Assert.Equal(s_gpt2Reference, generated);
    }
}
