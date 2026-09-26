using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// Real-weight checks for gpt-oss (GptOssForwardPass.cs): finite logits, a greedy parity
/// receipt against llama-server, and a decode-speed benchmark. gpt-oss is admitted to
/// <see cref="ModelCompatibility"/> on the parity receipt below (2026-09-26).
/// </summary>
public sealed class GptOssRealWeightSmokeTests : HeavyTestBase
{
    private const string ModelFile = "gpt-oss-20b-MXFP4.gguf";

    [Fact]
    public void GptOss_LoadsAndProducesFiniteLogits_ForOneToken()
    {
        var path = FindModel();
        Assert.SkipWhen(path is null, $"{ModelFile} is required for this smoke test.");

        using var model = GgufModel.Open(path!);
        var metadata = model.Metadata;

        Assert.Equal("gpt-oss", Convert.ToString(metadata["general.architecture"]));

        var hp = GptOssHyperparams.FromModel(model);
        int vocabSize = hp.VocabSize;
        Assert.True(hp.NumLayer is 24 or 36, $"unexpected layer count {hp.NumLayer} (reference only recognizes 24=20B/36=120B)");

        using var fwd = new GptOssForwardPass(model, hp);

        Console.WriteLine($"Wq dtype: {model.FindTensor("blk.0.attn_q.weight")?.DType}");
        Console.WriteLine($"Wo dtype: {model.FindTensor("blk.0.attn_output.weight")?.DType}");
        Console.WriteLine($"GateInp dtype: {model.FindTensor("blk.0.ffn_gate_inp.weight")?.DType}");
        Console.WriteLine($"GateExps dtype: {model.FindTensor("blk.0.ffn_gate_exps.weight")?.DType}");
        Console.WriteLine($"DownExps dtype: {model.FindTensor("blk.0.ffn_down_exps.weight")?.DType}");
        Console.WriteLine($"Output dtype: {model.FindTensor("output.weight")?.DType}");

        var logits = fwd.Forward(token: 100, position: 0);

        Assert.Equal(vocabSize, logits.Length);
        bool anyNonZero = false;
        foreach (float v in logits)
        {
            Assert.False(float.IsNaN(v), "logit was NaN");
            Assert.False(float.IsInfinity(v), "logit was infinite");
            if (v != 0f) anyNonZero = true;
        }
        Assert.True(anyNonZero, "all logits were exactly zero -- suspicious, likely a wiring bug");
    }

    /// <summary>
    /// Greedy parity receipt against llama-server (vendored tools/llama.cpp, same GGUF, raw
    /// completion, temperature 0, top_k 1, captured 2026-09-26 with return_tokens:true). The
    /// prompt tokenization is checked too (it exercises the "gpt-4o" pre-tokenizer cascade).
    /// </summary>
    [Fact]
    public void GptOss_TeacherForcedMatchesLlamaServer_24Tokens()
    {
        var path = FindModel();
        Assert.SkipWhen(path is null, $"{ModelFile} is required for this parity test.");

        using var model = GgufModel.Open(path!);
        Assert.Equal("gpt-4o", Convert.ToString(model.Metadata["tokenizer.ggml.pre"]));

        var tokenizer = GgufTokenizer.FromGgufModel(model);
        var promptTokens = tokenizer.Encode("The capital of France is");
        Assert.Equal([976, 9029, 328, 10128, 382], promptTokens);

        using var fwd = new GptOssForwardPass(model, GptOssHyperparams.FromModel(model));

        ReadOnlySpan<float> logits = default;
        int pos = 0;
        foreach (int t in promptTokens)
            logits = fwd.Forward(t, pos++);

        // " Paris.\"\n    # Test with a non-existent page\n    content = fetch_wikipedia_page_content(\"ThisPageDoesNot"
        int[] expected =
        [
            12650, 14396, 271, 1069, 4674, 483, 261, 2893, 130142, 3011, 198, 271,
            3100, 314, 12011, 3567, 18249, 13263, 16500, 568, 2500, 3325, 28133, 2874,
        ];
        // Teacher-forced: feed llama-server's own greedy tokens and require each to sit within
        // MaxGap logits of our top logit. An exact free-running match is too brittle here —
        // step 1 is a 0.02-logit tie in llama.cpp itself (14396 -1.852 vs 3692 -1.872), and
        // llama.cpp's -fa on/off alone moves logits by up to 0.13.
        const float MaxGap = 0.25f;
        int exact = 0;
        float worstGap = 0f;
        for (int i = 0; i < expected.Length; i++)
        {
            int argmax = Argmax(logits);
            float gap = logits[argmax] - logits[expected[i]];
            if (argmax == expected[i]) exact++;
            worstGap = MathF.Max(worstGap, gap);
            Assert.True(gap <= MaxGap, $"step {i}: reference token {expected[i]} is {gap:F3} logits below our argmax {argmax}");
            logits = fwd.Forward(expected[i], pos++);
        }

        Console.WriteLine($"Teacher-forced: {exact}/{expected.Length} exact argmax, worst gap {worstGap:F3}");
    }


    [Fact]
    public void GptOss_Benchmark_DecodeSpeed()
    {
        var path = FindModel();
        Assert.SkipWhen(path is null, $"{ModelFile} is required for this smoke test.");

        using var model = GgufModel.Open(path!);
        var hp = GptOssHyperparams.FromModel(model);

        using var fwd = new GptOssForwardPass(model, hp);

        // Warmup (1 token)
        var logits = fwd.Forward(100, 0);

        int decodeTokens = 16;
        long bytesBefore = GC.GetAllocatedBytesForCurrentThread();
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var tokenSw = System.Diagnostics.Stopwatch.StartNew();
        for (int i = 0; i < decodeTokens; i++)
        {
            tokenSw.Restart();
            int next = Argmax(logits);
            logits = fwd.Forward(next, 1 + i);
            Console.WriteLine($"Token {i + 1}/{decodeTokens} (id={next}): {tokenSw.Elapsed.TotalMilliseconds:F1} ms");
        }
        sw.Stop();
        long bytesAfter = GC.GetAllocatedBytesForCurrentThread();

        double totalMs = sw.Elapsed.TotalMilliseconds;
        double msPerToken = totalMs / decodeTokens;
        double tokPerSec = decodeTokens / sw.Elapsed.TotalSeconds;
        long allocBytesPerToken = (bytesAfter - bytesBefore) / decodeTokens;

        Console.WriteLine($"\n=======================================================");
        Console.WriteLine($"[GPT-OSS 20B MXFP4 CPU SPEED] Sustained Speed: {tokPerSec:F2} T/S ({msPerToken:F2} ms/tok)");
        Console.WriteLine($"[GPT-OSS 20B MXFP4 CPU SPEED] Total: {totalMs:F2} ms for {decodeTokens} tokens | Alloc: {allocBytesPerToken / 1024.0:F1} KB/token");
        Console.WriteLine($"=======================================================\n");
    }

    private static int Argmax(ReadOnlySpan<float> logits)
    {
        int best = 0;
        float bestVal = logits[0];
        for (int i = 1; i < logits.Length; i++)
        {
            if (logits[i] > bestVal) { bestVal = logits[i]; best = i; }
        }
        return best;
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
