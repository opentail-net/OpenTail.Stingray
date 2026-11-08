using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Phase 0.2 diagnostic (docs/2-coverage/2026-11-07-coverage-completion-programme-plan.md): runs the
/// five-token prompt "The capital of France is" through Qwen4ExpForwardPass and dumps l_last_N and
/// result_output at the last position, for comparison with the llama.cpp eval-callback dump.
/// Opt-in: skipped unless QWEN_STINGRAY_DUMP_DIR names an existing directory. Not a parity assertion.
/// </summary>
public sealed class Qwen4ExpRefDumpTests
{
    // Token IDs as printed by llama-eval-callback for "The capital of France is" (BOS-less, 5 tokens).
    private static readonly int[] PromptTokens = { 760, 6511, 314, 9338, 369 };

    private static string? FindModel() => OpenTail.Stingray.Engine.Verification.ModelLocator.FindDeep(
        "Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf");

    /// <summary>
    /// Phase 0.2: feed llama.cpp's own final_output-0 into Stingray's layer-0 ssm_out weights (dequantised f32, dot per row)
    /// and compare with llama.cpp's linear_attn_out-0. Separates a weight/orientation error from an input error.
    /// </summary>
    [Fact]
    public unsafe void SsmOutLayer0_ReferenceInput_VersusReferenceOutput()
    {
        string? refDir = Environment.GetEnvironmentVariable("QWEN_REF_DUMP_DIR");
        Assert.SkipUnless(refDir is not null && Directory.Exists(refDir),
            "Set QWEN_REF_DUMP_DIR to the llama.cpp eval-callback dump directory.");
        string? path = FindModel();
        Assert.SkipUnless(path is not null, "Qwen3.8-Flash-Next UD-IQ1_S checkpoint is required (paged, 72.5 GB).");

        using var model = OpenTail.Stingray.Core.GgufModel.Open(path!);
        int totalLayers = (int)Convert.ToInt64(model.Metadata["qwen4exp.block_count"]);
        var hp = Qwen4ExpHyperparams.FromMetadata(model.Metadata, totalLayers);
        var tensors = Qwen4ExpTensorSet.Load(model, hp, totalLayers);
        var ssmOut = tensors.Layers[0].SsmOut!.Value;

        int inDim = 6144;
        int outDim = (int)(ssmOut.Info.ElementCount / inDim);
        var w = new float[ssmOut.Info.ElementCount];
        long byteSize = OpenTail.Stingray.Core.DTypeInfo.ByteSize((int)ssmOut.Info.ElementCount, ssmOut.DType);
        OpenTail.Stingray.Cpu.Dequantize.ToFloat32(new ReadOnlySpan<byte>(ssmOut.DataPtr, checked((int)byteSize)), w, ssmOut.DType, w.Length);

        var finalOut = ReadLastToken(Path.Combine(refDir!, "final_output_0.bin"), inDim);
        var refLinear = ReadLastToken(Path.Combine(refDir!, "linear_attn_out_0.bin"), outDim);

        var y = new double[outDim];
        for (int r = 0; r < outDim; r++)
        {
            double acc = 0;
            for (int c = 0; c < inDim; c++) acc += (double)w[r * inDim + c] * finalOut[c];
            y[r] = acc;
        }

        double num = 0, den = 0, dot = 0, ny = 0;
        for (int r = 0; r < outDim; r++)
        {
            double d = y[r] - refLinear[r];
            num += d * d;
            den += (double)refLinear[r] * refLinear[r];
            dot += y[r] * refLinear[r];
            ny += y[r] * y[r];
        }
        Console.WriteLine($"ssm_out(ref final_output) vs ref linear_attn_out: rel={Math.Sqrt(num / den):G6} cos={dot / Math.Sqrt(ny * den):F6}");
    }

    /// <summary>
    /// Phase 0.2 acceptance check: greedy continuation of "The capital of France is" for QWEN_GREEDY_STEPS tokens.
    /// Prints token IDs and text so they can be compared with llama.cpp's greedy output. Opt-in.
    /// </summary>
    [Fact]
    public void GreedyContinuation_PrintsTokensForComparison()
    {
        string? stepsText = Environment.GetEnvironmentVariable("QWEN_GREEDY_STEPS");
        Assert.SkipUnless(int.TryParse(stepsText, out int steps) && steps > 0,
            "Set QWEN_GREEDY_STEPS to the number of greedy tokens to generate.");
        string? path = FindModel();
        Assert.SkipUnless(path is not null, "Qwen3.8-Flash-Next UD-IQ1_S checkpoint is required (paged, 72.5 GB).");

        using var model = OpenTail.Stingray.Core.GgufModel.Open(path!);
        int totalLayers = (int)Convert.ToInt64(model.Metadata["qwen4exp.block_count"]);
        var hp = Qwen4ExpHyperparams.FromMetadata(model.Metadata, totalLayers);
        var tensors = Qwen4ExpTensorSet.Load(model, hp, totalLayers);
        using var forwardPass = new Qwen4ExpForwardPass(model, hp, tensors);
        var tokenizer = OpenTail.Stingray.Core.GgufTokenizer.FromGgufModel(model);

        string prompt = Environment.GetEnvironmentVariable("QWEN_GREEDY_PROMPT") ?? "The capital of France is";
        var ids = tokenizer.Encode(prompt).ToList();
        int pos = 0;
        ReadOnlySpan<float> logits = default;
        foreach (int t in ids) logits = forwardPass.Forward(t, pos++);

        var generated = new List<int>();
        for (int s = 0; s < steps; s++)
        {
            int best = 0;
            for (int i = 1; i < logits.Length; i++) if (logits[i] > logits[best]) best = i;
            generated.Add(best);
            if (Environment.GetEnvironmentVariable("QWEN_GREEDY_TOP") == "1")
            {
                float[] lg = logits.ToArray();
                var top = Enumerable.Range(0, lg.Length).OrderByDescending(i => lg[i]).Take(5).ToArray();
                Console.WriteLine($"GREEDY_STEP {s} top5: " + string.Join(" ", top.Select(i => $"{i}:{lg[i]:F4}")) + $" eog:{string.Join(",", tokenizer.EogTokenIds)}");
            }
            if (tokenizer.EogTokenIds.Contains(best)) break;
            logits = forwardPass.Forward(best, pos++);
        }

        Console.WriteLine("GREEDY_PROMPT " + prompt);
        Console.WriteLine("GREEDY_IDS " + string.Join(",", generated));
        Console.WriteLine("GREEDY_TEXT " + tokenizer.Decode(generated));
    }

    private static float[] ReadLastToken(string path, int width)
    {
        var all = File.ReadAllBytes(path);
        int floats = all.Length / 4;
        int offset = floats - width; // last token is the last slice of a [width, n_tokens] tensor
        var result = new float[width];
        Buffer.BlockCopy(all, offset * 4, result, 0, width * 4);
        return result;
    }

    [Fact]
    public void PromptPass_DumpsLastPositionResidualsAndLogits()
    {
        string? outDir = Environment.GetEnvironmentVariable("QWEN_STINGRAY_DUMP_DIR");
        Assert.SkipUnless(outDir is not null && Directory.Exists(outDir),
            "Set QWEN_STINGRAY_DUMP_DIR to an existing directory to run the Qwen dump.");
        string? path = FindModel();
        Assert.SkipUnless(path is not null, "Qwen3.8-Flash-Next UD-IQ1_S checkpoint is required (paged, 72.5 GB).");

        string? customPrompt = Environment.GetEnvironmentVariable("QWEN_DUMP_PROMPT");
        using var tokModel = OpenTail.Stingray.Core.GgufModel.Open(path!);
        int[] toks = customPrompt is null
            ? PromptTokens
            : OpenTail.Stingray.Core.GgufTokenizer.FromGgufModel(tokModel).Encode(customPrompt).ToArray();
        int last = toks.Length - 1;
        Console.WriteLine("DUMP_TOKENS " + string.Join(",", toks));
        if (Environment.GetEnvironmentVariable("QWEN_DECODE") == "1")
        {
            var tk = OpenTail.Stingray.Core.GgufTokenizer.FromGgufModel(tokModel);
            foreach (int id in new[] { 8964, 48017, 20403, 42120, 271, 198, 32, 13, 33 })
                Console.WriteLine($"DECODE {id} => [{tk.Decode(new[] { id })}]");
        }
        Environment.SetEnvironmentVariable("STINGRAY_Q4E_DUMP_DIR", outDir);
        Environment.SetEnvironmentVariable("STINGRAY_Q4E_DUMP_POS", last.ToString());
        try
        {
            using var model = OpenTail.Stingray.Core.GgufModel.Open(path!);
            int totalLayers = (int)Convert.ToInt64(model.Metadata["qwen4exp.block_count"]);
            var hp = Qwen4ExpHyperparams.FromMetadata(model.Metadata, totalLayers);
            var tensors = Qwen4ExpTensorSet.Load(model, hp, totalLayers);
            using var forwardPass = new Qwen4ExpForwardPass(model, hp, tensors);

            ReadOnlySpan<float> logits = default;
            for (int pos = 0; pos <= last; pos++)
            {
                logits = forwardPass.Forward(toks[pos], pos);
            }

            Assert.False(float.IsNaN(logits[0]) || float.IsInfinity(logits[0]));
            Assert.True(File.Exists(Path.Combine(outDir!, "result_output.bin")));
            Assert.True(File.Exists(Path.Combine(outDir!, $"l_last_{totalLayers - 2}.bin")));
        }
        finally
        {
            Environment.SetEnvironmentVariable("STINGRAY_Q4E_DUMP_DIR", null);
            Environment.SetEnvironmentVariable("STINGRAY_Q4E_DUMP_POS", null);
        }
    }
}
