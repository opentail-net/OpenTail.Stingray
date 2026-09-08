using OpenTail.Stingray.Audio.VibeVoice;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>ONE-OFF DEBUG: dumps config.json + prediction_head tensor names for VibeVoice TTS's
/// packed GGUF. Real ground truth for the diffusion head's real config numbers.</summary>
public sealed class VibeVoiceTtsDumpDebugTest : HeavyTestBase
{
    private static string? FindRepoFile(string relPath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relPath);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void DumpConfigAndPredictionHeadTensorNames()
    {
        string? path = FindRepoFile("models/_models/vibevoice-tts/VibeVoice-1.5B-GGUF/vibevoice-1.5b-q8_0.gguf");
        Assert.SkipUnless(path != null, "vibevoice-1.5b-q8_0.gguf not found");

        using var model = OpenTail.Stingray.Core.GgufModel.Open(path!);
        string? outDir = FindRepoFile("docs/audio-review-progress.md");
        string root = Path.GetDirectoryName(outDir!)!;

        if (model.Metadata.TryGetValue("audiocpp.embedded_files.names", out var namesObj) && namesObj is object[] names)
        {
            var offsets = (object[])model.Metadata["audiocpp.embedded_files.offsets"];
            var data = (object[])model.Metadata["audiocpp.embedded_files.data"];
            var bytes = data.Select(o => (byte)Convert.ToInt64(o)).ToArray();
            Console.Error.WriteLine($"[VibeVoiceTtsDump] embedded files: {string.Join(" | ", names.Cast<string>())}");
            for (int i = 0; i < names.Length; i++)
            {
                string name = (string)names[i];
                if (name != "config.json") continue;
                long start = Convert.ToInt64(offsets[i]);
                long end = i + 1 < offsets.Length ? Convert.ToInt64(offsets[i + 1]) : bytes.Length;
                string content = System.Text.Encoding.UTF8.GetString(bytes, (int)start, (int)(end - start));
                File.WriteAllText(Path.Combine(root, "..", "vibevoice-tts-config.json"), content);
            }
        }

        if (model.Metadata.TryGetValue("audiocpp.tensor_names", out var tnObj) && tnObj is object[] tensorNames)
        {
            var sb = new System.Text.StringBuilder();
            for (int i = 0; i < tensorNames.Length; i++)
            {
                string name = (string)tensorNames[i];
                if (!name.Contains("prediction_head", StringComparison.Ordinal)) continue;
                var t = model.Tensors[i];
                sb.AppendLine($"{name}\t{string.Join(",", t.Dimensions)}\t{t.DType}");
            }
            File.WriteAllText(Path.Combine(root, "..", "vibevoice-tts-prediction-head-tensor-names.txt"), sb.ToString());
            Console.Error.WriteLine($"[VibeVoiceTtsDump] wrote prediction_head tensor names, {tensorNames.Length} total tensors");
        }
    }

    [Fact]
    public void CompareEmbeddingsAndTraceHidden()
    {
        string? path = FindRepoFile("models/_models/vibevoice-tts/VibeVoice-1.5B-GGUF/vibevoice-1.5b-q8_0.gguf");
        Assert.SkipUnless(path != null, "vibevoice-1.5b-q8_0.gguf not found");

        using var model = OpenTail.Stingray.Core.GgufModel.Open(path!);
        var source = new OpenTail.Stingray.Audio.Rvc.RvcPackedTensorSource(model);
        var emb = source.GetTensor("model.language_model.embed_tokens.weight");
        const int hiddenDim = 1536;

        int t0 = 15226;
        float e0_0 = emb[t0 * hiddenDim + 0];
        float e0_39 = emb[t0 * hiddenDim + 39];
        Console.Error.WriteLine($"[EMB CHECK] Token {t0} emb[0]={e0_0} (ref: -0.00393677), emb[39]={e0_39} (ref: -0.0283203)");

        int tLast = 151652;
        float eL_0 = emb[tLast * hiddenDim + 0];
        float eL_39 = emb[tLast * hiddenDim + 39];
        Console.Error.WriteLine($"[EMB CHECK] Token {tLast} emb[0]={eL_0} (ref: -0.013855), emb[39]={eL_39} (ref: -0.0032959)");

        // Compare prompt token ids
        string dir = Path.Combine(Path.GetTempPath(), "stingray-vibevoice-tts-tokenizer");
        var result = HuggingFaceTokenizerSource.Load(dir);
        var tokenizer = GgufTokenizer.FromSource(result.Source!);
        string script = "Speaker 1: Hello there, this is a real end to end test of speech synthesis.";
        var promptTokenIds = VibeVoiceTtsPromptBuilder.BuildPrompt(text => [.. tokenizer.Encode(text)], script, 151652);
        Console.Error.WriteLine($"[PROMPT] C# prompt length = {promptTokenIds.Length}");
        Console.Error.WriteLine($"[PROMPT] tokens = [{string.Join(", ", promptTokenIds)}]");

        const int NumLayers = 28, NumHeads = 12, NumKvHeads = 2, HeadDim = 128, FfDim = 8960, VocabSize = 151936;
        const float RopeTheta = 1_000_000f, RmsNormEps = 1e-6f;
        using var llm = new VibeVoiceLlmTensorSource(source, NumLayers, hiddenDim, NumHeads, NumKvHeads, HeadDim, FfDim, VocabSize, RopeTheta, RmsNormEps);
        var hp = ModelHyperparams.FromGgufMetadata(llm.Metadata);
        Assert.True(hp.HasAttnBias, "VibeVoice Qwen2 backbone must have attention bias enabled.");

        using var backend = new CpuBackend();
        using var fwd = new ForwardPass(llm, backend, hp);

        fwd.Prefill(promptTokenIds);
        var csLastHidden = fwd.LastHidden.ToArray();
        Console.Error.WriteLine($"[HIDDEN CHECK] C# step 0 hidden[0]={csLastHidden[0]} (ref: -0.391388), hidden[39]={csLastHidden[39]} (ref: 0.947536)");
        Console.Error.WriteLine($"[HIDDEN CHECK] C# step 0 hidden[78]={csLastHidden[78]} (ref: -1.39805), hidden[117]={csLastHidden[117]} (ref: 0.714557)");

        // Reference sample indices and values from C++ trace:
        int[] sampleIndices = [
            0, 39, 78, 117, 157, 196, 235, 275, 314, 353, 393, 432, 471, 511, 512, 558, 604, 651, 697, 744,
            790, 837, 883, 930, 976, 1023, 1024, 1063, 1102, 1141, 1181, 1220, 1259, 1299, 1338, 1377, 1417, 1456, 1495, 1535
        ];
        float[] refValues = [
            -0.391388f, 0.947536f, -1.39805f, 0.714557f, 0.467165f, 0.33005f, -2.58657f, -1.96559f, -0.427127f, 1.17882f,
            -0.774252f, 1.43499f, -0.78333f, 0.995172f, 0.919917f, 1.99151f, -1.10308f, -0.217178f, -0.28154f, -0.636229f,
            -0.128229f, 1.54485f, -0.943278f, -1.21248f, 0.0203801f, 2.31171f, -1.2374f, 1.14996f, 0.164772f, -0.469612f,
            -0.3092f, -2.067f, -1.08506f, 0.0182186f, -0.431231f, 0.505418f, -0.276733f, 1.57315f, 1.21501f, -2.10977f
        ];

        double dot = 0, normCs = 0, normRef = 0;
        var diffs = new System.Text.StringBuilder();
        for (int i = 0; i < sampleIndices.Length; i++)
        {
            int idx = sampleIndices[i];
            float cs = csLastHidden[idx];
            float rf = refValues[i];
            dot += cs * rf;
            normCs += cs * cs;
            normRef += rf * rf;
            diffs.AppendLine($"[{idx}] cs={cs,10:F6} ref={rf,10:F6} diff={Math.Abs(cs - rf),10:F6}");
        }
        double cosine = dot / (Math.Sqrt(normCs) * Math.Sqrt(normRef));
        diffs.AppendLine($"Cosine similarity over 40 sample points = {cosine:F6}");

        string reportPath = Path.Combine(Path.GetTempPath(), "vibevoice-tts-hidden0-comparison.txt");
        File.WriteAllText(reportPath, diffs.ToString());

        Assert.True(cosine > 0.99, $"Step 0 hidden cosine similarity {cosine:F6} too low vs reference C++!");
    }
}
