
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>TEMPORARY debug test: isolates whether OUR OWN LLM's newly-generated speech tokens
/// (with the new promptText/promptSpeechTokens conditioning) are the remaining source of
/// "gibberish", now that frontend extraction (mel/embedding/prompt-tokens) and flow-encoder/DiT/
/// HiFT math are BOTH proven bit-exact/correct against the reference. Uses the reference's real
/// promptTokens/embedding/promptFeat (proven identical to our own extraction already) but OUR OWN
/// LLM-generated new tokens, run through the same math chain that produced clean speech when fed
/// the reference's own new tokens (CosyVoice3FullChainFromRefInputsDebugTest).</summary>
public sealed class CosyVoice3OurLlmWithRefPromptDebugTest : HeavyTestBase
{
    private static string? FindModelPath(string relPath)
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

    private static int[] ReadInts(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var ints = new int[bytes.Length / sizeof(int)];
        Buffer.BlockCopy(bytes, 0, ints, 0, bytes.Length);
        return ints;
    }

    private static float[] ReadFloats(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var floats = new float[bytes.Length / sizeof(float)];
        Buffer.BlockCopy(bytes, 0, floats, 0, bytes.Length);
        return floats;
    }

    [Fact]
    public void OurLlm_WithRefPromptConditioning_ProducesAudio()
    {
        string dumpDir = FindModelPath("examples/cosyvoice.cpp/mu.bin") is { } p ? Path.GetDirectoryName(p)! : "";
        Assert.SkipUnless(!string.IsNullOrEmpty(dumpDir), "reference dumps not found");
        string? ggufPath = FindModelPath("models/cosyvoice3/CosyVoice3-2512_F16.gguf");
        Assert.SkipUnless(ggufPath != null, "CosyVoice3 GGUF not found");

        var refPromptTokens = ReadInts(Path.Combine(dumpDir, "prompttokens.bin"));
        var refNewTokens = ReadInts(Path.Combine(dumpDir, "newtokens.bin"));
        var embedding = ReadFloats(Path.Combine(dumpDir, "embedding.bin"));
        var promptFeatFrameMajor = ReadFloats(Path.Combine(dumpDir, "promptfeat.bin"));

        using var rawModel = GgufModel.Open(ggufPath!);
        var llmSource = new CosyVoice3LlmTensorSource(rawModel);
        llmSource.EnableSpeechGenerationMode();

        // OUR OWN LLM, conditioned on the REAL reference promptText/promptSpeechTokens.
        var ourNewTokens = CosyVoice3Llm.GenerateSpeechTokens(rawModel, llmSource, "This is a test of voice synthesis.", 200,
            promptText: "this is a test of voice cloning", promptSpeechTokens: refPromptTokens);

        string tokMsg = $"[OURLLM] ourNewTokens.Length={ourNewTokens.Length} refNewTokens.Length={refNewTokens.Length} " +
                        $"ourFirst20=[{string.Join(",", ourNewTokens[..Math.Min(20, ourNewTokens.Length)])}] " +
                        $"refFirst20=[{string.Join(",", refNewTokens[..Math.Min(20, refNewTokens.Length)])}]";
        Console.WriteLine(tokMsg);

        var flowWeights = new CosyVoice3FlowEncoderWeights(rawModel);
        var ditWeights = new CosyVoice3DiTWeights(rawModel);
        using var hiftWeights = new CosyVoice3HiftWeights(ggufPath!);

        int[] jointTokens = [.. refPromptTokens, .. ourNewTokens];
        var (mu, spks) = CosyVoice3FlowEncoder.ComputeMuAndSpks(flowWeights, jointTokens, embedding);

        const int melDim = 80;
        int numFrames = mu.Length / melDim;
        int promptFrames = Math.Min(promptFeatFrameMajor.Length / melDim, numFrames);
        var cond = new float[mu.Length];
        Array.Copy(promptFeatFrameMajor, 0, cond, 0, promptFrames * melDim);

        var spksBroadcast = new float[numFrames * melDim];
        for (int f = 0; f < numFrames; f++)
            Array.Copy(spks, 0, spksBroadcast, f * melDim, melDim);

        var rng = new Random(42);
        var mel = CosyVoice3DiTModel.SolveFlowMatchingOde(ditWeights, cond, mu, spksBroadcast, numFrames, odeSteps: 10, rng, cfgRate: 0.7f);

        var melChannelFirst = new float[mel.Length];
        for (int f = 0; f < numFrames; f++)
            for (int c = 0; c < melDim; c++)
                melChannelFirst[c * numFrames + f] = mel[f * melDim + c];

        var wav = CosyVoiceHiftVocoder.Generate(hiftWeights, melChannelFirst, numFrames, rng);

        float peak = 0f;
        for (int i = 0; i < wav.Length; i++) peak = MathF.Max(peak, MathF.Abs(wav[i]));
        if (peak > 1e-4f && peak < 0.8f)
        {
            float gain = 0.85f / peak;
            for (int i = 0; i < wav.Length; i++) wav[i] *= gain;
        }

        string repoRoot = Directory.GetParent(Directory.GetParent(Directory.GetParent(ggufPath!)!.FullName)!.FullName)!.FullName;
        var result = new AudioGenerationResult(wav, 24000);
        result.SaveWav(Path.Combine(repoRoot, "docs", "audio-samples", "cosyvoice3-ourllm-refprompt.wav"));
        File.WriteAllText(Path.Combine(Path.GetTempPath(), "ourllm_refprompt_result.txt"), tokMsg);
    }
}
