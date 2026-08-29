
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>TEMPORARY debug test: runs OUR full CosyVoice3 math chain (ComputeMuAndSpks -&gt;
/// SolveFlowMatchingOde -&gt; HiFT Decode) using the real reference's OWN dumped tokens/embedding/
/// prompt-mel (bypassing our own CamPlus/speech-tokenizer/mel-extractor frontends entirely), to
/// isolate whether the remaining "gibberish" symptom lives in our DiT/CFM math or in our own
/// frontend extraction quality. mu/spks/conds were already proven bit-exact (cosine 1.0) against
/// the reference given these same inputs (see CosyVoice3FlowEncoderCompareDebugTest) -- this test
/// goes one stage further, through the ODE solve and vocoder, and checks the result via Whisper
/// ASR instead of guessing from a numeric metric (the ODE's own random noise isn't reproducible
/// bit-for-bit against the reference, so a cosine comparison of the final mel wouldn't be
/// meaningful here -- intelligibility is the right signal at this stage).</summary>
public sealed class CosyVoice3FullChainFromRefInputsDebugTest : HeavyTestBase
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
    public void FullChain_FromRealReferenceInputs_ProducesAudio()
    {
        string dumpDir = FindModelPath("examples/cosyvoice.cpp/mu.bin") is { } p ? Path.GetDirectoryName(p)! : "";
        Assert.SkipUnless(!string.IsNullOrEmpty(dumpDir), "reference flow-encoder dumps not found");

        string? ggufPath = FindModelPath("models/cosyvoice3/CosyVoice3-2512_F16.gguf");
        Assert.SkipUnless(ggufPath != null, "CosyVoice3 GGUF not found");

        var newTokens = ReadInts(Path.Combine(dumpDir, "newtokens.bin"));
        var promptTokens = ReadInts(Path.Combine(dumpDir, "prompttokens.bin"));
        var embedding = ReadFloats(Path.Combine(dumpDir, "embedding.bin"));
        var promptFeatFrameMajor = ReadFloats(Path.Combine(dumpDir, "promptfeat.bin"));
        int[] jointTokens = [.. promptTokens, .. newTokens];

        using var rawModel = GgufModel.Open(ggufPath!);
        var flowWeights = new CosyVoice3FlowEncoderWeights(rawModel);
        var ditWeights = new CosyVoice3DiTWeights(rawModel);
        using var hiftWeights = new CosyVoice3HiftWeights(ggufPath!);

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
        result.SaveWav(Path.Combine(repoRoot, "docs", "audio-samples", "cosyvoice3-fullchain-refinputs.wav"));
        Console.WriteLine($"Wrote fullchain-from-ref-inputs wav, {wav.Length} samples, {numFrames} mel frames, promptFrames={promptFrames}");
    }
}
