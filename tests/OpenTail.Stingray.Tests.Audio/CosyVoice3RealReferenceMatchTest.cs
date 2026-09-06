
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Generates CosyVoice3 output with the EXACT same reference audio (b.wav), reference
/// text, target text, and seed used in the real examples/audio.cpp reference run
/// (cosyvoice3-REFERENCE-audiocpp-real.wav, confirmed by ear to sound correct/clean), for the
/// most direct possible A/B comparison against our own port's output on identical inputs.</summary>
public sealed class CosyVoice3RealReferenceMatchTest : HeavyTestBase
{
    private static string? FindRepoFile(string relativePath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativePath);
            if (File.Exists(p) || Directory.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void Generate_MatchingRealReferenceInputs()
    {
        string? modelPath = FindRepoFile("models/cosyvoice3/CosyVoice3-2512_F16-with-noise.gguf");
        Assert.SkipUnless(modelPath != null, "CosyVoice3 GGUF model not found");
        string? refAudio = FindRepoFile("docs/audio-samples/cosyvoice3-ref-b.wav");
        Assert.SkipUnless(refAudio != null, "reference b.wav not found (copy it from examples/audio.cpp/assets/resources/b.wav)");
        string? outDir = FindRepoFile("docs/audio-samples");
        Assert.SkipUnless(outDir != null, "docs/audio-samples directory not found");

        using var pipeline = CosyVoice3Pipeline.Load(modelPath!);

        const string targetText = "Hello, I will make some lunch, darling!";
        const string referenceText = "Some call me nature. Others call me Mother Nature. I have been here for over four and a half billion years.";

        var pcm = pipeline.Generate(targetText, maxNewSpeechTokens: 200, odeSteps: 10, seed: 42,
            referenceAudioPath: refAudio, cfgRate: 0.7f, referenceText: referenceText, temperature: 0.8f, pitchScale: 1.0f);

        Assert.NotEmpty(pcm);
        string path = Path.Combine(outDir!, "cosyvoice3-OURS-frozen-noise-fix.wav");
        new OpenTail.Stingray.Audio.AudioGenerationResult(pcm, 24000).SaveWav(path);
        Console.Error.WriteLine($"[CosyVoice3RealMatch] samples={pcm.Length} seconds={pcm.Length / 24000.0:F2}");
    }

    /// <summary>Generalization check for the RoPE-heads fix (numRopeHeads:1 in
    /// CosyVoice3DiTModel.Attention): a SECOND, independent text/seed pair (different from
    /// Generate_MatchingRealReferenceInputs' "Hello, I will make some lunch, darling!"/seed 42),
    /// with its own real reference run captured via cosyvoice3-REFERENCE-gen2.wav, so the fix's
    /// per-channel agreement with the reference isn't just an artifact of the one sample it was
    /// found and tuned against.</summary>
    [Fact]
    public void Generate_SecondIndependentSample_ChannelStatsVsReference()
    {
        string? modelPath = FindRepoFile("models/cosyvoice3/CosyVoice3-2512_F16-with-noise.gguf");
        Assert.SkipUnless(modelPath != null, "CosyVoice3 GGUF model not found");
        string? refAudio = FindRepoFile("docs/audio-samples/cosyvoice3-ref-b.wav");
        Assert.SkipUnless(refAudio != null, "reference b.wav not found");
        string? outDir = FindRepoFile("docs/audio-samples");
        Assert.SkipUnless(outDir != null, "docs/audio-samples directory not found");
        string? refWavPath = FindRepoFile("docs/audio-samples/cosyvoice3-REFERENCE-gen2.wav");
        Assert.SkipUnless(refWavPath != null, "cosyvoice3-REFERENCE-gen2.wav not found (real C++ reference run)");

        using var pipeline = CosyVoice3Pipeline.Load(modelPath!);

        const string targetText = "The weather today is absolutely beautiful, perfect for a long walk in the park.";
        const string referenceText = "Some call me nature. Others call me Mother Nature. I have been here for over four and a half billion years.";

        var pcm = pipeline.Generate(targetText, maxNewSpeechTokens: 200, odeSteps: 10, seed: 123,
            referenceAudioPath: refAudio, cfgRate: 0.7f, referenceText: referenceText, temperature: 0.8f, pitchScale: 1.0f);

        Assert.NotEmpty(pcm);
        string ourPath = Path.Combine(outDir!, "cosyvoice3-OURS-gen2.wav");
        new OpenTail.Stingray.Audio.AudioGenerationResult(pcm, 24000).SaveWav(ourPath);
        Console.Error.WriteLine($"[Gen2] samples={pcm.Length} seconds={pcm.Length / 24000.0:F2}");

        var (refSamples, _, _) = WavReader.ReadWav(refWavPath!);
        var refMel = CosyVoiceMelExtractor.Shared.ExtractMel(refSamples);
        var ourMel = CosyVoiceMelExtractor.Shared.ExtractMel(pcm);
        const int melDim = 80;
        int refFrames = refMel.Length / melDim;
        int ourFrames = ourMel.Length / melDim;

        int worstOutlier = -1; double worstRatio = 1.0;
        var sb = new System.Text.StringBuilder("[Gen2ChannelStats] ratios=");
        for (int c = 0; c < melDim; c++)
        {
            double refSum = 0, refSumSq = 0;
            for (int f = 0; f < refFrames; f++) { double v = refMel[f * melDim + c]; refSum += v; refSumSq += v * v; }
            double refMean = refSum / refFrames;
            double refStd = Math.Sqrt(Math.Max(0, refSumSq / refFrames - refMean * refMean));

            double ourSum = 0, ourSumSq = 0;
            for (int f = 0; f < ourFrames; f++) { double v = ourMel[f * melDim + c]; ourSum += v; ourSumSq += v * v; }
            double ourMean = ourSum / ourFrames;
            double ourStd = Math.Sqrt(Math.Max(0, ourSumSq / ourFrames - ourMean * ourMean));

            double ratio = refStd > 1e-6 ? ourStd / refStd : 0;
            sb.Append(ratio.ToString("F2")).Append(',');
            if (Math.Abs(Math.Log(Math.Max(ratio, 1e-6))) > Math.Abs(Math.Log(Math.Max(worstRatio, 1e-6))))
            {
                worstRatio = ratio;
                worstOutlier = c;
            }
        }
        Console.Error.WriteLine(sb.ToString());
        Console.Error.WriteLine($"[Gen2ChannelStats] worstOutlierChannel={worstOutlier} ratio={worstRatio:F2}");
    }

    [Fact]
    public void CompareWaveformsChannelStats()
    {
        string? refPath = FindRepoFile("docs/audio-samples/cosyvoice3-REFERENCE-audiocpp-real.wav");
        string? ourPath = FindRepoFile("docs/audio-samples/cosyvoice3-OURS-frozen-noise-fix.wav");
        Assert.SkipUnless(refPath != null && ourPath != null, "WAV files not found");

        var (refSamples, refSr, _) = WavReader.ReadWav(refPath!);
        var (ourSamples, ourSr, _) = WavReader.ReadWav(ourPath!);

        var refMel = CosyVoiceMelExtractor.Shared.ExtractMel(refSamples);
        var ourMel = CosyVoiceMelExtractor.Shared.ExtractMel(ourSamples);

        string? modelPath = FindRepoFile("models/cosyvoice3/CosyVoice3-2512_F16-with-noise.gguf");
        string? refAudio = FindRepoFile("docs/audio-samples/cosyvoice3-ref-b.wav");
        if (modelPath != null && refAudio != null)
        {
            var (bSamples, bSr, _) = WavReader.ReadWav(refAudio);
            var bMel = CosyVoiceMelExtractor.Shared.ExtractMel(bSamples);
            string onnxPath = Path.Combine(Path.GetDirectoryName(modelPath)!, "frontend-onnx", "speech_tokenizer_v3.onnx");
            var bTokens = CosyVoiceSpeechTokenizer.Extract(onnxPath, bSamples) ?? [];
            File.AppendAllText(Path.Combine(Path.GetDirectoryName(refPath!)!, "channel_stats_comparison.txt"),
                $"\n[Reference b.wav breakdown] bMel.Frames={bMel.Length / 80}, bTokens.Count={bTokens.Length}, bTokens*2={bTokens.Length * 2}\n");
        }

        const int melDim = 80;
        int refFrames = refMel.Length / melDim;
        int ourFrames = ourMel.Length / melDim;

        var sb = new System.Text.StringBuilder();
        sb.AppendLine($"Channel Stats Comparison: refFrames={refFrames}, ourFrames={ourFrames}");
        sb.AppendLine($"{"Ch",4} | {"Ref Mean",9} | {"Our Mean",9} | {"Ref Std",8} | {"Our Std",8} | {"Ratio (Our/Ref)",15}");
        sb.AppendLine(new string('-', 65));

        for (int c = 0; c < melDim; c++)
        {
            double refSum = 0, refSumSq = 0;
            for (int f = 0; f < refFrames; f++)
            {
                double v = refMel[f * melDim + c];
                refSum += v;
                refSumSq += v * v;
            }
            double refMean = refSum / refFrames;
            double refVar = Math.Max(0, refSumSq / refFrames - refMean * refMean);
            double refStd = Math.Sqrt(refVar);

            double ourSum = 0, ourSumSq = 0;
            for (int f = 0; f < ourFrames; f++)
            {
                double v = ourMel[f * melDim + c];
                ourSum += v;
                ourSumSq += v * v;
            }
            double ourMean = ourSum / ourFrames;
            double ourVar = Math.Max(0, ourSumSq / ourFrames - ourMean * ourMean);
            double ourStd = Math.Sqrt(ourVar);

            double ratio = refStd > 1e-6 ? ourStd / refStd : 0;
            sb.AppendLine($"{c,4} | {refMean,9:F3} | {ourMean,9:F3} | {refStd,8:F3} | {ourStd,8:F3} | {ratio,15:F2}x");
        }

        string outTxt = Path.Combine(Path.GetDirectoryName(refPath)!, "channel_stats_comparison.txt");
        File.WriteAllText(outTxt, sb.ToString());

        if (modelPath != null && refAudio != null)
        {
            var (bSamples, bSr, _) = WavReader.ReadWav(refAudio);
            var bMel = CosyVoiceMelExtractor.Shared.ExtractMel(bSamples);
            string onnxPath = Path.Combine(Path.GetDirectoryName(modelPath)!, "frontend-onnx", "speech_tokenizer_v3.onnx");
            var bTokensRaw = CosyVoiceSpeechTokenizer.Extract(onnxPath, bSamples) ?? [];
            var bSamples16k = AudioResampler.Resample(bSamples, bSr, CosyVoiceSpeechTokenizer.SampleRate);
            var bTokensResampled = CosyVoiceSpeechTokenizer.Extract(onnxPath, bSamples16k) ?? [];

            File.AppendAllText(outTxt,
                $"\n[Reference b.wav breakdown]\n" +
                $"  bMel.Frames: {bMel.Length / 80} (extracted at 24kHz)\n" +
                $"  bTokens (raw 24kHz unresampled): {bTokensRaw.Length} tokens -> {bTokensRaw.Length * 2} frames\n" +
                $"  bTokens (proper 16kHz resampled, what CosyVoice3Pipeline.ExtractPromptTokens actually runs): {bTokensResampled.Length} tokens -> {bTokensResampled.Length * 2} frames\n");
        }
        Console.Error.WriteLine(sb.ToString());
    }
}
