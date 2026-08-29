using System;
using System.IO;
using OpenTail.Stingray.Audio.CosyVoice;
using OpenTail.Stingray.Audio.Primitives;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>TEMPORARY debug test: feeds the real reference's own dumped speech_feat/excitation
/// (`speech_feat_dump2.bin`/`excitation_dump2.bin`, produced by the same single reference CLI run
/// as `stage0_dump2.bin`) through our C# HiFTVocoderKernels.DecodeForTest with
/// STINGRAY_DUMP_STAGE0_PATH set, then computes cosine similarity between our end-of-stage-0
/// tensor and the reference's -- to check whether the `source_downs` causal-left-pad fix actually
/// moved the previously-measured 0.568 divergence.</summary>
public sealed class CosyVoiceHiftStage0CompareDebugTest : HeavyTestBase
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

    [Fact]
    public void Stage0_MatchesReference_AfterSourceDownPadFix()
    {
        string? ggufPath = FindModelPath("models/cosyvoice3/CosyVoice3-2512_F16.gguf");
        string? melDumpPath = FindModelPath("examples/cosyvoice.cpp/speech_feat_dump2.bin");
        string? excDumpPath = FindModelPath("examples/cosyvoice.cpp/excitation_dump2.bin");
        string? refStage0Path = FindModelPath("examples/cosyvoice.cpp/stage0_dump2.bin");
        Assert.SkipUnless(ggufPath != null && melDumpPath != null && excDumpPath != null && refStage0Path != null,
            "CosyVoice3 GGUF or reference speech_feat/excitation/stage0 dumps not found");

        const int melDim = 80;
        var melBytes = File.ReadAllBytes(melDumpPath!);
        int numFrames = (melBytes.Length / sizeof(float)) / melDim;
        var melFrameMajor = new float[melBytes.Length / sizeof(float)];
        Buffer.BlockCopy(melBytes, 0, melFrameMajor, 0, melBytes.Length);
        var melChannelFirst = new float[melFrameMajor.Length];
        for (int f = 0; f < numFrames; f++)
            for (int c = 0; c < melDim; c++)
                melChannelFirst[c * numFrames + f] = melFrameMajor[f * melDim + c];

        var excBytes = File.ReadAllBytes(excDumpPath!);
        var excitation = new float[excBytes.Length / sizeof(float)];
        Buffer.BlockCopy(excBytes, 0, excitation, 0, excBytes.Length);

        string ourStage0Path = Path.Combine(Path.GetTempPath(), "our_stage0.bin");
        if (File.Exists(ourStage0Path)) File.Delete(ourStage0Path);
        Environment.SetEnvironmentVariable("STINGRAY_DUMP_STAGE0_PATH", ourStage0Path);
        try
        {
            using var hiftWeights = new CosyVoice3HiftWeights(ggufPath!);
            _ = HiFTVocoderKernels.DecodeForTest(hiftWeights, melChannelFirst, numFrames, excitation, excitation.Length, melDim);
        }
        finally
        {
            Environment.SetEnvironmentVariable("STINGRAY_DUMP_STAGE0_PATH", null);
        }

        Assert.True(File.Exists(ourStage0Path), "our stage0 dump was not written");

        var refBytes = File.ReadAllBytes(refStage0Path!);
        var refStage0 = new float[refBytes.Length / sizeof(float)];
        Buffer.BlockCopy(refBytes, 0, refStage0, 0, refBytes.Length);

        var ourBytes = File.ReadAllBytes(ourStage0Path);
        var ourStage0 = new float[ourBytes.Length / sizeof(float)];
        Buffer.BlockCopy(ourBytes, 0, ourStage0, 0, ourBytes.Length);

        int n = Math.Min(refStage0.Length, ourStage0.Length);
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < n; i++)
        {
            dot += (double)refStage0[i] * ourStage0[i];
            na += (double)refStage0[i] * refStage0[i];
            nb += (double)ourStage0[i] * ourStage0[i];
        }
        double cosine = dot / (Math.Sqrt(na) * Math.Sqrt(nb) + 1e-12);
        string msg = $"[COMPARE] ref.Length={refStage0.Length} our.Length={ourStage0.Length} compared={n} cosine={cosine:F6}";
        Console.WriteLine(msg);
        File.WriteAllText(Path.Combine(Path.GetTempPath(), "stage0_compare_result.txt"), msg);
    }
}
