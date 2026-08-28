using System;
using System.Globalization;
using System.IO;
using System.Linq;
using OpenTail.Stingray.Audio.FishSpeech;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// TEMP debug harness (docs/audio-review-progress.md's Fish Speech "gargly" investigation,
/// 2026-08-28, per ChatGPT-suggested "Test A"): feeds the REAL reference's own generated 10xT
/// code sequence (captured directly from `examples/s2.cpp`'s own C++ generation, via a temporary
/// env-var-gated dump added to `s2_pipeline.cpp` and reverted after use) into our C# codec
/// decoder, bypassing the AR entirely. This isolates whether the "gargly" symptom is a codec-side
/// bug (this decode should match the reference's own decode of the SAME codes) or an AR-side bug
/// (our codes differ from what a healthy AR would produce). TODO remove once resolved.
/// </summary>
public sealed class FishSpeechCodecReferenceCodesDebugTests : HeavyTestBase
{
    private static string? FindModelPath(string fileName)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "models", fileName);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static string? FindCodesFile()
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "scratch_ref_codes.txt");
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static string? FindOutDir()
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "docs", "audio-samples");
            if (Directory.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void DecodeReferenceCodes_ThroughOurCodec()
    {
        string? modelPath = FindModelPath("s2-pro-q4_k_m.gguf");
        string? codesPath = FindCodesFile();
        string? outDir = FindOutDir();
        Assert.SkipUnless(modelPath != null && codesPath != null && outDir != null,
            "S2 Pro GGUF, scratch_ref_codes.txt, or docs/audio-samples not found");

        var lines = File.ReadAllLines(codesPath!);
        var header = lines[0].Split(' ', StringSplitOptions.RemoveEmptyEntries);
        int nFrames = int.Parse(header[0], CultureInfo.InvariantCulture);
        int numCodebooks = int.Parse(header[1], CultureInfo.InvariantCulture);
        Console.WriteLine($"nFrames={nFrames} numCodebooks={numCodebooks}");

        var semanticCodes = new int[nFrames];
        var residualCodes = new int[numCodebooks - 1][];
        for (int cb = 0; cb < numCodebooks - 1; cb++) residualCodes[cb] = new int[nFrames];

        for (int f = 0; f < nFrames; f++)
        {
            var row = lines[f + 1].Split(' ', StringSplitOptions.RemoveEmptyEntries);
            semanticCodes[f] = int.Parse(row[0], CultureInfo.InvariantCulture);
            for (int cb = 0; cb < numCodebooks - 1; cb++)
                residualCodes[cb][f] = int.Parse(row[cb + 1], CultureInfo.InvariantCulture);
        }

        using var codecWeights = new FishSpeechCodecWeights(modelPath!);
        var pcm = FishSpeechCodec.Decode(codecWeights, semanticCodes, residualCodes);

        int nan = pcm.Count(float.IsNaN);
        int inf = pcm.Count(float.IsInfinity);
        Console.WriteLine($"pcm.Length={pcm.Length} nan={nan} inf={inf} durationSec={pcm.Length / 44100.0:F2}");

        // Peak normalize to 0.85, matching FishSpeechFullPipeline.Synthesize's own post-processing,
        // so this is directly comparable (same gain convention) to our normal pipeline output.
        float peak = 0f;
        for (int i = 0; i < pcm.Length; i++) { float a = MathF.Abs(pcm[i]); if (a > peak) peak = a; }
        if (peak > 1e-4f && peak < 0.8f)
        {
            float gain = 0.85f / peak;
            for (int i = 0; i < pcm.Length; i++) pcm[i] *= gain;
        }

        var result = new OpenTail.Stingray.Audio.AudioGenerationResult(pcm, 44100);
        string outPath = Path.Combine(outDir!, "fishspeech-lunch-CSHARP-decode-of-reference-codes.wav");
        result.SaveWav(outPath);
        Console.WriteLine($"saved {outPath}");
    }
}
