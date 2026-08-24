using System;
using System.IO;
using System.Text;
using OpenTail.Stingray.Audio.Kokoro;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Throwaway dev diagnostic: checks ABSOLUTE magnitude (not just cosine similarity, which is scale-invariant and can't catch a uniform scale bug) of PredictDurations against the real golden ReduceSum output.</summary>
public sealed class KokoroDurationMagnitudeCheckTests
{
    private static string? FindRepoFile(string relativePath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativePath);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static float[] ReadNpyFloat32(string path)
    {
        byte[] data = File.ReadAllBytes(path);
        byte major = data[6];
        int headerLen; int headerStart;
        if (major == 1) { headerLen = data[8] | (data[9] << 8); headerStart = 10; }
        else { headerLen = data[8] | (data[9] << 8) | (data[10] << 16) | (data[11] << 24); headerStart = 12; }
        int dataStart = headerStart + headerLen;
        int floatCount = (data.Length - dataStart) / 4;
        var result = new float[floatCount];
        Buffer.BlockCopy(data, dataStart, result, 0, floatCount * 4);
        return result;
    }

    [Fact]
    public void CheckDurationMagnitudeRatio()
    {
        string? modelPath = FindRepoFile("models/kokoro-82m-q8_0.gguf");
        string? goldenDurSumPath = FindRepoFile("scratch-llamacpp-ref/kokoro_golden_durenc/encoder_predictor_ReduceSum_output_0.npy");
        if (modelPath is null || goldenDurSumPath is null) return;

        using var weights = new KokoroWeights(modelPath);
        int[] inputIds = [0, 50, 83, 54, 156, 57, 135, 3, 16, 65, 156, 0];
        int t = inputIds.Length;

        var fullStyle = new float[2 * weights.StyleDim];
        fullStyle[0] = 0.1f; fullStyle[1] = -0.2f; fullStyle[2] = 0.3f; fullStyle[3] = 0.05f; fullStyle[4] = -0.1f;
        var predictorStyle = new float[weights.StyleDim];
        Array.Copy(fullStyle, weights.StyleDim, predictorStyle, 0, weights.StyleDim);

        float[] lastHidden = KokoroBertEncoder.Forward(weights, inputIds);
        float[] dEn = KokoroBertEncoder.ProjectToWorkingDim(weights, lastHidden, t);
        float[] d = KokoroProsodyPredictor.EncodeDuration(weights, dEn, predictorStyle, t);
        float[] durations = KokoroProsodyPredictor.PredictDurations(weights, d, t);

        float[] goldenDurSums = ReadNpyFloat32(goldenDurSumPath);

        var sb = new StringBuilder();
        sb.AppendLine("idx\tours\tgolden\tratio");
        for (int i = 0; i < durations.Length; i++)
        {
            float ratio = goldenDurSums[i] != 0 ? durations[i] / goldenDurSums[i] : float.NaN;
            sb.AppendLine($"{i}\t{durations[i]:F4}\t{goldenDurSums[i]:F4}\t{ratio:F4}");
        }
        File.WriteAllText(@"C:\Git-Public\OpenTail.Stingray\scratch-llamacpp-ref\duration_magnitude_check.txt", sb.ToString());
    }
}
