using System;
using System.IO;
using System.Text;
using OpenTail.Stingray.Audio.MeloTTS;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Verifies MeloTextEncoder (VITS2 enc_p: token+tone+language embeddings + zero'd bert/ja_bert
/// bias + speaker-conditioned windowed relative-position Transformer + proj to mu/logs) against
/// real ONNX weights and real onnxruntime golden output
/// (scratch-llamacpp-ref/melo_golden_textenc.py's `/enc_p/proj/Conv_output_0`, [1,384,T]).
/// </summary>
public sealed class MeloTextEncoderTests : HeavyTestBase
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
        if (data[0] != 0x93 || Encoding.ASCII.GetString(data, 1, 5) != "NUMPY")
            throw new InvalidDataException($"Not a .npy file: {path}");
        byte major = data[6];
        int headerLen;
        int headerStart;
        if (major == 1)
        {
            headerLen = data[8] | (data[9] << 8);
            headerStart = 10;
        }
        else
        {
            headerLen = data[8] | (data[9] << 8) | (data[10] << 16) | (data[11] << 24);
            headerStart = 12;
        }
        int dataStart = headerStart + headerLen;
        int floatCount = (data.Length - dataStart) / 4;
        var result = new float[floatCount];
        Buffer.BlockCopy(data, dataStart, result, 0, floatCount * 4);
        return result;
    }

    private static double CosineSimilarity(float[] a, float[] b)
    {
        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += (double)a[i] * b[i];
            normA += (double)a[i] * a[i];
            normB += (double)b[i] * b[i];
        }
        return dot / (Math.Sqrt(normA) * Math.Sqrt(normB));
    }

    [Fact]
    public void MeloTextEncoder_RealOnnxWeights_MatchesOnnxGoldenOutput()
    {
        string? modelPath = FindRepoFile("models/melotts-zh_en.onnx");
        string? goldenPath = FindRepoFile("scratch-llamacpp-ref/melo_golden_textenc/enc_p_proj_Conv_output_0.npy");
        if (modelPath is null || goldenPath is null) return;

        var weights = new MeloOnnxWeights(modelPath);

        // Matches melo_golden_textenc.py's fixed input_ids/tones/sid exactly.
        int[] tokens = [1, 5, 10, 20, 30, 40, 50, 2];
        int[] tones = [0, 1, 2, 3, 4, 5, 6, 0];
        const int speakerId = 0;

        var (_, mu, logs) = MeloTextEncoder.Forward(weights, tokens, tones, speakerId, out float[] preEncoderX);

        int t = tokens.Length;
        int dim = weights.HiddenDim;

        string? preEncoderGoldenPath = FindRepoFile("scratch-llamacpp-ref/melo_golden_textenc/enc_p_Mul_output_0.npy");
        if (preEncoderGoldenPath is not null)
        {
            float[] goldenTimeMajor = ReadNpyFloat32(preEncoderGoldenPath); // [1, T, dim] time-major
            var goldenChannelFirst = new float[dim * t];
            for (int ti = 0; ti < t; ti++)
                for (int c = 0; c < dim; c++)
                    goldenChannelFirst[c * t + ti] = goldenTimeMajor[ti * dim + c];
            double preCosine = CosineSimilarity(preEncoderX, goldenChannelFirst);
            Assert.True(preCosine > 0.999, $"Pre-encoder embedding-sum cosine similarity {preCosine} too low vs golden ONNX /enc_p/Mul_output_0.");
        }

        Assert.Equal(dim * t, mu.Length);
        Assert.Equal(dim * t, logs.Length);

        float[] golden = ReadNpyFloat32(goldenPath); // [1, 384, T] = [mu; logs] concatenated on channel dim
        Assert.Equal(2 * dim * t, golden.Length);

        var goldenMu = new float[dim * t];
        var goldenLogs = new float[dim * t];
        Array.Copy(golden, 0, goldenMu, 0, dim * t);
        Array.Copy(golden, dim * t, goldenLogs, 0, dim * t);

        double muCosine = CosineSimilarity(mu, goldenMu);
        double logsCosine = CosineSimilarity(logs, goldenLogs);
        Assert.True(muCosine > 0.999, $"TextEncoder mu cosine similarity {muCosine} too low vs golden ONNX output.");
        Assert.True(logsCosine > 0.999, $"TextEncoder logs cosine similarity {logsCosine} too low vs golden ONNX output.");
    }
}
