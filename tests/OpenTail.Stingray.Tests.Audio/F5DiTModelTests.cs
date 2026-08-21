using System;
using System.IO;
using System.Text;
using OpenTail.Stingray.Audio.F5TTS;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Verifies F5-TTS's DiT transformer against real weights and real PyTorch reference output
/// (scratch-llamacpp-ref/f5_golden_dit.py -- unlike the ONNX-exported VITS-family pipelines, this
/// golden dump runs the ACTUAL PyTorch reference source (examples/f5-tts-py) loaded with the real
/// safetensors checkpoint, not an ONNX re-export, since F5-TTS ships as safetensors with working
/// PyTorch source available). Checks text_embed, input_embed, time_embed, and the final velocity
/// output separately against their own golden targets, not just the final output.
///
/// IMPORTANT for anyone extending this dump: every `np.save(..., tensor.numpy())` in the
/// generating script MUST use `tensor.contiguous().numpy()`. A non-contiguous tensor (e.g. the
/// result of `.permute(...)`) gets saved by numpy with `fortran_order: True`, which this (and any
/// naive) flat-byte `.npy` reader silently misinterprets as plain row-major data -- discovered via
/// a real bisection here: three intermediate dumps (`input_embed`, and two ad hoc debug dumps)
/// were saved via non-contiguous tensors, making otherwise-correct C# code look ~100% wrong
/// (cosine near 0) until traced back to the dump script, not the port.
/// </summary>
public sealed class F5DiTModelTests : HeavyTestBase
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

    private static (byte[] Payload, char Descr) ParseNpyHeader(byte[] data)
    {
        if (data[0] != 0x93 || Encoding.ASCII.GetString(data, 1, 5) != "NUMPY")
            throw new InvalidDataException("Not a .npy file");
        byte major = data[6];
        int headerLen, headerStart;
        if (major == 1) { headerLen = data[8] | (data[9] << 8); headerStart = 10; }
        else { headerLen = data[8] | (data[9] << 8) | (data[10] << 16) | (data[11] << 24); headerStart = 12; }
        string header = Encoding.ASCII.GetString(data, headerStart, headerLen);
        if (header.Contains("'fortran_order': True"))
            throw new InvalidDataException("fortran_order npy files are not supported by this flat-byte reader -- regenerate the dump with tensor.contiguous().numpy().");
        char descr = header.Contains("'<f4'") ? 'f' : header.Contains("'<i8'") ? 'i' : '?';
        int dataStart = headerStart + headerLen;
        var payload = new byte[data.Length - dataStart];
        Array.Copy(data, dataStart, payload, 0, payload.Length);
        return (payload, descr);
    }

    private static float[] ReadNpyFloat32(string path)
    {
        var (payload, descr) = ParseNpyHeader(File.ReadAllBytes(path));
        Assert.Equal('f', descr);
        var result = new float[payload.Length / 4];
        Buffer.BlockCopy(payload, 0, result, 0, payload.Length);
        return result;
    }

    private static int[] ReadNpyInt64AsInt32(string path)
    {
        var (payload, descr) = ParseNpyHeader(File.ReadAllBytes(path));
        Assert.Equal('i', descr);
        int count = payload.Length / 8;
        var result = new int[count];
        for (int i = 0; i < count; i++) result[i] = (int)BitConverter.ToInt64(payload, i * 8);
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
    public void F5DiT_RealWeights_MatchesPyTorchGoldenOutput()
    {
        string? modelPath = FindRepoFile("models/f5tts_base.safetensors");
        string? dir = FindRepoFile("scratch-llamacpp-ref/f5_golden_dit/velocity.npy");
        if (modelPath is null || dir is null) return;
        string baseDir = Path.GetDirectoryName(dir)!;

        var weights = new F5TtsWeights(modelPath);

        float[] x = ReadNpyFloat32(Path.Combine(baseDir, "input_x.npy"));
        float[] cond = ReadNpyFloat32(Path.Combine(baseDir, "input_cond.npy"));
        int[] tokens = ReadNpyInt64AsInt32(Path.Combine(baseDir, "input_text.npy"));
        const float timestep = 0.3f;
        const int numFrames = 20;

        float[] goldenTextEmbed = ReadNpyFloat32(Path.Combine(baseDir, "text_embed.npy"));
        float[] goldenTimeEmbed = ReadNpyFloat32(Path.Combine(baseDir, "time_embed.npy"));
        float[] goldenInputEmbed = ReadNpyFloat32(Path.Combine(baseDir, "input_embed.npy"));
        float[] goldenVelocity = ReadNpyFloat32(Path.Combine(baseDir, "velocity.npy"));

        float[] textEmbed = F5TextEmbedding.Forward(weights, tokens, numFrames);
        Assert.True(CosineSimilarity(textEmbed, goldenTextEmbed) > 0.999, "text_embed mismatch");

        float[] timeEmbed = F5TimestepEmbedding.Forward(weights, timestep);
        Assert.True(CosineSimilarity(timeEmbed, goldenTimeEmbed) > 0.999, "time_embed mismatch");

        float[] inputEmbed = F5InputEmbedding.Forward(weights, x, cond, textEmbed, numFrames);
        Assert.True(CosineSimilarity(inputEmbed, goldenInputEmbed) > 0.999, "input_embed mismatch");

        float[] velocity = F5DiTModel.ForwardVelocity(weights, x, cond, tokens, timestep, numFrames);
        foreach (float v in velocity) Assert.False(float.IsNaN(v) || float.IsInfinity(v), "velocity must be finite");

        double cosine = CosineSimilarity(velocity, goldenVelocity);
        Assert.True(cosine > 0.99, $"Final velocity cosine similarity {cosine} too low vs golden PyTorch output.");
    }
}
