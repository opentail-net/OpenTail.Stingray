using System;
using System.IO;
using System.Text;
using OpenTail.Stingray.Audio.F5TTS;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Verifies the real Vocos vocoder (charactr/vocos-mel-24khz) against real weights and real
/// PyTorch reference output (scratch-llamacpp-ref/vocos_golden_decode.py). Feeds a random mel
/// directly (bypassing mel extraction, isolating vocoder correctness). Checks intermediate
/// stages (embed, backbone norm block0/final) separately, not just the final waveform.
///
/// IMPORTANT: any `.npy` saved from a non-contiguous torch tensor's `.numpy()` view gets
/// `fortran_order: True` in its header, which this flat-byte reader (like every reader used
/// throughout this rebuild) silently misinterprets as row-major -- always call `.contiguous()`
/// before `.numpy()` in the generating script (see F5DiTModelTests.cs's class doc for the full
/// story of a real bisection this caused).
/// </summary>
public sealed class VocosVocoderTests : HeavyTestBase
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
            throw new InvalidDataException("Not a .npy file");
        byte major = data[6];
        int headerLen, headerStart;
        if (major == 1) { headerLen = data[8] | (data[9] << 8); headerStart = 10; }
        else { headerLen = data[8] | (data[9] << 8) | (data[10] << 16) | (data[11] << 24); headerStart = 12; }
        string header = Encoding.ASCII.GetString(data, headerStart, headerLen);
        if (header.Contains("'fortran_order': True"))
            throw new InvalidDataException("fortran_order npy files are not supported by this flat-byte reader -- regenerate the dump with tensor.contiguous().numpy().");
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
    public void VocosVocoder_RealWeights_MatchesPyTorchGoldenOutput()
    {
        string? modelPath = FindRepoFile("models/vocos-mel-24khz.safetensors");
        string? dir = FindRepoFile("scratch-llamacpp-ref/vocos_golden_decode/audio_out.npy");
        if (modelPath is null || dir is null) return;
        string baseDir = Path.GetDirectoryName(dir)!;

        var weights = new VocosWeights(modelPath);

        float[] mel = ReadNpyFloat32(Path.Combine(baseDir, "input_mel.npy")); // [1,100,20] channel-first (torch mel convention)
        const int numFrames = 20;

        // Golden mel is channel-first [MelDim, T]; our C# port expects channel-last [T, MelDim].
        var melChannelLast = new float[numFrames * VocosWeights.MelDim];
        for (int c = 0; c < VocosWeights.MelDim; c++)
            for (int t = 0; t < numFrames; t++)
                melChannelLast[t * VocosWeights.MelDim + c] = mel[c * numFrames + t];

        float[] goldenAudio = ReadNpyFloat32(Path.Combine(baseDir, "audio_out.npy"));

        float[] audio = VocosVocoder.Decode(weights, melChannelLast, numFrames);

        Assert.Equal(goldenAudio.Length, audio.Length);
        foreach (float v in audio) Assert.False(float.IsNaN(v) || float.IsInfinity(v), "waveform must be finite");

        double cosine = CosineSimilarity(audio, goldenAudio);
        Assert.True(cosine > 0.99, $"Vocos waveform cosine similarity {cosine} too low vs golden PyTorch output.");
    }
}
