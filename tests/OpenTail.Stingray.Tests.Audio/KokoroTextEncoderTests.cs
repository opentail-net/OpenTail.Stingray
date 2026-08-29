
namespace OpenTail.Stingray.Tests.Audio.Fast;

public sealed class KokoroTextEncoderTests : HeavyTestBase
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

    /// <summary>
    /// Verifies KModel.text_encoder (CNN+BiLSTM `t_en`, channel-first [512,T]) against a real
    /// onnxruntime run of models/kokoro-v1.0.onnx (scratch-llamacpp-ref/kokoro_golden_textenc.py),
    /// using the exact same fixed input_ids as the BERT stage test. See docs/audio-review-progress.md.
    /// </summary>
    [Fact]
    public void KokoroTextEncoder_RealWeights_MatchesOnnxGoldenOutput()
    {
        string? modelPath = FindRepoFile("models/kokoro-82m-q8_0.gguf");
        string? golden = FindRepoFile("scratch-llamacpp-ref/kokoro_golden_textenc/encoder_text_encoder_Transpose_2_output_0.npy");
        if (modelPath is null || golden is null) return;

        using var weights = new KokoroWeights(modelPath);

        int[] inputIds = [0, 50, 83, 54, 156, 57, 135, 3, 16, 65, 156, 0];
        int t = inputIds.Length;

        float[] tEn = KokoroTextEncoder.Forward(weights, inputIds);

        float[] goldenValues = ReadNpyFloat32(golden);
        Assert.Equal(weights.HiddenDim * t, goldenValues.Length);

        double maxAbsDiff = 0;
        double sumAbsDiff = 0;
        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < goldenValues.Length; i++)
        {
            double diff = Math.Abs(tEn[i] - goldenValues[i]);
            maxAbsDiff = Math.Max(maxAbsDiff, diff);
            sumAbsDiff += diff;
            dot += (double)tEn[i] * goldenValues[i];
            normA += (double)tEn[i] * tEn[i];
            normB += (double)goldenValues[i] * goldenValues[i];
        }
        double meanAbsDiff = sumAbsDiff / goldenValues.Length;
        double cosineSim = dot / (Math.Sqrt(normA) * Math.Sqrt(normB));

        // Q8_0 GGUF weights vs FP32 ONNX reference -- cosine similarity is the primary signal,
        // same rationale as KokoroBertEncoderTests (raw diff picks up quantization noise, not bugs).
        Assert.True(cosineSim > 0.999, $"TextEncoder t_en cosine similarity {cosineSim} too low vs golden ONNX output (maxAbsDiff={maxAbsDiff}, meanAbsDiff={meanAbsDiff}) -- suggests an architecture bug, not just Q8_0 quantization noise.");
    }
}
