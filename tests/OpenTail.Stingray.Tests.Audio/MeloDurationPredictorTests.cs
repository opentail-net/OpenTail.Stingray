
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Verifies MeloTTS's blended duration predictor (sdp + dp) against real ONNX weights and real
/// onnxruntime golden output (scratch-llamacpp-ref/melo_golden_durpred.py). Checks the SDP half
/// and the plain DP half SEPARATELY against their own golden node outputs (not just the final
/// blend), since sdp_ratio-weighted blending could mask a broken half if only the blend were
/// checked -- e.g. sdp_ratio=0.2 would let an 80%-wrong DP half still pass a loose cosine bar.
/// </summary>
public sealed class MeloDurationPredictorTests : HeavyTestBase
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
    public void MeloDurationPredictor_RealOnnxWeights_MatchesOnnxGoldenLogw()
    {
        string? modelPath = FindRepoFile("models/melotts-zh_en.onnx");
        string? noisePath = FindRepoFile("scratch-llamacpp-ref/melo_golden_durpred/sdp_RandomNormalLike_output_0.npy");
        string? sdpLogwPath = FindRepoFile("scratch-llamacpp-ref/melo_golden_durpred/sdp_Split_output_0.npy");
        string? dpLogwPath = FindRepoFile("scratch-llamacpp-ref/melo_golden_durpred/dp_proj_Conv_output_0.npy");
        if (modelPath is null || noisePath is null || sdpLogwPath is null || dpLogwPath is null) return;

        var weights = new MeloOnnxWeights(modelPath);

        // Matches melo_golden_durpred.py's fixed input_ids/tones/sid exactly.
        int[] tokens = [1, 5, 10, 20, 30, 40, 50, 2];
        int[] tones = [0, 1, 2, 3, 4, 5, 6, 0];
        const int speakerId = 0;
        int t = tokens.Length;

        var (encoderHidden, _, _) = MeloTextEncoder.Forward(weights, tokens, tones, speakerId);

        var g = new float[weights.GinChannels];
        Array.Copy(weights.EmbGWeight, speakerId * weights.GinChannels, g, 0, weights.GinChannels);

        float[] noise = ReadNpyFloat32(noisePath); // [1,2,T] raw N(0,1), channel-first
        Assert.Equal(2 * t, noise.Length);
        const float noiseScaleW = 0.8f; // matches melo_golden_durpred.py's noise_scale_w

        // Isolate the SDP half via reflection-free direct call: PredictLogDuration blends, so
        // verify each half by calling it with sdp_ratio=1 (pure SDP) and sdp_ratio=0 (pure DP).
        float[] logwSdp = MeloDurationPredictor.PredictLogDuration(weights, encoderHidden, t, g, noise, noiseScaleW, sdpRatio: 1.0f);
        float[] logwDp = MeloDurationPredictor.PredictLogDuration(weights, encoderHidden, t, g, noise, noiseScaleW, sdpRatio: 0.0f);

        float[] goldenSdpLogw = ReadNpyFloat32(sdpLogwPath);
        float[] goldenDpLogw = ReadNpyFloat32(dpLogwPath);
        Assert.Equal(t, goldenSdpLogw.Length);
        Assert.Equal(t, goldenDpLogw.Length);

        foreach (float v in logwSdp) Assert.False(float.IsNaN(v) || float.IsInfinity(v), "sdp logw must be finite");
        foreach (float v in logwDp) Assert.False(float.IsNaN(v) || float.IsInfinity(v), "dp logw must be finite");

        double sdpCosine = CosineSimilarity(logwSdp, goldenSdpLogw);
        double dpCosine = CosineSimilarity(logwDp, goldenDpLogw);
        Assert.True(sdpCosine > 0.99, $"SDP logw cosine similarity {sdpCosine} too low vs golden ONNX output.");
        Assert.True(dpCosine > 0.99, $"DP logw cosine similarity {dpCosine} too low vs golden ONNX output.");
    }
}
