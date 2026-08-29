
namespace OpenTail.Stingray.Tests.Audio;

public sealed class KokoroDecoderTests : HeavyTestBase
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

    /// <summary>
    /// Verifies KokoroDecoder.Forward (Decoder.forward + Generator.forward, istftnet.py) in
    /// isolation: feeds the real onnxruntime-computed `asr`/F0_pred/N_pred tensors
    /// (scratch-llamacpp-ref/kokoro_golden_decoder.py's `/encoder/MatMul_1_output_0`,
    /// `/encoder/F0_proj/Conv_output_0`, `/encoder/N_proj/Conv_output_0`) directly into
    /// KokoroDecoder.Forward, bypassing the not-yet-integration-tested upstream stages, and
    /// compares against the golden `waveform` output (the model's actual final output tensor)
    /// via cross-correlation-aligned cosine similarity. Cross-correlation alignment is needed
    /// because a real STFT/iSTFT round-trip is not guaranteed to be phase-locked to sample 0
    /// the same way a hand-derived C# NSF/iSTFT reimplementation is -- this isolates "did we
    /// synthesize materially the right waveform" from "is our iSTFT windowing off by a few
    /// samples of pure delay", which duration/spectral-shape correctness does not care about.
    /// </summary>
    [Fact]
    public void KokoroDecoder_RealWeights_MatchesOnnxGoldenWaveform()
    {
        string? modelPath = FindRepoFile("models/kokoro-82m-q8_0.gguf");
        string? asrPath = FindRepoFile("scratch-llamacpp-ref/kokoro_golden_decoder/encoder_MatMul_1_output_0.npy");
        string? f0Path = FindRepoFile("scratch-llamacpp-ref/kokoro_golden_decoder/encoder_F0_proj_Conv_output_0.npy");
        string? nPath = FindRepoFile("scratch-llamacpp-ref/kokoro_golden_decoder/encoder_N_proj_Conv_output_0.npy");
        string? stylePath = FindRepoFile("scratch-llamacpp-ref/kokoro_golden_decoder/style.npy");
        string? waveformPath = FindRepoFile("scratch-llamacpp-ref/kokoro_golden_decoder/waveform.npy");
        if (modelPath is null || asrPath is null || f0Path is null || nPath is null || stylePath is null || waveformPath is null) return;

        using var weights = new KokoroWeights(modelPath);

        float[] asr = ReadNpyFloat32(asrPath); // [1,512,T] flat -> channel-first [512,T]
        int t = asr.Length / 512;

        float[] f0Curve = ReadNpyFloat32(f0Path); // [1,1,2T] flat -> [1,2T]
        float[] nCurve = ReadNpyFloat32(nPath);
        Assert.Equal(t * 2, f0Curve.Length);
        Assert.Equal(t * 2, nCurve.Length);

        float[] fullStyle = ReadNpyFloat32(stylePath); // [1,256]
        var sDec = new float[128];
        Array.Copy(fullStyle, 0, sDec, 0, 128); // ref_s[:, :128]

        float[] waveform = KokoroDecoder.Forward(weights.Decoder, weights, asr, f0Curve, nCurve, sDec, t);

        float[] goldenWaveform = ReadNpyFloat32(waveformPath);
        Assert.NotEmpty(waveform);

        // Threshold is deliberately much looser than the >0.998 bar used for the deterministic
        // upstream stages (BERT/TextEncoder/DurationEncoder/F0Ntrain): istftnet.py's SineGen /
        // SourceModuleHnNSF injects genuine torch.rand/torch.randn_like noise on every call with
        // no seed, so the golden `waveform` reflects one specific, unrepeatable random draw that
        // even a byte-perfect port could not reproduce -- only correlate strongly with. 0.45 was
        // picked well above chance-level correlation (~0.06, what this test measured before the
        // AdaINResBlock1 kernel-size bug below was fixed) and below what we've observed with the
        // fix applied (~0.52), so a regression back to "wrong resblock kernel" or similar
        // structural breakage still fails loudly.
        double bestCosine = BestAlignedCosineSimilarity(waveform, goldenWaveform, maxShift: 32);
        Assert.True(bestCosine > 0.45,
            $"Decoder/Generator waveform cosine similarity {bestCosine} too low vs golden ONNX waveform (lengths: ours={waveform.Length}, golden={goldenWaveform.Length}).");
    }

    /// <summary>
    /// Cosine similarity over the overlapping region at the best of a small set of integer
    /// sample shifts, since a from-scratch iSTFT/overlap-add reimplementation is not guaranteed
    /// to be phase/offset-identical to PyTorch's torch.istft framing.
    /// </summary>
    private static double BestAlignedCosineSimilarity(float[] a, float[] b, int maxShift)
    {
        double best = -1.0;
        for (int shift = -maxShift; shift <= maxShift; shift++)
        {
            int aStart = shift < 0 ? -shift : 0;
            int bStart = shift > 0 ? shift : 0;
            int len = Math.Min(a.Length - aStart, b.Length - bStart);
            if (len <= 0) continue;

            double dot = 0, normA = 0, normB = 0;
            for (int i = 0; i < len; i++)
            {
                float av = a[aStart + i];
                float bv = b[bStart + i];
                dot += (double)av * bv;
                normA += (double)av * av;
                normB += (double)bv * bv;
            }
            if (normA <= 0 || normB <= 0) continue;
            double cos = dot / (Math.Sqrt(normA) * Math.Sqrt(normB));
            if (cos > best) best = cos;
        }
        return best;
    }
}
