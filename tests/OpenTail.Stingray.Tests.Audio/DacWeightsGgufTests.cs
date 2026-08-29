
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Cross-format conformance test: the real community `ecyht2/parler-tts-mini-v1-GGUF` conversion's
/// DAC codec tensors (`audio_encoder.*`, a genuinely different flatter naming convention than the
/// Safetensors checkpoint -- see <see cref="DacWeights(GgufModel)"/>'s doc comment), loaded via
/// the new GGUF constructor, should decode the SAME real codes to (near-)identical PCM as the
/// already golden-verified Safetensors-loaded DAC. The Safetensors path is the oracle here
/// (already proven against a real external oracle in `ParlerDacTests`), not a fresh external
/// reference.
/// </summary>
public sealed class DacWeightsGgufTests : HeavyTestBase
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

    [Fact]
    public void Decode_GgufWeights_MatchesSafetensorsGoldenOutput()
    {
        string? stPath = FindModelPath("parler-tts-mini-v1.safetensors");
        string? ggufPath = FindModelPath("parler-tts-mini-v1-Q8_0.gguf");
        Assert.SkipUnless(stPath != null, "models/parler-tts-mini-v1.safetensors not found");
        Assert.SkipUnless(ggufPath != null, "models/parler-tts-mini-v1-Q8_0.gguf not found");

        // Deterministic real codes: 6 timesteps, 9 codebooks, in-range values.
        int t = 6;
        var codes = new int[DacWeights.NumCodebooks][];
        var rng = new Random(11);
        for (int cb = 0; cb < DacWeights.NumCodebooks; cb++)
        {
            codes[cb] = new int[t];
            for (int i = 0; i < t; i++) codes[cb][i] = rng.Next(0, DacWeights.CodebookSize);
        }

        float[] stPcm, ggufPcm;
        using (var loader = SafetensorsLoader.Open(stPath!))
        {
            var stWeights = new DacWeights(loader);
            stPcm = DacDecoder.Decode(stWeights, codes);
        }
        using (var model = GgufModel.Open(ggufPath!))
        {
            var ggufWeights = new DacWeights(model);
            ggufPcm = DacDecoder.Decode(ggufWeights, codes);
        }

        Assert.Equal(stPcm.Length, ggufPcm.Length);

        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < stPcm.Length; i++)
        {
            dot += ggufPcm[i] * stPcm[i];
            normA += ggufPcm[i] * ggufPcm[i];
            normB += stPcm[i] * stPcm[i];
        }
        double cosine = dot / (Math.Sqrt(normA) * Math.Sqrt(normB));
        Assert.True(cosine > 0.99, $"GGUF-vs-Safetensors DAC cosine {cosine} too low");
    }
}
