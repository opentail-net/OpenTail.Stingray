
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Cross-format conformance test: the real community `ecyht2/parler-tts-mini-v1-GGUF` conversion
/// of the SAME `parler-tts/parler-tts-mini-v1` checkpoint this project already golden-verified
/// from Safetensors, loaded via <see cref="ParlerDecoderWeights"/>'s new GGUF constructor, should
/// produce (near-)identical decoder output to the already golden-verified Safetensors path on the
/// same real deterministic input -- the Safetensors path IS the oracle here (already proven
/// against a real external PyTorch oracle in `ParlerDecoderTests`), not a fresh external
/// reference. See docs/audio-review-progress.md's GGUF-expansion entries for the full derivation,
/// including the real, confirmed limitation that this GGUF conversion has NO text_encoder (T5)
/// tensors -- decoder/DAC only.
/// </summary>
public sealed class ParlerDecoderGgufTests : HeavyTestBase
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
    public void Forward_GgufWeights_MatchesSafetensorsGoldenOutput()
    {
        string? stPath = FindModelPath("parler-tts-mini-v1.safetensors");
        string? ggufPath = FindModelPath("parler-tts-mini-v1-Q8_0.gguf");
        Assert.SkipUnless(stPath != null, "models/parler-tts-mini-v1.safetensors not found");
        Assert.SkipUnless(ggufPath != null, "models/parler-tts-mini-v1-Q8_0.gguf not found");

        // Same deterministic fixture as ParlerDecoderTests: 4-position, all-0.05 fake encoder hidden.
        var encoderHidden = new float[4][];
        for (int i = 0; i < 4; i++)
        {
            var row = new float[ParlerDecoderWeights.HiddenDim];
            for (int d = 0; d < row.Length; d++) row[d] = 0.05f;
            encoderHidden[i] = row;
        }

        // Deterministic codebook-id sequence, small T for a quick but real cross-format check.
        int t = 5;
        var codebookIds = new int[t][];
        var rng = new Random(7);
        for (int i = 0; i < t; i++)
        {
            codebookIds[i] = new int[ParlerDecoderWeights.NumCodebooks];
            for (int cb = 0; cb < ParlerDecoderWeights.NumCodebooks; cb++)
                codebookIds[i][cb] = rng.Next(0, ParlerDecoderWeights.InputVocabSize);
        }

        float[][] stOutput, ggufOutput;
        using (var loader = SafetensorsLoader.Open(stPath!))
        {
            var stWeights = new ParlerDecoderWeights(loader);
            var inputEmbeds = new float[t][];
            for (int i = 0; i < t; i++) inputEmbeds[i] = ParlerDecoder.EmbedStep(stWeights, codebookIds[i], i);
            stOutput = ParlerDecoder.Forward(stWeights, inputEmbeds, encoderHidden);
        }

        using (var model = GgufModel.Open(ggufPath!))
        {
            var ggufWeights = new ParlerDecoderWeights(model);
            var inputEmbeds = new float[t][];
            for (int i = 0; i < t; i++) inputEmbeds[i] = ParlerDecoder.EmbedStep(ggufWeights, codebookIds[i], i);
            ggufOutput = ParlerDecoder.Forward(ggufWeights, inputEmbeds, encoderHidden);
        }

        for (int i = 0; i < t; i++)
        {
            double dot = 0, normA = 0, normB = 0;
            for (int d = 0; d < ParlerDecoderWeights.HiddenDim; d++)
            {
                float a = ggufOutput[i][d];
                float b = stOutput[i][d];
                dot += a * b;
                normA += a * a;
                normB += b * b;
            }
            double cosine = dot / (Math.Sqrt(normA) * Math.Sqrt(normB));
            Assert.True(cosine > 0.99, $"position {i}: GGUF-vs-Safetensors cosine {cosine} too low");
        }
    }
}
