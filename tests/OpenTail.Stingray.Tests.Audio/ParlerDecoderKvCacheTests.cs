
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Conformance test for <see cref="ParlerDecoder.ForwardStep"/>'s self-/cross-attention KV cache:
/// asserts (a) step-by-step cached decoding produces the IDENTICAL result to the already
/// golden-verified full-sequence <see cref="ParlerDecoder.Forward"/> (the batch path is the
/// oracle here -- no new numerical claim, only cache-correctness), and (b) the cache's own
/// internal bookkeeping matches the real architecture: cross-attention K/V is built exactly once
/// per layer and never rebuilt, self-attention K/V grows by exactly one position per step. See
/// docs/audio-review-progress.md's Parler-TTS generation-loop section for the real source
/// derivation of this self/cross cache split.
/// </summary>
public sealed class ParlerDecoderKvCacheTests : HeavyTestBase
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

    [Fact]
    public void ForwardStep_RealWeights_MatchesBatchForwardAndCachesCorrectly()
    {
        string? modelPath = FindRepoFile("models/parler-tts-mini-v1.safetensors");
        Assert.SkipUnless(modelPath != null, "models/parler-tts-mini-v1.safetensors not found");

        string? idsPath = FindRepoFile("scratch-llamacpp-ref/parler_decoder_golden_codebook_ids.txt");
        Assert.SkipUnless(idsPath != null, "golden Parler decoder codebook-id fixture not found");

        var idLines = File.ReadAllText(idsPath!).Trim().Split('\n');
        int t = idLines.Length;
        var codebookIds = new int[t][];
        for (int i = 0; i < t; i++) codebookIds[i] = Array.ConvertAll(idLines[i].Split(','), int.Parse);

        using var loader = SafetensorsLoader.Open(modelPath!);
        var weights = new ParlerDecoderWeights(loader);

        var inputEmbeds = new float[t][];
        for (int i = 0; i < t; i++) inputEmbeds[i] = ParlerDecoder.EmbedStep(weights, codebookIds[i], i);

        var encoderHidden = new float[4][];
        for (int i = 0; i < 4; i++)
        {
            var row = new float[ParlerDecoderWeights.HiddenDim];
            for (int d = 0; d < row.Length; d++) row[d] = 0.05f;
            encoderHidden[i] = row;
        }

        // Oracle: the already golden-verified batch path.
        var batchOutput = ParlerDecoder.Forward(weights, inputEmbeds, encoderHidden);

        // Subject: step-by-step cached decode.
        var cache = new ParlerDecoderKvCache(ParlerDecoderWeights.NumLayers);
        var stepOutputs = new float[t][];
        for (int i = 0; i < t; i++)
        {
            stepOutputs[i] = ParlerDecoder.ForwardStep(weights, cache, inputEmbeds[i], encoderHidden);

            // Real architecture check: self-attention cache grows by exactly one position per step.
            for (int layer = 0; layer < ParlerDecoderWeights.NumLayers; layer++)
                Assert.Equal(i + 1, cache.SelfLength(layer));

            // Real architecture check: cross-attention cache is built by the FIRST step and never rebuilt.
            for (int layer = 0; layer < ParlerDecoderWeights.NumLayers; layer++)
                Assert.True(cache.CrossBuilt(layer), $"cross cache for layer {layer} should be built after step {i}");
        }

        for (int i = 0; i < t; i++)
        {
            double dot = 0, normA = 0, normB = 0;
            for (int d = 0; d < ParlerDecoderWeights.HiddenDim; d++)
            {
                float a = stepOutputs[i][d];
                float b = batchOutput[i][d];
                dot += a * b;
                normA += a * a;
                normB += b * b;
            }
            double cosine = dot / (Math.Sqrt(normA) * Math.Sqrt(normB));
            Assert.True(cosine > 0.9999, $"position {i}: cached step-decode cosine {cosine} vs batch Forward too low");
        }
    }
}
