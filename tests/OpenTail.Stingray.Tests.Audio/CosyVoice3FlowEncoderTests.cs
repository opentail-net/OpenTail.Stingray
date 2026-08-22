using System;
using System.IO;
using OpenTail.Stingray.Audio.CosyVoice;
using OpenTail.Stingray.Core;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real numeric golden verification for <see cref="CosyVoice3FlowEncoder.ComputeMuAndSpks"/> --
/// real GGUF weights, real oracle transcribed from `examples/cosyvoice.cpp`'s
/// `PreLookaheadLayer::build_cgraph`/`CausalMaskedDiffWithDiT::build_cgraph_encode`
/// (`scratch-llamacpp-ref/cosyvoice3_flowencoder_golden.py`), same technique used throughout
/// this doc (real GGUF weights via `gguf.GGUFReader`, deterministic real speech-token input,
/// cosine similarity against the C# port). First numeric (not just structural) golden test for
/// any CosyVoice3 stage.
/// </summary>
public sealed class CosyVoice3FlowEncoderTests : HeavyTestBase
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
    public void ComputeMuAndSpks_RealWeights_MatchesGoldenOracle()
    {
        string? modelPath = FindRepoFile("models/cosyvoice3/CosyVoice3-2512_F16.gguf");
        Assert.SkipUnless(modelPath != null, "models/cosyvoice3/CosyVoice3-2512_F16.gguf not found");

        string? tokensPath = FindRepoFile("scratch-llamacpp-ref/cosyvoice3_flowencoder_golden_tokens.txt");
        string? muPath = FindRepoFile("scratch-llamacpp-ref/cosyvoice3_flowencoder_golden_mu.txt");
        string? spksPath = FindRepoFile("scratch-llamacpp-ref/cosyvoice3_flowencoder_golden_spks.txt");
        Assert.SkipUnless(tokensPath != null && muPath != null && spksPath != null, "golden flow-encoder fixture not found");

        var speechTokens = Array.ConvertAll(File.ReadAllText(tokensPath!).Trim().Split(','), int.Parse);

        var muLines = File.ReadAllText(muPath!).Split('\n');
        var muDims = muLines[0].Trim().Split(',');
        int goldenT = int.Parse(muDims[0]);
        int goldenDim = int.Parse(muDims[1]);
        var muParts = muLines[1].Trim().Split(',');
        var goldenMu = Array.ConvertAll(muParts, float.Parse);

        var goldenSpks = Array.ConvertAll(File.ReadAllText(spksPath!).Trim().Split(','), float.Parse);

        using var model = GgufModel.Open(modelPath!);
        var weights = new CosyVoice3FlowEncoderWeights(model);

        var (mu, spks) = CosyVoice3FlowEncoder.ComputeMuAndSpks(weights, speechTokens, new float[CosyVoice3FlowEncoderWeights.SpeakerEmbedDim]);

        Assert.Equal(goldenT * goldenDim, mu.Length);
        Assert.Equal(goldenDim, spks.Length);

        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < mu.Length; i++)
        {
            dot += mu[i] * goldenMu[i];
            normA += mu[i] * mu[i];
            normB += goldenMu[i] * goldenMu[i];
        }
        double muCosine = dot / (Math.Sqrt(normA) * Math.Sqrt(normB));
        Assert.True(muCosine > 0.999, $"mu cosine similarity {muCosine} too low vs golden oracle");

        double dotS = 0, normAS = 0, normBS = 0;
        for (int i = 0; i < spks.Length; i++)
        {
            dotS += spks[i] * goldenSpks[i];
            normAS += spks[i] * spks[i];
            normBS += goldenSpks[i] * goldenSpks[i];
        }
        double spksCosine = dotS / (Math.Sqrt(normAS) * Math.Sqrt(normBS));
        Assert.True(spksCosine > 0.999, $"spks cosine similarity {spksCosine} too low vs golden oracle");
    }
}
