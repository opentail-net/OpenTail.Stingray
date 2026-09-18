using OpenTail.Stingray.Diffusion.Flux2;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// First real-weight smoke test for FLUX.2's DiT: real IWeightLoader wiring landed 2026-09-18
/// (docs/087) -- this confirms every real tensor name/shape in the checkpoint resolves and the
/// full double+single block stack runs to completion without crashing, producing finite output.
/// NOT a coherence check (no real Mistral text encoder or VAE wired yet -- text conditioning is
/// synthetic here) -- matches the same "structurally sound, not yet coherent" milestone
/// HunyuanVideo/Qwen Image reached before their own text-conditioning wiring landed.
/// </summary>
public sealed class Flux2RealWeightsTests
{
    private const string ModelFileName = "flux2-dev-Q4_K_S.gguf";

    private static string? FindModelPath(string fileName)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "models", "_models", fileName);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void Flux2DiT_RealWeights_ForwardPassProducesFiniteOutput()
    {
        string? modelPath = FindModelPath(ModelFileName);
        Assert.SkipUnless(modelPath != null, "models/_models/flux2-dev-Q4_K_S.gguf not found");

        using var weights = GgufWeightLoader.Open(modelPath!);

        var p = new Flux2Params(); // real defaults, confirmed against the checkpoint (docs/087)
        var dit = new Flux2DiT(weights, p);

        // Small target: 2x2 patch grid (4 image tokens) keeps this a real but fast smoke test.
        int patchH = 2, patchW = 2;
        int nTarget = patchH * patchW;
        var targetLatent = new float[nTarget * p.InChannels];
        var rng = new Random(42);
        for (int i = 0; i < targetLatent.Length; i++) targetLatent[i] = (float)(rng.NextDouble() - 0.5) * 0.1f;

        var targetPositions = new int[nTarget * 4];
        int idx = 0;
        for (int y = 0; y < patchH; y++)
            for (int x = 0; x < patchW; x++)
            {
                targetPositions[idx * 4 + 1] = y;
                targetPositions[idx * 4 + 2] = x;
                idx++;
            }

        int nTxt = 4;
        var textEmbeds = new float[nTxt * p.ContextInDim];
        for (int i = 0; i < textEmbeds.Length; i++) textEmbeds[i] = (float)(rng.NextDouble() - 0.5) * 0.1f;
        var textPositions = new int[nTxt * 4];
        for (int i = 0; i < nTxt; i++) textPositions[i * 4 + 3] = i;

        var velocity = dit.Forward(
            targetLatent, targetPositions,
            refLatents: null, refPositions: null,
            textEmbeds, textPositions,
            pooledEmbed: Array.Empty<float>(),
            timestep: 0.5f,
            guidance: 3.5f);

        Assert.Equal(nTarget * p.OutChannels, velocity.Length);
        foreach (var v in velocity) Assert.True(float.IsFinite(v), "FLUX.2 DiT output contains NaN/Inf");
    }
}
