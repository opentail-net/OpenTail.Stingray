namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// First real-weights coverage for Wan2.2's dual-model Low/High-Noise checkpoints (docs/094 Phase 8).
/// <see cref="OpenTail.Stingray.Diffusion.Wan.WanPipeline.Generate"/> already has real swap logic
/// (`highNoiseTransformer`/`highNoiseBoundary`) — no Wan2.2 checkpoint existed on this machine
/// before this pass's download, so this was entirely unverified.
///
/// <para><b>Deliberately minimal scope this pass</b>: <see cref="OpenTail.Stingray.Diffusion.Wan.WanModel"/>'s
/// constructor takes explicit `dim`/`numHeads`/`numLayers` with NO auto-detection (unlike
/// `QwenImageModel.DetectNumLayers`), defaulting to Wan2.1-1.3B's own real config (dim=1536,
/// numHeads=12, numLayers=30) — which is almost certainly wrong for a 14B-active-parameter A14B
/// model. Constructing a full `WanModel` with guessed hyperparameters risks a silent shape
/// mismatch or subtly wrong compute that LOOKS like it ran, exactly the kind of unverified claim
/// this project's own discipline warns against. This test therefore only verifies the checkpoint
/// itself is real and parseable (matching `WanVideoRealWeightsTests`'s own minimal bar for Wan2.1)
/// -- constructing the real `WanModel` with CONFIRMED real dims (read from the checkpoint's own
/// tensor shapes, e.g. `blocks.0.self_attn.q.weight`'s output dim, plus the real head-dim
/// convention for this architecture) is the next real step, not attempted here.</para>
/// </summary>
public sealed class Wan22DualModelRealWeightsTests
{
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
    public void Wan22_DualModel_RealCheckpoints_ParseAndExposeTensors()
    {
        string? lowNoisePath = FindModelPath("wan2.2_t2v_low_noise_14B_Q4_K_S.gguf");
        string? highNoisePath = FindModelPath("wan2.2_t2v_high_noise_14B_Q4_K_S.gguf");
        if (lowNoisePath is null || highNoisePath is null)
        {
            return; // checkpoints not yet downloaded -- see docs/094 Phase 8's background pull
        }

        using var lowModel = GgufModel.Open(lowNoisePath);
        Assert.True(lowModel.Tensors.Count > 0, "Wan2.2 low-noise GGUF must contain tensors");
        var lowQ = lowModel.FindTensor("blocks.0.self_attn.q.weight");
        Assert.True(lowQ is not null,
            "Wan2.2 low-noise checkpoint must use the same 'blocks.N.self_attn.*' naming WanModel expects");

        using var highModel = GgufModel.Open(highNoisePath);
        Assert.True(highModel.Tensors.Count > 0, "Wan2.2 high-noise GGUF must contain tensors");
        var highQ = highModel.FindTensor("blocks.0.self_attn.q.weight");
        Assert.True(highQ is not null,
            "Wan2.2 high-noise checkpoint must use the same 'blocks.N.self_attn.*' naming WanModel expects");

        // Real, useful signal even at this minimal scope: derive the real hidden dim directly from
        // the checkpoint's own q-projection output shape, and log it so the next pass has a
        // confirmed real number instead of needing to re-derive it.
        Console.WriteLine($"[Wan2.2] blocks.0.self_attn.q.weight shape: [{string.Join(",", lowQ!.Value.Dimensions)}]");
    }
}
