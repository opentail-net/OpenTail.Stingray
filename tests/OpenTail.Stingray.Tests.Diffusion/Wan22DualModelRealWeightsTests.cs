using System.Diagnostics;
using OpenTail.Stingray.Diffusion.Wan;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// First real-weights coverage for Wan2.2's dual-model Low/High-Noise checkpoints (docs/094 Phase 8).
/// <see cref="WanPipeline.Generate"/> already has real swap logic (`highNoiseTransformer`/
/// `highNoiseBoundary`) — no Wan2.2 checkpoint existed on this machine before this pass's download,
/// so this was entirely unverified.
///
/// <para><b>Correction found while writing this test</b>: an earlier, more cautious version of this
/// file assumed <see cref="WanModel"/> had NO auto-detection for `dim`/`numHeads`/`numLayers`
/// (unlike <c>QwenImageModel.DetectNumLayers</c>) and would need real hyperparameters worked out by
/// hand before a full forward pass could be trusted. That was wrong — a closer read of
/// <c>WanModel.DetectConfig</c> shows it ALREADY has real, checkpoint-driven auto-detection matching
/// Wan2.2-A14B exactly: block-count scanning for `numLayers`, `patch_embedding.weight`'s own shape
/// for `dim`, and a hardcoded `dim==5120 -> numHeads=40, ffnDim=13824` branch alongside the existing
/// `dim==1536` (Wan2.1) case. Confirmed against the real checkpoint via `stingray list-tensors`:
/// `blocks.0.self_attn.q.weight` is `[5120,5120]`, `blocks.0.ffn.0.weight` is `[5120,13824]`, and
/// block 39 is the last real block (block 40 doesn't exist) — all match `DetectConfig`'s hardcoded
/// values exactly. No guessing needed; this test constructs a real <see cref="WanModel"/> directly.
/// </para>
/// </summary>
public sealed class Wan22DualModelRealWeightsTests
{
    private readonly ITestOutputHelper _output;

    public Wan22DualModelRealWeightsTests(ITestOutputHelper output)
    {
        _output = output;
    }

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
            _output.WriteLine("[Wan22DualModelRealWeightsTests] Checkpoints missing, skipping."); Console.WriteLine("[Wan22DualModelRealWeightsTests] Checkpoints missing, skipping.");
            return;
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

        _output.WriteLine($"[Wan2.2] blocks.0.self_attn.q.weight shape: [{string.Join(",", lowQ!.Value.Dimensions)}]"); Console.WriteLine($"[Wan2.2] blocks.0.self_attn.q.weight shape: [{string.Join(",", lowQ!.Value.Dimensions)}]");
    }

    [Fact]
    public void Wan22_LowNoiseModel_RealWeights_ConfigAutoDetectedCorrectly_ForwardProducesFiniteVelocity()
    {
        string? lowNoisePath = FindModelPath("wan2.2_t2v_low_noise_14B_Q4_K_S.gguf");
        if (lowNoisePath is null)
        {
            _output.WriteLine("[Wan22DualModelRealWeightsTests] Checkpoint missing, skipping."); Console.WriteLine("[Wan22DualModelRealWeightsTests] Checkpoint missing, skipping.");
            return;
        }

        using var weights = GgufWeightLoader.Open(lowNoisePath);
        using var model = new WanModel(weights, prefix: "");

        // Real config, confirmed against the checkpoint's own tensor shapes via `list-tensors`
        // before writing this assertion (see class doc comment) -- not assumed.
        Assert.Equal(40, model.NumLayers);
        Assert.Equal(5120, model.Dim);
        Assert.Equal(40, model.NumHeads);
        Assert.Equal(128, model.HeadDim);
        Assert.Equal(13824, model.FfnDim);
        _output.WriteLine($"[Wan2.2 low-noise] Auto-detected config: numLayers={model.NumLayers}, dim={model.Dim}, numHeads={model.NumHeads}, headDim={model.HeadDim}, ffnDim={model.FfnDim}"); Console.WriteLine($"[Wan2.2 low-noise] Auto-detected config: numLayers={model.NumLayers}, dim={model.Dim}, numHeads={model.NumHeads}, headDim={model.HeadDim}, ffnDim={model.FfnDim}");

        // Minimal real forward pass -- 64x64 (8x8 latent, patch-embedded down further), 1 frame,
        // a single denoising step. Deliberately tiny: this is a first-ever coverage smoke test for
        // a 14B-active-parameter checkpoint on CPU, not a quality or performance benchmark.
        const int latH = 8, latW = 8, latC = 16;
        var rng = new Random(42);
        var latent = new float[latC * latH * latW];
        for (int i = 0; i < latent.Length; i++) latent[i] = (float)(rng.NextDouble() * 2 - 1);

        int seqLen = 4;
        var textContext = new float[seqLen * WanModel.TextDim];
        for (int i = 0; i < textContext.Length; i++) textContext[i] = (float)(rng.NextDouble() * 0.1);

        var sw = Stopwatch.StartNew();
        var velocity = model.Forward(latent, 1000f, textContext, numFrames: 1, latH, latW);
        sw.Stop();
        _output.WriteLine($"[Wan2.2 low-noise] Forward (64x64, 1 frame) took {sw.ElapsedMilliseconds} ms ({sw.Elapsed.TotalSeconds:F1}s)"); Console.WriteLine($"[Wan2.2 low-noise] Forward (64x64, 1 frame) took {sw.ElapsedMilliseconds} ms ({sw.Elapsed.TotalSeconds:F1}s)");

        Assert.All(velocity, v => Assert.True(float.IsFinite(v), "DiT velocity output must be finite"));
        double sumSq = 0;
        foreach (var v in velocity) sumSq += (double)v * v;
        double rms = Math.Sqrt(sumSq / velocity.Length);
        _output.WriteLine($"[Wan2.2 low-noise] Velocity RMS: {rms:E6}"); Console.WriteLine($"[Wan2.2 low-noise] Velocity RMS: {rms:E6}");
        Assert.True(rms > 1e-6, $"DiT velocity RMS too small ({rms}), likely degenerate/all-zero");
    }
}
