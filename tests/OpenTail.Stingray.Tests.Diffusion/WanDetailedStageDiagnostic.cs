using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.TextEncoders;
using OpenTail.Stingray.Diffusion.Wan;
using OpenTail.Stingray.Vulkan;
using System.Numerics.Tensors;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class WanDetailedStageDiagnostic
{
    private readonly ITestOutputHelper _output;

    public WanDetailedStageDiagnostic(ITestOutputHelper output)
    {
        _output = output;
    }

    private static string FindModelPath(string relativePath)
    {
        var candidates = new[]
        {
            Path.Combine("..", "..", "..", "..", "..", relativePath),
            Path.Combine("..", "..", "..", relativePath),
            relativePath,
            Path.Combine(AppContext.BaseDirectory, relativePath),
            Path.Combine(@"c:\Git-Public\OpenTail.Stingray", relativePath)
        };
        foreach (var c in candidates)
        {
            if (File.Exists(c)) return Path.GetFullPath(c);
        }
        return relativePath;
    }

    private static readonly string LogFile = Path.Combine(@"c:\Git-Public\OpenTail.Stingray", "stage_diagnostics_report.txt");

    private void Log(string msg)
    {
        _output.WriteLine(msg);
        Console.WriteLine(msg);
        File.AppendAllText(LogFile, msg + Environment.NewLine);
    }

    private static (float mean, float std, float min, float max, float l2Norm) Stats(ReadOnlySpan<float> span)
    {
        if (span.IsEmpty) return (0, 0, 0, 0, 0);
        float mean = 0, sumSq = 0, min = span[0], max = span[0];
        for (int i = 0; i < span.Length; i++)
        {
            float v = span[i];
            mean += v;
            sumSq += v * v;
            if (v < min) min = v;
            if (v > max) max = v;
        }
        mean /= span.Length;
        float std = MathF.Sqrt(MathF.Max(0, sumSq / span.Length - mean * mean));
        float l2Norm = MathF.Sqrt(sumSq);
        return (mean, std, min, max, l2Norm);
    }

    [Fact]
    public void DetailedStageInspection()
    {
        string tokPath = FindModelPath(Path.Combine("models", "wan2.1", "umt5-tokenizer.json"));
        string encPath = FindModelPath(Path.Combine("models", "wan2.1", "models_t5_umt5-xxl-enc-bf16.safetensors"));
        string ditPath = FindModelPath(Path.Combine("models", "wan2.1", "wan2.1-t2v-1.3b-dit.safetensors"));

        if (!File.Exists(tokPath) || !File.Exists(encPath) || !File.Exists(ditPath))
        {
            Log("Models missing, skipping detailed inspection.");
            return;
        }

        using var vulkan = new VulkanBackend();

        // 1. Prompt & Tokenization
        var tokenizer = T5Tokenizer.FromFile(tokPath, maxLen: 512);
        string prompt = "a red apple on a wooden table, photorealistic";
        var tokens = tokenizer.Tokenize(prompt);
        var uncondTokens = tokenizer.Tokenize("");
        Log($"[Stage 1] Cond Tokens ({tokens.Length}): [{string.Join(", ", tokens)}]");
        Log($"[Stage 1] Uncond Tokens ({uncondTokens.Length}): [{string.Join(", ", uncondTokens)}]");

        // 2. UMT5 Encoding
        using var umt5 = new UMT5Encoder(encPath);
        var condContext = umt5.EncodeGpu(tokens, vulkan);
        var uncondContext = umt5.EncodeGpu(uncondTokens, vulkan);

        var sCond = Stats(condContext);
        var sUncond = Stats(uncondContext);
        Log($"[Stage 2] Cond UMT5 Stats: mean={sCond.mean:F4}, std={sCond.std:F4}, norm={sCond.l2Norm:F2}, range=[{sCond.min:F3}, {sCond.max:F3}]");
        Log($"[Stage 2] Uncond UMT5 Stats: mean={sUncond.mean:F4}, std={sUncond.std:F4}, norm={sUncond.l2Norm:F2}, range=[{sUncond.min:F3}, {sUncond.max:F3}]");

        // 3. Load DiT weights
        using var weights = SafetensorsLoader.Open(ditPath);
        using var model = new WanModel(weights, "", backend: vulkan);
        int dim = model.Dim;
        int heads = model.NumHeads;
        int headDim = model.HeadDim;
        int ffnDim = model.FfnDim;
        int layers = model.NumLayers;
        Log($"[Stage 3] DiT Config: Dim={dim}, Heads={heads}, HeadDim={headDim}, FfnDim={ffnDim}, Layers={layers}");

        // 4. Inspect Text Projection on Cond & Uncond
        // Create workspace
        int numFrames = 1, latH = 32, latW = 32;
        int patchH = latH / 2, patchW = latW / 2;
        int numTokens = numFrames * patchH * patchW; // 256
        int condTxt = condContext.Length / WanModel.TextDim;
        int uncondTxt = uncondContext.Length / WanModel.TextDim;

        var (ropeCos, ropeSin) = WanRoPE.Compute3DRoPECompact(numFrames, patchH, patchW, headDim);
        using var gpuWs = new WanGpuWorkspace(vulkan, numTokens, dim, ffnDim, layers, condTxt, ropeCos, ropeSin, headDim);
        using var gpuWeights = model.GetOrCreateGpuWeights();

        model.PrecomputeCrossKvCacheGpu(condContext, gpuWs, gpuWeights, vulkan);

        // Inspect Block 0 Cross KV - Cond
        var k0Host = new float[condTxt * dim];
        var v0Host = new float[condTxt * dim];
        vulkan.Download(gpuWs.CrossKvCache[0].K, k0Host);
        vulkan.Download(gpuWs.CrossKvCache[0].V, v0Host);
        var sK0 = Stats(k0Host);
        var sV0 = Stats(v0Host);
        Log($"[Stage 4: CrossKV Cond Block 0] K stats: mean={sK0.mean:F4}, std={sK0.std:F4}, norm={sK0.l2Norm:F2}");
        Log($"[Stage 4: CrossKV Cond Block 0] V stats: mean={sV0.mean:F4}, std={sV0.std:F4}, norm={sV0.l2Norm:F2}");

        // Also inspect mid-layer (block 15) KV
        var k15Host = new float[condTxt * dim];
        vulkan.Download(gpuWs.CrossKvCache[15].K, k15Host);
        var sK15 = Stats(k15Host);
        Log($"[Stage 4: CrossKV Cond Block 15] K stats: mean={sK15.mean:F4}, std={sK15.std:F4}, norm={sK15.l2Norm:F2}");

        // 5. Inspect Timestep Embedding at t=1000 and t=500
        var t1000 = DiffusionOps.SinusoidalTimestepEmbedding(1000.0f);
        var sT1000 = Stats(t1000);
        Log($"[Stage 5: Timestep 1000] Sinusoidal stats: mean={sT1000.mean:F4}, std={sT1000.std:F4}, norm={sT1000.l2Norm:F2}");

        // 6. Test Single Step Velocity
        var latent = new float[16 * numFrames * latH * latW];
        var rng = new Random(42);
        for (int i = 0; i < latent.Length; i++)
        {
            double u1 = 1.0 - rng.NextDouble(), u2 = 1.0 - rng.NextDouble();
            latent[i] = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }
        var sLat0 = Stats(latent);
        Log($"[Stage 6: Latent Init] mean={sLat0.mean:F4}, std={sLat0.std:F4}, norm={sLat0.l2Norm:F2}");

        var condVel = model.ForwardGpu(latent, 1000.0f, condContext, numFrames, latH, latW, gpuWs, gpuWeights, vulkan);
        var sCondVel = Stats(condVel);
        Log($"[Stage 6: Cond Velocity t=1000] mean={sCondVel.mean:F4}, std={sCondVel.std:F4}, norm={sCondVel.l2Norm:F2}, range=[{sCondVel.min:F3}, {sCondVel.max:F3}]");

        using var uncondGpuWs = new WanGpuWorkspace(vulkan, numTokens, dim, ffnDim, layers, uncondTxt, ropeCos, ropeSin, headDim);
        model.PrecomputeCrossKvCacheGpu(uncondContext, uncondGpuWs, gpuWeights, vulkan);

        // Compare uncond KV vs cond KV to verify cross-attention carries distinct signals
        var uncondK0Host = new float[uncondTxt * dim];
        vulkan.Download(uncondGpuWs.CrossKvCache[0].K, uncondK0Host);
        var sUncondK0 = Stats(uncondK0Host);
        Log($"[Stage 4: CrossKV Uncond Block 0] K stats: mean={sUncondK0.mean:F4}, std={sUncondK0.std:F4}, norm={sUncondK0.l2Norm:F2}");
        Log($"[Stage 4: KV Signal] Cond K0 norm={sK0.l2Norm:F2} vs Uncond K0 norm={sUncondK0.l2Norm:F2} (ratio={sUncondK0.l2Norm / sK0.l2Norm:F3})");

        var uncondVel = model.ForwardGpu(latent, 1000.0f, uncondContext, numFrames, latH, latW, uncondGpuWs, gpuWeights, vulkan);
        var sUncondVel = Stats(uncondVel);
        Log($"[Stage 6: Uncond Velocity t=1000] mean={sUncondVel.mean:F4}, std={sUncondVel.std:F4}, norm={sUncondVel.l2Norm:F2}");

        // Compute CFG difference
        var cfgDiff = new float[condVel.Length];
        for (int i = 0; i < cfgDiff.Length; i++) cfgDiff[i] = condVel[i] - uncondVel[i];
        var sDiff = Stats(cfgDiff);
        Log($"[Stage 6: CFG Signal (Cond - Uncond)] norm={sDiff.l2Norm:F4}, std={sDiff.std:F6}, maxAbs={MathF.Max(MathF.Abs(sDiff.min), MathF.Abs(sDiff.max)):F6}");

        // Check if CFG difference is essentially ZERO!
        if (sDiff.l2Norm < 1e-3f)
        {
            Log("!!! CRITICAL: CFG difference (Cond - Uncond) is nearly ZERO! The DiT is completely ignoring text conditioning!");
        }
        else
        {
            Log($"[Stage 6: CFG Signal Ratio] Diff norm / Cond norm = {sDiff.l2Norm / sCondVel.l2Norm:P2}");
        }
    }
}
