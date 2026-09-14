using OpenTail.Stingray.Diffusion.TextEncoders;
using OpenTail.Stingray.Diffusion.Wan;
using OpenTail.Stingray.Vulkan;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// A genuinely new diagnostic for the Priority-0 Wan accuracy investigation (docs/081): FLUX's
/// T5-XXL encoder (`Diffusion.TextEncoders.T5Encoder`) is now proven correct end-to-end (real,
/// visually-confirmed coherent FLUX output, 2026-09-14, docs/071's re-verification) and outputs
/// the SAME 4096-dim embedding space Wan's UMT5 does. This test feeds FLUX's proven-correct T5
/// embeddings into Wan's DiT cross-attention INSTEAD of UMT5's own output -- semantically wrong
/// (different vocab/register) but dimensionally valid. If the decoded image shows ANY real
/// spatial structure (even wrong content), that proves Wan's DiT itself is capable of using a
/// text-conditioning signal to produce coherent output, isolating the remaining bug to UMT5
/// specifically. If it's still pure noise, that points the bug back at the DiT/Euler loop
/// (or somewhere shared by any conditioning source), not UMT5.
/// </summary>
public sealed class WanCrossSubstituteT5DiagnosticTest
{
    private static string? FindModelPath(string relativePath)
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

    [Fact]
    public void Generate_UsingFluxT5EmbeddingsInsteadOfUmt5_CheckForAnyRealStructure()
    {
        string? t5Path = FindModelPath(Path.Combine("models", "flux1-schnell", "t5xxl_fp8_e4m3fn.safetensors"));
        string? t5TokPath = FindModelPath(Path.Combine("models", "flux1-schnell", "tokenizer_t5", "tokenizer.json"));
        string? ditPath = FindModelPath(Path.Combine("models", "wan2.1", "wan2.1-t2v-1.3b-dit.safetensors"));
        string? vaePath = FindModelPath(Path.Combine("models", "wan2.1", "Wan2.1_VAE.safetensors"));

        if (t5Path is null || t5TokPath is null || ditPath is null || vaePath is null ||
            !File.Exists(t5Path) || !File.Exists(t5TokPath) || !File.Exists(ditPath) || !File.Exists(vaePath))
        {
            Console.WriteLine("Missing model file(s), skipping.");
            return;
        }

        using var vulkan = new VulkanBackend();

        // Encode via FLUX's own, now-proven-correct T5-XXL, padded to 256 (FLUX's own real
        // fixed-length convention, docs/056) -- length doesn't need to match Wan's real 226,
        // this is a structural probe, not a production path.
        var tokenizer = T5Tokenizer.FromFile(t5TokPath, maxLen: 256);
        string prompt = "a red apple on a wooden table, photorealistic";
        var rawTokens = tokenizer.Tokenize(prompt);
        var tokens = new int[256];
        Array.Copy(rawTokens, tokens, Math.Min(rawTokens.Length, 256));

        using var t5 = new T5Encoder(t5Path);
        var fluxT5Context = t5.EncodeGpu(tokens, vulkan);
        Console.WriteLine($"[CrossSubstitute] FLUX T5 context: length={fluxT5Context.Length} ({256}x4096)");

        float mean = 0, sumSq = 0;
        for (int i = 0; i < fluxT5Context.Length; i++) { mean += fluxT5Context[i]; sumSq += fluxT5Context[i] * fluxT5Context[i]; }
        mean /= fluxT5Context.Length;
        float std = MathF.Sqrt(sumSq / fluxT5Context.Length - mean * mean);
        Console.WriteLine($"[CrossSubstitute] FLUX T5 context stats: mean={mean:F4} std={std:F4}");

        using var pipeline = WanPipeline.Load(ditPath, vaePath, vulkan);

        var frames = pipeline.Generate(
            prompt: prompt,
            width: 256,
            height: 256,
            numFrames: 1,
            steps: 4,
            guidance: 1.0f, // isolate: no CFG cancellation confound, already ruled out separately
            flowShift: 3.0f,
            seed: 42,
            outputPath: "wan_cross_substitute_flux_t5.png",
            progress: (step, total) => Console.WriteLine($"[CrossSubstitute Denoise Step {step}/{total}]"),
            textContext: fluxT5Context);

        Console.WriteLine($"[CrossSubstitute] Generated frame count: {frames.Count}");
        float rMean = 0, gMean = 0, bMean = 0;
        int pixelCount = 256 * 256;
        for (int i = 0; i < pixelCount; i++)
        {
            rMean += frames[0][0 * pixelCount + i];
            gMean += frames[0][1 * pixelCount + i];
            bMean += frames[0][2 * pixelCount + i];
        }
        rMean /= pixelCount; gMean /= pixelCount; bMean /= pixelCount;
        Console.WriteLine($"[CrossSubstitute] VAE Output RGB Means: R={rMean:F4}, G={gMean:F4}, B={bMean:F4}");
    }
}
