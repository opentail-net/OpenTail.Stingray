using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion;
using OpenTail.Stingray.Diffusion.LTXVideo;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Isolation test (docs/077, 2026-09-14): decodes a hand-built standard-normal latent, at the SAME
/// real 512x512 scale used by the post-timestep-fix end-to-end run that produced a structured (not
/// random) but still visually-wrong image (`docs/diffusion-samples/ltx_video_apple_flipfix.png`),
/// directly through <see cref="LtxVaeDecoder"/> -- bypassing the DiT and scheduler entirely, mirroring
/// the same-purpose <c>WanVaeOnlyIsolationTests</c> approach used earlier this session for Wan. If
/// this alone reproduces the SAME banding/color-block character, that implicates the VAE decoder
/// itself (at this larger scale than its own 2x2-latent golden test covers) rather than the DiT
/// output or scheduler math -- both of which are now numerically golden-verified.
/// </summary>
public sealed class LtxVaeNoiseIsolationTest
{
    private const string ModelFileName = "ltx-video-2b-v0.9.1.safetensors";

    private static string? FindModelPath(string fileName)
    {
        string[] absoluteCandidates =
        {
            $@"C:\Git-Public\OpenTail.Stingray\models\{fileName}",
            $@"C:\p\opentail-llm\models\{fileName}",
            $@"E:\models\{fileName}",
        };
        foreach (var p in absoluteCandidates)
            if (File.Exists(p)) return p;

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
    public void DecodeStandardNormalLatent_At512_SavesForVisualInspection()
    {
        string? modelPath = FindModelPath(ModelFileName);
        if (modelPath is null) return;

        using var loader = SafetensorsLoader.Open(modelPath);
        var vae = new LtxVaeDecoder(loader);

        const int inChannels = 128;
        const int patchH = 16; // 512 / 32
        const int patchW = 16;
        const int numLatentFrames = 1;
        int spatialSize = patchH * patchW;

        var stdOfMeans = loader.ReadF32("vae.per_channel_statistics.std-of-means");
        var meanOfMeans = loader.ReadF32("vae.per_channel_statistics.mean-of-means");

        var rng = new Random(42);
        // Token-major [numTokens, C] standard-normal latent, matching LtxVideoPipeline's own
        // Box-Muller initial-noise construction exactly (same distribution DiT output should
        // resemble after denoising, for a fair isolation of the un-normalize+decode stage alone).
        var latentsTokenMajor = new float[numLatentFrames * spatialSize * inChannels];
        for (int i = 0; i < latentsTokenMajor.Length; i++)
        {
            float u1 = Math.Max(1e-7f, rng.NextSingle());
            float u2 = rng.NextSingle();
            latentsTokenMajor[i] = MathF.Sqrt(-2.0f * MathF.Log(u1)) * MathF.Cos(2.0f * MathF.PI * u2);
        }

        // Real un-normalize + token-major -> channel-first conversion, copied verbatim from
        // LtxVideoPipeline.GenerateVideo's own real logic.
        var chFirst = new float[inChannels * numLatentFrames * spatialSize];
        for (int f = 0; f < numLatentFrames; f++)
        {
            int tokenBase = f * spatialSize;
            for (int p = 0; p < spatialSize; p++)
            {
                int srcOff = (tokenBase + p) * inChannels;
                for (int c = 0; c < inChannels; c++)
                {
                    float v = latentsTokenMajor[srcOff + c];
                    v = v * stdOfMeans[c] + meanOfMeans[c];
                    chFirst[(c * numLatentFrames + f) * spatialSize + p] = v;
                }
            }
        }

        var video = vae.Decode(chFirst, decodeTimestep: 0f, numLatentFrames, patchH, patchW);

        int outF = 8 * (numLatentFrames - 1) + 1;
        int outH = patchH * LtxVaeDecoder.SpatialScale;
        int outW = patchW * LtxVaeDecoder.SpatialScale;
        int outSpatial = outH * outW;

        Assert.Equal(1, outF);
        Assert.Equal(512, outH);
        Assert.Equal(512, outW);

        var frameRgb = new float[3 * outSpatial];
        for (int c = 0; c < 3; c++)
        {
            int srcBase = c * outSpatial; // f=0
            int dstBase = c * outSpatial;
            for (int p = 0; p < outSpatial; p++)
                frameRgb[dstBase + p] = Math.Clamp((video[srcBase + p] + 1.0f) * 0.5f, 0.0f, 1.0f);
        }

        string outPath = Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..",
            "docs", "diffusion-samples", "ltx_vae_noise_isolation.png");
        outPath = Path.GetFullPath(outPath);
        PngWriter.Write(outPath, frameRgb, outW, outH);
        Console.WriteLine($"[LTX VAE noise isolation] wrote {outPath}");
    }
}
