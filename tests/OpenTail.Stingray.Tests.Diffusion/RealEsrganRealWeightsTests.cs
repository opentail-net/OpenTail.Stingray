
namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class RealEsrganRealWeightsTests
{
    private const string ModelFileName = "RealESRGAN_x4plus.safetensors";

    private static string? FindModelPath(string modelFile)
    {
        string[] absoluteCandidates =
        {
            $@"C:\Git-Public\OpenTail.Stingray\models\{modelFile}",
            $@"C:\p\opentail-llm\models\{modelFile}",
            $@"E:\models\{modelFile}",
        };
        foreach (var p in absoluteCandidates)
        {
            if (File.Exists(p) && CanOpenFile(p)) return p;
        }

        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "models", modelFile);
            if (File.Exists(p) && CanOpenFile(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static bool CanOpenFile(string path)
    {
        try
        {
            using var fs = File.Open(path, FileMode.Open, FileAccess.Read, FileShare.None);
            return fs.Length > 1024 * 1024;
        }
        catch
        {
            return false;
        }
    }

    [Fact]
    public void RealEsrgan_RealWeights_LoadAndUpscale_4xScalingValid()
    {
        string? modelPath = FindModelPath(ModelFileName);
        if (modelPath is null)
        {
            return;
        }

        using var upscaler = RRDBNet.Load(modelPath);

        Assert.Equal(4, upscaler.Scale);

        // Test small 8x8 RGB patch upscale (8x8 -> 32x32)
        int inW = 8;
        int inH = 8;
        float[] inputRgb = new float[3 * inH * inW];
        for (int i = 0; i < inputRgb.Length; i++)
        {
            inputRgb[i] = 0.5f + 0.5f * MathF.Sin(i * 0.1f);
        }

        var (upscaledPixels, outW, outH) = upscaler.Upscale(inputRgb, inW, inH, tileSize: 64, overlap: 4);

        Assert.Equal(32, outW);
        Assert.Equal(32, outH);
        Assert.Equal(3 * 32 * 32, upscaledPixels.Length);

        for (int i = 0; i < upscaledPixels.Length; i++)
        {
            Assert.False(float.IsNaN(upscaledPixels[i]), $"NaN at pixel {i}");
            Assert.False(float.IsInfinity(upscaledPixels[i]), $"Infinity at pixel {i}");
        }
    }
}
