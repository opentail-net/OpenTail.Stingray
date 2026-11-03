using System.Diagnostics;
using OpenTail.Stingray.Diffusion.LTXVideo;

namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class LtxVideoBenchmarkTests : HeavyTestBase
{
    private const string ModelFileName = "ltx-video-2b-v0.9.1.safetensors";

    private static string? FindModelPath(string fileName) => OpenTail.Stingray.Tests.Audio.AudioTestPaths.FindModelFile(fileName);

    [Fact]
    public void Benchmark_LtxVideoModel_Forward_Profile()
    {
        string? modelPath = FindModelPath(ModelFileName);
        if (modelPath is null) Assert.Skip("modelPath not found (checkpoint or fixture missing).");

        using var loader = SafetensorsLoader.Open(modelPath);
        var model = new LtxVideoModel(loader);

        // Realistic size: numFrames=2 (latent frames), patchH=8, patchW=8 -> 128 tokens
        // and patchH=16, patchW=16 -> 512 tokens (512x512 video 9 frames)
        int numFrames = 2, patchH = 8, patchW = 8;
        int numTokens = numFrames * patchH * patchW;
        int textTokens = 64;

        var latents = new float[numTokens * model.InChannels];
        var caption = new float[textTokens * model.CaptionChannels];
        var rng = new Random(42);
        for (int i = 0; i < latents.Length; i++) latents[i] = rng.NextSingle() * 2f - 1f;
        for (int i = 0; i < caption.Length; i++) caption[i] = rng.NextSingle() * 2f - 1f;

        // Warmup
        model.Forward(latents, 500f, caption, numFrames, patchH, patchW);

        // Benchmark runs
        const int runs = 3;
        var sw = Stopwatch.StartNew();
        for (int r = 0; r < runs; r++)
        {
            model.Forward(latents, 500f, caption, numFrames, patchH, patchW);
        }
        sw.Stop();
        double msPerForward = sw.Elapsed.TotalMilliseconds / runs;
        Console.WriteLine($"[BENCHMARK] LtxVideoModel.Forward ({numTokens} tokens, 28 blocks): {msPerForward:F1} ms/pass");
    }

    [Fact]
    public void Benchmark_LtxVaeDecoder_Decode_Profile()
    {
        string? modelPath = FindModelPath(ModelFileName);
        if (modelPath is null) Assert.Skip("modelPath not found (checkpoint or fixture missing).");

        using var loader = SafetensorsLoader.Open(modelPath);
        var decoder = new LtxVaeDecoder(loader);

        // Latent size: F=1, H=4, W=4 -> decodes to [3, 1, 128, 128]
        int f = 1, h = 4, w = 4;
        var latents = new float[LtxVaeDecoder.LatentChannels * f * h * w];
        var rng = new Random(42);
        for (int i = 0; i < latents.Length; i++) latents[i] = rng.NextSingle() * 2f - 1f;

        // Warmup
        decoder.Decode(latents, 0f, f, h, w, injectNoise: false);

        // Benchmark runs
        const int runs = 3;
        var sw = Stopwatch.StartNew();
        for (int r = 0; r < runs; r++)
        {
            decoder.Decode(latents, 0f, f, h, w, injectNoise: false);
        }
        sw.Stop();
        double msPerDecode = sw.Elapsed.TotalMilliseconds / runs;
        Console.WriteLine($"[BENCHMARK] LtxVaeDecoder.Decode (F={f}, H={h}, W={w}): {msPerDecode:F1} ms/decode");
    }
}
