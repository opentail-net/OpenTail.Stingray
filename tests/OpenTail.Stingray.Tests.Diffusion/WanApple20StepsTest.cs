using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.TextEncoders;
using OpenTail.Stingray.Diffusion.Wan;
using OpenTail.Stingray.Vulkan;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class WanApple20StepsTest
{
    private readonly ITestOutputHelper _output;

    public WanApple20StepsTest(ITestOutputHelper output)
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

    [Fact]
    public void GenerateApple_20Steps_Wan21()
    {
        string tokPath = FindModelPath(Path.Combine("models", "wan2.1", "umt5-tokenizer.json"));
        string encPath = FindModelPath(Path.Combine("models", "wan2.1", "models_t5_umt5-xxl-enc-bf16.safetensors"));
        string ditPath = FindModelPath(Path.Combine("models", "wan2.1", "wan2.1-t2v-1.3b-dit.safetensors"));
        string vaePath = FindModelPath(Path.Combine("models", "wan2.1", "Wan2.1_VAE.safetensors"));

        if (!File.Exists(tokPath) || !File.Exists(encPath) || !File.Exists(ditPath) || !File.Exists(vaePath))
        {
            return;
        }

        using var vulkan = new VulkanBackend();
        var swTotal = System.Diagnostics.Stopwatch.StartNew();

        // 1. Text conditioning
        var swText = System.Diagnostics.Stopwatch.StartNew();
        var tokenizer = T5Tokenizer.FromFile(tokPath, maxLen: 512);
        string prompt = "a red apple on a wooden table, photorealistic, high quality";
        var tokens = tokenizer.Tokenize(prompt);
        var uncondTokens = tokenizer.Tokenize("");

        using var umt5 = new UMT5Encoder(encPath);
        var rawCond = umt5.EncodeGpu(tokens, vulkan);
        var rawUncond = umt5.EncodeGpu(uncondTokens, vulkan);
        swText.Stop();
        Console.WriteLine($"[TestProfile] Text encoding (cond + uncond) took {swText.ElapsedMilliseconds} ms");

        // Real diffusers WanPipeline._get_t5_prompt_embeds: zero-pad the ENCODED embeddings (not
        // the token ids) up to a fixed max_sequence_length=226 for cross-attention -- see
        // docs/081, 2026-09-14 update #9, and WanAppleDiagnosticTests's own real diagnostic of this.
        const int fixedLen = 226;
        const int txtDim = 4096;
        var condContext = new float[fixedLen * txtDim];
        Array.Copy(rawCond, condContext, Math.Min(rawCond.Length, fixedLen * txtDim));
        var uncondContext = new float[fixedLen * txtDim];
        Array.Copy(rawUncond, uncondContext, Math.Min(rawUncond.Length, fixedLen * txtDim));

        // 2. Load Pipeline
        var swLoad = System.Diagnostics.Stopwatch.StartNew();
        using var pipeline = WanPipeline.Load(ditPath, vaePath, vulkan);
        swLoad.Stop();
        Console.WriteLine($"[TestProfile] Pipeline load took {swLoad.ElapsedMilliseconds} ms");

        string outputPath = Path.Combine(@"c:\Git-Public\OpenTail.Stingray", "wan_apple_20steps.png");

        // 3. Generate 20 steps
        var swGen = System.Diagnostics.Stopwatch.StartNew();
        pipeline.Generate(
            prompt: prompt,
            negativePrompt: "",
            width: 256,
            height: 256,
            numFrames: 1,
            steps: 20,
            guidance: 6.0f,
            flowShift: 3.0f,
            seed: 42,
            outputPath: outputPath,
            textContext: condContext,
            negativeTextContext: uncondContext);
        swGen.Stop();
        swTotal.Stop();
        Console.WriteLine($"[TestProfile] pipeline.Generate total took {swGen.ElapsedMilliseconds} ms");
        Console.WriteLine($"[TestProfile] Full test total took {swTotal.ElapsedMilliseconds} ms");

        Assert.True(File.Exists(outputPath), "Output image was not generated.");
    }
}
