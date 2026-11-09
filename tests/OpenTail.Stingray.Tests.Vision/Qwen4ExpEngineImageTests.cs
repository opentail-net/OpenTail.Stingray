using System.Text;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Engine.Verification;
using OpenTail.Stingray.Vision;

namespace OpenTail.Stingray.Tests.Vision;

/// <summary>
/// Qwen 3.8 plan, Phase 5 (server image path): the engine's own image splice (<see cref="InferenceEngine.EnableQwenImageInput"/>
/// and the prompt marker rewrite) on the real qwen4exp checkpoint. The image is set by QWEN_TEST_IMAGE (a PNG); skipped
/// unless the image, the trunk and the mmproj are present. Checks the answer names the two colours. The direct pass path is
/// covered by <see cref="Qwen4ExpImageAnswerTests"/>.
/// </summary>
public sealed class Qwen4ExpEngineImageTests
{
    private static string? FindTrunk() => ModelLocator.FindDeep("Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf");

    private static string? FindMmproj() => ModelLocator.FindDeep("mmproj-Qwen3.8-Flash-Next-F16.gguf");

    [Fact]
    public async Task EngineImageAnswer_NamesTheColours_ThroughTheEngineSplice()
    {
        string? image = Environment.GetEnvironmentVariable("QWEN_TEST_IMAGE");
        Assert.SkipUnless(image is not null && File.Exists(image), "Set QWEN_TEST_IMAGE to a PNG path.");
        string? trunkPath = FindTrunk();
        Assert.SkipUnless(trunkPath is not null, "Qwen3.8-Flash-Next UD-IQ1_S checkpoint is required (paged, 72.5 GB).");
        string? mmprojPath = FindMmproj();
        Assert.SkipUnless(mmprojPath is not null, "mmproj-Qwen3.8-Flash-Next-F16.gguf is required (904 MB).");

        using var model = GgufModel.Open(trunkPath!);
        int totalLayers = (int)Convert.ToInt64(model.Metadata["qwen4exp.block_count"]);
        var hp = Qwen4ExpHyperparams.FromMetadata(model.Metadata, totalLayers);
        var tensors = Qwen4ExpTensorSet.Load(model, hp, totalLayers);
        var pass = new Qwen4ExpForwardPass(model, hp, tensors);
        var tok = GgufTokenizer.FromGgufModel(model);

        int think = Assert.Single(tok.Encode("<think>"));
        int endThink = Assert.Single(tok.Encode("</think>"));
        int open = Assert.Single(tok.Encode("<|vision_start|>"));
        int close = Assert.Single(tok.Encode("<|vision_end|>"));
        int pad = Assert.Single(tok.Encode("<|image_pad|>"));

        // The engine owns the forward pass and the projector; disposing the engine releases both.
        using var engine = new InferenceEngine(pass, tok, "qwen4exp", think, endThink, pass);
        engine.EnableQwenImageInput(QwenVlVisionModel.Open(mmprojPath!), open, close, pad);
        Assert.True(engine.SupportsImageInput);

        // The server writes the generic marker; the engine turns it into the Qwen marker sequence.
        const string prompt = "<|im_start|>user\n<|image|>Which colours are in this picture? Answer in one short sentence.<|im_end|>\n<|im_start|>assistant\n";
        byte[] png = File.ReadAllBytes(image!);
        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 96 };

        var text = new StringBuilder();
        await foreach (var chunk in engine.GenerateImageChunksAsync(prompt, new[] { png }, sp))
            if (chunk.Kind == GenerateChunkKind.Text) text.Append(chunk.Text);
        string answer = text.ToString();

        Console.WriteLine($"ENGINE_IMAGE_ANSWER {answer}");
        string lower = answer.ToLowerInvariant();
        Assert.Contains("red", lower);
        Assert.Contains("blue", lower);
    }
}
