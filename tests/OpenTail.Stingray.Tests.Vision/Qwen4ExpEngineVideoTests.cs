using System.Linq;
using System.Text;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Engine.Verification;
using OpenTail.Stingray.Vision;

namespace OpenTail.Stingray.Tests.Vision;

/// <summary>
/// Qwen 3.8 plan, Phase 5.3 (video through the engine): the engine's own video splice (<see cref="InferenceEngine.GenerateVideoChunksAsync"/>)
/// on the real qwen4exp checkpoint. The clip is set by QWEN_TEST_VIDEO (an MP4); ffmpeg decodes it here, as the host would.
/// Skipped unless the clip, ffmpeg, the trunk and the mmproj are present. Checks the answer names both colours; the direct
/// pass path is covered by <see cref="Qwen4ExpVideoAnswerTests"/>.
/// </summary>
public sealed class Qwen4ExpEngineVideoTests
{
    private const int Side = 224;
    private const float Fps = 4.0f;
    private const int MaxNewTokens = 96;
    private const string Question = "What colors appear in this video, and in what order? Answer in one short sentence.";

    private static string? FindTrunk() => ModelLocator.FindDeep("Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf");

    private static string? FindMmproj() => ModelLocator.FindDeep("mmproj-Qwen3.8-Flash-Next-F16.gguf");

    [Fact]
    public async Task EngineVideoAnswer_NamesTheColours_ThroughTheEngineSplice()
    {
        string? clip = Environment.GetEnvironmentVariable("QWEN_TEST_VIDEO");
        Assert.SkipUnless(clip is not null && File.Exists(clip), "Set QWEN_TEST_VIDEO to an MP4 path.");
        string? ffmpeg = FfmpegVideoFrameSource.Locate();
        Assert.SkipUnless(ffmpeg is not null, "ffmpeg is required (STINGRAY_FFMPEG or PATH).");
        string? trunkPath = FindTrunk();
        Assert.SkipUnless(trunkPath is not null, "Qwen3.8-Flash-Next UD-IQ1_S checkpoint is required (paged, 72.5 GB).");
        string? mmprojPath = FindMmproj();
        Assert.SkipUnless(mmprojPath is not null, "mmproj-Qwen3.8-Flash-Next-F16.gguf is required (904 MB).");

        // Host side: decode and convert to the projector's input, as the server will.
        var source = new FfmpegVideoFrameSource(ffmpeg!);
        var chw = source.ReadFrames(clip!, Fps, Side, Side).Select(f => VideoTemporalPairs.RgbToChw(f, Side, Side)).ToList();
        Assert.NotEmpty(chw);

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

        using var engine = new InferenceEngine(pass, tok, "qwen4exp", think, endThink, pass);
        engine.EnableQwenImageInput(QwenVlVisionModel.Open(mmprojPath!), open, close, pad);

        // The prompt carries one video placeholder; the engine expands its layout during prefill.
        string prompt = "<|im_start|>user\n<|video_pad|>" + Question + "<|im_end|>\n<|im_start|>assistant\n";
        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = MaxNewTokens };
        var videos = new[] { new QwenVideoInput(chw, Side, Fps) };

        var text = new StringBuilder();
        await foreach (var c in engine.GenerateVideoChunksAsync(prompt, videos, sp))
            if (c.Kind == GenerateChunkKind.Text) text.Append(c.Text);
        string answer = text.ToString();

        Console.WriteLine($"ENGINE_VIDEO_ANSWER frames={chw.Count} answer={answer}");
        string lower = answer.ToLowerInvariant();
        Assert.Contains("red", lower);
        Assert.Contains("blue", lower);
    }
}
