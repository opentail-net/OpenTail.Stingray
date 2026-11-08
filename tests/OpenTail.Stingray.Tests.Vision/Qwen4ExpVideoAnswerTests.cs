using System.Linq;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Engine.Verification;
using OpenTail.Stingray.Vision;

namespace OpenTail.Stingray.Tests.Vision;

/// <summary>
/// Phase 5.3 step (e) (Qwen 3.8 plan): qwen4exp answers a question about a short clip through the video path. Frames come
/// from ffmpeg at the reference's 4 fps, the two-frame projector runs per chunk, and the prompt follows the reference layout
/// (<see cref="VideoPromptLayout"/>): "Video:", then each chunk wrapped in the vision markers, with timestamp texts between
/// chunks, then the question. M-RoPE positions follow the chunk rule. Oracle: the pinned reference llama-mtmd-cli --video on
/// the same clip, prompt and greedy settings. Opt-in: needs QWEN_TEST_VIDEO (an MP4 path), ffmpeg, the trunk and the
/// projector; skipped visibly otherwise. QWEN_VIDEO_ANSWER_FILE, when set, receives the answer text for comparison.
/// </summary>
public sealed class Qwen4ExpVideoAnswerTests
{
    private const int Side = 224;
    private const double Fps = 4.0;
    private const int MaxNewTokens = 40;
    private const string Question = "What colors appear in this video, and in what order? Answer in one short sentence.";

    private static string? FindTrunk() => ModelLocator.FindDeep("Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf");

    private static string? FindMmproj() => ModelLocator.FindDeep("mmproj-Qwen3.8-Flash-Next-F16.gguf");

    [Fact]
    public void VideoQuestion_AnswersAboutTheClip()
    {
        string? clip = Environment.GetEnvironmentVariable("QWEN_TEST_VIDEO");
        Assert.SkipUnless(clip is not null && File.Exists(clip), "Set QWEN_TEST_VIDEO to an MP4 path.");
        string? ffmpeg = FfmpegVideoFrameSource.Locate();
        Assert.SkipUnless(ffmpeg is not null, "ffmpeg is required (STINGRAY_FFMPEG or PATH).");
        string? trunkPath = FindTrunk();
        Assert.SkipUnless(trunkPath is not null, "Qwen3.8-Flash-Next UD-IQ1_S checkpoint is required (paged, 72.5 GB).");
        string? mmprojPath = FindMmproj();
        Assert.SkipUnless(mmprojPath is not null, "mmproj-Qwen3.8-Flash-Next-F16.gguf is required (904 MB).");

        // The clip must already be at the test size, so neither side resizes it.
        var source = new FfmpegVideoFrameSource(ffmpeg);
        var info = source.Probe(clip!);
        Assert.Equal(Side, info.Width);
        Assert.Equal(Side, info.Height);
        var frames = source.ReadFrames(clip!, Fps, Side, Side).Select(f => VideoTemporalPairs.RgbToChw(f, Side, Side)).ToList();
        var segments = VideoPromptLayout.Segments(VideoPromptLayout.Parts(frames.Count, (float)Fps));
        Console.WriteLine($"VIDEO frames={frames.Count} segments={segments.Count} chunks={segments.Count(s => s.Frames is not null)}");

        using var vision = QwenVlVisionModel.Open(mmprojPath!);
        var encoder = new QwenVlVisionEncoder(vision);
        int grid = Side / (vision.PatchSize * 2); // patch size from the mmproj, 2x2 merge
        int chunkTokens = grid * grid;

        using var model = GgufModel.Open(trunkPath!);
        int totalLayers = (int)Convert.ToInt64(model.Metadata["qwen4exp.block_count"]);
        var hp = Qwen4ExpHyperparams.FromMetadata(model.Metadata, totalLayers);
        var tensors = Qwen4ExpTensorSet.Load(model, hp, totalLayers);
        using var pass = new Qwen4ExpForwardPass(model, hp, tensors);
        var tok = GgufTokenizer.FromGgufModel(model);
        int[] pad = tok.Encode("<|image_pad|>").ToArray();
        Assert.Single(pad);

        // No empty think block, to match llama-mtmd-cli's generation prompt (see Qwen4ExpImageAnswerTests).
        var state = pass.CreateSequenceState();
        int pos = 0;
        ReadOnlySpan<float> logits = default;
        foreach (int id in tok.Encode("<|im_start|>user\n").ToArray()) logits = pass.Forward(id, pos++, state);

        foreach (var seg in segments)
        {
            if (seg.Text is not null)
            {
                foreach (int id in tok.Encode(seg.Text).ToArray()) logits = pass.Forward(id, pos++, state);
                continue;
            }

            int a = seg.Frames![0], b = seg.Frames.Length > 1 ? seg.Frames[1] : a;
            float[] soft = encoder.ForwardVideo(frames[a], frames[b], Side, Side, out int nTok);
            Assert.Equal(chunkTokens, nTok);

            foreach (int id in tok.Encode("<|vision_start|>").ToArray()) logits = pass.Forward(id, pos++, state);
            int basePos = pos;
            var placeholders = Enumerable.Repeat(pad[0], nTok).ToArray();
            var posT = Enumerable.Repeat(basePos, nTok).ToArray();
            var posY = Enumerable.Range(0, nTok).Select(i => basePos + i / grid).ToArray();
            var posX = Enumerable.Range(0, nTok).Select(i => basePos + i % grid).ToArray();
            pass.ForwardImageChunk(soft, placeholders, posT, posY, posX, state);
            pos = basePos + grid; // square grid: one chunk advances the text position by max(nx, ny) = grid
            foreach (int id in tok.Encode("<|vision_end|>").ToArray()) logits = pass.Forward(id, pos++, state);
        }

        foreach (int id in tok.Encode(Question + "<|im_end|>\n<|im_start|>assistant\n").ToArray())
            logits = pass.Forward(id, pos++, state);

        var generated = new List<int>();
        for (int s = 0; s < MaxNewTokens; s++)
        {
            int best = 0;
            for (int i = 1; i < logits.Length; i++) if (logits[i] > logits[best]) best = i;
            if (tok.EogTokenIds.Contains(best)) break;
            generated.Add(best);
            logits = pass.Forward(best, pos++, state);
        }

        string answer = tok.Decode(generated);
        Console.WriteLine($"ANSWER_IDS {string.Join(",", generated)}");
        Console.WriteLine($"ANSWER_TEXT {answer}");
        if (Environment.GetEnvironmentVariable("QWEN_VIDEO_ANSWER_FILE") is { } outFile)
            File.WriteAllText(outFile, answer);
        Assert.False(string.IsNullOrWhiteSpace(answer));
    }
}
