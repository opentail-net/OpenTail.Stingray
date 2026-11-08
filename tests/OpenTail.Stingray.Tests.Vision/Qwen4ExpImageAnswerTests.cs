using System.Linq;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Vision;

namespace OpenTail.Stingray.Tests.Vision;

/// <summary>
/// Phase 5.2 end to end: qwen4exp answers a question about an image through the Qwen 3.8 projector and the layer-major
/// image chunk. Oracle: the pinned reference llama-mtmd-cli on the same image and prompt (scratchpad). Opt-in: needs
/// QWEN_TEST_IMAGE (a PNG path) plus the trunk and projector; skipped visibly otherwise.
/// </summary>
public sealed class Qwen4ExpImageAnswerTests
{
    private static string? FindTrunk() => OpenTail.Stingray.Engine.Verification.ModelLocator.FindDeep(
        "Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf");

    private static string? FindMmproj() => OpenTail.Stingray.Engine.Verification.ModelLocator.FindDeep(
        "mmproj-Qwen3.8-Flash-Next-F16.gguf");

    [Fact]
    public void ImageQuestion_AnswersAboutTheImage()
    {
        string? image = Environment.GetEnvironmentVariable("QWEN_TEST_IMAGE");
        Assert.SkipUnless(image is not null && File.Exists(image), "Set QWEN_TEST_IMAGE to a PNG path.");
        string? trunkPath = FindTrunk();
        Assert.SkipUnless(trunkPath is not null, "Qwen3.8-Flash-Next UD-IQ1_S checkpoint is required (paged, 72.5 GB).");
        string? mmprojPath = FindMmproj();
        Assert.SkipUnless(mmprojPath is not null, "mmproj-Qwen3.8-Flash-Next-F16.gguf is required (904 MB).");

        // Image -> CHW in [-1, 1] (mean 0.5, std 0.5), then the projector.
        byte[] rgb = ImageIO.LoadRgb(image!, out int w, out int h);
        var chw = new float[3 * w * h];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                for (int c = 0; c < 3; c++)
                    chw[c * w * h + y * w + x] = 2f * rgb[(y * w + x) * 3 + c] / 255f - 1f;

        // Noise probe (Phase 0.3): a deterministic relative perturbation of the input pixels, as the text probe does.
        if (double.TryParse(Environment.GetEnvironmentVariable("QWEN_IMAGE_PERTURB_EPS"), System.Globalization.NumberStyles.Float,
                System.Globalization.CultureInfo.InvariantCulture, out double eps) && eps != 0)
        {
            for (int i = 0; i < chw.Length; i++)
            {
                double u = Math.Sin(i * 12.9898) * 43758.5453;
                u -= Math.Floor(u);
                chw[i] *= (float)(1.0 + eps * (2 * u - 1));
            }
        }

        using var vision = QwenVlVisionModel.Open(mmprojPath!);
        float[] soft = new QwenVlVisionEncoder(vision).Forward(chw, w, h, out int nImg);
        int nx = w / 32, ny = h / 32; // patch 16, spatial merge 2
        Assert.Equal(nx * ny, nImg);
        int embd = soft.Length / nImg;
        Assert.Equal(2560, embd);

        using var model = GgufModel.Open(trunkPath!);
        int totalLayers = (int)Convert.ToInt64(model.Metadata["qwen4exp.block_count"]);
        var hp = Qwen4ExpHyperparams.FromMetadata(model.Metadata, totalLayers);
        var tensors = Qwen4ExpTensorSet.Load(model, hp, totalLayers);
        using var pass = new Qwen4ExpForwardPass(model, hp, tensors);
        var tok = GgufTokenizer.FromGgufModel(model);

        int[] pre = tok.Encode("<|im_start|>user\n<|vision_start|>").ToArray();
        int[] pad = tok.Encode("<|image_pad|>").ToArray();
        // Default: the prompt ends with an empty think block. QWEN_IMAGE_NO_THINK_BLOCK=1 drops it, to match llama-mtmd-cli's
        // generation prompt (the model then generates the think block itself).
        string postText = Environment.GetEnvironmentVariable("QWEN_IMAGE_NO_THINK_BLOCK") == "1"
            ? "<|vision_end|>What shapes are in the image?<|im_end|>\n<|im_start|>assistant\n"
            : "<|vision_end|>What shapes are in the image?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        int[] post = tok.Encode(postText).ToArray();
        Assert.Single(pad);
        Console.WriteLine($"PROMPT pre={pre.Length} image={nImg} grid={nx}x{ny} post={post.Length} pre_ids={string.Join(",", pre)} post_ids={string.Join(",", post)}");

        var state = pass.CreateSequenceState();
        int pos0 = pre.Length;
        for (int i = 0; i < pre.Length; i++) pass.Forward(pre[i], i, state);

        var placeholders = Enumerable.Repeat(pad[0], nImg).ToArray();
        var posT = Enumerable.Repeat(pos0, nImg).ToArray();
        var posY = Enumerable.Range(0, nImg).Select(i => pos0 + i / nx).ToArray();
        var posX = Enumerable.Range(0, nImg).Select(i => pos0 + i % nx).ToArray();
        pass.ForwardImageChunk(soft, placeholders, posT, posY, posX, state);

        int textPos = pos0 + Math.Max(nx, ny);
        ReadOnlySpan<float> logits = default;
        for (int j = 0; j < post.Length; j++) logits = pass.Forward(post[j], textPos + j, state);

        var generated = new List<int>();
        for (int s = 0; s < 40; s++)
        {
            if (Environment.GetEnvironmentVariable("QWEN_LOGIT_DIR") is { } dir)
            {
                var buf = new byte[logits.Length * sizeof(float)];
                System.Runtime.InteropServices.MemoryMarshal.AsBytes(logits).CopyTo(buf);
                File.WriteAllBytes(Path.Combine(dir, $"step_{s}.bin"), buf);
            }
            int best = 0;
            for (int i = 1; i < logits.Length; i++) if (logits[i] > logits[best]) best = i;
            if (tok.EogTokenIds.Contains(best)) break;
            generated.Add(best);
            logits = pass.Forward(best, textPos + post.Length + s, state);
        }

        string answer = tok.Decode(generated);
        Console.WriteLine($"ANSWER_IDS {string.Join(",", generated)}");
        Console.WriteLine($"ANSWER_TEXT {answer}");
        Assert.Contains("red", answer.ToLowerInvariant());
    }
}
