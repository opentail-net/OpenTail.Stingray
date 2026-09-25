using System.Numerics.Tensors;
using OpenTail.Stingray.Vision.Clip;

namespace OpenTail.Stingray.Tests.Vision;

/// <summary>
/// Standalone CLIP (<see cref="ClipModel"/>) on openai/clip-vit-base-patch32 against Xenova/clip-vit-base-patch32's
/// ONNX export of the same weights (text_embeds, image_embeds, logits_per_image on identical ids and pixels), plus the
/// tokenizer on the HF docs' "a photo of a cat" ids and zero-shot labels on real photos.
/// </summary>
public sealed class ClipModelTests
{
    private static string? Find(string rel)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, rel);
            if (File.Exists(p) || Directory.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static readonly string[] Labels = ["a photo of a cup", "a photo of a dog", "a photo of a beach", "a photo of a city street", "a photo of a tent"];

    [Fact]
    public void Clip_MatchesOnnx_AndZeroShotClassifiesPhotos()
    {
        string? dir = Find("models/_models/hf/openai__clip-vit-base-patch32");
        string? cup = Find("examples/flux/assets/cup.png");
        Assert.SkipUnless(dir != null && File.Exists(Path.Combine(dir, "model.safetensors")) && File.Exists(Path.Combine(dir, "onnx", "model.onnx")) && cup != null,
            "clip-vit-base-patch32 (safetensors + ONNX) or test photos not found");

        using var clip = ClipModel.Load(dir!);
        Assert.Equal([49406, 320, 1125, 539, 320, 2368, 49407], clip.Tokenizer.Encode("a photo of a cat"));
        Assert.Equal([49406, 320, 1125, 539, 320, 1929, 49407], clip.Tokenizer.Encode("A  photo of a DOG"));
        Console.WriteLine($"[Clip] \"It's 2024!\" -> {string.Join(",", clip.Tokenizer.Encode("It's 2024!"))}");

        using var onnx = new OnnxModelSession(Path.Combine(dir!, "onnx", "model.onnx"));
        Console.WriteLine($"[Clip] onnx inputs {string.Join(",", onnx.InputNames)} outputs {string.Join(",", onnx.OutputNames)}");
        var textEmb = Labels.Select(clip.TextEmbedding).ToArray();

        string[] photos = ["examples/flux/assets/cup.png", "examples/CogVideo/inference/gradio_composite_demo/example_images/beach.png",
            "examples/CogVideo/inference/gradio_composite_demo/example_images/street.png", "examples/CogVideo/inference/gradio_composite_demo/example_images/camping.png"];
        int[] expected = [0, 2, 3, 4];
        for (int pi = 0; pi < photos.Length; pi++)
        {
            string? path = Find(photos[pi]);
            if (path is null) continue;
            var rgb = ImageIO.LoadRgb(path, out int w, out int h);
            var chw = clip.Preprocess(rgb, w, h);
            var img = clip.ImageEmbedding(chw);
            var logits = clip.ZeroShotLogits(img, textEmb);

            // ONNX on the same pixels and each label's ids (one text at a time, so no padding semantics are involved).
            float minTextCos = 1f, imgCos = 1f, maxLogitDiff = 0f;
            for (int li = 0; li < Labels.Length; li++)
            {
                int[] ids = clip.Tokenizer.Encode(Labels[li]);
                var r = onnx.Run(("input_ids", ids.Select(x => (long)x).ToArray(), [1, ids.Length]),
                    ("attention_mask", Enumerable.Repeat(1L, ids.Length).ToArray(), [1, ids.Length]),
                    ("pixel_values", chw, [1, 3, clip.ImageSize, clip.ImageSize]));
                minTextCos = Math.Min(minTextCos, TensorPrimitives.CosineSimilarity(textEmb[li], r["text_embeds"]));
                imgCos = Math.Min(imgCos, TensorPrimitives.CosineSimilarity(img, r["image_embeds"]));
                maxLogitDiff = Math.Max(maxLogitDiff, Math.Abs(logits[li] - r["logits_per_image"][0]));
            }
            int best = Array.IndexOf(logits, logits.Max());
            Console.WriteLine($"[Clip] {Path.GetFileName(path)}: text cos {minTextCos:F7}, image cos {imgCos:F7}, logit maxDiff {maxLogitDiff:E2}; " +
                $"zero-shot -> \"{Labels[best]}\" ({string.Join(", ", logits.Select(l => l.ToString("F1")))})");
            Assert.True(minTextCos > 0.99999f, $"text embedding cos {minTextCos}");
            Assert.True(imgCos > 0.99999f, $"image embedding cos {imgCos}");
            Assert.True(maxLogitDiff < 0.01f, $"logit diff {maxLogitDiff}");
            Assert.Equal(expected[pi], best);
        }
    }
}
