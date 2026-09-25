using System.Numerics.Tensors;
using System.Text.Json;
using OpenTail.Stingray.Vision.Classification;

namespace OpenTail.Stingray.Tests.Vision;

/// <summary>
/// timm MobileNetV3-small and EfficientNet-B0 (<see cref="TimmEfficientNetClassifier"/>). MobileNetV3's logits are compared
/// with onnx-community's ONNX export of the same checkpoint on identical input tensors; EfficientNet-B0 has no ONNX export
/// of the timm weights, so it shares the ONNX-verified block code and is checked by top-5 on real photos
/// (<c>examples/flux/assets/cup.png</c>: a paper coffee cup in sand; CogVideo example photos).
/// </summary>
public sealed class TimmEfficientNetClassifierTests
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

    private static readonly string[] Photos =
    [
        "examples/flux/assets/cup.png",
        "examples/CogVideo/inference/gradio_composite_demo/example_images/beach.png",
        "examples/CogVideo/inference/gradio_composite_demo/example_images/street.png",
        "examples/CogVideo/inference/gradio_composite_demo/example_images/camping.png",
    ];

    private static Dictionary<int, string> Labels(string onnxDir)
    {
        using var doc = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(onnxDir, "config.json")));
        return doc.RootElement.GetProperty("id2label").EnumerateObject().ToDictionary(p => int.Parse(p.Name), p => p.Value.GetString()!);
    }

    private static int[] Top(float[] logits, int k) => logits.Select((v, i) => (v, i)).OrderByDescending(x => x.v).Take(k).Select(x => x.i).ToArray();

    [Fact]
    public void MobileNetV3Small_MatchesOnnx_And_EfficientNetB0_ClassifiesPhotos()
    {
        string? mnDir = Find("models/_models/hf/timm__mobilenetv3_small_100.lamb_in1k");
        string? onnxDir = Find("models/_models/hf/onnx-community__mobilenetv3_small_100.lamb_in1k");
        string? effDir = Find("models/_models/hf/timm__efficientnet_b0.ra_in1k");
        string? cup = Find(Photos[0]);
        Assert.SkipUnless(mnDir != null && onnxDir != null && effDir != null && cup != null, "timm checkpoints, the ONNX export or the test photos not found");

        var labels = Labels(onnxDir!);
        using var mn = TimmEfficientNetClassifier.Load(mnDir!);
        using var eff = TimmEfficientNetClassifier.Load(effDir!);
        using var onnx = new OnnxModelSession(Path.Combine(onnxDir!, "onnx", "model.onnx"));
        string input = onnx.InputNames.First();

        foreach (string rel in Photos)
        {
            string? path = Find(rel);
            if (path is null) continue;
            var rgb = ImageIO.LoadRgb(path, out int w, out int h);
            var chw = ImageClassificationPreprocessor.ResizeCenterCropNormalize(rgb, w, h, mn.InputSize, mn.CropPct, mn.Mean, mn.Std, mn.Bicubic);
            var ours = mn.Logits(chw, mn.InputSize, mn.InputSize);
            var reference = onnx.Run((input, chw, [1, 3, mn.InputSize, mn.InputSize])).Values.First();
            float maxAbs = 0;
            for (int i = 0; i < ours.Length; i++) maxAbs = Math.Max(maxAbs, Math.Abs(ours[i] - reference[i]));
            float cos = TensorPrimitives.CosineSimilarity(ours, reference);
            var effLogits = eff.Classify(rgb, w, h);
            string Fmt(float[] l) => string.Join(", ", Top(l, 3).Select(i => $"{labels[i]} ({i})"));
            Console.WriteLine($"[Timm] {Path.GetFileName(path)}: MobileNetV3 vs ONNX maxAbs {maxAbs:E2} cos {cos:F7}\n" +
                $"  mobilenetv3: {Fmt(ours)}\n  onnx       : {Fmt(reference)}\n  effnet-b0  : {Fmt(effLogits)}");
            Assert.True(cos > 0.99999f, $"{rel}: cosine {cos}");
            Assert.Equal(Top(reference, 5), Top(ours, 5));

            if (rel == Photos[0])
            {
                // A white lidded paper cup: both models (and the ONNX) rank "bucket, pail" first, cup / coffee mug next.
                Assert.Contains(Top(ours, 3), i => i is 968 or 504);
                Assert.Contains(Top(effLogits, 3), i => i is 968 or 504);
            }
        }
    }
}
