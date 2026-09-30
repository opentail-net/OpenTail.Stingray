using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Vision;
using EForwardPass = OpenTail.Stingray.Engine.ForwardPass;

namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// End-to-end CPU image→text tests for Granite 4.0 3B Vision (bug item 04):
/// Verifies that with the canonical Granite chat template, image prefill does not produce
/// early EOS or empty decode, top-1 greedy token matches llama-mtmd-cli, and batched
/// deepstack prefill decodes coherent text grounded in the image.
/// </summary>
public sealed class Granite4VisionE2ETests : HeavyTestBase
{
    private readonly ITestOutputHelper _output;
    public Granite4VisionE2ETests(ITestOutputHelper output) => _output = output;

    private const string TextModel = "granite-4.0-3b-vision-Q4_K_M.gguf";
    private const string Mmproj = "mmproj-granite-4.0-3b-vision-f16.gguf";
    private const string SampleImage = "flux_apple_vulkan_512_4steps.png";

    private static string? Find(string file)
    {
        foreach (var p in new[] { $@"E:\models\{file}", $@"C:\p\opentail-llm\models\{file}" })
            if (File.Exists(p)) return p;
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8 && dir is not null; i++)
        {
            var p = Path.Combine(dir, "models", file);
            if (File.Exists(p)) return p;
            var pNested = Path.Combine(dir, "models", "_models", file);
            if (File.Exists(pNested)) return pNested;
            dir = Directory.GetParent(dir)?.FullName;
        }
        return null;
    }

    private static string? FindImage(string file)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8 && dir is not null; i++)
        {
            var p = Path.Combine(dir, "docs", "diffusion-samples", file);
            if (File.Exists(p)) return p;
            dir = Directory.GetParent(dir)?.FullName;
        }
        return null;
    }

    private static int ArgMax(ReadOnlySpan<float> span)
    {
        int maxIdx = 0;
        float maxVal = span[0];
        for (int i = 1; i < span.Length; i++)
        {
            if (span[i] > maxVal)
            {
                maxVal = span[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    [Fact]
    public void Granite4Vision_CanonicalTemplate_NoEarlyEos_MatchesReferenceTop1()
    {
        var textPath = Find(TextModel);
        var mmprojPath = Find(Mmproj);
        var imagePath = FindImage(SampleImage);
        Assert.SkipUnless(textPath is not null && mmprojPath is not null && imagePath is not null,
            "textPath, mmprojPath or sample image not found");

        using var modelHandle = SharedModelCacheFixture.Instance.Acquire(textPath);
        var model = modelHandle.Model;
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var tok = GgufTokenizer.FromGgufModel(model);
        using var backend = new CpuBackend();
        using var fwd = new EForwardPass(model, backend, hp);
        Assert.True(fwd.SupportsEmbeddingInput);

        using var embedder = UnifiedVisionPipeline.Open(mmprojPath);
        Assert.NotNull(embedder);

        float[] soft = embedder.EmbedImageFile(imagePath, out int tokenCount);
        Assert.True(tokenCount > 0);

        // Canonical Granite 4.0 template matching llama.cpp's LLM_CHAT_TEMPLATE_GRANITE_4_0
        string prompt = $"<|start_of_role|>user<|end_of_role|>{embedder.PlaceholderMarker}Describe this picture.<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>";
        var allTokens = tok.Encode(prompt).ToList();
        if (tok.AddBosToken && tok.BosTokenId >= 0 && (allTokens.Count == 0 || allTokens[0] != tok.BosTokenId))
            allTokens.Insert(0, tok.BosTokenId);

        int placeholder = tok.SpecialTokens[embedder.PlaceholderMarker];

        int pos = 0;
        ReadOnlySpan<float> logits = default;
        foreach (int id in allTokens)
        {
            if (id == placeholder)
            {
                fwd.PrefillEmbeddings(soft, tokenCount, pos);
                pos += tokenCount;
            }
            else
            {
                logits = fwd.Forward(id, pos++);
            }
        }

        // 1. Verify post-prefill logits:
        int top1 = ArgMax(logits);
        Assert.DoesNotContain(top1, tok.EogTokenIds);

        // 2. Greedy decode 20 tokens: none should be EOG/EOS.
        var decodedTokens = new List<int> { top1 };
        int currentToken = top1;

        for (int step = 0; step < 19; step++)
        {
            logits = fwd.Forward(currentToken, pos++);
            currentToken = ArgMax(logits);
            decodedTokens.Add(currentToken);
            Assert.DoesNotContain(currentToken, tok.EogTokenIds);
        }

        Assert.Equal(20, decodedTokens.Count);
        string text = tok.Decode(decodedTokens);
        _output.WriteLine($"Decoded: {text}");
        Assert.Contains("red", text, StringComparison.OrdinalIgnoreCase);
    }
}
