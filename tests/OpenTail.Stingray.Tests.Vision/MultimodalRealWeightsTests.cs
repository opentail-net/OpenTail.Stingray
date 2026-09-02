
namespace OpenTail.Stingray.Tests.Vision;

public sealed class MultimodalRealWeightsTests
{
    private static string? FindModelPath(string fileName)
    {
        string[] candidates =
        {
            $@"C:\Git-Public\OpenTail.Stingray\models\{fileName}",
            $@"C:\p\opentail-llm\models\{fileName}",
            $@"E:\models\{fileName}",
        };
        foreach (var p in candidates)
        {
            if (File.Exists(p)) return p;
        }
        return null;
    }

    private static void ValidateEmbeddingRigorously(
        float[] tokensA, float[] tokensB,
        int tokenCount, int embeddingDim)
    {
        Assert.NotNull(tokensA);
        Assert.NotNull(tokensB);
        Assert.True(tokenCount > 0, "Token count must be positive");
        Assert.Equal(tokenCount * embeddingDim, tokensA.Length);
        Assert.Equal(tokenCount * embeddingDim, tokensB.Length);

        // 1. Full-buffer numerical sanity (zero NaN, zero Inf, non-zero values)
        bool hasNonZeroA = false;
        bool hasNonZeroB = false;
        double sumA = 0, sumSqA = 0;

        for (int i = 0; i < tokensA.Length; i++)
        {
            float va = tokensA[i];
            float vb = tokensB[i];

            Assert.False(float.IsNaN(va), $"tokensA[{i}] is NaN");
            Assert.False(float.IsInfinity(va), $"tokensA[{i}] is Infinity");
            Assert.False(float.IsNaN(vb), $"tokensB[{i}] is NaN");
            Assert.False(float.IsInfinity(vb), $"tokensB[{i}] is Infinity");

            if (Math.Abs(va) > 1e-7f) hasNonZeroA = true;
            if (Math.Abs(vb) > 1e-7f) hasNonZeroB = true;

            sumA += va;
            sumSqA += (double)va * va;
        }

        Assert.True(hasNonZeroA, "Embedding A is completely zero");
        Assert.True(hasNonZeroB, "Embedding B is completely zero");

        // 2. Statistical variance / dispersion check (embeddings must have real feature spread)
        double meanA = sumA / tokensA.Length;
        double varianceA = (sumSqA / tokensA.Length) - (meanA * meanA);
        double stdA = Math.Sqrt(Math.Max(0, varianceA));
        Assert.True(stdA > 0.0001, $"Embedding standard deviation too low ({stdA}), indicates flat or degenerate output");

        // 3. Image differentiation (cosine similarity between distinct inputs must not be 1.0)
        float dot = TensorPrimitives.Dot(tokensA.AsSpan(), tokensB.AsSpan());
        float normA = MathF.Sqrt(TensorPrimitives.SumOfSquares(tokensA.AsSpan()));
        float normB = MathF.Sqrt(TensorPrimitives.SumOfSquares(tokensB.AsSpan()));

        Assert.True(normA > 0.01f, $"Norm A is too small: {normA}");
        Assert.True(normB > 0.01f, $"Norm B is too small: {normB}");

        float cosSim = dot / (normA * normB);
        Assert.True(cosSim < 0.9999f, $"Embeddings for distinct images are identical (cosSim = {cosSim})");

        // 4. L2 distance between distinct image representations
        float l2DistSq = TensorPrimitives.Distance(tokensA.AsSpan(), tokensB.AsSpan());
        Assert.True(l2DistSq > 0.01f, $"L2 distance between distinct images is too small: {l2DistSq}");
    }

    private static (byte[] imgA, byte[] imgB) CreateTestImagePair(int width, int height)
    {
        var imgA = new byte[width * height * 3];
        var imgB = new byte[width * height * 3];

        // Image A: Diagonal RGB gradient
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int idx = (y * width + x) * 3;
                imgA[idx]     = (byte)((x * 255) / width);
                imgA[idx + 1] = (byte)((y * 255) / height);
                imgA[idx + 2] = (byte)(((x + y) * 128) / (width + height));
            }
        }

        // Image B: Inverted checkerboard pattern
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int idx = (y * width + x) * 3;
                bool check = ((x / 16) + (y / 16)) % 2 == 0;
                imgB[idx]     = check ? (byte)240 : (byte)20;
                imgB[idx + 1] = check ? (byte)30  : (byte)220;
                imgB[idx + 2] = check ? (byte)200 : (byte)50;
            }
        }

        return (imgA, imgB);
    }

    [Fact]
    public void Llava_RealWeights_LoadsAndEmbedsImage()
    {
        string? path = FindModelPath("mmproj-llava-v1.5-7b-f16.gguf");
        if (path is null) return;

        using var embedder = UnifiedVisionPipeline.Open(path);
        Assert.NotNull(embedder);

        var (imgA, imgB) = CreateTestImagePair(336, 336);
        var tokensA = embedder.EmbedImage(imgA, 336, 336, out int tokenCountA);
        var tokensB = embedder.EmbedImage(imgB, 336, 336, out int tokenCountB);

        Assert.Equal(tokenCountA, tokenCountB);
        ValidateEmbeddingRigorously(tokensA, tokensB, tokenCountA, embedder.EmbeddingDim);
    }

    [Fact]
    public void Pixtral_RealWeights_LoadsAndEmbedsImage()
    {
        string? path = FindModelPath("mmproj-pixtral-12b-f16.gguf");
        if (path is null) return;

        using var embedder = UnifiedVisionPipeline.Open(path);
        Assert.NotNull(embedder);

        var (imgA, imgB) = CreateTestImagePair(128, 128);
        var tokensA = embedder.EmbedImage(imgA, 128, 128, out int tokenCountA);
        var tokensB = embedder.EmbedImage(imgB, 128, 128, out int tokenCountB);

        Assert.Equal(tokenCountA, tokenCountB);
        ValidateEmbeddingRigorously(tokensA, tokensB, tokenCountA, embedder.EmbeddingDim);
    }

    [Fact]
    public void HunyuanVl_RealWeights_LoadsAndEmbedsImage()
    {
        string? path = FindModelPath("mmproj-hunyuanocr-q8_0.gguf") ?? FindModelPath("mmproj-HunyuanOCR-Q8_0.gguf");
        if (path is null) return;

        using var embedder = UnifiedVisionPipeline.Open(path);
        Assert.NotNull(embedder);

        var (imgA, imgB) = CreateTestImagePair(378, 378);
        var tokensA = embedder.EmbedImage(imgA, 378, 378, out int tokenCountA);
        var tokensB = embedder.EmbedImage(imgB, 378, 378, out int tokenCountB);

        Assert.Equal(tokenCountA, tokenCountB);
        ValidateEmbeddingRigorously(tokensA, tokensB, tokenCountA, embedder.EmbeddingDim);
    }

    [Fact]
    public void Step3Vl_RealWeights_LoadsAndEmbedsImage()
    {
        string? path = FindModelPath("mmproj-step3-flash-f16.gguf") ?? FindModelPath("mmproj-step3.7-flash-f16.gguf");
        if (path is null) return;

        using var embedder = UnifiedVisionPipeline.Open(path);
        Assert.NotNull(embedder);

        var (imgA, imgB) = CreateTestImagePair(378, 378);
        var tokensA = embedder.EmbedImage(imgA, 378, 378, out int tokenCountA);
        var tokensB = embedder.EmbedImage(imgB, 378, 378, out int tokenCountB);

        Assert.Equal(tokenCountA, tokenCountB);
        ValidateEmbeddingRigorously(tokensA, tokensB, tokenCountA, embedder.EmbeddingDim);
    }

    [Fact]
    public void Exaone4_RealWeights_LoadsAndEmbedsImage()
    {
        string? path = FindModelPath("mmproj-exaone-4.5-q8_0.gguf") ?? FindModelPath("EXAONE-4.5-33B.mmproj-Q8_0.gguf");
        if (path is null) return;

        using var embedder = UnifiedVisionPipeline.Open(path);
        Assert.NotNull(embedder);

        var (imgA, imgB) = CreateTestImagePair(280, 280);
        var tokensA = embedder.EmbedImage(imgA, 280, 280, out int tokenCountA);
        var tokensB = embedder.EmbedImage(imgB, 280, 280, out int tokenCountB);

        Assert.Equal(tokenCountA, tokenCountB);
        ValidateEmbeddingRigorously(tokensA, tokensB, tokenCountA, embedder.EmbeddingDim);
    }

    [Fact]
    public void MimoVl_RealWeights_LoadsAndEmbedsImage()
    {
        string? path = FindModelPath("mmproj-mimovl-7b-q8_0.gguf") ?? FindModelPath("MiMo-VL-7B-SFT.mmproj-Q8_0.gguf");
        if (path is null) return;

        using var embedder = UnifiedVisionPipeline.Open(path);
        Assert.NotNull(embedder);

        var (imgA, imgB) = CreateTestImagePair(280, 280);
        var tokensA = embedder.EmbedImage(imgA, 280, 280, out int tokenCountA);
        var tokensB = embedder.EmbedImage(imgB, 280, 280, out int tokenCountB);

        Assert.Equal(tokenCountA, tokenCountB);
        ValidateEmbeddingRigorously(tokensA, tokensB, tokenCountA, embedder.EmbeddingDim);
    }

    [Fact]
    public void YoutuVl_RealWeights_LoadsAndEmbedsImage()
    {
        string? path = FindModelPath("mmproj-youtuvl-4b-q8_0.gguf") ?? FindModelPath("Youtu-VL-4B-Instruct.mmproj-Q8_0.gguf");
        if (path is null) return;

        using var embedder = UnifiedVisionPipeline.Open(path);
        Assert.NotNull(embedder);

        var (imgA, imgB) = CreateTestImagePair(280, 280);
        var tokensA = embedder.EmbedImage(imgA, 280, 280, out int tokenCountA);
        var tokensB = embedder.EmbedImage(imgB, 280, 280, out int tokenCountB);

        Assert.Equal(tokenCountA, tokenCountB);
        ValidateEmbeddingRigorously(tokensA, tokensB, tokenCountA, embedder.EmbeddingDim);
    }

    [Fact]
    public void KimiVl_RealWeights_LoadsAndEmbedsImage()
    {
        string? path = FindModelPath("mmproj-kimivl-q8_0.gguf") ?? FindModelPath("Kimi-VL-A3B-Thinking-2506.mmproj-Q8_0.gguf");
        if (path is null) return;

        using var embedder = UnifiedVisionPipeline.Open(path);
        Assert.NotNull(embedder);

        var (imgA, imgB) = CreateTestImagePair(280, 280);
        var tokensA = embedder.EmbedImage(imgA, 280, 280, out int tokenCountA);
        var tokensB = embedder.EmbedImage(imgB, 280, 280, out int tokenCountB);

        Assert.Equal(tokenCountA, tokenCountB);
        ValidateEmbeddingRigorously(tokensA, tokensB, tokenCountA, embedder.EmbeddingDim);
    }

    [Fact]
    public void Qwen2_5_Vl_RealWeights_LoadsAndEmbedsImage()
    {
        string? path = FindModelPath("mmproj-qwen2.5-vl-7b-f16.gguf");
        if (path is null) return;

        using var embedder = UnifiedVisionPipeline.Open(path);
        Assert.NotNull(embedder);

        var (imgA, imgB) = CreateTestImagePair(280, 280);
        var tokensA = embedder.EmbedImage(imgA, 280, 280, out int tokenCountA);
        var tokensB = embedder.EmbedImage(imgB, 280, 280, out int tokenCountB);

        Assert.Equal(tokenCountA, tokenCountB);
        ValidateEmbeddingRigorously(tokensA, tokensB, tokenCountA, embedder.EmbeddingDim);
    }

    [Fact]
    public void MiniCpmV_RealWeights_LoadsAndEmbedsImage()
    {
        string? path = FindModelPath("mmproj-minicpm-v-2_6-f16.gguf");
        if (path is null) return;

        using var embedder = UnifiedVisionPipeline.Open(path);
        Assert.NotNull(embedder);

        var (imgA, imgB) = CreateTestImagePair(448, 448);
        var tokensA = embedder.EmbedImage(imgA, 448, 448, out int tokenCountA);
        var tokensB = embedder.EmbedImage(imgB, 448, 448, out int tokenCountB);

        Assert.Equal(tokenCountA, tokenCountB);
        ValidateEmbeddingRigorously(tokensA, tokensB, tokenCountA, embedder.EmbeddingDim);
    }

    [Fact]
    public void Glm4V_RealWeights_LoadsAndEmbedsImage()
    {
        string? path = FindModelPath("mmproj-glm-4.6v-q4.gguf");
        if (path is null) return;

        using var embedder = UnifiedVisionPipeline.Open(path);
        Assert.NotNull(embedder);

        var (imgA, imgB) = CreateTestImagePair(280, 280);
        var tokensA = embedder.EmbedImage(imgA, 280, 280, out int tokenCountA);
        var tokensB = embedder.EmbedImage(imgB, 280, 280, out int tokenCountB);

        Assert.Equal(tokenCountA, tokenCountB);
        ValidateEmbeddingRigorously(tokensA, tokensB, tokenCountA, embedder.EmbeddingDim);
    }

    // ---- Checkpoints present locally with real weights but never previously exercised by any
    // test (2026-09-02 coverage sweep) -- same "sitting untested" shape KimiVl/MiniCpmV/YoutuVl/
    // HunyuanVl were in before real bugs were found in each. Differentiation-level only, matching
    // this file's existing bar; a real bug found via any of these should get the same
    // golden-verification treatment (scripts/<name>_ref.py + *EmbedderParityTests.cs) the others
    // got, not just a fix.

    [Fact]
    public void PaddleOcr_RealWeights_LoadsAndEmbedsImage()
    {
        string? path = FindModelPath("PaddleOCR-VL-1.6-GGUF-mmproj.gguf") ?? FindModelPath("mmproj-paddleocr-vl-1.6.gguf");
        if (path is null) return;

        using var embedder = UnifiedVisionPipeline.Open(path);
        Assert.NotNull(embedder);

        var (imgA, imgB) = CreateTestImagePair(280, 280);
        var tokensA = embedder.EmbedImage(imgA, 280, 280, out int tokenCountA);
        var tokensB = embedder.EmbedImage(imgB, 280, 280, out int tokenCountB);

        Assert.Equal(tokenCountA, tokenCountB);
        ValidateEmbeddingRigorously(tokensA, tokensB, tokenCountA, embedder.EmbeddingDim);
    }

    [Fact]
    public void DotsOcr_RealWeights_LoadsAndEmbedsImage()
    {
        string? path = FindModelPath("mmproj-dots.ocr-Q8_0.gguf");
        if (path is null) return;

        using var embedder = UnifiedVisionPipeline.Open(path);
        Assert.NotNull(embedder);

        var (imgA, imgB) = CreateTestImagePair(280, 280);
        var tokensA = embedder.EmbedImage(imgA, 280, 280, out int tokenCountA);
        var tokensB = embedder.EmbedImage(imgB, 280, 280, out int tokenCountB);

        Assert.Equal(tokenCountA, tokenCountB);
        ValidateEmbeddingRigorously(tokensA, tokensB, tokenCountA, embedder.EmbeddingDim);
    }

    [Fact]
    public void Granite4Vision_RealWeights_LoadsAndEmbedsImage()
    {
        string? path = FindModelPath("mmproj-granite-4.0-3b-vision-f16.gguf");
        if (path is null) return;

        using var embedder = UnifiedVisionPipeline.Open(path);
        Assert.NotNull(embedder);

        // Native image_size=384, patch_size=16 (24x24 patch grid) -- confirmed via list-metadata.
        var (imgA, imgB) = CreateTestImagePair(384, 384);
        var tokensA = embedder.EmbedImage(imgA, 384, 384, out int tokenCountA);
        var tokensB = embedder.EmbedImage(imgB, 384, 384, out int tokenCountB);

        Assert.Equal(tokenCountA, tokenCountB);
        ValidateEmbeddingRigorously(tokensA, tokensB, tokenCountA, embedder.EmbeddingDim);
    }

    [Fact]
    public void GraniteVision3_RealWeights_LoadsAndEmbedsImage()
    {
        // clip.projector_type is real "mlp" on this checkpoint (confirmed via list-metadata), not
        // "granite4_vision" -- routes through LlavaAdapter, not Granite4Adapter. Testing via
        // UnifiedVisionPipeline.Open so it exercises whatever the real autodetect routes to.
        string? path = FindModelPath("mmproj-granite-vision-3.2-2b-f16.gguf");
        if (path is null) return;

        using var embedder = UnifiedVisionPipeline.Open(path);
        Assert.NotNull(embedder);

        var (imgA, imgB) = CreateTestImagePair(336, 336);
        var tokensA = embedder.EmbedImage(imgA, 336, 336, out int tokenCountA);
        var tokensB = embedder.EmbedImage(imgB, 336, 336, out int tokenCountB);

        Assert.Equal(tokenCountA, tokenCountB);
        ValidateEmbeddingRigorously(tokensA, tokensB, tokenCountA, embedder.EmbeddingDim);
    }

    [Fact]
    public void Llama4_RealWeights_LoadsAndEmbedsImage()
    {
        string? path = FindModelPath("mmproj-llama-4-scout-17b-16e-instruct-f16.gguf");
        if (path is null) return;

        using var embedder = UnifiedVisionPipeline.Open(path);
        Assert.NotNull(embedder);

        // Native image_size=336, patch_size=14 (24x24 patch grid) -- confirmed via list-metadata.
        var (imgA, imgB) = CreateTestImagePair(336, 336);
        var tokensA = embedder.EmbedImage(imgA, 336, 336, out int tokenCountA);
        var tokensB = embedder.EmbedImage(imgB, 336, 336, out int tokenCountB);

        Assert.Equal(tokenCountA, tokenCountB);
        ValidateEmbeddingRigorously(tokensA, tokensB, tokenCountA, embedder.EmbeddingDim);
    }

    [Fact]
    public void Nemotron_RealWeights_LoadsAndEmbedsImage()
    {
        string? path = FindModelPath("mmproj-nemotron-nano-12b-v2-vl-bf16.gguf");
        if (path is null) return;

        using var embedder = UnifiedVisionPipeline.Open(path);
        Assert.NotNull(embedder);

        // patch_size=16; using 256x256 (16x16 patch grid) rather than the native 512 max to keep
        // the test fast -- dynamic/tiled preprocessors in this codebase generally accept smaller
        // inputs than their configured max.
        var (imgA, imgB) = CreateTestImagePair(256, 256);
        var tokensA = embedder.EmbedImage(imgA, 256, 256, out int tokenCountA);
        var tokensB = embedder.EmbedImage(imgB, 256, 256, out int tokenCountB);

        Assert.Equal(tokenCountA, tokenCountB);
        ValidateEmbeddingRigorously(tokensA, tokensB, tokenCountA, embedder.EmbeddingDim);
    }

    [Fact]
    public void Gemma4V_E4B_RealWeights_LoadsAndEmbedsImage()
    {
        string? path = FindModelPath("gemma-4-E4B-it-mmproj.gguf");
        if (path is null) return;

        using var embedder = UnifiedVisionPipeline.Open(path);
        Assert.NotNull(embedder);

        // Native image_size=224, patch_size=16 (14x14 patch grid) -- confirmed via list-metadata.
        var (imgA, imgB) = CreateTestImagePair(224, 224);
        var tokensA = embedder.EmbedImage(imgA, 224, 224, out int tokenCountA);
        var tokensB = embedder.EmbedImage(imgB, 224, 224, out int tokenCountB);

        Assert.Equal(tokenCountA, tokenCountB);
        ValidateEmbeddingRigorously(tokensA, tokensB, tokenCountA, embedder.EmbeddingDim);
    }
}
