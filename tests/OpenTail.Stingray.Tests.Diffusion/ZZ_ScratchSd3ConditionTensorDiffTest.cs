using Xunit;
using OpenTail.Stingray.Diffusion.SD3;

namespace OpenTail.Stingray.Tests.Diffusion;

// Scratch diagnostic (2026-09-21): NOT for commit -- deleted after use. Dumps this port's own
// assembled context/pooledY tensors and directly diffs them, per-region, against the real
// stable-diffusion.cpp reference's own dumped tensors (zz_scratch_cpp_cond_*.bin, produced by a
// temporary SD_DUMP_CONDITION_PREFIX hook in examples/stable-diffusion.cpp).
public sealed class ZZ_ScratchSd3ConditionTensorDiffTest
{
    private readonly ITestOutputHelper _output;
    public ZZ_ScratchSd3ConditionTensorDiffTest(ITestOutputHelper output) => _output = output;

    private static string P(string relativePath)
    {
        var candidates = new[]
        {
            Path.Combine("..", "..", "..", "..", "..", relativePath),
            Path.Combine("..", "..", "..", relativePath),
            relativePath,
            Path.Combine(@"c:\Git-Public\OpenTail.Stingray", relativePath)
        };
        foreach (var c in candidates)
            if (File.Exists(c)) return Path.GetFullPath(c);
        return relativePath;
    }

    private static float[] LoadBin(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var arr = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, arr, 0, bytes.Length);
        return arr;
    }

    private void Stats(string label, ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        int n = Math.Min(a.Length, b.Length);
        double sumA = 0, sumB = 0, sumSqA = 0, sumSqB = 0, dot = 0, normA = 0, normB = 0, maxAbsDiff = 0, sumSqDiff = 0;
        for (int i = 0; i < n; i++)
        {
            float av = a[i], bv = b[i];
            sumA += av; sumB += bv;
            sumSqA += av * av; sumSqB += bv * bv;
            dot += (double)av * bv;
            normA += (double)av * av; normB += (double)bv * bv;
            double diff = Math.Abs(av - bv);
            if (diff > maxAbsDiff) maxAbsDiff = diff;
            sumSqDiff += diff * diff;
        }
        double cos = dot / (Math.Sqrt(normA) * Math.Sqrt(normB) + 1e-12);
        double rms = Math.Sqrt(sumSqDiff / n);
        _output.WriteLine($"[{label}] n={n} A(mean={sumA / n:F4} std={Math.Sqrt(sumSqA / n - (sumA / n) * (sumA / n)):F4}) " +
                          $"B(mean={sumB / n:F4} std={Math.Sqrt(sumSqB / n - (sumB / n) * (sumB / n)):F4}) " +
                          $"cosine={cos:F6} maxAbsDiff={maxAbsDiff:E4} rmsDiff={rms:E4}");
        Console.WriteLine($"[{label}] n={n} A(mean={sumA / n:F4} std={Math.Sqrt(sumSqA / n - (sumA / n) * (sumA / n)):F4}) " +
                          $"B(mean={sumB / n:F4} std={Math.Sqrt(sumSqB / n - (sumB / n) * (sumB / n)):F4}) " +
                          $"cosine={cos:F6} maxAbsDiff={maxAbsDiff:E4} rmsDiff={rms:E4}");
    }

    [Fact]
    public void Scratch_CompareContextAndPooledY()
    {
        string ditPath = P(Path.Combine("models", "sd3.5_medium-Q4_K_M.gguf"));
        string clipLPath = P(Path.Combine("models", "sd35-medium-aux", "text_encoder", "model.fp16.safetensors"));
        string clipGPath = P(Path.Combine("models", "sd35-medium-aux", "text_encoder_2", "model.fp16.safetensors"));
        string vaePath = P(Path.Combine("models", "sd35-medium-aux", "vae", "diffusion_pytorch_model.safetensors"));
        string tokPath = P(Path.Combine("models", "flux1-schnell", "tokenizer_clip", "tokenizer.json"));
        string t5Path = P(Path.Combine("models", "flux1-schnell", "t5xxl_fp8_e4m3fn.safetensors"));
        string t5TokPath = P(Path.Combine("models", "flux1-schnell", "tokenizer_t5", "tokenizer.json"));

        string cppContextCond = P(Path.Combine("docs", "diffusion-samples", "zz_scratch_cpp_cond_realt5_context_cond.bin"));
        string cppPooledCond = P(Path.Combine("docs", "diffusion-samples", "zz_scratch_cpp_cond_realt5_pooledY_cond.bin"));
        string cppContextUncond = P(Path.Combine("docs", "diffusion-samples", "zz_scratch_cpp_cond_realt5_context_uncond.bin"));
        string cppPooledUncond = P(Path.Combine("docs", "diffusion-samples", "zz_scratch_cpp_cond_realt5_pooledY_uncond.bin"));

        if (!File.Exists(ditPath) || !File.Exists(cppContextCond))
        {
            _output.WriteLine("Missing checkpoints or C++ dump files, skipping.");
            return;
        }

        using var pipeline = Sd3Pipeline.LoadSeparate(clipLPath, clipGPath, ditPath, vaePath, tokPath, backend: null,
            t5EncoderPath: File.Exists(t5Path) ? t5Path : null,
            t5TokenizerPath: File.Exists(t5TokPath) ? t5TokPath : null);

        var (condContext, condPooledY, numTextTokens) = pipeline.EncodePromptForTesting("a red apple on a wooden table");
        var (uncondContext, uncondPooledY, _) = pipeline.EncodePromptForTesting("");

        _output.WriteLine($"C# numTextTokens={numTextTokens} (77 CLIP + T5) -- total context tokens = 77 + {numTextTokens - 77} T5");

        var cppContextCondArr = LoadBin(cppContextCond);
        var cppPooledCondArr = LoadBin(cppPooledCond);
        var cppContextUncondArr = LoadBin(cppContextUncond);
        var cppPooledUncondArr = LoadBin(cppPooledUncond);

        _output.WriteLine($"C++ context floats={cppContextCondArr.Length} (implies {cppContextCondArr.Length / 4096} tokens x 4096 channels, C++ tensor is [channel-fastest], ours is [token-major] -- see note below)");

        // NOTE: C++ sd::Tensor stores [hidden=4096 fastest, token, batch] -- i.e. for a given token,
        // all 4096 channels are contiguous... actually shape=[4096,154,1] means ne0=4096 (fastest),
        // ne1=154 (token). So flat index = channel + 4096*token -- SAME convention as our C# arrays
        // (token-major, channel-contiguous within a token row) since our BuildContext also stores
        // [token, channel] with channel fastest per row. Direct comparison should be valid as-is.

        Stats("pooledY_cond_FULL_2048", condPooledY, cppPooledCondArr);
        Stats("pooledY_cond_CLIP-L_0-767", condPooledY.AsSpan(0, 768), cppPooledCondArr.AsSpan(0, 768));
        Stats("pooledY_cond_CLIP-G_768-2047", condPooledY.AsSpan(768, 1280), cppPooledCondArr.AsSpan(768, 1280));

        Stats("pooledY_uncond_FULL_2048", uncondPooledY, cppPooledUncondArr);

        int totalTokens = numTextTokens; // 77 (CLIP) + T5MaxTokens
        int t5Tokens = totalTokens - 77;
        _output.WriteLine($"Comparing context: our totalTokens={totalTokens}, cpp totalTokens={cppContextCondArr.Length / 4096}");

        int compareTokens = Math.Min(totalTokens, cppContextCondArr.Length / 4096);
        Stats("context_cond_FULL", condContext.AsSpan(0, compareTokens * 4096), cppContextCondArr.AsSpan(0, compareTokens * 4096));

        // CLIP rows (0..76)
        Stats("context_cond_CLIProws_0-76_ch0-767_CLIPL", condContext.AsSpan(0, 77 * 4096), cppContextCondArr.AsSpan(0, 77 * 4096));

        if (compareTokens > 77)
        {
            int t5Compare = compareTokens - 77;
            Stats("context_cond_T5rows", condContext.AsSpan(77 * 4096, t5Compare * 4096), cppContextCondArr.AsSpan(77 * 4096, t5Compare * 4096));
        }
    }
}
