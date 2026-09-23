using Xunit;
using OpenTail.Stingray.Diffusion.TextEncoders;

namespace OpenTail.Stingray.Tests.Diffusion;

// Scratch diagnostic (2026-09-21): T5 encoder bisection test. Dumps internal checkpoints
// (E0, A0, F0, final) for direct comparison against the C++ reference's own dumps.
// NOT for commit -- temporary diagnostic, delete after T5 divergence is isolated and fixed.
public sealed class ZZ_ScratchT5BisectionDumpTest
{
    private readonly ITestOutputHelper _output;
    public ZZ_ScratchT5BisectionDumpTest(ITestOutputHelper output) => _output = output;

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
            if (File.Exists(c) || Directory.Exists(c)) return Path.GetFullPath(c);
        return relativePath;
    }

    [Fact]
    public void Scratch_DumpT5Checkpoints()
    {
        string t5Path = P(Path.Combine("models", "flux1-schnell", "t5xxl_fp8_e4m3fn.safetensors"));
        string t5TokPath = P(Path.Combine("models", "flux1-schnell", "tokenizer_t5", "tokenizer.json"));
        string dumpDir = P(Path.Combine("docs", "diffusion-samples"));

        if (!File.Exists(t5Path) || !File.Exists(t5TokPath))
        {
            _output.WriteLine("Missing T5 checkpoint or tokenizer, skipping.");
            return;
        }

        Directory.CreateDirectory(dumpDir);

        // Set the dump prefix env var BEFORE loading the encoder
        string dumpPrefix = Path.Combine(dumpDir, "zz_scratch_csharp_t5");
        Environment.SetEnvironmentVariable("STINGRAY_T5_DUMP_PREFIX", dumpPrefix);

        try
        {
            using var t5 = new T5Encoder(t5Path);
            var tokenizer = T5Tokenizer.FromFile(t5TokPath, maxLen: 77);

            string prompt = "a red apple on a wooden table";
            _output.WriteLine($"Encoding prompt: \"{prompt}\"");

            var tokens = tokenizer.Tokenize(prompt);
            _output.WriteLine($"Raw tokens ({tokens.Length}): {string.Join(", ", tokens)}");
            
            // Pad to 77 tokens (SD3's T5MaxTokens) with PAD token (0)
            var paddedTokens = new int[77];
            Array.Copy(tokens, paddedTokens, Math.Min(tokens.Length, 77));
            for (int i = tokens.Length; i < 77; i++)
                paddedTokens[i] = 0; // PAD token
            
            _output.WriteLine($"Padded to 77 tokens");

            var encoded = t5.Encode(paddedTokens, tokens.Length); // validLen = actual token count before padding
            _output.WriteLine($"Encoded shape: [77, 4096] = {encoded.Length} floats");

            // Also encode the empty string (uncond)
            _output.WriteLine("\nEncoding empty string (uncond)...");
            var emptyTokens = tokenizer.Tokenize("");
            _output.WriteLine($"Raw empty tokens ({emptyTokens.Length}): {string.Join(", ", emptyTokens)}");
            
            var paddedEmptyTokens = new int[77];
            Array.Copy(emptyTokens, paddedEmptyTokens, Math.Min(emptyTokens.Length, 77));
            for (int i = emptyTokens.Length; i < 77; i++)
                paddedEmptyTokens[i] = 0;
            
            var emptyEncoded = t5.Encode(paddedEmptyTokens, emptyTokens.Length);
            _output.WriteLine($"Empty encoded shape: [77, 4096] = {emptyEncoded.Length} floats");

            _output.WriteLine($"\nDumps written to: {dumpDir}");
            _output.WriteLine("Check for files: zz_scratch_csharp_t5_E0_*.bin, A0_*.bin, F0_*.bin, final_*.bin");
        }
        finally
        {
            Environment.SetEnvironmentVariable("STINGRAY_T5_DUMP_PREFIX", null);
        }
    }

    [Fact]
    public void Scratch_CompareT5CheckpointsAgainstReference()
    {
        string dumpDir = P(Path.Combine("docs", "diffusion-samples"));
        
        // Expected C++ dumps (to be created)
        string cppE0Cond = Path.Combine(dumpDir, "zz_scratch_cpp_t5_E0_cond.bin");
        string cppA0Cond = Path.Combine(dumpDir, "zz_scratch_cpp_t5_A0_cond.bin");
        string cppF0Cond = Path.Combine(dumpDir, "zz_scratch_cpp_t5_F0_cond.bin");
        string cppFinalCond = Path.Combine(dumpDir, "zz_scratch_cpp_t5_final_cond.bin");

        // C# dumps (from previous test)
        string csE0Cond = Path.Combine(dumpDir, "zz_scratch_csharp_t5_E0_0.bin");
        string csA0Cond = Path.Combine(dumpDir, "zz_scratch_csharp_t5_A0_0.bin");
        string csF0Cond = Path.Combine(dumpDir, "zz_scratch_csharp_t5_F0_0.bin");
        string csFinalCond = Path.Combine(dumpDir, "zz_scratch_csharp_t5_final_0.bin");

        var checkpoints = new[] { ("E0", cppE0Cond, csE0Cond), ("A0", cppA0Cond, csA0Cond), 
                                   ("F0", cppF0Cond, csF0Cond), ("final", cppFinalCond, csFinalCond) };

        foreach (var (name, cppPath, csPath) in checkpoints)
        {
            if (!File.Exists(cppPath))
            {
                _output.WriteLine($"[{name}] C++ dump not found: {cppPath} - skipping comparison");
                continue;
            }

            if (!File.Exists(csPath))
            {
                _output.WriteLine($"[{name}] C# dump not found: {csPath} - skipping comparison");
                continue;
            }

            var cppData = LoadBin(cppPath);
            var csData = LoadBin(csPath);

            _output.WriteLine($"\n=== Checkpoint {name} ===");
            Stats(name, csData, cppData);
        }
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
}
