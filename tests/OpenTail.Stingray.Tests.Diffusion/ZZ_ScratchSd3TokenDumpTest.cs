using Xunit;
using OpenTail.Stingray.Diffusion.TextEncoders;

namespace OpenTail.Stingray.Tests.Diffusion;

// Scratch diagnostic (2026-09-21): NOT for commit -- deleted after use. Dumps this port's own
// CLIP/T5 tokenizer output for direct comparison against the real stable-diffusion.cpp reference's
// token IDs (captured via a temporary SD_DUMP_TOKENS hook in examples/stable-diffusion.cpp).
public sealed class ZZ_ScratchSd3TokenDumpTest
{
    private readonly ITestOutputHelper _output;
    public ZZ_ScratchSd3TokenDumpTest(ITestOutputHelper output) => _output = output;

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

    private void Dump(string label, int[] ids)
    {
        string s = $"[{label}] (n={ids.Length}): {string.Join(",", ids)}";
        _output.WriteLine(s);
        Console.WriteLine(s);
    }

    [Fact]
    public void Scratch_DumpTokens()
    {
        string clipTokPath = P(Path.Combine("models", "flux1-schnell", "tokenizer_clip", "tokenizer.json"));
        string t5TokPath = P(Path.Combine("models", "flux1-schnell", "tokenizer_t5", "tokenizer.json"));

        if (!File.Exists(clipTokPath) || !File.Exists(t5TokPath))
        {
            _output.WriteLine("Missing tokenizer files, skipping.");
            return;
        }

        var clipTok = ClipTokenizer.FromFile(clipTokPath);
        var t5Tok = T5Tokenizer.FromFile(t5TokPath, maxLen: 77);

        foreach (var prompt in new[] { "a red apple on a wooden table", "" })
        {
            _output.WriteLine($"=== prompt: \"{prompt}\" ===");
            Dump("clip", clipTok.Tokenize(prompt));
            Dump("t5", t5Tok.Tokenize(prompt));
        }
    }
}
