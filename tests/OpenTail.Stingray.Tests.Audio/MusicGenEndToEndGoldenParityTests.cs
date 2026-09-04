using OpenTail.Stingray.Audio.MusicGen;
using OpenTail.Stingray.Audio.Primitives;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Full end-to-end chained numeric golden verification for MusicGen: real text prompt -&gt; real
/// T5 text encoder -&gt; real cross-attention conditioning -&gt; real greedy delayed-pattern decoder
/// generation -&gt; real EnCodec-32kHz decode -&gt; mono PCM, run through the ACTUAL
/// <see cref="MusicGenGenerator.Generate"/> entry point end to end and compared against
/// `scratch-llamacpp-ref/musicgen_e2e_golden.py`, which chains the SAME three real stages already
/// independently golden-verified in <see cref="MusicGenTextEncoderGoldenParityTests"/>,
/// <see cref="MusicGenDecoderGoldenParityTests"/>, and
/// <see cref="MusicGenEncodecDecoderGoldenParityTests"/> into one numpy program.
///
/// Closes gap 3/3 of the 2026-09-04 MusicGen numeric-parity closure -- this is the one check the
/// isolated per-component tests cannot catch: that the real stages compose correctly (T5 hidden
/// states flow through the real `enc_to_dec_proj` into cross-attention with the right shapes/
/// values, the delayed-pattern greedy loop produces the right codebook token grid, and that clean
/// grid feeds the real EnCodec decoder) using the SAME public generation entry point real callers
/// use, not a hand-wired test-only pipeline.
///
/// <para>Generation config chosen so the oracle needs no RNG: `guidanceScale = 1.0f` (exactly at
/// the CFG threshold -- <see cref="MusicGenGenerator"/>'s `useCfg = guidanceScale &gt; 1.0f` is
/// false, disabling the unconditional branch entirely) and `topK = 1` (deterministic greedy
/// argmax). `durationSeconds = 2 / MusicGenConfig.FrameRate` requests exactly 2 frames.</para>
/// </summary>
public sealed class MusicGenEndToEndGoldenParityTests : HeavyTestBase
{
    private const string Prompt = "electronic dance music";
    private static readonly int[] ExpectedPromptIds = [3031, 2595, 723, 1];

    private static string? FindRepoFile(string relativePath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativePath);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void Generate_RealWeights_MatchesGoldenChainedOutput()
    {
        string? modelPath = FindRepoFile("models/musicgen-small/musicgen-small.safetensors");
        Assert.SkipUnless(modelPath != null, "models/musicgen-small/musicgen-small.safetensors not found");

        string? tokenizerPath = FindRepoFile("models/musicgen-small/t5-base-tokenizer.json");
        Assert.SkipUnless(tokenizerPath != null, "models/musicgen-small/t5-base-tokenizer.json not found");

        string? codesPath = FindRepoFile("scratch-llamacpp-ref/musicgen_e2e_golden_codes.txt");
        string? pcmPath = FindRepoFile("scratch-llamacpp-ref/musicgen_e2e_golden_pcm.txt");
        Assert.SkipUnless(codesPath != null && pcmPath != null,
            "golden MusicGen e2e files not found (re-run scratch-llamacpp-ref/musicgen_e2e_golden.py)");

        // Guard against tokenizer drift: the oracle's fixed prompt ids were captured once via the
        // real t5-base tokenizer.json using the `tokenizers` Python library; assert the real C#
        // T5Tokenizer produces the IDENTICAL ids for the identical prompt before trusting the rest
        // of the chain.
        var tokenizer = T5Tokenizer.FromFile(tokenizerPath!);
        var actualPromptIds = tokenizer.Tokenize(Prompt);
        Assert.Equal(ExpectedPromptIds, actualPromptIds);

        var codeLines = File.ReadAllText(codesPath!).Trim().Split('\n');
        var goldenCodes = new int[codeLines.Length][];
        for (int i = 0; i < codeLines.Length; i++) goldenCodes[i] = Array.ConvertAll(codeLines[i].Split(','), int.Parse);
        int frames = goldenCodes[0].Length;

        var pcmLines = File.ReadAllText(pcmPath!).Split('\n');
        int goldenLen = int.Parse(pcmLines[0].Trim());
        var goldenParts = pcmLines[1].Trim().Split(',');
        Assert.Equal(goldenLen, goldenParts.Length);
        var golden = new float[goldenLen];
        for (int i = 0; i < golden.Length; i++) golden[i] = float.Parse(goldenParts[i]);

        using var loader = SafetensorsLoader.Open(modelPath!);
        var textEncoderWeights = MusicGenTextEncoderWeights.Load(loader);
        var transformerWeights = new MusicGenTransformerWeights(loader);
        var codecWeights = MusicGenEncodecDecoderWeights.Load(loader);

        var generator = new MusicGenGenerator(textEncoderWeights, tokenizer, transformerWeights, codecWeights);

        float durationSeconds = (float)frames / MusicGenConfig.FrameRate;
        var pcm = generator.Generate(Prompt, durationSeconds, seed: 0, guidanceScale: 1.0f, topK: 1);

        Assert.Equal(goldenLen, pcm.Length);

        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < goldenLen; i++)
        {
            float a = pcm[i];
            float b = golden[i];
            dot += a * b;
            normA += a * a;
            normB += b * b;
        }
        double cosine = dot / (Math.Sqrt(normA) * Math.Sqrt(normB));
        Assert.True(cosine > 0.999, $"cosine similarity {cosine} too low vs golden MusicGen end-to-end output");
    }
}
