
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real-weights coverage for the real-BPE-tokenizer-backed <see cref="QwenAsrTokenizer"/> (see
/// docs/audio-review-progress.md's QwenASR section for how the previous fake char-level vocab
/// and fictional timestamp-token range were found and replaced).
/// </summary>
public sealed class QwenAsrTokenizerTests : HeavyTestBase
{
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
    public void RealTokenizer_EncodesAndDecodes_ChatMlPromptRoundTrip()
    {
        string? path = FindRepoFile("models/qwen3-asr-0.6b-q4_k.gguf");
        Assert.SkipUnless(path != null, "models/qwen3-asr-0.6b-q4_k.gguf not found");

        using var weights = new QwenAsrWeights(path!);
        var tokenizer = new QwenAsrTokenizer(weights);

        // Real template, matching examples/audio.cpp's Qwen3ASRTextTokenizer::build_prompt (fixed
        // 2026-09-12 -- the previous version of this test asserted the OLD, unverified invented
        // template, which turned out to be the real cause of a severe correctness bug: see
        // docs/00-current-work.md's 2026-09-12 entry. `language` seeds "language {lang}<asr_text>"
        // as a forced assistant-turn prefix, not prose inside the user turn.
        string prompt = tokenizer.FormatPrompt(numAudioTokens: 5, language: "en");
        int[] tokens = tokenizer.Encode(prompt);

        Assert.NotEmpty(tokens);
        // Real special tokens should collapse to single ids, not be BPE-shredded per character.
        Assert.Contains(151644, tokens); // <|im_start|>
        Assert.Contains(weights.AudioStartTokenId, tokens);
        Assert.Contains(weights.AudioPadTokenId, tokens);
        Assert.Contains(weights.AudioEndTokenId, tokens);
        // Exactly 5 audio_pad tokens for numAudioTokens=5, not fewer (which would mean the
        // special token got merged into surrounding text instead of recognized standalone).
        int padCount = 0;
        foreach (var t in tokens) if (t == weights.AudioPadTokenId) padCount++;
        Assert.Equal(5, padCount);

        // <asr_text> must also collapse to a single real token, not get BPE-shredded (the exact
        // regression found and fixed this session -- see QwenAsrWeights.BuildTokenizer's doc
        // comment).
        string decoded = tokenizer.Decode(tokens);
        Assert.Contains("language en", decoded);
        Assert.Contains("<asr_text>", decoded);
    }

    [Fact]
    public void RealTokenizer_DecodeWithTimestamps_ProducesSingleSegmentSpanningDuration()
    {
        string? path = FindRepoFile("models/qwen3-asr-0.6b-q4_k.gguf");
        Assert.SkipUnless(path != null, "models/qwen3-asr-0.6b-q4_k.gguf not found");

        using var weights = new QwenAsrWeights(path!);
        var tokenizer = new QwenAsrTokenizer(weights);

        int[] tokens = tokenizer.Encode("hello world");
        var (text, segments) = tokenizer.DecodeWithTimestamps(tokens, TimeSpan.Zero, TimeSpan.FromSeconds(2));

        Assert.False(string.IsNullOrWhiteSpace(text));
        Assert.Single(segments);
        Assert.Equal(TimeSpan.Zero, segments[0].Start);
        Assert.Equal(TimeSpan.FromSeconds(2), segments[0].End);
    }
}
