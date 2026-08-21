using System.IO;
using OpenTail.Stingray.Audio.QwenASR;
using Xunit;

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

        string prompt = tokenizer.FormatPrompt(numAudioTokens: 5, language: "en", taskInstruction: "Transcribe the audio speech into text.");
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

        string decoded = tokenizer.Decode(tokens);
        Assert.Contains("Transcribe", decoded);
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
