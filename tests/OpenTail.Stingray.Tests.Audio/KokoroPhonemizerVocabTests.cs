using OpenTail.Stingray.Audio.Kokoro;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Verifies KokoroPhonemizer.Tokenize produces the REAL Kokoro-82M model's own token ids, not a
/// locally-guessed enumeration. "hello" is a real, independent cross-check: cmudict's real entry
/// ("hello  HH AH0 L OW1") -&gt; this port's IPA ("hˈələˈoʊ"... -&gt; "həlˈoʊ") -&gt; real vocab ids
/// [50, 83, 54, 156, 57, 135] for [h, ə, l, ˈ, o, ʊ] -- which is EXACTLY the first six ids of
/// scratch-llamacpp-ref/kokoro_golden.py's own hand-picked fixed test input
/// ([0, 50, 83, 54, 156, 57, 135, 3, 16, 65, 156, 0]), an independently authored golden fixture
/// that was never written with this test in mind. That match is strong evidence the real vocab
/// table transcribed from Kokoro-82M's published config.json is wired correctly end-to-end.
/// </summary>
public sealed class KokoroPhonemizerVocabTests
{
    [Fact]
    public void Tokenize_Hello_MatchesRealKokoroVocabIds()
    {
        var phonemizer = new KokoroPhonemizer();
        string ipa = phonemizer.TextToPhonemes("hello");
        int[] tokens = phonemizer.Tokenize(ipa);

        // [0]=BOS pad, then real per-character ids, then [^1]=EOS pad.
        int[] middle = tokens[1..^1];
        Assert.Equal([50, 83, 54, 156, 57, 135], middle);
    }
}
