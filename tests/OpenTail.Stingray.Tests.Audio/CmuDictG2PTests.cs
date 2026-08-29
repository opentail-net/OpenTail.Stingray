
namespace OpenTail.Stingray.Tests.Audio;

public sealed class CmuDictG2PTests
{
    [Theory]
    [InlineData("hello")]
    [InlineData("world")]
    [InlineData("running")]
    [InlineData("native")]
    [InlineData("entirely")]
    [InlineData("python")]
    [InlineData("anywhere")]
    public void TryLookup_RealDictionaryWords_Succeed(string word)
    {
        bool found = CmuDictG2P.TryLookup(word, out string ipa);
        Assert.True(found, $"'{word}' should be in the real cmudict dictionary");
        Assert.False(string.IsNullOrEmpty(ipa));
    }

    [Fact]
    public void TryLookup_Hello_ProducesRealisticIpa()
    {
        Assert.True(CmuDictG2P.TryLookup("hello", out string ipa));
        // Real cmudict entry: "hello  HH AH0 L OW1" -> IPA should carry the primary-stress mark
        // on the final syllable and a schwa (unstressed AH0), not the stressed ʌ.
        Assert.Contains("ˈoʊ", ipa);
        Assert.Contains("ə", ipa);
    }

    [Fact]
    public void TryLookup_UnknownWord_ReturnsFalse()
    {
        // "opentail" is a made-up compound not in cmudict.
        Assert.False(CmuDictG2P.TryLookup("zzqxnotaword", out _));
    }

    [Fact]
    public void TryLookup_IsCaseInsensitive()
    {
        Assert.True(CmuDictG2P.TryLookup("HELLO", out string upper));
        Assert.True(CmuDictG2P.TryLookup("hello", out string lower));
        Assert.Equal(lower, upper);
    }

    [Fact]
    public void Dictionary_HasRealisticCoverage()
    {
        // cmudict.dict has ~135k entries; a loose lower bound catches a truncated/corrupt embed.
        int found = 0;
        string[] sample = ["the", "a", "is", "of", "to", "and", "in", "that", "have", "for"];
        foreach (var w in sample)
            if (CmuDictG2P.TryLookup(w, out _)) found++;
        Assert.Equal(sample.Length, found);
    }
}
