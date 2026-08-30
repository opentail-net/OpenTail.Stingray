
namespace OpenTail.Stingray.Tests.Audio.Fast;

/// <summary>SCRATCH/throwaway diagnostic. Delete after use.</summary>
public sealed class ParlerScratchTests : HeavyTestBase
{
    [Fact]
    public void Tokenizer_Loads_And_Encodes()
    {
        string? tokDir = FindRepoDir("scratch-llamacpp-ref/parler-tokenizer");
        Assert.SkipUnless(tokDir != null, "parler-tokenizer dir not found");

        var tokenizer = UnigramTokenizer.FromTokenizerJson(Path.Combine(tokDir!, "tokenizer.json"));
        var ids = tokenizer.Encode("A female speaker with a clear voice.");
        Assert.NotEmpty(ids);
    }

    private static string? FindRepoDir(string relativePath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativePath);
            if (Directory.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }
}
