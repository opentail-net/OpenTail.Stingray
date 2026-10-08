using OpenTail.Stingray.Engine.Verification;

namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

public sealed class ModelLocatorTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "stingray-locator-" + Guid.NewGuid().ToString("N"));

    public ModelLocatorTests() => Directory.CreateDirectory(_root);

    public void Dispose() { try { Directory.Delete(_root, recursive: true); } catch { /* best effort */ } }

    private string Make(string relativeDir, string? file = null)
    {
        string dir = Path.Combine(_root, relativeDir);
        Directory.CreateDirectory(dir);
        if (file is not null) File.WriteAllText(Path.Combine(dir, file), "x");
        return dir;
    }

    [Fact]
    public void Find_ReturnsTheFirstNameThatExists_TryingNamesInOrder()
    {
        string a = Make("a", "second.gguf");
        string b = Make("b", "first.gguf");
        // "first.gguf" is wanted before "second.gguf", so the file in b wins even though a is listed first.
        Assert.Equal(Path.Combine(b, "first.gguf"), ModelLocator.Find([a, b], ["first.gguf", "second.gguf"]));
        Assert.Equal(Path.Combine(a, "second.gguf"), ModelLocator.Find([a, b], ["missing.gguf", "second.gguf"]));
    }

    [Fact]
    public void Find_ReturnsNull_WhenNothingMatches_AndTheMissNamesWhatWasWantedAndWhereItLooked()
    {
        string a = Make("a");
        Assert.Null(ModelLocator.Find([a], ["nope.gguf"]));
        string text = ModelLocator.DescribeMiss(["nope.gguf", "alt.gguf"], [a]);
        Assert.Contains("nope.gguf", text);
        Assert.Contains("alt.gguf", text);
        Assert.Contains(a, text);
        Assert.Contains("STINGRAY_MODEL_DIRS", text);
    }

    [Fact]
    public void Roots_ListsOnlyExistingFolders_WithoutDuplicates_AndHonoursTheEnvironmentVariable()
    {
        string extra = Make("env-models");
        string missing = Path.Combine(_root, "does-not-exist");
        string? old = Environment.GetEnvironmentVariable("STINGRAY_MODEL_DIRS");
        try
        {
            Environment.SetEnvironmentVariable("STINGRAY_MODEL_DIRS", string.Join(Path.PathSeparator, extra, missing, extra));
            var roots = ModelLocator.Roots();
            Assert.Equal(extra, roots[0]);                                   // highest priority
            Assert.DoesNotContain(missing, roots);
            Assert.Equal(roots.Count, roots.Distinct(StringComparer.OrdinalIgnoreCase).Count());
            Assert.All(roots, r => Assert.True(Directory.Exists(r), r));
        }
        finally { Environment.SetEnvironmentVariable("STINGRAY_MODEL_DIRS", old); }
    }

    [Fact]
    public void Roots_FindsModelsAndModelsUnderModelsInAnAncestorOfTheStartDirectory()
    {
        string nested = Make(Path.Combine("repo", "models", "_models"), "ancestor.gguf");
        string start = Make(Path.Combine("repo", "tests", "bin", "Debug"));
        var roots = ModelLocator.Roots(start);
        Assert.Contains(nested, roots);
        Assert.Equal(Path.Combine(nested, "ancestor.gguf"), ModelLocator.Find(roots, ["ancestor.gguf"]));
    }
}
