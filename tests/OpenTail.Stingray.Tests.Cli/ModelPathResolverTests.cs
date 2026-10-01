namespace OpenTail.Stingray.Tests.Cli;

public sealed class ModelPathResolverTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ot-resolver-" + Guid.NewGuid().ToString("N"));
    public ModelPathResolverTests() => Directory.CreateDirectory(_dir);
    public void Dispose() { try { Directory.Delete(_dir, true); } catch (IOException) { } }

    private string Touch(string name) { string p = Path.Combine(_dir, name); File.WriteAllBytes(p, [0]); return p; }

    [Fact]
    public void FindFirst_ReturnsTheFirstExistingCandidateInPreferenceOrder()
    {
        string second = Touch("b.gguf");
        string third = Touch("c.gguf");
        Assert.Equal(second, ModelPathResolver.FindFirst([Path.Combine(_dir, "a.gguf"), second, third]));
    }

    [Fact]
    public void FindFirst_ReturnsNullWhenNothingExists() =>
        Assert.Null(ModelPathResolver.FindFirst([Path.Combine(_dir, "nope.gguf")]));

    [Fact]
    public void FindFirst_WithDirMarker_RequiresTheMarkerFileInsideTheDirectory()
    {
        string withMarker = Path.Combine(_dir, "good"); Directory.CreateDirectory(withMarker);
        File.WriteAllBytes(Path.Combine(withMarker, "model.safetensors"), [0]);
        string withoutMarker = Path.Combine(_dir, "empty"); Directory.CreateDirectory(withoutMarker);

        Assert.Equal(withMarker, ModelPathResolver.FindFirst([withoutMarker, withMarker], "model.safetensors"));
        Assert.Null(ModelPathResolver.FindFirst([withoutMarker], "model.safetensors"));
    }

    [Fact]
    public void Resolve_ThrowsAnActionableMessageWhenNothingIsFound()
    {
        var search = new ModelSearch("Foo", "a Foo .gguf checkpoint", "models/foo.gguf", [Path.Combine(_dir, "missing.gguf")], Extra: "Needs --ref-audio.");
        var ex = Assert.Throws<ArgumentException>(() => ModelPathResolver.Resolve(search));
        Assert.Contains("No Foo model found", ex.Message, StringComparison.Ordinal);
        Assert.Contains("--model (-m)", ex.Message, StringComparison.Ordinal);
        Assert.Contains("models/foo.gguf", ex.Message, StringComparison.Ordinal);
        Assert.EndsWith("Needs --ref-audio.", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void RequireExisting_PassesThroughAnExistingFileAndRejectsAMissingOne()
    {
        string p = Touch("k.gguf");
        Assert.Equal(p, ModelPathResolver.RequireExisting("Kokoro", p));
        var ex = Assert.Throws<ArgumentException>(() => ModelPathResolver.RequireExisting("Kokoro", Path.Combine(_dir, "gone.gguf")));
        Assert.Contains("Kokoro model file not found", ex.Message, StringComparison.Ordinal);
    }
}
