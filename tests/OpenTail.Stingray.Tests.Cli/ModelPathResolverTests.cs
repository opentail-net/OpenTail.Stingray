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

    private static (int? Result, string Out, string Err) Check(string? path)
    {
        var originalOut = Console.Out;
        var originalErr = Console.Error;
        var o = new StringWriter();
        var e = new StringWriter();
        Console.SetOut(o);
        Console.SetError(e);
        try { return (ModelPathResolver.TryRequireModelFile(path, out int code) ? null : code, o.ToString(), e.ToString()); }
        finally { Console.SetOut(originalOut); Console.SetError(originalErr); }
    }

    [Fact]
    public void CheckModelFile_ExistingFile_IsAccepted_WithoutOutput()
    {
        var (result, o, e) = Check(Touch("m.gguf"));
        Assert.Null(result);
        Assert.Equal("", o + e);
    }

    [Fact]
    public void CheckModelFile_NoPath_IsAUsageError_OnStderr()
    {
        var (result, o, e) = Check(null);
        Assert.Equal(64, result);
        Assert.Contains("-m <path>", e, StringComparison.Ordinal);
        Assert.Equal("", o);
    }

    [Fact]
    public void CheckModelFile_MissingFile_IsNoInput_AndNamesThePath()
    {
        string missing = Path.Combine(_dir, "gone.gguf");
        var (result, o, e) = Check(missing);
        Assert.Equal(66, result);
        Assert.Contains("gone.gguf", e, StringComparison.Ordinal);
        Assert.Equal("", o);
    }

    [Fact]
    public void Commands_DistinguishAMissingModelArgumentFromAMissingFile()
    {
        static int Run(ICommand command, params string[] args)
        {
            var originalOut = Console.Out; var originalErr = Console.Error;
            Console.SetOut(new StringWriter()); Console.SetError(new StringWriter());
            try { return command.Run(args, CancellationToken.None); }
            finally { Console.SetOut(originalOut); Console.SetError(originalErr); }
        }
        string text = Touch("corpus.txt");   // perplexity validates -f before it looks at the model
        var factories = new (string Name, Func<ICommand> Create, string[] Extra)[]
        {
            ("list-tensors", () => new ListTensorsCommand(), []),
            ("list-metadata", () => new ListMetadataCommand(), []),
            ("admit-arch", () => new AdmitArchCommand(), []),
            ("perplexity", () => new PerplexityCommand(), ["-f", text]),
        };
        foreach (var (name, create, extra) in factories)
        {
            Assert.True(Run(create(), extra) == 64, $"{name}: no -m should be a usage error");
            Assert.True(Run(create(), [.. extra, "-m", Path.Combine(_dir, "gone.gguf")]) == 66, $"{name}: a missing file should be NoInput");
        }
    }
}
