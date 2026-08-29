
namespace OpenTail.Stingray.Tests.Cli;

/// <summary>
/// `list-models` is read-only enumeration, so the cases that matter are the awkward ones: an empty
/// directory, a missing directory, and a file that looks like a model but is not.
/// </summary>
public sealed class ListModelsCommandTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ot-models-" + Guid.NewGuid().ToString("N"));

    public ListModelsCommandTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { if (Directory.Exists(_dir)) Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private (int Exit, string Output) Run(params string[] args)
    {
        var command = (ICommand)new ListModelsCommand();
        var original = Console.Out;
        var writer = new StringWriter();
        Console.SetOut(writer);
        try { return (command.Run(args, CancellationToken.None), writer.ToString()); }
        finally { Console.SetOut(original); }
    }

    [Fact]
    public void EmptyDirectoryIsNotAnError()
    {
        var (exit, output) = Run("-d", _dir);
        Assert.Equal(0, exit);
        Assert.Contains("No .gguf files", output, StringComparison.Ordinal);
    }

    [Fact]
    public void MissingDirectoryFails()
    {
        var (exit, _) = Run("-d", Path.Combine(_dir, "does-not-exist"));
        Assert.NotEqual(0, exit);
    }

    /// <summary>
    /// A truncated or corrupt .gguf is the case worth surfacing: present on disk, plausible name,
    /// and unusable. It must be REPORTED rather than crashing the listing or being silently hidden.
    /// </summary>
    [Fact]
    public void CorruptModelIsReportedRatherThanThrowing()
    {
        File.WriteAllText(Path.Combine(_dir, "broken.gguf"), "not actually a gguf");
        var (exit, output) = Run("-d", _dir, "--deep");
        Assert.Equal(0, exit);
        Assert.Contains("broken.gguf", output, StringComparison.Ordinal);
        Assert.Contains("UNREADABLE", output, StringComparison.Ordinal);
    }

    [Fact]
    public void ShallowListingDoesNotOpenTheFile()
    {
        File.WriteAllText(Path.Combine(_dir, "broken.gguf"), "not actually a gguf");
        var (exit, output) = Run("-d", _dir);
        Assert.Equal(0, exit);
        Assert.Contains("broken.gguf", output, StringComparison.Ordinal);
        Assert.DoesNotContain("UNREADABLE", output, StringComparison.Ordinal);
    }
}
