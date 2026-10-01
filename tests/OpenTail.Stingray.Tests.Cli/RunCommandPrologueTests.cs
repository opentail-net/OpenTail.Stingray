namespace OpenTail.Stingray.Tests.Cli;

/// <summary>
/// Characterization of <c>run</c>'s argument handling BEFORE a model is touched (docs: the 1,500-line Execute is being
/// split into phases; these pin the early-exit behaviour so the split cannot change it). Every case returns before any
/// model load, so no checkpoint is needed and nothing here can silently skip.
/// </summary>
public sealed class RunCommandPrologueTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ot-run-prologue-" + Guid.NewGuid().ToString("N"));
    public RunCommandPrologueTests() => Directory.CreateDirectory(_dir);
    public void Dispose() { try { Directory.Delete(_dir, true); } catch (IOException) { } }

    private static (int Exit, string Out, string Err) Run(params string[] args)
    {
        var command = (ICommand)new RunCommand();
        var originalOut = Console.Out;
        var originalErr = Console.Error;
        var stdout = new StringWriter();
        var stderr = new StringWriter();
        Console.SetOut(stdout);
        Console.SetError(stderr);
        try { return (command.Run(args, CancellationToken.None), stdout.ToString(), stderr.ToString()); }
        finally { Console.SetOut(originalOut); Console.SetError(originalErr); }
    }

    [Fact]
    public void Explain_WithoutAuto_IsRejectedBeforeAnythingElse()
    {
        var (exit, o, e) = Run("--explain", "-m", Path.Combine(_dir, "irrelevant.gguf"));
        Assert.Equal(1, exit);
        Assert.Contains("--explain requires --auto", e, StringComparison.Ordinal);
        Assert.DoesNotContain("--explain requires --auto", o, StringComparison.Ordinal);
    }

    [Fact]
    public void MissingPromptFile_IsReportedOnStderr()
    {
        string missing = Path.Combine(_dir, "no-such-prompt.txt");
        var (exit, o, e) = Run("-f", missing, "-m", Path.Combine(_dir, "irrelevant.gguf"));
        Assert.Equal(1, exit);
        Assert.Contains("Prompt file not found", e, StringComparison.Ordinal);
        Assert.Contains("no-such-prompt.txt", e, StringComparison.Ordinal);
        Assert.DoesNotContain("Prompt file not found", o, StringComparison.Ordinal);
    }

    [Fact]
    public void UnknownDevice_IsRejected()
    {
        var (exit, _, e) = Run("--device", "banana", "-m", Path.Combine(_dir, "irrelevant.gguf"));
        Assert.Equal(1, exit);
        Assert.Contains("Error:", e, StringComparison.Ordinal);
    }

    [Fact]
    public void NCpuMoe_IsRejectedWithTheRationale()
    {
        var (exit, _, e) = Run("--n-cpu-moe", "3", "-m", Path.Combine(_dir, "irrelevant.gguf"));
        Assert.Equal(1, exit);
        Assert.Contains("not supported yet", e, StringComparison.Ordinal);
        Assert.Contains("--cpu-moe", e, StringComparison.Ordinal);
    }

    [Fact]
    public void MissingModel_IsReportedOnStderr()
    {
        var (exit, o, e) = Run("-m", Path.Combine(_dir, "gone.gguf"), "-p", "hi");
        Assert.Equal(1, exit);
        Assert.Contains("No model file or package directory found", e, StringComparison.Ordinal);
        Assert.DoesNotContain("No model file", o, StringComparison.Ordinal);
    }

    [Fact]
    public void PromptFile_IsReadBeforeTheModelIsChecked()
    {
        // An existing prompt file is consumed (no "not found"), and the run then stops at the missing model.
        string prompt = Path.Combine(_dir, "prompt.txt");
        File.WriteAllText(prompt, "hello");
        var (exit, _, e) = Run("-f", prompt, "-m", Path.Combine(_dir, "gone.gguf"));
        Assert.Equal(1, exit);
        Assert.DoesNotContain("Prompt file not found", e, StringComparison.Ordinal);
        Assert.Contains("No model file or package directory found", e, StringComparison.Ordinal);
    }
}
