
namespace OpenTail.Stingray.Tests.Cli;

/// <summary>
/// `show-template` answers "what does the model actually receive?" — a question otherwise only
/// answerable by running a generation and inferring the formatting backwards from the output.
/// These pin the failure modes rather than the happy path, since the reference model is not
/// guaranteed present in every environment.
/// </summary>
public sealed class ShowTemplateCommandTests
{
    private static (int Exit, string Out, string Err) Run(params string[] args)
    {
        var command = (ICommand)new ShowTemplateCommand();
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
    public void RequiresAModelPath()
    {
        var (exit, _, _) = Run();
        Assert.NotEqual(0, exit);
    }

    /// <summary>
    /// A missing or unparseable model must report rather than throw, and must use a non-zero exit
    /// so a script can act on it.
    /// </summary>
    [Fact]
    public void MissingModelReportsAndFailsCleanly()
    {
        string path = Path.Combine(Path.GetTempPath(), "ot-no-such-" + Guid.NewGuid().ToString("N") + ".gguf");
        var (exit, _, err) = Run("-m", path);
        Assert.NotEqual(0, exit);
        Assert.Contains("show-template", err, StringComparison.Ordinal);
    }

    /// <summary>
    /// The reference model ships a ChatML template. Skipped rather than failed when the model is
    /// absent, so the suite stays green on a machine without it.
    /// </summary>
    [Fact]
    public void RendersTheReferenceModelTemplate()
    {
        // Resolve against the repo, not the test output directory — a relative "models/..." never
        // matches from bin/, so the test would silently skip on the very machine that has the model.
        string? model = null;
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "models", "SmolLM2-1.7B-Instruct-Q4_K_M.gguf");
            if (File.Exists(candidate)) { model = candidate; break; }
        }
        Assert.SkipUnless(model is not null, "reference model not present in this environment");

        var (exit, output, _) = Run("-m", model, "-p", "Ping", "--system", "Be brief.");
        Assert.Equal(0, exit);
        Assert.Contains("Ping", output, StringComparison.Ordinal);
        Assert.Contains("Be brief.", output, StringComparison.Ordinal);
        // The generation prompt must be present, or the model would be asked to continue the user
        // turn rather than reply.
        Assert.Contains("assistant", output, StringComparison.Ordinal);
    }
}
