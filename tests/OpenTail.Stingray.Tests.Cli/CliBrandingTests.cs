using System.Diagnostics;

namespace OpenTail.Stingray.Tests.Cli;

/// <summary>
/// The executable is <c>stingray</c>, so the help it prints must say <c>stingray</c>, not the retired <c>opentail-llm-cli</c>
/// (front-door plan B5 item 3). Runs the CLI built beside this test assembly out of process; no model is loaded.
/// </summary>
public sealed class CliBrandingTests
{
    private const string RetiredName = "opentail-llm-cli";

    private static string RunHelp(params string[] args)
    {
        string exe = Path.Combine(AppContext.BaseDirectory, OperatingSystem.IsWindows() ? "stingray.exe" : "stingray");
        Assert.True(File.Exists(exe), $"CLI binary not found: {exe}");
        var psi = new ProcessStartInfo(exe)
        {
            RedirectStandardOutput = true, RedirectStandardError = true, RedirectStandardInput = true, UseShellExecute = false,
        };
        foreach (string a in args) psi.ArgumentList.Add(a);
        using var p = Process.Start(psi)!;
        p.StandardInput.Close();
        var stdout = p.StandardOutput.ReadToEndAsync();
        var stderr = p.StandardError.ReadToEndAsync();
        Assert.True(p.WaitForExit(TimeSpan.FromSeconds(60)), "--help did not finish within 60 s");
        Assert.Equal(0, p.ExitCode);
        return stdout.Result + "\n" + stderr.Result;
    }

    [Fact]
    public void RootHelp_UsesTheExecutableName()
    {
        string help = RunHelp("--help");
        Assert.Contains("stingray <OPTIONS> [COMMAND]", help, StringComparison.Ordinal);
        Assert.DoesNotContain(RetiredName, help, StringComparison.Ordinal);
    }

    // Help and error text wrap at hyphens, so the rendered help cannot prove every message is clean; scanning the source can.
    [Fact]
    public void NoSourceFile_MentionsTheRetiredName()
    {
        var root = new DirectoryInfo(AppContext.BaseDirectory);
        while (root is not null && !File.Exists(Path.Combine(root.FullName, "OpenTail.Stingray.slnx"))) root = root.Parent;
        Assert.NotNull(root);

        var offenders = Directory.EnumerateFiles(Path.Combine(root!.FullName, "src"), "*.cs", SearchOption.AllDirectories)
            .Where(f => !f.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}", StringComparison.Ordinal))
            .Where(f => File.ReadAllText(f).Contains(RetiredName, StringComparison.Ordinal))
            .Select(f => Path.GetRelativePath(root.FullName, f))
            .ToList();
        Assert.True(offenders.Count == 0, "still mention " + RetiredName + ": " + string.Join(", ", offenders));
    }
}
