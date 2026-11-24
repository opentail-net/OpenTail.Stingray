using System.Diagnostics;

namespace OpenTail.Stingray.Tests.Cli;

/// <summary>
/// Teardown of a hybrid run (GPU layers plus CPU layers, MoE) must not let a queued expert preload run after the model it reads
/// has been freed. Runs the real binary out of process, three times, and asserts that stderr never reports a preload failure.
/// Heavy: skipped unless <c>STINGRAY_RUN_HEAVY_TESTS=1</c>, and skips visibly when the OLMoE checkpoint or the optimized CLI build
/// is not on this machine. Build the CLI with <c>dotnet build src/OpenTail.Stingray.Cli -c Release</c> before running it.
/// </summary>
public sealed class HybridTeardownPrefetchTests
{
    // The failure appeared in 3 of 8 runs on the unfixed build (A32, 2026-11-24): six repetitions make a miss unlikely.
    private const int Repetitions = 6;

    [Fact]
    public void OlmoeHybridRun_TeardownReportsNoPreloadFailure()
    {
        Assert.SkipUnless(Environment.GetEnvironmentVariable("STINGRAY_RUN_HEAVY_TESTS") == "1",
            "heavy: loads a real model; set STINGRAY_RUN_HEAVY_TESTS=1");
        string? model = OpenTail.Stingray.Engine.Verification.ModelLocator.FindDeep("OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf");
        Assert.SkipWhen(model is null, "OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf is not on this machine");
        string exe = CliExe();
        Assert.SkipUnless(File.Exists(exe), $"CLI binary not built: {exe}");

        for (int i = 1; i <= Repetitions; i++)
        {
            var (exit, stdout, stderr) = Run(exe, model!);
            Assert.True(exit == 0, $"run {i}: exit {exit}\n--- stdout\n{stdout}\n--- stderr\n{stderr}");
            Assert.Contains("[HybridForwardPass] GPU:", stderr, StringComparison.Ordinal);   // the hybrid path actually ran
            Assert.DoesNotContain("[MoEPrefetcher]", stderr, StringComparison.Ordinal);
        }
    }

    private static string RepoRoot()
    {
        for (var d = new DirectoryInfo(AppContext.BaseDirectory); d is not null; d = d.Parent)
            if (File.Exists(Path.Combine(d.FullName, "OpenTail.Stingray.slnx"))) return d.FullName;
        throw new DirectoryNotFoundException("repository root not found");
    }

    private static string CliExe()
    {
        string name = OperatingSystem.IsWindows() ? "stingray.exe" : "stingray";
        string? overridePath = Environment.GetEnvironmentVariable("STINGRAY_CLI_EXE");
        if (!string.IsNullOrEmpty(overridePath)) return overridePath;
        return Path.Combine(RepoRoot(), "src", "OpenTail.Stingray.Cli", "bin", "Release", "net10.0", name);
    }

    private static (int Exit, string Out, string Err) Run(string exe, string model)
    {
        var psi = new ProcessStartInfo(exe)
        {
            RedirectStandardOutput = true, RedirectStandardError = true, RedirectStandardInput = true, UseShellExecute = false,
            WorkingDirectory = RepoRoot(),
        };
        foreach (var a in new[]
        {
            "-m", model, "-p", "Write one short sentence about the Eiffel Tower.", "-n", "32", "--temp", "0", "-g", "8",
            "--single-turn", "--no-display-prompt",
        })
            psi.ArgumentList.Add(a);
        using var p = Process.Start(psi)!;
        p.StandardInput.Close();   // a run must never wait for interactive input
        var stdout = p.StandardOutput.ReadToEndAsync();
        var stderr = p.StandardError.ReadToEndAsync();
        if (!p.WaitForExit(TimeSpan.FromMinutes(5)))
        {
            p.Kill(true);
            p.WaitForExit();
            Assert.Fail("run timed out after 5 minutes");
        }
        return (p.ExitCode, stdout.Result, stderr.Result);
    }
}
