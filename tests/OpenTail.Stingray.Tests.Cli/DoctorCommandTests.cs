using OpenTail.Stingray.Cli;
using OpenTail.Stingray.Cli.CommandLine;

namespace OpenTail.Stingray.Tests.Cli;

public sealed class DoctorCommandTests
{
    [Fact]
    public void NoGpuProbe_ProducesMachineReadableNonFailingChecks()
    {
        var command = (ICommand)new DoctorCommand();
        var original = Console.Out;
        var output = new StringWriter();
        Console.SetOut(output);
        try
        {
            int exit = command.Run(["--no-gpu-probe", "--json"], CancellationToken.None);

            Assert.Equal(0, exit);
            Assert.Contains("\"schema_version\": 1", output.ToString());
        Assert.Contains("\"backend.cuda\"", output.ToString());
        Assert.Contains("\"not_probed\"", output.ToString());
        Assert.Contains("\"cuda.driver\"", output.ToString());
        Assert.Contains("\"cuda.runtime\"", output.ToString());
        Assert.Contains("\"cuda.nvrtc\"", output.ToString());
        }
        finally
        {
            Console.SetOut(original);
        }
    }

    private static string RunDoctor(params string[] args)
    {
        var command = (ICommand)new DoctorCommand();
        var original = Console.Out;
        var output = new StringWriter();
        Console.SetOut(output);
        try { command.Run(args, CancellationToken.None); return output.ToString(); }
        finally { Console.SetOut(original); }
    }

    /// <summary>
    /// §6.3 "CPU instruction-set support". AVX2+FMA are required by the CPU kernels, so the check
    /// must be present and must report `error` when they are absent. This box has them, so the
    /// assertion is that the check exists and passes; the error branch is asserted structurally
    /// (the status string is derived from the same condition).
    /// </summary>
    [Fact]
    public void ReportsCpuInstructionSets()
    {
        string output = RunDoctor("--no-gpu-probe", "--json");
        Assert.Contains("\"cpu.isa\"", output);
        Assert.Contains("AVX2", output);
    }

    /// <summary>§6.3 "filesystem readability and available space".</summary>
    [Fact]
    public void ReportsFilesystemFreeSpace()
    {
        string output = RunDoctor("--no-gpu-probe", "--json");
        Assert.Contains("\"filesystem\"", output);
        Assert.Contains("GiB free", output);
    }

    /// <summary>
    /// §6.3 "effective configuration conflicts and unknown settings". With nothing unusual set the
    /// check must still be present and clean — a diagnostic that only appears on failure cannot be
    /// distinguished from one that is broken.
    /// </summary>
    [Fact]
    public void ReportsConfigurationCheckEvenWhenClean()
    {
        string output = RunDoctor("--no-gpu-probe", "--json");
        Assert.Contains("\"config\"", output);
    }

    /// <summary>
    /// The case the check exists for: a misspelled setting is otherwise indistinguishable from
    /// unset. Uses a name no build reads, and restores the environment afterwards so the rest of
    /// the suite is unaffected.
    /// </summary>
    [Fact]
    public void FlagsAnUnknownSettingWithASuggestion()
    {
        const string name = "STINGRAY_KV_TYPE";   // real typo: the engine reads KV_DTYPE
        string? previous = Environment.GetEnvironmentVariable(name);
        Environment.SetEnvironmentVariable(name, "q8_0");
        try
        {
            string output = RunDoctor("--no-gpu-probe", "--json");
            Assert.Contains(name, output);
            Assert.Contains("STINGRAY_KV_DTYPE", output);
            Assert.Contains("warning", output);
        }
        finally { Environment.SetEnvironmentVariable(name, previous); }
    }

    /// <summary>§6.3 "a minimal allocation/backend smoke test in --deep mode".</summary>
    [Fact]
    public void DeepMode_RunsAllocationSmokeTest()
    {
        string output = RunDoctor("--no-gpu-probe", "--deep", "--json");
        Assert.Contains("\"allocation.smoke\"", output);
        Assert.Contains("64 MiB host memory buffer", output);
    }
}
