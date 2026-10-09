using OpenTail.Stingray.Engine.Verification;

namespace OpenTail.Stingray.Tests.Cli;

public sealed class VerifyGoldensCommandTests : IDisposable
{
    private readonly string _tempDir = Path.Combine(Path.GetTempPath(), "ot-verify-goldens-" + Guid.NewGuid().ToString("N"));

    public VerifyGoldensCommandTests() => Directory.CreateDirectory(_tempDir);

    public void Dispose()
    {
        try { if (Directory.Exists(_tempDir)) Directory.Delete(_tempDir, recursive: true); } catch (IOException) { }
    }

    private (int Exit, string Output) Run(params string[] args)
    {
        var command = (ICommand)new VerifyGoldensCommand();
        var originalOut = Console.Out;
        var originalErr = Console.Error;
        var writer = new StringWriter();
        Console.SetOut(writer);
        Console.SetError(writer);
        try { return (command.Run(args, CancellationToken.None), writer.ToString()); }
        finally
        {
            Console.SetOut(originalOut);
            Console.SetError(originalErr);
        }
    }

    [Fact]
    public void MissingDirectoryFailsGracefully()
    {
        var (exit, output) = Run("-d", Path.Combine(_tempDir, "nonexistent"));
        Assert.Equal(1, exit);
        Assert.Contains("Goldens directory not found", output, StringComparison.Ordinal);
    }

    [Fact]
    public void EmptyDirectoryReportsNoGoldens()
    {
        var (exit, output) = Run("-d", _tempDir);
        Assert.Equal(0, exit);
        Assert.Contains("No golden files found", output, StringComparison.Ordinal);
    }

    [Fact]
    public void MissingModelSkipsCleanlyAndWritesBaseline()
    {
        // Write a synthetic golden referencing a nonexistent model
        var golden = new GoldenFile
        {
            Architecture = "synthetic-arch",
            Model = new GoldenModel
            {
                FileName = "nonexistent-model-file-12345.gguf",
                Source = "test-owner/test-model"
            },
            Cases =
            [
                new GoldenCase
                {
                    Name = "c1",
                    PromptTokens = [1, 2, 3],
                    Tokens = [4, 5, 6]
                }
            ]
        };

        string goldenPath = Path.Combine(_tempDir, "synthetic.golden.json");
        golden.Save(goldenPath);

        string baselinePath = Path.Combine(_tempDir, "baseline.json");

        var (exit, output) = Run("-d", _tempDir, "--baseline", baselinePath);
        Assert.Equal(0, exit);
        Assert.Contains("synthetic.golden.json", output, StringComparison.Ordinal);
        Assert.Contains("SKIPPED", output, StringComparison.Ordinal);
        Assert.True(File.Exists(baselinePath));

        var baseline = GoldenBaselineFile.Load(baselinePath);
        Assert.Single(baseline.Entries);
        var entry = baseline.Entries[0];
        Assert.Equal("synthetic-arch", entry.Architecture);
        Assert.Equal("Skipped", entry.Verdict);
        Assert.False(entry.Passed);
    }

    [Fact]
    public void StrictModeFailsOnMissingModel()
    {
        var golden = new GoldenFile
        {
            Architecture = "synthetic-arch",
            Model = new GoldenModel { FileName = "nonexistent-model-file-54321.gguf" }
        };

        golden.Save(Path.Combine(_tempDir, "strict-test.golden.json"));
        var (exit, _) = Run("-d", _tempDir, "--strict");
        Assert.Equal(1, exit);
    }

    [Fact]
    public void DiffOptionReportsComparison()
    {
        var golden = new GoldenFile
        {
            Architecture = "diff-arch",
            Model = new GoldenModel { FileName = "nonexistent-model-diff.gguf" }
        };
        golden.Save(Path.Combine(_tempDir, "diff-test.golden.json"));

        // Prior baseline
        var priorBaseline = new GoldenBaselineFile
        {
            Entries =
            [
                new GoldenBaselineEntry
                {
                    Architecture = "diff-arch",
                    GoldenFile = "diff-test.golden.json",
                    ModelFile = "nonexistent-model-diff.gguf",
                    Verdict = "Exact",
                    Passed = true,
                    DecodeTokensPerSecond = 50.0
                }
            ]
        };
        string baselinePath = Path.Combine(_tempDir, "prior-baseline.json");
        priorBaseline.Save(baselinePath);

        var (exit, output) = Run("-d", _tempDir, "--diff", baselinePath);
        Assert.Equal(0, exit);
        Assert.Contains("Baseline Diff Comparison", output, StringComparison.Ordinal);
        Assert.Contains("diff-arch", output, StringComparison.Ordinal);
    }
}
