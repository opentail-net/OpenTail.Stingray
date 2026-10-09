using OpenTail.Stingray.Cli;
using OpenTail.Stingray.Core.Catalog;

namespace OpenTail.Stingray.Tests.Cli;

public sealed class FrontDoorTaskCommandsTests
{
    [Fact]
    public void CatalogTaskResolver_ResolvesDefaultsForKnownTasks()
    {
        using var tempDir = new TempDir();
        var home = new ModelHome(tempDir.Path);

        // Missing state
        Assert.False(CatalogTaskResolver.TryResolve("chat", null, null, out _, out string? chatErr, home));
        Assert.Contains("stingray setup chat", chatErr!, StringComparison.Ordinal);

        Assert.False(CatalogTaskResolver.TryResolve("speak", null, null, out _, out string? speakErr, home));
        Assert.Contains("stingray setup speak", speakErr!, StringComparison.Ordinal);

        Assert.False(CatalogTaskResolver.TryResolve("transcribe", null, null, out _, out string? transcribeErr, home));
        Assert.Contains("stingray setup transcribe", transcribeErr!, StringComparison.Ordinal);
    }

    [Fact]
    public void CatalogTaskResolver_ResolvesCaseInsensitiveModelId()
    {
        using var tempDir = new TempDir();
        var home = new ModelHome(tempDir.Path);

        Assert.False(CatalogTaskResolver.TryResolve("chat", "QWEN2.5-0.5B", null, out _, out string? err, home));
        Assert.Contains("qwen2.5-0.5b", err!, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void CatalogTaskResolver_RejectsUnknownModelId()
    {
        using var tempDir = new TempDir();
        var home = new ModelHome(tempDir.Path);

        Assert.False(CatalogTaskResolver.TryResolve("chat", "nonexistent-model-xyz", null, out _, out string? err, home));
        Assert.Contains("unknown model 'nonexistent-model-xyz'", err!, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("qwen2.5-0.5b", err!, StringComparison.Ordinal);
    }

    [Fact]
    public void CatalogTaskResolver_AcceptsValidModelFileOverride()
    {
        using var tempDir = new TempDir();
        string dummyFile = Path.Combine(tempDir.Path, "custom.gguf");
        File.WriteAllText(dummyFile, "dummy");

        Assert.True(CatalogTaskResolver.TryResolve("chat", null, dummyFile, out var resolved, out string? err));
        Assert.Null(err);
        Assert.NotNull(resolved);
        Assert.Equal(Path.GetFullPath(dummyFile), resolved.ModelPath);
        Assert.Null(resolved.Entry);
    }

    [Fact]
    public void CatalogTaskResolver_RejectsMissingModelFileOverride()
    {
        Assert.False(CatalogTaskResolver.TryResolve("chat", null, "missing_file_xyz.gguf", out var resolved, out string? err));
        Assert.Null(resolved);
        Assert.NotNull(err);
        Assert.Contains("specified model file not found", err, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void CatalogTaskResolver_DetectsPartialOrDamagedInstallation()
    {
        using var tempDir = new TempDir();
        var home = new ModelHome(tempDir.Path);
        var entry = ModelCatalog.DefaultFor("chat")!;

        // Write a file with wrong size
        string targetFile = home.PathOf(entry.MainFile);
        Directory.CreateDirectory(Path.GetDirectoryName(targetFile)!);
        File.WriteAllBytes(targetFile, [1, 2, 3]);

        Assert.False(CatalogTaskResolver.TryResolve("chat", null, null, out var resolved, out string? err, home));
        Assert.Null(resolved);
        Assert.Contains("incomplete or damaged", err!, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("stingray setup chat", err!, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void TranscribeCommand_FailsGracefully_WhenInputMissingOrNotFound()
    {
        var cmd = new TranscribeCommand();
        int exitNoInput = ((ICommand)cmd).Run([], CancellationToken.None);
        Assert.Equal(ExitCodes.Usage, exitNoInput);

        int exitNotFound = ((ICommand)cmd).Run(["--input", "nonexistent_file_123.wav"], CancellationToken.None);
        Assert.Equal(ExitCodes.NoInput, exitNotFound);
    }

    [Fact]
    public void SpeakCommand_FailsGracefully_WhenTextMissing()
    {
        var cmd = new SpeakCommand();
        int exitNoText = ((ICommand)cmd).Run([], CancellationToken.None);
        Assert.Equal(ExitCodes.Usage, exitNoText);
    }

    [Fact]
    public void ChatCommand_FailsGracefully_WhenModelNotInstalled()
    {
        var cmd = new ChatCommand();
        int exit = ((ICommand)cmd).Run(["--model", "unknown-chat-model"], CancellationToken.None);
        Assert.Equal(ExitCodes.Usage, exit);
    }

    private sealed class TempDir : IDisposable
    {
        public string Path { get; } = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "stingray_test_" + Guid.NewGuid().ToString("N"));
        public TempDir() => Directory.CreateDirectory(Path);
        public void Dispose()
        {
            try { Directory.Delete(Path, recursive: true); } catch { }
        }
    }
}
