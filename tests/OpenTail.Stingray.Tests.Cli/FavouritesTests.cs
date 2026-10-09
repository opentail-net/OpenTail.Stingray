using OpenTail.Stingray.Cli;
using OpenTail.Stingray.Core.Catalog;
using Xunit;

namespace OpenTail.Stingray.Tests.Cli;

public sealed class FavouritesTests
{
    private sealed class TempDir : IDisposable
    {
        public string Path { get; }

        public TempDir()
        {
            Path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "stingray-fav-test-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(Path);
        }

        public void Dispose()
        {
            try
            {
                if (Directory.Exists(Path))
                    Directory.Delete(Path, recursive: true);
            }
            catch
            {
                // Best-effort cleanup
            }
        }
    }

    [Fact]
    public void SetAndClear_WritesAndRemovesFavouritesAtomically()
    {
        using var temp = new TempDir();

        // Initially no favourites
        Assert.True(Favourites.TryGetAll(out var allInitial, out string? initialError, temp.Path));
        Assert.Null(initialError);
        Assert.Empty(allInitial);

        Assert.True(Favourites.TryGet("chat", out string? initialChat, out string? chatErr, temp.Path));
        Assert.Null(chatErr);
        Assert.Null(initialChat);

        // Set favourite for chat
        Favourites.Set("chat", "qwen2.5-0.5b", temp.Path);

        string filePath = Favourites.FilePath(temp.Path);
        Assert.True(File.Exists(filePath), "favourites.json should exist after Set.");

        Assert.True(Favourites.TryGet("chat", out string? resolvedFav, out string? readErr, temp.Path));
        Assert.Null(readErr);
        Assert.Equal("qwen2.5-0.5b", resolvedFav);

        Assert.True(Favourites.TryGetAll(out var allAfterSet, out _, temp.Path));
        Assert.Single(allAfterSet);
        Assert.Equal("qwen2.5-0.5b", allAfterSet["chat"]);

        // Clear favourite for chat
        Favourites.Clear("chat", temp.Path);

        Assert.True(Favourites.TryGet("chat", out string? clearedFav, out _, temp.Path));
        Assert.Null(clearedFav);

        Assert.True(Favourites.TryGetAll(out var allAfterClear, out _, temp.Path));
        Assert.Empty(allAfterClear);
    }

    [Fact]
    public void ResolutionOrder_ExplicitModelFile_TakesHighestPrecedence()
    {
        using var temp = new TempDir();
        string dummyFile = Path.Combine(temp.Path, "custom-weights.gguf");
        File.WriteAllText(dummyFile, "dummy weights");

        // Even with a favourite set in config
        Favourites.Set("chat", "qwen2.5-0.5b", temp.Path);

        // Explicit model-file overrides everything (even if modelId is also supplied)
        Assert.True(CatalogTaskResolver.TryResolve(
            "chat",
            modelId: "qwen2.5-0.5b",
            modelFile: dummyFile,
            out var resolved,
            out string? err,
            customConfigDir: temp.Path));

        Assert.Null(err);
        Assert.NotNull(resolved);
        Assert.Equal(Path.GetFullPath(dummyFile), resolved.ModelPath);
        Assert.Null(resolved.Entry);
    }

    [Fact]
    public void ResolutionOrder_ExplicitModelId_TakesPrecedenceOverFavourite()
    {
        using var temp = new TempDir();
        var home = new ModelHome(temp.Path);

        // Set favourite to valid chat model
        Favourites.Set("chat", "qwen2.5-0.5b", temp.Path);

        // Pass an explicit unknown model id: it must fail on that id, proving explicit flag was checked before favourite
        Assert.False(CatalogTaskResolver.TryResolve(
            "chat",
            modelId: "unknown-explicit-model",
            modelFile: null,
            out _,
            out string? err,
            customHome: home,
            customConfigDir: temp.Path));

        Assert.Contains("unknown model 'unknown-explicit-model'", err!, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ResolutionOrder_Favourite_TakesPrecedenceOverDefault()
    {
        using var temp = new TempDir();
        var home = new ModelHome(temp.Path);

        // Set a favourite that does not match the catalog: it must fail with the favourite error,
        // proving it did NOT silently fall back to the catalog default.
        string favFile = Favourites.FilePath(temp.Path);
        File.WriteAllText(favFile, "{\n  \"chat\": \"custom-chat-nonexistent\"\n}");

        Assert.False(CatalogTaskResolver.TryResolve(
            "chat",
            modelId: null,
            modelFile: null,
            out _,
            out string? err,
            customHome: home,
            customConfigDir: temp.Path));

        Assert.Contains("favourite for 'chat' ('custom-chat-nonexistent') is not a valid catalogue model", err!);
        Assert.Contains("stingray models use chat <id>", err!);
    }

    [Fact]
    public void ResolutionOrder_FallbackToDefault_WhenNoFavouriteConfigured()
    {
        using var temp = new TempDir();
        var home = new ModelHome(temp.Path);

        // Empty config directory, no favourites.json
        Assert.False(CatalogTaskResolver.TryResolve(
            "chat",
            modelId: null,
            modelFile: null,
            out _,
            out string? err,
            customHome: home,
            customConfigDir: temp.Path));

        // Falls back to catalog default for chat ("qwen2.5-0.5b"), which is not installed in temp home
        Assert.Contains("Model for chat ('qwen2.5-0.5b') is not installed", err!);
        Assert.Contains("stingray setup chat", err!);
    }

    [Fact]
    public void StaleFavourite_UnknownModel_ReturnsNamedErrorWithFixInstruction()
    {
        using var temp = new TempDir();
        var home = new ModelHome(temp.Path);

        string favFile = Favourites.FilePath(temp.Path);
        File.WriteAllText(favFile, "{\"chat\": \"obsolete-model\"}");

        Assert.False(CatalogTaskResolver.TryResolve(
            "chat",
            modelId: null,
            modelFile: null,
            out _,
            out string? err,
            customHome: home,
            customConfigDir: temp.Path));

        Assert.Contains("favourite for 'chat' ('obsolete-model') is not a valid catalogue model for task 'chat'", err!);
        Assert.Contains("Fix with: stingray models use chat <id> (or 'stingray models use chat --clear')", err!);
    }

    [Fact]
    public void StaleFavourite_CrossTaskModel_ReturnsNamedErrorWithFixInstruction()
    {
        using var temp = new TempDir();
        var home = new ModelHome(temp.Path);

        // 'piper-lessac' is a speak model, not a chat model
        string favFile = Favourites.FilePath(temp.Path);
        File.WriteAllText(favFile, "{\"chat\": \"piper-lessac\"}");

        Assert.False(CatalogTaskResolver.TryResolve(
            "chat",
            modelId: null,
            modelFile: null,
            out _,
            out string? err,
            customHome: home,
            customConfigDir: temp.Path));

        Assert.Contains("favourite for 'chat' ('piper-lessac') is not a valid catalogue model for task 'chat'", err!);
        Assert.Contains("Fix with: stingray models use chat <id> (or 'stingray models use chat --clear')", err!);
    }

    [Fact]
    public void CorruptFile_ReturnsNamedErrorAndNeverOverwrites()
    {
        using var temp = new TempDir();
        var home = new ModelHome(temp.Path);

        string favFile = Favourites.FilePath(temp.Path);
        const string corruptContent = "{ this is not valid json !!";
        File.WriteAllText(favFile, corruptContent);

        // TryGet returns false with named error
        Assert.False(Favourites.TryGet("chat", out _, out string? readErr, temp.Path));
        Assert.Contains("Error reading favourites file", readErr!);

        // TryResolve returns false with named error
        Assert.False(CatalogTaskResolver.TryResolve(
            "chat",
            modelId: null,
            modelFile: null,
            out _,
            out string? resolveErr,
            customHome: home,
            customConfigDir: temp.Path));
        Assert.Contains("Error reading favourites file", resolveErr!);

        // Set and Clear refuse to overwrite the corrupt file and throw InvalidOperationException
        var setEx = Assert.Throws<InvalidOperationException>(() => Favourites.Set("chat", "qwen2.5-0.5b", temp.Path));
        Assert.Contains("Cannot update favourites: Error reading favourites file", setEx.Message);

        var clearEx = Assert.Throws<InvalidOperationException>(() => Favourites.Clear("chat", temp.Path));
        Assert.Contains("Cannot update favourites: Error reading favourites file", clearEx.Message);

        // Content on disk is untouched
        Assert.Equal(corruptContent, File.ReadAllText(favFile));
    }

    [Fact]
    public void EmptyFile_ReturnsNamedErrorAndNeverOverwrites()
    {
        using var temp = new TempDir();
        string favFile = Favourites.FilePath(temp.Path);
        File.WriteAllText(favFile, "   \r\n");

        Assert.False(Favourites.TryGet("chat", out _, out string? readErr, temp.Path));
        Assert.Contains("file is empty", readErr!);

        var ex = Assert.Throws<InvalidOperationException>(() => Favourites.Set("chat", "qwen2.5-0.5b", temp.Path));
        Assert.Contains("Cannot update favourites: Error reading favourites file", ex.Message);
        Assert.Contains("file is empty", ex.Message);

        Assert.Equal("   \r\n", File.ReadAllText(favFile));
    }

    [Fact]
    public void AtomicWrite_LeavesNoTmpFilesBehind()
    {
        using var temp = new TempDir();

        Favourites.Set("chat", "qwen2.5-0.5b", temp.Path);
        Favourites.Set("speak", "piper-lessac", temp.Path);
        Favourites.Clear("chat", temp.Path);

        var files = Directory.GetFiles(temp.Path);
        Assert.Single(files);
        Assert.Equal(Favourites.FileName, Path.GetFileName(files[0]));
        Assert.DoesNotContain(files, f => f.EndsWith(".tmp", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void ConfigDirectory_RespectsEnvironmentVariable()
    {
        const string customPath = @"C:\CustomStingrayConfigDir";
        string resolved = Favourites.ConfigDirectory(envName => envName == "STINGRAY_CONFIG_DIR" ? customPath : null);
        Assert.Equal(Path.GetFullPath(customPath), resolved);
    }

    [Fact]
    public void ModelsCommand_ExecuteUse_ValidationsAndOperations()
    {
        using var temp = new TempDir();
        string origEnv = Environment.GetEnvironmentVariable("STINGRAY_CONFIG_DIR") ?? "";
        Environment.SetEnvironmentVariable("STINGRAY_CONFIG_DIR", temp.Path);

        try
        {
            var cmd = new ModelsCommand();

            // 1. Missing task
            int exit1 = cmd.Run(["use"]);
            Assert.Equal(1, exit1);

            // 2. Unknown task
            int exit2 = cmd.Run(["use", "unsupported-task"]);
            Assert.Equal(1, exit2);

            // 3. Task without id or --clear
            int exit3 = cmd.Run(["use", "chat"]);
            Assert.Equal(1, exit3);

            // 4. Unknown model id
            int exit4 = cmd.Run(["use", "chat", "nonexistent-model"]);
            Assert.Equal(1, exit4);

            // 5. Wrong task for model (piper-lessac is speak, not chat)
            int exit5 = cmd.Run(["use", "chat", "piper-lessac"]);
            Assert.Equal(1, exit5);

            // 6. Valid set
            int exit6 = cmd.Run(["use", "chat", "qwen2.5-0.5b"]);
            Assert.Equal(0, exit6);

            Assert.True(Favourites.TryGet("chat", out string? configuredFav, out _));
            Assert.Equal("qwen2.5-0.5b", configuredFav);

            // 7. Clear
            int exit7 = cmd.Run(["use", "chat", "--clear"]);
            Assert.Equal(0, exit7);

            Assert.True(Favourites.TryGet("chat", out string? clearedFav, out _));
            Assert.Null(clearedFav);
        }
        finally
        {
            Environment.SetEnvironmentVariable("STINGRAY_CONFIG_DIR", string.IsNullOrEmpty(origEnv) ? null : origEnv);
        }
    }

    [Fact]
    public void TryResolveOrOffer_DoesNotPromptInstallOnStaleOrCorruptFavourite()
    {
        using var temp = new TempDir();
        string favFile = Favourites.FilePath(temp.Path);
        File.WriteAllText(favFile, "{\"chat\": \"obsolete-model\"}");

        // In interactive mode, a stale favourite must NOT offer to download the default model
        bool resolved = CatalogTaskResolver.TryResolveOrOffer(
            "chat",
            modelId: null,
            modelFile: null,
            out var res,
            out string? err,
            ct: default,
            interactive: true,
            customConfigDir: temp.Path);

        Assert.False(resolved);
        Assert.Null(res);
        Assert.Contains("favourite for 'chat' ('obsolete-model') is not a valid catalogue model", err!);
    }
}
