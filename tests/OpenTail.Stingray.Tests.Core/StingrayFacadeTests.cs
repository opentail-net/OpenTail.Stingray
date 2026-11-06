#nullable enable

using OpenTail.Stingray;
using OpenTail.Stingray.Core.Catalog;
using Xunit;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// <see cref="Stingray.CreateAsync"/>: the one-call chat entry point. Model resolution branches
/// that don't need network access (unknown id, licence-gated catalog entry) are tested directly;
/// the "resolves an existing file and returns a ready session" path uses this file's synthetic
/// GGUF fixture and stops short of actually generating (see <see cref="ModelLifecycleTests"/>'s
/// own remarks on why driving real forward-pass construction over that fixture crashes the
/// process — nothing here touches <c>ChatAsync</c>/<c>AskAsync</c> for the same reason).
/// </summary>
public sealed class StingrayFacadeTests : IDisposable
{
    private readonly string _path = Path.Combine(Path.GetTempPath(), "stingray-facade-" + Guid.NewGuid().ToString("N") + ".gguf");

    public StingrayFacadeTests()
    {
        File.WriteAllBytes(_path, SyntheticGguf.Build(3,
        [
            SyntheticGguf.KvStr("general.architecture", "llama"),
            SyntheticGguf.KvU32("llama.block_count", 1),
            SyntheticGguf.KvU32("llama.context_length", 64),
            SyntheticGguf.KvU32("llama.embedding_length", 8),
            SyntheticGguf.KvU32("llama.attention.head_count", 2),
            SyntheticGguf.KvStrArray("tokenizer.ggml.tokens", "a", "b", "c"),
        ], [new("token_embd.weight", [8, 3], 0, 0)], dataBytes: 1024));
    }

    public void Dispose() { try { File.Delete(_path); } catch (IOException) { } }

    [Fact]
    public async Task CreateAsync_WithAnExistingFilePath_ResolvesDirectly_AndReturnsAReadySession()
    {
        using var session = await Stingray.CreateAsync(_path);

        Assert.NotNull(session.Context);
        Assert.Same(session.Model, session.Context.Model);
        Assert.Empty(session.History);
        Assert.Equal(1, session.Model.ActiveContextCount);
    }

    [Fact]
    public async Task CreateAsync_UnknownIdOrPath_ThrowsFileNotFoundException_NamingKnownTasks()
    {
        var ex = await Assert.ThrowsAsync<FileNotFoundException>(
            () => Stingray.CreateAsync("not-a-real-path-or-catalog-id-" + Guid.NewGuid().ToString("N")));

        foreach (string task in ModelCatalog.Tasks)
            Assert.Contains(task, ex.Message);
    }

    [Fact]
    public async Task CreateAsync_LicenceGatedCatalogEntry_ThrowsWithoutDownloading_UntilAccepted()
    {
        // piper-lessac is a real catalog entry with LicenceNeedsConsent=true; resolving it by id
        // hits the licence check before any network call, so this needs no network access.
        var entry = ModelCatalog.Find("piper-lessac");
        Assert.NotNull(entry);
        Assert.True(entry!.LicenceNeedsConsent);

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(
            () => Stingray.CreateAsync("piper-lessac"));

        Assert.Contains("piper-lessac", ex.Message);
        Assert.Contains(nameof(StingrayOptions.AcceptLicence), ex.Message);
    }

    [Fact]
    public async Task Dispose_IsIdempotent_AndFreesTheUnderlyingModel()
    {
        var session = await Stingray.CreateAsync(_path);
        var model = session.Model;

        session.Dispose();
        session.Dispose();   // must not throw a second time

        Assert.Equal(0, model.ActiveContextCount);
        Assert.Throws<ObjectDisposedException>(() => model.CreateContext());
    }

    [Fact]
    public async Task DisposeAsync_DelegatesToDispose()
    {
        var session = await Stingray.CreateAsync(_path);
        var model = session.Model;

        await session.DisposeAsync();

        Assert.Equal(0, model.ActiveContextCount);
    }
}
