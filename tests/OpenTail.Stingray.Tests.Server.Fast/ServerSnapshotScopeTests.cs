namespace OpenTail.Stingray.Tests.Server.Fast;

/// <summary>
/// The compatibility snapshot must describe what this host serves and what the loaded model is,
/// not the full product surface or the configured fallback hint.
/// </summary>
public sealed class ServerSnapshotScopeTests
{
    [Fact]
    public void Create_WithoutRouteList_ReportsTheCompleteSurface()
    {
        var snapshot = ServerCompatibilitySnapshot.Create(new OpenTailStingrayServerOptions(), new FakeInferenceEngine("m"));

        Assert.True(snapshot.Api.OpenAiChatCompletions);
        Assert.True(snapshot.Api.Health);
        Assert.True(snapshot.Api.OpenAiRerank);
    }

    [Fact]
    public void Create_ReportsOnlyTheRoutesTheHostMapped()
    {
        string[] mapped = ["/health", "/v1/chat/completions"];

        var snapshot = ServerCompatibilitySnapshot.Create(
            new OpenTailStingrayServerOptions(), new FakeInferenceEngine("m"), sessions: null, mapped);

        Assert.True(snapshot.Api.Health);
        Assert.True(snapshot.Api.OpenAiChatCompletions);
        Assert.False(snapshot.Api.Metrics);
        Assert.False(snapshot.Api.OpenAiResponses);
        Assert.False(snapshot.Api.AnthropicMessages);
        Assert.False(snapshot.Api.OpenAiEmbeddings);
        Assert.False(snapshot.Api.OpenAiRerank);
        Assert.False(snapshot.Api.SessionLifecycle);
    }

    [Fact]
    public void Create_PrefersTheLoadedArchitectureOverTheConfiguredHint()
    {
        var options = new OpenTailStingrayServerOptions { Architecture = "qwen2" };

        var fallback = ServerCompatibilitySnapshot.Create(options, new FakeInferenceEngine("m"));
        var loaded = ServerCompatibilitySnapshot.Create(options, new FakeInferenceEngine("m"),
            sessions: null, mappedRoutes: null, loadedArchitecture: "llama");

        Assert.Equal("qwen2", fallback.Architecture);
        Assert.Equal("llama", loaded.Architecture);
    }
}
