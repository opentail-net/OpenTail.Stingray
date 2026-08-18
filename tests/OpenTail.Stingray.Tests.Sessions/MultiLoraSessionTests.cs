using OpenTail.Stingray.Core.Lora;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Sessions;
using Xunit;

namespace OpenTail.Stingray.Tests.Sessions;

public sealed class MultiLoraSessionTests
{
    [Fact]
    public async Task InferenceSession_ActiveLora_CanBeAssignedAndCleared()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var session = new InferenceSession(cache);

        Assert.Null(session.ActiveLora);

        var layer = new LoraLayer("q_proj", 0, new float[] { 1f, 0f }, new float[] { 1f, 0f }, 2, 2, 1, 1f);
        using var adapter = new LoraAdapter("coder-adapter", "coder.safetensors", new[] { layer });

        session.ActiveLora = adapter;
        Assert.Same(adapter, session.ActiveLora);

        session.ActiveLora = null;
        Assert.Null(session.ActiveLora);
    }

    [Fact]
    public async Task ConcurrentSessions_HaveIndependentActiveLoraBindings()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var sessionA = new InferenceSession(cache);
        await using var sessionB = new InferenceSession(cache);

        using var adapter1 = new LoraAdapter("adapter-1", "1.safetensors", Array.Empty<LoraLayer>());
        using var adapter2 = new LoraAdapter("adapter-2", "2.safetensors", Array.Empty<LoraLayer>());

        sessionA.ActiveLora = adapter1;
        sessionB.ActiveLora = adapter2;

        Assert.Equal("adapter-1", sessionA.ActiveLora?.Id);
        Assert.Equal("adapter-2", sessionB.ActiveLora?.Id);
        Assert.NotSame(sessionA.ActiveLora, sessionB.ActiveLora);
    }
}
