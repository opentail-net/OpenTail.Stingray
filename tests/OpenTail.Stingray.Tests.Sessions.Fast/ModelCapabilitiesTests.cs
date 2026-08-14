using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Core.Capabilities;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Sessions;
using Xunit;

namespace OpenTail.Stingray.Tests.Sessions.Fast;

/// <summary>
/// Unit tests for <see cref="ModelCapabilities"/> discovery on <see cref="IInferenceRuntime"/> and <see cref="IInferenceSession"/>.
/// </summary>
public sealed class ModelCapabilitiesTests
{
    [Fact]
    public async Task Test01_RuntimeExposesModelCapabilities()
    {
        await using var runtime = new InferenceRuntime(totalPages: 100, pageSizeTokens: 32);

        Assert.NotNull(runtime.ModelCapabilities);
        Assert.Same(runtime.Capabilities.Model, runtime.ModelCapabilities);
        Assert.True(runtime.ModelCapabilities.ContextLength > 0);
        Assert.NotNull(runtime.ModelCapabilities.Architecture);
    }

    [Fact]
    public async Task Test02_SessionExposesModelCapabilities()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var session = new InferenceSession(cache);

        Assert.NotNull(session.ModelCapabilities);
        Assert.True(session.ModelCapabilities.ContextLength > 0);
        Assert.Equal("Transformer", session.ModelCapabilities.Architecture);
        Assert.True(session.ModelCapabilities.SupportsStructuredOutput);
    }

    [Fact]
    public async Task Test03_ContextLength_ReflectsModelNotSessionLimit()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var customModelCaps = new ModelCapabilities
        {
            Architecture = "Qwen2",
            ContextLength = 32768,
            SupportsToolCalling = true,
            SupportsVision = true
        };

        await using var session = new InferenceSession(
            cache,
            maxContextTokens: 1024,
            modelCapabilities: customModelCaps);

        // Session application limit is 1024
        Assert.Equal(1024, session.MaxContextTokens);

        // Model capability ContextLength remains 32768
        Assert.Equal(32768, session.ModelCapabilities.ContextLength);
        Assert.True(session.ModelCapabilities.SupportsToolCalling);
        Assert.True(session.ModelCapabilities.SupportsVision);
    }

    [Fact]
    public async Task Test04_FailClosed_ToolCallingAndVisionDefaultFalse()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var session = new InferenceSession(cache);

        // Default fail-closed semantics: unless forward pass / metadata indicates otherwise,
        // tool calling and vision default to false
        Assert.False(session.ModelCapabilities.SupportsToolCalling);
        Assert.False(session.ModelCapabilities.SupportsVision);
    }

    [Fact]
    public async Task Test05_VisionSupported_WhenEmbeddingInputSupported()
    {
        var mockFwd = new VisionCapableForwardPass();
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var runtime = new InferenceRuntime(cache, forwardPass: mockFwd);

        Assert.True(runtime.ModelCapabilities.SupportsEmbeddingInput);
        Assert.True(runtime.ModelCapabilities.SupportsVision);
    }

    [Fact]
    public async Task Test06_ForkedSession_InheritsParentModelCapabilities()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var customCaps = new ModelCapabilities
        {
            Architecture = "Llama3",
            ContextLength = 16384,
            SupportsToolCalling = true
        };

        await using var parent = new InferenceSession(cache, modelCapabilities: customCaps);
        await parent.AppendAsync(new int[] { 1, 2, 3 });

        await using var child = (InferenceSession)parent.Fork();

        Assert.NotNull(child.ModelCapabilities);
        Assert.Same(parent.ModelCapabilities, child.ModelCapabilities);
        Assert.Equal(16384, child.ModelCapabilities.ContextLength);
        Assert.True(child.ModelCapabilities.SupportsToolCalling);
    }

    [Fact]
    public async Task Test07_Immutability()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var session = new InferenceSession(cache);

        var caps1 = session.ModelCapabilities;
        var caps2 = session.ModelCapabilities;

        Assert.Same(caps1, caps2);
    }

    [Fact]
    public async Task Test08_NoGenerationHotPathRegression()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new FastMockForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd, ownsForwardPass: true);
        await session.AppendAsync(new int[] { 1, 2, 3 });

        var chunks = new List<GenerateChunk>();
        await foreach (var chunk in session.GenerateAsync(new SamplingParams { MaxNewTokens = 5 }))
        {
            chunks.Add(chunk);
        }

        Assert.Equal(5, chunks.Count);
        Assert.Equal(8, session.TokenCount);
    }

    private sealed class VisionCapableForwardPass : IForwardPass
    {
        public bool SupportsEmbeddingInput => true;
        public int VocabSize => 100;
        public int MaxSeqLen => 8192;

        public IForwardPass CreateContext() => this;
        public ReadOnlySpan<float> Forward(int token, int position) => new float[100];
        public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0) => new float[100];
        public void TruncateTo(int position) { }
        public void ResetCache() { }
        public void Dispose() { }
    }

    private sealed class FastMockForwardPass : IForwardPass
    {
        public int VocabSize => 100;
        public int MaxSeqLen => 2048;

        public IForwardPass CreateContext() => this;
        public ReadOnlySpan<float> Forward(int token, int position)
        {
            var res = new float[100];
            res[10] = 5.0f;
            return res;
        }

        public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0) => new float[100];
        public void TruncateTo(int position) { }
        public void ResetCache() { }
        public void Dispose() { }
    }
}
