using System;
using System.Threading.Tasks;
using OpenTail.Stingray.Core.Capabilities;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Sessions;
using Xunit;

namespace OpenTail.Stingray.Tests.Sessions.Fast;

public class CapabilityDiscoveryTests
{
    [Fact]
    public async Task Test1_TruthfulCapabilitySet()
    {
        await using var runtime = new InferenceRuntime(totalPages: 100, pageSizeTokens: 32);
        var caps = runtime.Capabilities;

        Assert.NotNull(caps);
        Assert.NotNull(caps.Model);
        Assert.NotNull(caps.Execution);
        Assert.NotNull(caps.State);
        Assert.NotNull(caps.Generation);
        Assert.NotNull(caps.Device);

        Assert.True(caps.State.SupportsPagedKvCache);
        Assert.True(caps.State.SupportsSessionFork);
        Assert.True(caps.State.SupportsSuspendResume);
        Assert.True(caps.State.SupportsSnapshotRestore);
        Assert.True(caps.Generation.SupportsSampling);
        Assert.True(caps.Device.SupportsCpu);
    }

    [Fact]
    public async Task Test2_UnsupportedCapability()
    {
        await using var runtime = new InferenceRuntime(totalPages: 100, pageSizeTokens: 32);
        var caps = runtime.Capabilities;

        // Default CPU runtime does not have CUDA or MTP enabled
        Assert.False(caps.Supports(InferenceCapability.Cuda));
        Assert.False(caps.Supports(InferenceCapability.Mtp));
        Assert.False(caps.Supports(InferenceCapability.EmbeddingInput));
    }

    [Fact]
    public async Task Test3_TypedPropertyAndEnumConsistency()
    {
        await using var runtime = new InferenceRuntime(totalPages: 100, pageSizeTokens: 32);
        var caps = runtime.Capabilities;

        Assert.Equal(caps.State.SupportsPagedKvCache, caps.Supports(InferenceCapability.PagedKvCache));
        Assert.Equal(caps.State.SupportsKvForking, caps.Supports(InferenceCapability.KvForking));
        Assert.Equal(caps.State.SupportsKvCopyOnWrite, caps.Supports(InferenceCapability.KvCopyOnWrite));
        Assert.Equal(caps.State.SupportsCheckpointRollback, caps.Supports(InferenceCapability.CheckpointRollback));
        Assert.Equal(caps.State.SupportsSessionFork, caps.Supports(InferenceCapability.SessionFork));
        Assert.Equal(caps.State.SupportsSuspendResume, caps.Supports(InferenceCapability.SuspendResume));
        Assert.Equal(caps.State.SupportsSnapshotRestore, caps.Supports(InferenceCapability.SnapshotRestore));
        Assert.Equal(caps.Device.SupportsCpu, caps.Supports(InferenceCapability.Cpu));
        Assert.Equal(caps.Device.SupportsCuda, caps.Supports(InferenceCapability.Cuda));
    }

    [Fact]
    public void Test4_EffectiveCapabilityComposition()
    {
        // Custom capability builder with MTP model but MTP execution false -> effective capability is false!
        var customCaps = new InferenceCapabilities
        {
            Model = new ModelCapabilities { Architecture = "Qwen2", ContextLength = 32768, SupportsMtp = true },
            Execution = new ExecutionCapabilities { SupportsSpeculativeDecoding = false },
            State = new StateCapabilities(),
            Generation = new GenerationCapabilities { SupportsSpeculativeSampling = true },
            Device = new DeviceCapabilities()
        };

        // SpeculativeSampling requires BOTH Generation.SupportsSpeculativeSampling AND Execution.SupportsSpeculativeDecoding!
        Assert.False(customCaps.Supports(InferenceCapability.SpeculativeSampling));
    }

    [Fact]
    public async Task Test5_Immutability()
    {
        await using var runtime = new InferenceRuntime(totalPages: 100, pageSizeTokens: 32);
        var caps1 = runtime.Capabilities;
        var caps2 = runtime.Capabilities;

        Assert.Same(caps1, caps2); // Reference equal and immutable
    }

    [Fact]
    public async Task Test6_SupportsAll_SupportsAny_Negotiation()
    {
        await using var runtime = new InferenceRuntime(totalPages: 100, pageSizeTokens: 32);
        var caps = runtime.Capabilities;

        Assert.True(caps.SupportsAll(
            InferenceCapability.PagedKvCache,
            InferenceCapability.SessionFork,
            InferenceCapability.Cpu));

        Assert.False(caps.SupportsAll(
            InferenceCapability.PagedKvCache,
            InferenceCapability.Cuda));

        Assert.True(caps.SupportsAny(
            InferenceCapability.Cuda,
            InferenceCapability.Cpu));

        Assert.False(caps.SupportsAny(
            InferenceCapability.Cuda,
            InferenceCapability.Mtp));
    }

    [Fact]
    public async Task Test7_RuntimeCapabilities_DerivedFromInstantiatedForwardPass()
    {
        var capabilityForwardPass = new CapabilityBearingTestForwardPass();
        using var cache = new CpuKvCache(100, 32);

        await using var runtime = new InferenceRuntime(cache, forwardPass: capabilityForwardPass);

        // Assert runtime capabilities reflect the actual instantiated forward pass properties!
        Assert.True(runtime.Capabilities.Model.SupportsEmbeddingInput);
        Assert.True(runtime.Capabilities.Supports(InferenceCapability.EmbeddingInput));
        Assert.Equal("CapabilityBearingTestForwardPass", runtime.Capabilities.Device.Backend);
    }

    private sealed class CapabilityBearingTestForwardPass : OpenTail.Stingray.Core.IForwardPass
    {
        public bool SupportsEmbeddingInput => true;
        public int VocabSize => 100;
        public int MaxSeqLen => 2048;

        public OpenTail.Stingray.Core.IForwardPass CreateContext() => this;
        public System.ReadOnlySpan<float> Forward(int token, int position) => new float[100];
        public System.ReadOnlySpan<float> Prefill(System.Collections.Generic.IReadOnlyList<int> tokens, int startPos = 0) => new float[100];
        public void TruncateTo(int position) { }
        public void ResetCache() { }
        public void Dispose() { }
    }
}
