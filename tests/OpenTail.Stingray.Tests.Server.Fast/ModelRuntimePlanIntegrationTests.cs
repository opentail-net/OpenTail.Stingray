using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Engine.Runtime;
using OpenTail.Stingray.Server;
using Xunit;

namespace OpenTail.Stingray.Tests.Server.Fast;

public sealed class ModelRuntimePlanIntegrationTests
{
    [Fact]
    public void ModelRuntime_ExposesInstanceAndPlan_AndDisposesInstance()
    {
        var plan = ExecutionPlan.CreateSynthesized(
            architecture: "llama",
            decision: new ForwardPassDecision(ForwardPassKind.CpuDense, "cpu"),
            backend: ForwardPassBackend.Cpu,
            contextSize: 512,
            gpuLayers: 0);

        var fakeEngine = new DisposableFakeEngine("test-model");
        var loadedEngine = new LoadedEngine(fakeEngine, "llama", null);

        var modelRuntime = new ModelRuntime(
            new ModelId("test-model"),
            loadedEngine,
            estimatedModelBytes: 1024 * 1024,
            instance: null);

        Assert.Equal("test-model", modelRuntime.Id.Value);
        Assert.Same(loadedEngine, modelRuntime.Loaded);
        Assert.Same(fakeEngine, modelRuntime.Engine);
        Assert.Null(modelRuntime.Instance);
        Assert.Null(modelRuntime.Plan);

        // Verify disposal
        modelRuntime.Dispose();
        Assert.Equal(ModelRuntimeState.Disposed, modelRuntime.State);
        Assert.True(fakeEngine.Disposed);
    }

    private sealed class DisposableFakeEngine(string modelId) : IInferenceEngine, IDisposable
    {
        public bool Disposed { get; private set; }
        public string ModelId => modelId;
        public int QueueDepth => 0;
        public int ActiveRequests => 0;
        public bool PrefixCacheEnabled => false;
        public long PrefillTokensReused => 0;

        public async IAsyncEnumerable<GenerateChunk> GenerateChunksAsync(
            string prompt,
            SamplingParams sp,
            [EnumeratorCancellation] CancellationToken ct = default,
            string? canonicalHistoryPrefix = null)
        {
            await Task.Yield();
            yield return new GenerateChunk(GenerateChunkKind.Text, "ok");
        }

        public void Dispose()
        {
            Disposed = true;
        }
    }
}
