#nullable enable

using OpenTail.Stingray;
using Xunit;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Model and context ownership rules that must run everywhere. The equivalent tests in <see cref="PublicApiContractsTests"/> return early when no
/// real checkpoint is on disk (always the case in CI), so they never exercised the lock protocol; these use a synthetic GGUF.
/// </summary>
public sealed class ModelLifecycleTests : IDisposable
{
    private readonly string _path = Path.Combine(Path.GetTempPath(), "lifecycle-" + Guid.NewGuid().ToString("N") + ".gguf");

    public ModelLifecycleTests()
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
    public void ContextThenModel_DisposeInThatOrder_IsSafe_AndDoubleDisposeIsHarmless()
    {
        var model = Model.Load(_path);
        var ctx = model.CreateContext();
        Assert.Equal(1, model.ActiveContextCount);

        ctx.Dispose();
        Assert.Equal(0, model.ActiveContextCount);
        model.Dispose();
        ctx.Dispose();
        model.Dispose();
    }

    [Fact]
    public void ModelDispose_DisposesLiveContexts_AndRefusesNewOnes()
    {
        var model = Model.Load(_path);
        var ctx = model.CreateContext();

        model.Dispose();

        Assert.Equal(0, model.ActiveContextCount);
        Assert.Throws<ObjectDisposedException>(() => ctx.Reset());
        Assert.Throws<ObjectDisposedException>(() => model.CreateContext());
    }

    [Fact]
    public void ContextsFromOneModel_AreDisposedIndependently()
    {
        using var model = Model.Load(_path);
        var a = model.CreateContext();
        var b = model.CreateContext();
        var c = model.CreateContext();
        Assert.Equal(3, model.ActiveContextCount);

        b.Dispose();
        Assert.Equal(2, model.ActiveContextCount);
        Assert.Throws<ObjectDisposedException>(() => b.Reset());
        a.Reset();   // the siblings are unaffected
        c.Reset();
        a.Dispose();
        c.Dispose();
        Assert.Equal(0, model.ActiveContextCount);
    }

    [Fact]
    public async Task ExecutorOverADisposedContext_FailsWithObjectDisposed_NotAnEngineError()
    {
        using var model = Model.Load(_path);
        var ctx = model.CreateContext();
        var executor = new OpenTail.Stingray.Executors.StatelessExecutor(ctx);
        ctx.Dispose();

        await Assert.ThrowsAsync<ObjectDisposedException>(async () =>
        {
            await foreach (var _ in executor.InferAsync("hello")) { }
        });
    }

    [Fact]
    public async Task ConcurrentCreateAndModelDispose_CompletesWithinTheBound_AndLeavesNothingRegistered()
    {
        for (int round = 0; round < 25; round++)
        {
            var model = Model.Load(_path);
            var made = new System.Collections.Concurrent.ConcurrentBag<IModelContext>();
            using var barrier = new Barrier(3);

            var creators = Enumerable.Range(0, 2).Select(_ => Task.Run(() =>
            {
                barrier.SignalAndWait();
                for (int i = 0; i < 20; i++)
                {
                    try { made.Add(model.CreateContext()); }
                    catch (ObjectDisposedException) { break; }
                }
            })).ToArray();
            var disposer = Task.Run(() =>
            {
                barrier.SignalAndWait();
                Thread.Yield();
                model.Dispose();
            });

            var all = Task.WhenAll(creators.Append(disposer));
            Assert.Same(all, await Task.WhenAny(all, Task.Delay(TimeSpan.FromSeconds(30))));
            await all;

            Assert.Equal(0, model.ActiveContextCount);
            foreach (var ctx in made) Assert.Throws<ObjectDisposedException>(() => ctx.Reset());   // none survived the parent
        }
    }

    [Fact]
    public async Task ChildDisposeRacingParentDispose_DoesNotDeadlock()
    {
        for (int round = 0; round < 25; round++)
        {
            var model = Model.Load(_path);
            var contexts = Enumerable.Range(0, 8).Select(_ => model.CreateContext()).ToArray();
            using var barrier = new Barrier(2);

            var children = Task.Run(() => { barrier.SignalAndWait(); foreach (var c in contexts) c.Dispose(); });
            var parent = Task.Run(() => { barrier.SignalAndWait(); model.Dispose(); });

            var all = Task.WhenAll(children, parent);
            Assert.Same(all, await Task.WhenAny(all, Task.Delay(TimeSpan.FromSeconds(30))));
            await all;
            Assert.Equal(0, model.ActiveContextCount);
        }
    }

    [Fact]
    public void TurboQuantHeadDim_IsRefused_NotSilentlyIgnored()
    {
        using var model = Model.Load(_path);

        var ex = Assert.Throws<NotSupportedException>(() => model.CreateContext(new ContextParams { TurboQuantHeadDim = 128 }));

        Assert.Contains("TurboQuantHeadDim", ex.Message);
        Assert.Equal(0, model.ActiveContextCount);
        using var ok = model.CreateContext(new ContextParams { TurboQuantHeadDim = null });
    }

    [Theory]
    [InlineData("mlock")]
    [InlineData("nommap")]
    [InlineData("maingpu")]
    [InlineData("split")]
    public void UnsupportedModelParams_AreRefusedBeforeTheFileIsTouched(string option)
    {
        var p = new ModelParams("does-not-exist.gguf");
        p = option switch
        {
            "mlock" => p with { UseMemoryLock = true },
            "nommap" => p with { UseMemoryMap = false },
            "maingpu" => p with { MainGpu = 1 },
            _ => p with { TensorSplit = new[] { 0.5f, 0.5f } },
        };

        Assert.Throws<NotSupportedException>(() => Model.Load(p));
    }
}