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

    /// <summary>
    /// Regression (2026-11-06 review): <see cref="OpenTail.Stingray.Engine.Runtime.RuntimeInstance.Create(OpenTail.Stingray.Engine.ExecutionPlan, Model, OpenTail.Stingray.Engine.IExecutionTurnGate?)"/>
    /// must never dispose the caller-supplied <see cref="Model"/> on a failed construction. <see cref="ModelContext.CreateDefaultEngine"/>
    /// passes its shared <c>_model</c> into exactly this overload — a fix that disposed the model on
    /// failure (briefly shipped in commit 6b55da7) would invalidate every other context sharing it the
    /// moment any one context's engine construction failed.
    /// </summary>
    [Fact]
    public void RuntimeInstanceCreate_WithACallerSuppliedModel_LeavesItUsableAfterAFailedConstruction()
    {
        using var model = Model.Load(_path);

        // A second, independent live context sharing the same model — the one that must survive
        // a sibling context's failed engine construction untouched.
        var survivor = model.CreateContext();

        var planningCtx = (ModelContext)model.CreateContext();
        var plan = planningCtx.ExecutionPlan;

        var unexecutablePlan = plan with { IsExecutable = false };
        Assert.Throws<OpenTail.Stingray.Engine.Runtime.PlanNotExecutableException>(
            () => OpenTail.Stingray.Engine.Runtime.RuntimeInstance.Create(unexecutablePlan, model));

        // The shared model must not have been disposed by the failed Create() above, and the
        // unrelated sibling context must still be fully usable — a disposed Model refuses every
        // operation with ObjectDisposedException (see ModelDispose_DisposesLiveContexts_AndRefusesNewOnes
        // above), so Reset() succeeding here proves the model (and survivor) came through intact.
        // Stop short of actually building a real forward pass over this test's deliberately
        // minimal synthetic GGUF (1 block, garbage tensor bytes) — this file never exercises real
        // compute kernels, only Model/ModelContext lifecycle semantics.
        survivor.Reset();
        Assert.Equal(2, model.ActiveContextCount);

        survivor.Dispose();
        planningCtx.Dispose();
        Assert.Equal(0, model.ActiveContextCount);
    }

    /// <summary>
    /// Regression (2026-11-06 review): <see cref="ModelContext.CreateContinuousBatchingEngine"/> used to
    /// check <c>_disposed</c>, build the whole <see cref="RuntimeInstance"/> against the shared
    /// <see cref="Model"/> with NO lock held, and only then acquire <c>_lock</c> to register the result.
    /// <see cref="Model.Dispose"/> disposes every context by calling <c>ctx.Dispose()</c>, which needs
    /// that same per-context lock — so with the lock held throughout construction (the fix), a
    /// construction in flight on one thread must make <see cref="Model.Dispose"/> (running on another
    /// thread, trying to tear down that very context before freeing the shared tensor source) block
    /// until the construction finishes, rather than letting the model's tensor source be freed while
    /// construction is still reading it, or letting construction silently attach a fresh engine to a
    /// context that disposal already tore down.
    /// </summary>
    /// <remarks>
    /// This test cannot safely drive a real <see cref="ModelContext.CreateContinuousBatchingEngine"/>
    /// call end to end: doing so against this file's deliberately minimal synthetic GGUF crashes the
    /// process with a native access violation (confirmed while writing this test) — real forward-pass
    /// construction needs real weight data, which is exactly what the other tests in this file avoid by
    /// never touching <c>ctx.Engine</c> or the batching-engine APIs. Instead, this simulates "a
    /// construction is in flight" the way the production code's own fix depends on: holding the
    /// context's private per-instance lock (the same lock <see cref="ModelContext.CreateContinuousBatchingEngine"/>
    /// now holds for its whole duration) via reflection, and asserting <see cref="Model.Dispose"/> blocks
    /// on it rather than proceeding.
    /// </remarks>
    [Fact]
    public async Task ModelDispose_BlocksOnAContextsLock_InsteadOfTearingDownAnInFlightConstruction()
    {
        using var model = Model.Load(_path);
        var ctx = model.CreateContext();

        var lockField = typeof(ModelContext).GetField("_lock", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!;
        var ctxLock = lockField.GetValue(ctx)!;

        using var constructionEntered = new ManualResetEventSlim(false);
        using var releaseConstruction = new ManualResetEventSlim(false);

        var holder = Task.Run(() =>
        {
            lock (ctxLock)
            {
                constructionEntered.Set();
                releaseConstruction.Wait(TimeSpan.FromSeconds(10));
            }
        });

        Assert.True(constructionEntered.Wait(TimeSpan.FromSeconds(5)), "simulated construction never entered the lock");

        var disposeTask = Task.Run(model.Dispose);
        var settledEarly = await Task.WhenAny(disposeTask, Task.Delay(300));
        Assert.NotSame(disposeTask, settledEarly);   // Dispose() must not complete while the lock is held

        releaseConstruction.Set();
        await holder;
        await disposeTask.WaitAsync(TimeSpan.FromSeconds(10));

        Assert.Equal(0, model.ActiveContextCount);
    }

    private sealed class TrackingDisposable(string name, List<string> order) : IDisposable
    {
        public void Dispose() => order.Add(name);
    }

    /// <summary>
    /// Regression (2026-11-06 review): <see cref="OpenTail.Stingray.Engine.Runtime.RuntimeInstance.DisposeOnConstructionFailure"/>
    /// (the logic <c>RuntimeInstance.Create</c>'s catch block delegates to) must dispose a resource
    /// the architecture factory tracked on <c>loadContext.OwnedDisposables</c> even though
    /// <c>ConstructForwardPass</c> threw before <c>Create</c>'s own
    /// <c>ownedDisposables.AddRange(loadContext.OwnedDisposables)</c> ever ran — e.g.
    /// <c>CommonForwardPassFactory.CreateDense</c> tracks the forward pass it just built via
    /// <c>ctx.TrackDisposable()</c> before calling <c>EnableTurboQuant()</c>, which can then throw.
    /// Must also dispose in reverse-dependency order (the forward pass before the backends it runs
    /// on), matching <c>RuntimeInstance.Dispose()</c>'s own successful-teardown order.
    /// </summary>
    [Fact]
    public void DisposeOnConstructionFailure_FreesLoadContextTrackedResources_InReverseOrder()
    {
        var order = new List<string>();
        var backendA = new TrackingDisposable("backendA", order);
        var backendB = new TrackingDisposable("backendB", order);
        var owned = new List<IDisposable> { backendA, backendB };

        using var model = Model.Load(_path);
        var planningCtx = (ModelContext)model.CreateContext();
        var plan = planningCtx.ExecutionPlan;
        planningCtx.Dispose();

        var loadContext = new OpenTail.Stingray.Engine.ArchitectureLoadContext
        {
            Probe = new OpenTail.Stingray.Engine.ArchitectureProbe
            {
                Architecture = model.Architecture,
                TensorSource = model.TensorSource,
                IsGguf = model.IsGguf,
                Gguf = model.IsGguf ? model.Gguf : null,
            },
            Hyperparams = model.Hyperparams,
            Plan = plan,
        };
        // Simulates CommonForwardPassFactory.CreateDense tracking the forward pass it just built
        // and then throwing inside EnableTurboQuant, before Create's own AddRange ever ran — this
        // entry never made it into `owned`.
        var densePass = new TrackingDisposable("densePass", order);
        loadContext.TrackDisposable(densePass);

        OpenTail.Stingray.Engine.Runtime.RuntimeInstance.DisposeOnConstructionFailure(owned, loadContext);

        Assert.Equal(["densePass", "backendB", "backendA"], order);
    }

    [Fact]
    public void DisposeOnConstructionFailure_WithNoLoadContextYet_StillFreesOwnedDisposablesInReverseOrder()
    {
        var order = new List<string>();
        var backendA = new TrackingDisposable("backendA", order);
        var backendB = new TrackingDisposable("backendB", order);

        OpenTail.Stingray.Engine.Runtime.RuntimeInstance.DisposeOnConstructionFailure(
            [backendA, backendB], loadContext: null);

        Assert.Equal(["backendB", "backendA"], order);
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