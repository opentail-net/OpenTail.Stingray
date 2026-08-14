using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// <c>ForwardPass.MaxSeqLen</c> is the number every batching caller trusts to decide how long a
/// sequence may get: <c>ContinuousBatchingEngine</c> clamps admission to it, and
/// <c>HotSession.RunTurnAsync</c> refuses a turn whose projected positions exceed it. It must
/// therefore be the smallest real limit, not just one of them.
///
/// <para><b>The defect this pins.</b> It returned <c>_kvCache.MaxSeqLen</c>, which is the paged
/// cache's block capacity — <c>maxBlocks (8192) × PageSize (16) = 131,072</c> — and is unrelated to
/// the RoPE tables, which are allocated at <c>_ctxLen = min(maxContextLength, hp.ContextLength)</c>
/// positions. On a model with an 8192 trained context that is 131,072 versus 8,192: a caller
/// obeying the advertised limit could drive prefill sixteen times past the end of the RoPE tables.
/// The result was not an exception but an <c>AccessViolationException</c> inside
/// <c>ApplyRopeLayer</c> — process death, and on a different allocation layout it would have been
/// silent corruption instead.</para>
///
/// <para>Found by <c>tools/session-bench</c> attempting a 16K cold-TTFT baseline. No existing test
/// reached past the trained context, because every other test sizes its context at or below it —
/// which is exactly the region where the two numbers agree closely enough not to matter.</para>
///
/// <para>An out-of-bounds RoPE read cannot be asserted directly: an access violation cannot be
/// caught, it terminates the runner. So the contract is asserted on the advertised limit itself,
/// which is the thing callers actually consume.</para>
/// </summary>
public sealed class MaxSeqLenContractTests : HeavyTestBase
{
    private static string? FindModelPath(string filename = "SmolLM2-1.7B-Instruct-Q4_K_M.gguf")
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var candidate = Path.Combine(dir, "models", filename);
            if (File.Exists(candidate)) return candidate;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    /// <summary>
    /// Ask for far more context than the model was trained for. The engine clamps its scratch and
    /// RoPE tables to the trained length; the advertised maximum must clamp with them.
    /// </summary>
    [Fact]
    public void MaxSeqLen_NeverExceedsRopeTableCapacity_WhenRequestedContextExceedsTrained()
    {
        var path = FindModelPath();
        Assert.SkipUnless(path is not null, "model fixture not present in this environment");

        using var modelHandle = SharedModelCacheFixture.Instance.Acquire(path);
        var model = modelHandle.Model;
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        using var backend = new CpuBackend();

        int trained = hp.ContextLength;
        using var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: trained * 2);

        Assert.True(fwd.MaxSeqLen <= trained,
            $"MaxSeqLen advertises {fwd.MaxSeqLen} positions but the RoPE tables and attention "
            + $"scratch are sized for {trained}. A caller obeying the advertised limit walks off "
            + "the end of the RoPE tables.");
    }

    /// <summary>
    /// The ordinary case — requesting less than the trained context — must still be honoured
    /// exactly, so the clamp above cannot be "fixed" by pinning the limit to the trained length.
    /// </summary>
    [Fact]
    public void MaxSeqLen_HonoursASmallerRequestedContext()
    {
        var path = FindModelPath();
        Assert.SkipUnless(path is not null, "model fixture not present in this environment");

        using var modelHandle = SharedModelCacheFixture.Instance.Acquire(path);
        var model = modelHandle.Model;
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        using var backend = new CpuBackend();

        const int requested = 2048;
        Assert.True(requested < hp.ContextLength, "test presumes the request is below the trained context");
        using var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: requested);

        Assert.True(fwd.MaxSeqLen <= requested,
            $"MaxSeqLen advertises {fwd.MaxSeqLen} for a {requested}-position request.");
    }
}
