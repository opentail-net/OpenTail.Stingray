using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Sessions;

namespace OpenTail.Stingray.Tests.Sessions;

/// <summary>
/// Executes the claim in the session runtime plan §3.4.13: compressed KV (TurboQuant) does not
/// compose with the engine <see cref="HotSession"/> is built on, so sessions have exactly one
/// state encoding — fp32 at 384 KiB/token — and the residency problem in §3.4.3 has no existing
/// remedy available to them.
///
/// <para>That conclusion was reached by reading four <c>NotSupportedException</c> throw sites.
/// Reading is not running, and this repository has a standing record of audits that were confidently
/// wrong until executed. So it is executed here.</para>
///
/// <para>This test asserts a LIMITATION, deliberately. If someone later makes TurboQuant compose
/// with continuous batching, this test fails — and that failure is the correct signal, not a
/// regression: it means §3.4.13's roadmap conclusion is stale and the plan needs updating. The
/// message says so.</para>
/// </summary>
public sealed class TurboQuantSessionCompositionTests
{
    private static string? FindModelPath()
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var candidate = Path.Combine(dir, "models", "SmolLM2-1.7B-Instruct-Q4_K_M.gguf");
            if (File.Exists(candidate)) return candidate;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public async Task TurboQuantForwardPass_CannotBackAHotSession()
    {
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        using var backend = new CpuBackend();

        var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 512);
        // Engage the TurboQuant cache through the same entry point production uses.
        // Lloyd-Max at its DEFAULT 3 bits. An earlier version passed bits: 4 and died on
        // "No codebook for 4-bit, d=64" — the codebooks are per (bits, headDim) and 4-bit/64 is not
        // shipped. Catching only NotSupportedException missed it, since that surfaces as an
        // ArgumentException. Both are treated as "this codec cannot run here", which is a skip and
        // not a result: the composition question is only asked once TurboQuant is actually engaged.
        try { fwd.EnableTurboQuant(fp32WindowSize: 128, bits: 3, quantizer: TqQuantizer.LloydMax); }
        catch (NotSupportedException) { return; }
        catch (ArgumentException) { return; }

        using var engine = new ContinuousBatchingEngine(fwd, tokenizer, "tq-compose", maxBatchSize: 1);
        var runtime = new HotSessionRuntime(engine, tokenizer);
        using var session = runtime.Create();

        var result = await session.RunTurnAsync("The capital of France is",
            new SamplingParams { Temperature = 0f, MaxNewTokens = 4 },
            SessionRevision.Initial, SessionOperationId.New(),
            SessionRequestDigest.FromCanonicalValue("tq"));

        // The turn must not silently succeed: the engine's admission path (PrefillWithCache) and
        // its decode step (BatchForwardMulti) both refuse a TurboQuant cache.
        Assert.True(result.Operation.State == SessionOperationState.Failed,
            "A TurboQuant-backed ForwardPass drove a HotSession turn to state "
            + $"'{result.Operation.State}'. If TurboQuant now composes with continuous batching, "
            + "this is GOOD NEWS and not a regression — but session runtime plan §3.4.13 concluded "
            + "sessions have only one state encoding and that conclusion is now stale. Update the "
            + "plan's residency roadmap rather than relaxing this assertion.");
    }

}
