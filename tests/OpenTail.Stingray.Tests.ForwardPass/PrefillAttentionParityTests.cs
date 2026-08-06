using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// Chunked-vs-unchunked prefill parity at prompt lengths that actually reach the flash-64
/// attention path.
/// </summary>
/// <remarks>
/// <para><b>Why this exists separately from
/// <c>ContinuousBatchingTests.PrefillWithCache_Chunked_MatchesFull</c>.</b> That test uses a
/// 13-token prompt, and <c>PrefillCoreAttention</c> only dispatches to
/// <c>PrefillFlashAttention64</c> when <c>N &gt;= 256</c>. Every existing chunked-parity test is
/// therefore below the threshold and exercises only the incumbent per-token loop — the parity
/// contract that exists to catch numerics boundaries cannot see the flash path at all.</para>
///
/// <para><b>Why that matters.</b> Flash-64 uses online softmax (running max, <c>rescale =
/// exp(oldMax - newMax)</c>, accumulator rescaled per tile — ForwardPass.cs ~2539). That is
/// mathematically equivalent to score-then-softmax but not bit-identical. Selecting it by a size
/// threshold means a prompt admitted in chunks whose tail falls below 256 has some positions
/// computed one way and some the other. Real chunk sizes make this reachable: at
/// <c>STINGRAY_PREFILL_CHUNK</c> = 512, a ~600-token prompt splits 512 + 88.</para>
///
/// <para>This is the defect class <c>ForwardPass.cs:813-824</c> warns about, that the
/// <c>MinBatchForQ8Prefill</c> threshold had, and that the dual-Q8 gate had until 2026-08-02.
/// It is worth a test that can actually observe it.</para>
///
/// <para>Tolerance matches the existing chunked test: chunk boundaries change GEMM batch sizes and
/// therefore FP accumulation order regardless of attention, so this asserts close logits plus the
/// same argmax, not bit equality. A genuine kernel-selection divergence shows up far outside that
/// tolerance, not just below it.</para>
/// </remarks>
public sealed class PrefillAttentionParityTests
{
    private static string? FindModelPath(string filename = "SmolLM2-1.7B-Instruct-Q4_K_M.gguf")
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var candidate = Path.Combine(dir, "models", filename);
            if (File.Exists(candidate)) return candidate;
            var parent = Directory.GetParent(dir);
            if (parent == null) break;
            dir = parent.FullName;
        }
        return null;
    }

    /// <summary>Deterministic pseudo-prompt; content is irrelevant, length is the point.</summary>
    private static int[] MakeTokens(int count)
    {
        var t = new int[count];
        for (int i = 0; i < count; i++) t[i] = 1 + ((i * 7919) % 4096);
        return t;
    }

    /// <param name="chunk">Use <c>int.MaxValue</c> for a single unchunked pass.</param>
    private static float[] PrefillLogits(GgufModel model, CpuBackend backend, ModelHyperparams hp,
        int[] tokens, int chunk)
    {
        using var fwd = new Engine.ForwardPass(model, backend, hp);
        using var cache = fwd.CreateCache();
        float[] logits = [];
        for (int start = 0; start < tokens.Length; start += chunk)
        {
            int take = (int)Math.Min((long)chunk, tokens.Length - start);
            logits = fwd.PrefillWithCache(new ArraySegment<int>(tokens, start, take), cache,
                                          startPos: start).ToArray();
        }
        return logits;
    }

    /// <summary>
    /// 600 tokens in one pass (all positions flash-64) against 512 + 88 (flash-64 then the
    /// incumbent). This is the split a real chunked admission produces and the one no existing
    /// test covers.
    /// </summary>
    [Fact]
    public void ChunkedPrefill_MatchesUnchunked_AcrossFlash64Threshold()
    {
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        using var backend = new CpuBackend();

        int[] tokens = MakeTokens(600);

        float[] full = PrefillLogits(model, backend, hp, tokens, chunk: int.MaxValue);
        float[] split = PrefillLogits(model, backend, hp, tokens, chunk: 512);

        Assert.Equal(full.Length, split.Length);
        for (int i = 0; i < full.Length; i++)
            Assert.Equal(full[i], split[i], precision: 2);
        Assert.Equal(Sampler.Greedy(full), Sampler.Greedy(split));
    }

    /// <summary>
    /// Both arms above the threshold (300 + 300), so flash-64 handles every position in both. If
    /// this passes while the 512+88 case fails, the divergence is the kernel switch rather than
    /// ordinary chunk-boundary FP drift — which is the distinction that matters for diagnosis.
    /// </summary>
    [Fact]
    public void ChunkedPrefill_MatchesUnchunked_BothChunksAboveThreshold()
    {
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        using var backend = new CpuBackend();

        int[] tokens = MakeTokens(600);

        float[] full = PrefillLogits(model, backend, hp, tokens, chunk: int.MaxValue);
        float[] split = PrefillLogits(model, backend, hp, tokens, chunk: 300);

        Assert.Equal(full.Length, split.Length);
        for (int i = 0; i < full.Length; i++)
            Assert.Equal(full[i], split[i], precision: 2);
        Assert.Equal(Sampler.Greedy(full), Sampler.Greedy(split));
    }

    /// <summary>
    /// Control: the whole prompt below the threshold, so both arms use the incumbent loop
    /// throughout. Isolates "chunking itself is fine" from "the kernel switch is fine".
    /// </summary>
    [Fact]
    public void ChunkedPrefill_MatchesUnchunked_BothArmsBelowThreshold()
    {
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        using var backend = new CpuBackend();

        int[] tokens = MakeTokens(200);

        float[] full = PrefillLogits(model, backend, hp, tokens, chunk: int.MaxValue);
        float[] split = PrefillLogits(model, backend, hp, tokens, chunk: 64);

        Assert.Equal(full.Length, split.Length);
        for (int i = 0; i < full.Length; i++)
            Assert.Equal(full[i], split[i], precision: 2);
        Assert.Equal(Sampler.Greedy(full), Sampler.Greedy(split));
    }
}
