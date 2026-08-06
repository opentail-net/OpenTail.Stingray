namespace OpenTail.Stingray.Vulkan;

using OpenTail.Stingray.Core;

/// <summary>
/// Which Vulkan implementation the quantized batched matmul
/// (<see cref="VulkanBackend.MatMulBatched"/>) uses.
/// </summary>
/// <remarks>
/// <para><b>Path 1</b> is the incumbent and the only one that exists today: the
/// <c>MatVecBatched*</c> family. These are <i>matrix-vector</i> kernels with register blocking over
/// tokens — one workgroup per 8 output rows, <c>acc[MAX_NTOK]</c> accumulators in VGPRs, so a
/// weight word loaded from VRAM is reused across up to <c>MaxBatchVerifyK</c> = 16 tokens. Above 16
/// tokens the caller chunks, and each chunk re-streams the whole weight matrix.</para>
///
/// <para><b>Path 2</b> is reserved for a true quantized GEMM — a weight tile resident in shared
/// memory (LDS) with many activation columns streaming past it, the shape llama.cpp calls
/// <c>mul_mm</c> as distinct from <c>mul_mv</c>. <b>It is not implemented.</b> The enum member and
/// the dispatch seam exist so the idea can be measured against Path 1 the moment a kernel lands,
/// and abandoned by deleting one file if it does not pay.</para>
///
/// <para><b>Why this scaffold exists before the kernel does.</b> The CPU side spent a week
/// discovering that its prefill matmul was N matvecs where it should have been a GEMM, and the
/// single most expensive mistake in that work was reasoning about which path had run instead of
/// counting it — a null A/B is ambiguous between "no effect" and "never executed", and it was the
/// latter. <see cref="VulkanMatMulStats"/> answers that question up front, for Path 1 alone. The
/// number it produces (weight bytes streamed per token) is the one metric that decides whether
/// Path 2 is worth writing at all, and it is measurable today with no new kernel.</para>
/// </remarks>
public enum VulkanMatMulPath
{
    /// <summary>Incumbent <c>MatVecBatched*</c> kernels: register-blocked matvec, ≤16 tokens/pass.</summary>
    Path1 = 1,

    /// <summary>Shared-memory tiled quantized GEMM (<c>MatMulTiledQ4K</c>/<c>MatMulTiledQ6K</c>). Serves Q4_K and Q6_K; declines other dtypes to Path 1.</summary>
    Path2 = 2,
}

/// <summary>
/// Selects the active <see cref="VulkanMatMulPath"/>. Defaults from
/// <c>STINGRAY_VULKAN_MM_PATH</c> (<c>1</c> or <c>2</c>); settable at runtime so an A/B harness
/// can flip it between measurements without reloading the model.
/// </summary>
public static class VulkanMatMulPathConfig
{
    private static VulkanMatMulPath _current = ReadFromEnvironment();

    /// <summary>The active path. Defaults to <see cref="VulkanMatMulPath.Path1"/>.</summary>
    public static VulkanMatMulPath Current
    {
        get => _current;
        set => _current = value;
    }

    /// <summary>True when Path 2 should be attempted for this call.</summary>
    public static bool UsePath2 => _current == VulkanMatMulPath.Path2;

    /// <summary>
    /// The most tokens the active path's kernels can serve in one dispatch — and therefore the
    /// largest useful prefill chunk.
    /// </summary>
    /// <remarks>
    /// <para><b>This is the constant the whole exercise turns on.</b> Path 1's 16 is the
    /// <c>MAX_NTOK</c> of the <c>MatVecBatched*</c> shaders: <c>acc[MAX_NTOK]</c> is one VGPR per
    /// token, so it is a register-pressure limit and cannot simply be raised. Path 2's reason for
    /// existing is to move that limit somewhere with room — shared memory — and the number it
    /// reports here is the only thing that turns a tiled kernel into fewer passes over the
    /// weights.</para>
    ///
    /// <para>It is deliberately <i>not</i> <c>MaxBatchVerifyK</c>. That constant also bounds
    /// speculative-decode draft length, which is a property of the drafter and has nothing to do
    /// with how wide a matmul tile is; conflating them meant a wider matmul kernel could not be
    /// given a wider prefill chunk without also changing spec-decode behaviour. Path 2 stays at 16
    /// until a kernel actually raises it, so this split changes no behaviour on its own.</para>
    /// </remarks>
    public static int MaxTokensPerDispatch => _current switch
    {
        VulkanMatMulPath.Path2 => Path2MaxTokensPerDispatch,
        _ => Path1MaxTokensPerDispatch,
    };

    /// <summary>
    /// <c>MAX_NTOK</c> in the <c>MatVecBatched*</c> shaders. Changing this requires editing the
    /// GLSL and regenerating the SPIR-V table.
    /// </summary>
    public const int Path1MaxTokensPerDispatch = 16;

    /// <summary>
    /// Tokens per dispatch the Path 2 tiled kernels support — i.e. their BN. Still 16 despite the
    /// tile living in LDS rather than VGPRs: BN=32 was built and measured, and it halved weight
    /// traffic exactly as intended while running 34% SLOWER (49.8 vs 75.0 t/s), so 16 is an
    /// empirical optimum here and not an unexplored limit. See the BN=32 negative result in
    /// docs/cpu-architecture-kernel-opportunities.md before raising it again.
    /// </summary>
    public const int Path2MaxTokensPerDispatch = 16;

    /// <summary>Whether per-dispatch statistics are collected. Off unless explicitly enabled.</summary>
    /// <remarks>
    /// Counting costs a handful of <c>Interlocked</c> adds per GPU dispatch — immaterial next to the
    /// dispatch itself — but the flag keeps the default path byte-for-byte what it was, so a
    /// measurement can never be blamed on the measuring.
    /// </remarks>
    public static bool StatsEnabled { get; set; } =
        Environment.GetEnvironmentVariable("STINGRAY_VULKAN_MM_STATS") == "1";

    /// <summary>
    /// Re-reads <c>STINGRAY_VULKAN_MM_PATH</c>. Exposed for tests.
    /// </summary>
    /// <remarks>
    /// <b>Defaults to Path 1</b>, unlike the CPU's <c>GemmPathConfig</c> which defaults to Path 2.
    /// The CPU default was flipped only after Path 2 was measured faster end-to-end and no worse on
    /// perplexity. Here Path 2 does not exist yet, so the default must stay on the incumbent; when a
    /// kernel lands, it earns the default the same way — by measurement, not by being newer.
    /// </remarks>
    public static VulkanMatMulPath ReadFromEnvironment()
    {
        string? v = Environment.GetEnvironmentVariable("STINGRAY_VULKAN_MM_PATH");
        if (string.IsNullOrWhiteSpace(v)) return VulkanMatMulPath.Path1;
        v = v.Trim();
        if (v == "2" || v.Equals("path2", StringComparison.OrdinalIgnoreCase)) return VulkanMatMulPath.Path2;
        return VulkanMatMulPath.Path1;
    }
}

/// <summary>
/// Counts what the quantized batched matmul actually did, so the cost of Path 1 can be stated
/// rather than estimated.
/// </summary>
/// <remarks>
/// <para><b>The headline number is <see cref="TokensPerDispatch"/></b> — how many tokens each read
/// of a weight matrix was shared across. Path 1 is capped at <c>MaxBatchVerifyK</c> = 16 and
/// measures 15.8 on a 931-token prompt. A tiled GEMM given a prompt-sized chunk would measure in
/// the hundreds. The gap between those is the entire prize, it needs no external input to compute,
/// and it is bandwidth-independent — the same on an integrated Radeon as on a discrete card, which
/// matters because the only GPU available while this was written is an iGPU whose absolute
/// throughput generalises to nothing.</para>
///
/// <para><b>Bytes per token requires the prompt length and will not be guessed.</b>
/// <see cref="TotalTokenDispatches"/> counts token-dispatches (a 931-token prompt through 168
/// matmuls per chunk reports 156,416), which is <i>not</i> a token count. An early version of this
/// type divided by it and reported 357.6 KiB/token for a run that actually streamed 58.6 MiB/token
/// — off by the number of matmuls in the model. The first real measurement caught it, which is the
/// argument for measuring the instrument before trusting it. Callers pass the token count
/// explicitly to <see cref="WeightBytesPerToken"/>.</para>
///
/// <para><b><see cref="FallbackDispatches"/> is the cliff detector.</b> Any weight dtype other than
/// Q4_K/Q6_K falls to a loop of single-row matvecs with no weight amortization at all — the same
/// class of silent fast-path miss that cost 6.4 vs 51.7 t/s when SnapKV was wrongly excluded from
/// the batched trunk (see <c>GpuForwardPass.Prefill</c>), and that cost the CPU path 1.52x when FFN
/// gate+up were routed past the repacked kernel. A non-zero count here during a prefill that was
/// supposed to be fully amortized is a bug, not a tuning opportunity.</para>
///
/// <para>Counters are process-wide and additive. Call <see cref="Reset"/> before a measurement.</para>
/// </remarks>
public static class VulkanMatMulStats
{
    private static long s_path1Dispatches;
    private static long s_path2Dispatches;
    private static long s_path2Declines;
    private static long s_fallbackDispatches;
    private static long s_amortizedTokens;
    private static long s_fallbackTokens;
    private static long s_weightBytes;

    /// <summary>Dispatches that took a weight-amortizing Path 1 kernel (Q4_K / Q6_K).</summary>
    public static long Path1Dispatches => Interlocked.Read(ref s_path1Dispatches);

    /// <summary>Dispatches served by a Path 2 kernel. Zero until one exists.</summary>
    public static long Path2Dispatches => Interlocked.Read(ref s_path2Dispatches);

    /// <summary>Times Path 2 was selected but declined the shape, falling through to Path 1.</summary>
    public static long Path2Declines => Interlocked.Read(ref s_path2Declines);

    /// <summary>
    /// Dispatches that fell to the per-token single-row loop — no weight amortization. Expected to
    /// be zero on a dense Q4_K/Q6_K model taking the batched trunk.
    /// </summary>
    public static long FallbackDispatches => Interlocked.Read(ref s_fallbackDispatches);

    /// <summary>
    /// Sum of <c>nTok</c> over amortized dispatches. <b>Token-dispatches, not tokens</b> — a
    /// 931-token prompt through a 24-layer model with 7 trunk matmuls each reports ~156,000.
    /// </summary>
    public static long AmortizedTokenDispatches => Interlocked.Read(ref s_amortizedTokens);

    /// <summary>Sum of <c>nTok</c> over non-amortized (fallback) dispatches. Token-dispatches.</summary>
    public static long FallbackTokenDispatches => Interlocked.Read(ref s_fallbackTokens);

    /// <summary>
    /// Total weight bytes the GPU was asked to read, counting a fallback dispatch once per token
    /// because that is genuinely how many times it re-reads the matrix.
    /// </summary>
    public static long WeightBytes => Interlocked.Read(ref s_weightBytes);

    /// <summary>Total token-dispatches across both kinds of dispatch. Not a token count.</summary>
    public static long TotalTokenDispatches => AmortizedTokenDispatches + FallbackTokenDispatches;

    /// <summary>Total dispatches, amortized and fallback.</summary>
    public static long TotalDispatches => Path1Dispatches + Path2Dispatches + FallbackDispatches;

    /// <summary>
    /// <b>The headline metric:</b> mean tokens sharing one read of a weight matrix. Path 1's
    /// ceiling is <c>MaxBatchVerifyK</c> = 16; a tiled GEMM's ceiling is the prompt length. Needs no
    /// external input, so it cannot be mis-attributed the way a bytes-per-token figure can. Zero
    /// when nothing has been recorded.
    /// </summary>
    public static double TokensPerDispatch
    {
        get
        {
            long dispatches = TotalDispatches;
            return dispatches == 0 ? 0d : (double)TotalTokenDispatches / dispatches;
        }
    }

    /// <summary>
    /// Weight bytes streamed per prompt token — the cost Path 2 exists to reduce.
    /// </summary>
    /// <param name="promptTokens">
    /// Distinct tokens prefilled. Must come from the caller: this type only ever sees dispatches,
    /// and <see cref="TotalTokenDispatches"/> is emphatically not a substitute (see the type
    /// remarks for the measurement that proved it).
    /// </param>
    public static double WeightBytesPerToken(long promptTokens)
        => promptTokens <= 0 ? 0d : (double)WeightBytes / promptTokens;

    /// <summary>Clears every counter. Call before a measurement.</summary>
    public static void Reset()
    {
        Interlocked.Exchange(ref s_path1Dispatches, 0);
        Interlocked.Exchange(ref s_path2Dispatches, 0);
        Interlocked.Exchange(ref s_path2Declines, 0);
        Interlocked.Exchange(ref s_fallbackDispatches, 0);
        Interlocked.Exchange(ref s_amortizedTokens, 0);
        Interlocked.Exchange(ref s_fallbackTokens, 0);
        Interlocked.Exchange(ref s_weightBytes, 0);
    }

    /// <summary>Record a dispatch served by a weight-amortizing Path 1 kernel.</summary>
    public static void RecordPath1(int nTok, long rows, long cols, DType weightDType)
    {
        if (!VulkanMatMulPathConfig.StatsEnabled) return;
        Interlocked.Increment(ref s_path1Dispatches);
        Interlocked.Add(ref s_amortizedTokens, nTok);
        Interlocked.Add(ref s_weightBytes, WeightBytesFor(rows, cols, weightDType));
    }

    /// <summary>Record a dispatch served by a Path 2 kernel.</summary>
    public static void RecordPath2(int nTok, long rows, long cols, DType weightDType)
    {
        if (!VulkanMatMulPathConfig.StatsEnabled) return;
        Interlocked.Increment(ref s_path2Dispatches);
        Interlocked.Add(ref s_amortizedTokens, nTok);
        Interlocked.Add(ref s_weightBytes, WeightBytesFor(rows, cols, weightDType));
    }

    /// <summary>
    /// Record that Path 2 was asked for this shape and declined it.
    /// <para>The declined SHAPES are recorded, not just the count, because a decline falls through
    /// to Path 1 and Path 1 throws above nTok=16. Every decline is therefore a hard blocker on
    /// raising the token tile, and "declines=2" alone does not say whether those 2 are a real
    /// obstacle or a shape that cannot occur at a larger chunk. Distinct shapes only — a per-layer
    /// decline repeats thousands of times and would otherwise bury the signal.</para>
    /// </summary>
    public static void RecordPath2Decline(int nTok = 0, long rows = 0, long cols = 0,
        DType weightDType = DType.Float32)
    {
        if (!VulkanMatMulPathConfig.StatsEnabled) return;
        Interlocked.Increment(ref s_path2Declines);
        if (rows == 0) return;
        string shape = $"{weightDType} rows={rows} cols={cols} nTok={nTok}";
        lock (s_declineShapes) s_declineShapes.Add(shape);
    }

    private static readonly SortedSet<string> s_declineShapes = new(StringComparer.Ordinal);

    /// <summary>Distinct shapes Path 2 declined, for the stats report.</summary>
    public static IReadOnlyCollection<string> DeclineShapes
    {
        get { lock (s_declineShapes) return s_declineShapes.ToArray(); }
    }

    /// <summary>
    /// Record a dispatch that fell to the per-token single-row loop. Weight bytes are counted
    /// <paramref name="nTok"/> times — the matrix really is re-read once per token.
    /// </summary>
    public static void RecordFallback(int nTok, long rows, long cols, DType weightDType)
    {
        if (!VulkanMatMulPathConfig.StatsEnabled) return;
        Interlocked.Increment(ref s_fallbackDispatches);
        Interlocked.Add(ref s_fallbackTokens, nTok);
        Interlocked.Add(ref s_weightBytes, nTok * WeightBytesFor(rows, cols, weightDType));
    }

    /// <summary>
    /// Bytes one full read of a <paramref name="rows"/>x<paramref name="cols"/> weight matrix costs.
    /// </summary>
    /// <remarks>
    /// Block sizes are the GGUF on-disk layouts the shaders index directly: Q4_K is 144 bytes per
    /// 256 elements (<c>Shaders.cs:3753</c>), Q6_K is 210 (<c>Shaders.cs:3610</c>). Anything else is
    /// charged at 4 bytes/element — an approximation, and deliberately the pessimistic one, since
    /// every dtype that lands there is on the non-amortized path anyway.
    /// </remarks>
    public static long WeightBytesFor(long rows, long cols, DType weightDType)
    {
        long elements = rows * cols;
        return weightDType switch
        {
            DType.Q4_K => elements / 256 * 144,
            DType.Q6_K => elements / 256 * 210,
            _ => elements * 4,
        };
    }

    /// <summary>A compact human-readable summary, for diagnostics and test output.</summary>
    /// <param name="promptTokens">
    /// Distinct tokens prefilled, for the bytes-per-token line. Pass 0 if unknown — the line is
    /// then omitted rather than filled with a number that looks authoritative and is not.
    /// </param>
    public static string Report(long promptTokens)
    {
        double mib = WeightBytes / (1024d * 1024d);
        var shapes = DeclineShapes;
        string declineDetail = shapes.Count == 0
            ? ""
            : "\n  declined   : " + string.Join("; ", shapes);
        string perToken = promptTokens > 0
            ? $", {WeightBytesPerToken(promptTokens) / (1024d * 1024d):F1} MiB/token over {promptTokens} tokens"
            : "";
        return $"""
            [VulkanMatMul] path={VulkanMatMulPathConfig.Current}
              dispatches : path1={Path1Dispatches} path2={Path2Dispatches} declines={Path2Declines} fallback={FallbackDispatches}{declineDetail}
              amortization: {TokensPerDispatch:F2} tokens/dispatch ({TotalTokenDispatches} token-dispatches over {TotalDispatches} dispatches)
              weight I/O : {mib:F1} MiB total{perToken}
            """;
    }
}
