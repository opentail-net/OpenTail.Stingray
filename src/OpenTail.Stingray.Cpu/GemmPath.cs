namespace OpenTail.Stingray.Cpu;

/// <summary>
/// Which Q4_K repacked-GEMM implementation the batched prefill matmul uses.
/// </summary>
/// <remarks>
/// <para><b>Path 1</b> is the incumbent: <c>DotQ4Kx8_Q8KS_Avx2</c> / <c>_8In</c>, grown in the perf
/// loop (iterations 39–42). It uses the repacked <c>block_q4_Kx8</c> weight layout with our own
/// Q8_KS activations (eight sub-scales per super-block) and N separate activation pointers.</para>
///
/// <para><b>Path 2</b> is a deliberately literal port of llama.cpp's
/// <c>ggml_gemm_q4_K_8x8_q8_K</c> AVX2 arm (repack.cpp:2816–3487) — 16 token rows x 8 columns per
/// pass, <c>block_q8_Kx4</c> interleaved activations, the sp1/sp2 shuffle patterns, and fused
/// min correction. See docs/repack-gemm/port-log.md for the port record, every deviation from the
/// original, and the measurements.</para>
///
/// <para>The two paths exist side by side so they can be measured against each other. Path 2 is
/// permitted to decline any shape it does not yet handle, in which case the caller silently uses
/// Path 1 — so an incomplete Path 2 can never produce a wrong answer, only a slower one.</para>
/// </remarks>
public enum GemmPath
{
    /// <summary>Incumbent kernels (perf-loop iterations 39–42).</summary>
    Path1 = 1,

    /// <summary>Literal port of llama.cpp's AVX2 <c>ggml_gemm_q4_K_8x8_q8_K</c>.</summary>
    Path2 = 2,
}

/// <summary>
/// Selects the active <see cref="GemmPath"/>. Defaults from <c>STINGRAY_GEMM_PATH</c>
/// (<c>1</c> or <c>2</c>); settable at runtime so A/B harnesses can flip it between measurements
/// without reloading the model.
/// </summary>
public static class GemmPathConfig
{
    private static GemmPath _current = ReadFromEnvironment();

    /// <summary>The active path. Defaults to <see cref="GemmPath.Path2"/>.</summary>
    public static GemmPath Current
    {
        get => _current;
        set => _current = value;
    }

    /// <summary>True when Path 2 should be attempted for this call.</summary>
    public static bool UsePath2 => _current == GemmPath.Path2;

    /// <summary>
    /// Re-reads <c>STINGRAY_GEMM_PATH</c>. Exposed for tests.
    /// </summary>
    /// <remarks>
    /// <b>Defaults to Path 2 as of 2026-08-02.</b> It is faster (1.83x over Path 1 on the isolated
    /// matmul, 1.11x end-to-end) and measured no worse on quality: wikitext-2 test split, 8191
    /// scored tokens through the batched path, PPL 16.0484 for Path 2 against 16.0870 for Path 1
    /// and 16.0488 for the non-repacked baseline.
    /// <para>That last point contradicts the expectation going in. Path 2 quantises activations to
    /// Q8_K — one scale per 256-element super-block — where Path 1 uses Q8_KS with eight sub-scales,
    /// so Path 1 "should" be the more accurate of the two. Measurement says otherwise, on every
    /// position bucket. Recorded because the intuition is reasonable and someone will re-derive it.
    /// </para>
    /// </remarks>
    public static GemmPath ReadFromEnvironment()
    {
        string? v = Environment.GetEnvironmentVariable("STINGRAY_GEMM_PATH");
        if (string.IsNullOrWhiteSpace(v)) return GemmPath.Path2;
        v = v.Trim();
        if (v == "1" || v.Equals("path1", StringComparison.OrdinalIgnoreCase)) return GemmPath.Path1;
        return GemmPath.Path2;
    }
}
