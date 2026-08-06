namespace OpenTail.Stingray.Cuda;

/// <summary>
/// What a CUDA device can actually execute, derived purely from its compute capability.
/// </summary>
/// <remarks>
/// <para><b>Why this exists.</b> Kernel eligibility was previously decided by bare
/// <c>_smVersion &gt;= N</c> comparisons at the point of use, and the NVRTC module compiles every
/// kernel together for the exact detected architecture
/// (<c>--gpu-architecture=sm_{_smVersion}</c>, CudaBackend.cs:7182). A PTX instruction that the
/// target cannot encode therefore fails the whole compilation, and the failure throws — taking
/// down every custom kernel, not just the one that used it.</para>
///
/// <para>This type is deliberately <b>pure</b>: no P/Invoke, no device handle, no CUDA runtime. It
/// is a function from compute capability to a set of facts, so kernel-selection logic can be unit
/// tested against synthetic capability records for hardware nobody has to own. That is the only way
/// this area can be validated on a machine with no NVIDIA GPU — which is the machine it was written
/// on.</para>
///
/// <para><b>Arch requirements are from the PTX ISA, not from guesswork.</b> Each property below
/// cites the capability at which the instruction family becomes encodable. Getting these wrong in
/// the permissive direction reintroduces exactly the compile-time failure this is meant to prevent,
/// so they are conservative where the ISA is ambiguous.</para>
/// </remarks>
/// <param name="Major">Compute capability major version (e.g. 8 for Ampere).</param>
/// <param name="Minor">Compute capability minor version (e.g. 6 for sm_86).</param>
public readonly record struct CudaDeviceCaps(int Major, int Minor)
{
    /// <summary>Compute capability as the conventional two-digit form: sm_86 → 86.</summary>
    public int Sm => Major * 10 + Minor;

    /// <summary>Whether this record describes a usable device at all.</summary>
    public bool IsValid => Major > 0;

    /// <summary>Parse the conventional two-digit form (86 → 8.6).</summary>
    public static CudaDeviceCaps FromSm(int sm) => new(sm / 10, sm % 10);

    // ── Integer dot product ────────────────────────────────────────────────
    /// <summary><c>dp4a</c> — 8-bit dot product. Pascal GP102+ (sm_61).</summary>
    public bool SupportsDp4a => Sm >= 61;

    // ── Tensor-core MMA, fp16 multiplicands ────────────────────────────────
    /// <summary><c>mma.sync.m8n8k4.f16</c> — Volta (sm_70).</summary>
    public bool SupportsMmaF16M8N8K4 => Sm >= 70;

    /// <summary><c>mma.sync.m16n8k8.f16</c> — Turing (sm_75).</summary>
    public bool SupportsMmaF16M16N8K8 => Sm >= 75;

    /// <summary>
    /// <c>mma.sync.m16n8k16.f16</c> — <b>Ampere (sm_80)</b>, NOT Turing.
    /// Turing's widest fp16 shape is <c>m16n8k8</c>.
    /// </summary>
    public bool SupportsMmaF16M16N8K16 => Sm >= 80;

    // ── Tensor-core MMA, int8 multiplicands ────────────────────────────────
    /// <summary><c>mma.sync.m8n8k16.s8</c> — Turing (sm_75).</summary>
    public bool SupportsMmaS8M8N8K16 => Sm >= 75;

    /// <summary>
    /// <c>mma.sync.m16n8k32.s8</c> — <b>Ampere (sm_80)</b>, NOT Turing.
    /// Turing's widest int8 shape is <c>m8n8k16</c>.
    /// </summary>
    public bool SupportsMmaS8M16N8K32 => Sm >= 80;

    // ── Data types and async machinery ─────────────────────────────────────
    /// <summary>bfloat16 arithmetic and bf16 MMA — Ampere (sm_80).</summary>
    public bool SupportsBf16 => Sm >= 80;

    /// <summary>TF32 tensor-core path — Ampere (sm_80).</summary>
    public bool SupportsTf32 => Sm >= 80;

    /// <summary><c>cp.async</c> global→shared pipelining — Ampere (sm_80).</summary>
    public bool SupportsAsyncCopy => Sm >= 80;

    /// <summary>
    /// fp8 (E4M3/E5M2) through cuBLAS. Ada exposes the dtype but <c>cublasGemmEx</c> fp8 requires
    /// Hopper — this follows the stricter runtime requirement, matching CudaBackend.cs:11.
    /// </summary>
    public bool SupportsFp8Gemm => Sm >= 90;

    /// <summary><c>wgmma</c> warpgroup MMA — Hopper (sm_90).</summary>
    public bool SupportsWgmma => Sm >= 90;

    /// <summary>
    /// Whether the monolithic NVRTC module as currently written can compile for this device.
    /// </summary>
    /// <remarks>
    /// <b>This is the defect.</b> <c>CudaTextKernels.cs</c> contains unguarded
    /// <c>mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32</c> and
    /// <c>mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32</c>, both of which require sm_80. The
    /// backend nonetheless advertises operation from sm_53 (CudaBackend.cs:13). On anything below
    /// Ampere — Pascal, Volta <i>and Turing</i> — the compile fails and throws, so no custom kernel
    /// loads at all. Note the review that surfaced this named Pascal as "the obvious failure case";
    /// Turing fails too, so validating against a Turing card would wrongly appear to clear it.
    /// </remarks>
    public bool CanCompileMonolithicModule => SupportsMmaF16M16N8K16 && SupportsMmaS8M16N8K32;

    /// <summary>Human-readable architecture family, for diagnostics.</summary>
    public string Family => Sm switch
    {
        >= 90 => "Hopper/Blackwell",
        >= 89 => "Ada",
        >= 86 => "Ampere (consumer)",
        >= 80 => "Ampere",
        >= 75 => "Turing",
        >= 70 => "Volta",
        >= 60 => "Pascal",
        >= 50 => "Maxwell",
        _ => "pre-Maxwell/unknown",
    };

    public override string ToString() => $"sm_{Sm} ({Family})";
}
