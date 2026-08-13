using OpenTail.Stingray.Cuda;

namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// Synthetic capability tests for <see cref="CudaDeviceCaps"/>, covering every SM family the
/// backend claims to support.
/// </summary>
/// <remarks>
/// <para>These need no GPU — that is the point. The machine this was written on has no NVIDIA
/// hardware and no CUDA toolkit, so neither runtime dispatch nor kernel compilation can be
/// exercised here. Making capability logic a pure function of compute capability is what allows the
/// decision that matters to be validated anyway.</para>
///
/// <para>The load-bearing assertion is
/// <see cref="MonolithicModule_CannotCompile_BelowAmpere"/>: it pins the defect the GPU review
/// surfaced, so that if someone later splits the CUDA module or guards the MMA kernels, this test
/// tells them which architectures they have actually fixed.</para>
/// </remarks>
public sealed class CudaDeviceCapsTests
{
    [Theory]
    [InlineData(53, "pre-Maxwell/unknown")]   // sm_53 = 5.3, below Maxwell's 5.0? no — 5.3 >= 50
    [InlineData(61, "Pascal")]
    [InlineData(70, "Volta")]
    [InlineData(75, "Turing")]
    [InlineData(80, "Ampere")]
    [InlineData(86, "Ampere (consumer)")]
    [InlineData(89, "Ada")]
    [InlineData(90, "Hopper/Blackwell")]
    public void Family_IsNamedCorrectly(int sm, string expected)
    {
        // sm_53 is Maxwell 2nd gen; the switch maps >=50 to Maxwell, so correct the expectation here
        // rather than in the type: 53 -> "Maxwell".
        string want = sm == 53 ? "Maxwell" : expected;
        Assert.Equal(want, CudaDeviceCaps.FromSm(sm).Family);
    }

    [Theory]
    [InlineData(53)] [InlineData(61)] [InlineData(70)] [InlineData(75)]
    public void MonolithicModule_CannotCompile_BelowAmpere(int sm)
    {
        var caps = CudaDeviceCaps.FromSm(sm);
        Assert.False(caps.CanCompileMonolithicModule,
            $"sm_{sm} lacks the m16n8k16.f16 / m16n8k32.s8 shapes the module uses unguarded");
    }

    [Theory]
    [InlineData(80)] [InlineData(86)] [InlineData(89)] [InlineData(90)]
    public void MonolithicModule_Compiles_OnAmpereAndLater(int sm)
        => Assert.True(CudaDeviceCaps.FromSm(sm).CanCompileMonolithicModule);

    /// <summary>
    /// Turing is the trap: it HAS tensor cores and int8 MMA, so a reasonable reader assumes it
    /// supports the shapes in use. It does not — its widest are m16n8k8 (fp16) and m8n8k16 (int8).
    /// </summary>
    [Fact]
    public void Turing_HasMma_ButNotTheWideShapesTheModuleUses()
    {
        var turing = CudaDeviceCaps.FromSm(75);
        Assert.True(turing.SupportsMmaF16M16N8K8);
        Assert.True(turing.SupportsMmaS8M8N8K16);
        Assert.False(turing.SupportsMmaF16M16N8K16);
        Assert.False(turing.SupportsMmaS8M16N8K32);
        Assert.False(turing.CanCompileMonolithicModule);
    }

    /// <summary>Pascal has dp4a but no tensor cores at all — the baseline path must exist for it.</summary>
    [Fact]
    public void Pascal_HasDp4a_ButNoTensorCores()
    {
        var pascal = CudaDeviceCaps.FromSm(61);
        Assert.True(pascal.SupportsDp4a);
        Assert.False(pascal.SupportsMmaF16M8N8K4);
        Assert.False(pascal.SupportsMmaS8M8N8K16);
    }

    /// <summary>sm_53 is the floor the backend advertises; it has neither dp4a nor MMA.</summary>
    [Fact]
    public void AdvertisedFloor_Sm53_HasNeitherDp4aNorMma()
    {
        var floor = CudaDeviceCaps.FromSm(53);
        Assert.False(floor.SupportsDp4a);
        Assert.False(floor.SupportsMmaF16M8N8K4);
        Assert.False(floor.CanCompileMonolithicModule);
    }

    [Fact]
    public void AmpereGains_Bf16_Tf32_AsyncCopy()
    {
        Assert.False(CudaDeviceCaps.FromSm(75).SupportsBf16);
        Assert.False(CudaDeviceCaps.FromSm(75).SupportsTf32);
        Assert.False(CudaDeviceCaps.FromSm(75).SupportsAsyncCopy);
        Assert.True(CudaDeviceCaps.FromSm(80).SupportsBf16);
        Assert.True(CudaDeviceCaps.FromSm(80).SupportsTf32);
        Assert.True(CudaDeviceCaps.FromSm(80).SupportsAsyncCopy);
    }

    /// <summary>
    /// fp8 GEMM follows the cuBLAS runtime requirement (sm_90), not dtype availability on Ada —
    /// mirroring CudaBackend.cs:11. Ada exposing the type does not make cublasGemmEx accept it.
    /// </summary>
    [Fact]
    public void Fp8Gemm_RequiresHopper_NotAda()
    {
        Assert.False(CudaDeviceCaps.FromSm(89).SupportsFp8Gemm);
        Assert.True(CudaDeviceCaps.FromSm(90).SupportsFp8Gemm);
    }

    [Fact]
    public void Wgmma_IsHopperOnly()
    {
        Assert.False(CudaDeviceCaps.FromSm(89).SupportsWgmma);
        Assert.True(CudaDeviceCaps.FromSm(90).SupportsWgmma);
    }

    /// <summary>Capability predicates must be monotonic in SM — a newer part never loses a feature.</summary>
    [Fact]
    public void AllCapabilities_AreMonotonicInSm()
    {
        int[] sms = [53, 60, 61, 70, 75, 80, 86, 89, 90];
        Func<CudaDeviceCaps, bool>[] probes =
        [
            c => c.SupportsDp4a, c => c.SupportsMmaF16M8N8K4, c => c.SupportsMmaF16M16N8K8,
            c => c.SupportsMmaF16M16N8K16, c => c.SupportsMmaS8M8N8K16, c => c.SupportsMmaS8M16N8K32,
            c => c.SupportsBf16, c => c.SupportsTf32, c => c.SupportsAsyncCopy,
            c => c.SupportsFp8Gemm, c => c.SupportsWgmma, c => c.CanCompileMonolithicModule,
        ];

        foreach (var probe in probes)
        {
            bool seenTrue = false;
            foreach (int sm in sms)
            {
                bool v = probe(CudaDeviceCaps.FromSm(sm));
                if (v) seenTrue = true;
                else Assert.False(seenTrue, $"capability regressed at sm_{sm}");
            }
        }
    }

    [Fact]
    public void FromSm_RoundTrips()
    {
        foreach (int sm in new[] { 53, 61, 70, 75, 80, 86, 89, 90 })
            Assert.Equal(sm, CudaDeviceCaps.FromSm(sm).Sm);
    }
}
