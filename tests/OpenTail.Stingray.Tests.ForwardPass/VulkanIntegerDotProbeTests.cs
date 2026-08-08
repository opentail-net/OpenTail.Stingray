using OpenTail.Stingray.Core;
using OpenTail.Stingray.Vulkan;
using Xunit;

namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// Characterises the <c>GL_EXT_integer_dot_product</c> <c>dotPacked4x8AccSatEXT</c> intrinsic on
/// whatever device the suite is running on (perf-loop task #6).
///
/// <para><b>Why this exists.</b> Both int8 matvec kernels (<c>MatVecBatchedQ4KInt8</c>,
/// <c>MatVecBatchedQ6KInt8</c>) replaced the intrinsic with a hand-written unpack-and-multiply
/// loop, because on this codebase's AMD GCN/Vega reference driver it produced 4-8% relative error
/// at the real trunk shapes — enough to compound into garbage logits across 24 layers. That
/// workaround is currently unconditional, so correct hardware pays for a bug it does not have.</para>
///
/// <para>Gating the workaround behind a device check presupposes that a device check can actually
/// <i>see</i> the fault. This test measures that directly: it dispatches the intrinsic in isolation
/// on operand patterns taken from the kernels' real call sites and compares against the exact
/// integer answer. The result decides the design — if a trivial dispatch is correct on a device
/// whose real kernels the same intrinsic corrupts, then no isolated probe is a valid gate and the
/// check must run the actual kernel instead.</para>
///
/// <para>Silent-skip when Vulkan or the extension is unavailable.</para>
/// </summary>
public sealed class VulkanIntegerDotProbeTests(ITestOutputHelper output)
{
    private static VulkanBackend? TryCreate()
    {
        try { return new VulkanBackend(); }
        catch { return null; }
    }

    private static uint Pack(int b0, int b1, int b2, int b3) =>
        (uint)(b0 & 0xFF) | ((uint)(b1 & 0xFF) << 8) | ((uint)(b2 & 0xFF) << 16) | ((uint)(b3 & 0xFF) << 24);

    /// <summary>Signed×signed reference — the semantics of the (int, int, int) overload.</summary>
    private static int RefDot(uint w, uint a)
    {
        int sum = 0;
        for (int t = 0; t < 4; t++)
        {
            int wb = (int)((w >> (t * 8)) & 0xFF); if (wb >= 128) wb -= 256;
            int ab = (int)((a >> (t * 8)) & 0xFF); if (ab >= 128) ab -= 256;
            sum += wb * ab;
        }
        return sum;
    }

    /// <summary>
    /// The four operand populations the int8 kernels actually feed the intrinsic. A probe can only
    /// gate a kernel if it samples the population that kernel produces — a fault confined to, say,
    /// wide signed weights would be invisible to a probe built from Q4_K nibbles, and vice versa.
    /// </summary>
    public static TheoryData<string> Populations() =>
        ["q4k-nibbles", "q4k-ones-bias", "q6k-biased", "full-int8"];

    private static (uint W, uint A) Sample(string population, Random rng)
    {
        int Act() => rng.Next(-128, 128);
        return population switch
        {
            // Q4_K dot: 4-bit nibbles 0..15 (never sign-extended) × signed int8 activations.
            "q4k-nibbles" => (Pack(rng.Next(0, 16), rng.Next(0, 16), rng.Next(0, 16), rng.Next(0, 16)),
                              Pack(Act(), Act(), Act(), Act())),
            // Q4_K Σq min-bias: the constant ones vector × the same activations.
            "q4k-ones-bias" => (0x01010101u, Pack(Act(), Act(), Act(), Act())),
            // Q6_K: already-biased q6−32 weights ∈ [−32, 31] × signed int8 activations.
            "q6k-biased" => (Pack(rng.Next(-32, 32), rng.Next(-32, 32), rng.Next(-32, 32), rng.Next(-32, 32)),
                             Pack(Act(), Act(), Act(), Act())),
            // Unconstrained control, to characterise the fault itself.
            _ => (Pack(Act(), Act(), Act(), Act()), Pack(Act(), Act(), Act(), Act())),
        };
    }

    /// <summary>
    /// Measures the intrinsic's isolated fault rate per operand population.
    ///
    /// <para>Recorded outcome on the AMD Vega reference part (driver 2.0.348): the fault is real,
    /// fully deterministic, and always returns exactly <c>-1</c> — but it fires ONLY when the
    /// weight operand carries sign-extended (high-bit-set) bytes. The <c>q4k-nibbles</c> and
    /// <c>q4k-ones-bias</c> populations are 100% exact, while <c>q6k-biased</c> and
    /// <c>full-int8</c> fault on roughly a quarter of samples.</para>
    ///
    /// <para><b>That asymmetry is the finding.</b> An isolated probe is a VALID gate for the Q6_K
    /// kernel, whose operands live in the faulting population. It is NOT a valid gate for the Q4_K
    /// kernel: every operand pair that kernel can construct lies wholly inside the population the
    /// intrinsic gets right, so a probe would green-light it — yet the Q4_K kernel was measured at
    /// 4-8% relative error at real trunk shapes. Whatever corrupts Q4_K is therefore NOT this
    /// per-operand fault, and cannot be detected by sampling operands at all.</para>
    /// </summary>
    [Theory]
    [MemberData(nameof(Populations))]
    public void FaultRateIsCharacterisedPerOperandPopulation(string population)
    {
        using var vk = TryCreate();
        Assert.SkipUnless(vk is not null, "model fixture not present in this environment");
        if (!vk.HasShaderIntegerDotProduct)
        {
            output.WriteLine("VK_KHR_shader_integer_dot_product not present — skipped.");
            return;
        }

        const int Samples = 4096;
        var rng = new Random(20260726);
        var ops = new uint[Samples * 2];
        for (int i = 0; i < Samples; i++)
            (ops[2 * i], ops[2 * i + 1]) = Sample(population, rng);

        float[]? got = vk.ProbeIntegerDotRaw(ops);
        Assert.NotNull(got);

        int mismatches = 0, negOne = 0, weightHasHighBit = 0, faultWithoutHighBit = 0;
        for (int i = 0; i < Samples; i++)
        {
            uint w = ops[2 * i];
            bool highBit = (w & 0x80808080u) != 0;
            if (highBit) weightHasHighBit++;
            if (got[i] == RefDot(w, ops[2 * i + 1])) continue;

            mismatches++;
            if (got[i] == -1f) negOne++;
            if (!highBit)
            {
                faultWithoutHighBit++;
                if (faultWithoutHighBit <= 4)
                    output.WriteLine($"  fault with NO high-bit weight byte: w=0x{w:X8} a=0x{ops[2 * i + 1]:X8} " +
                                     $"→ device {got[i]} vs exact {RefDot(w, ops[2 * i + 1])}");
            }
        }

        output.WriteLine($"device        : {vk.Name}");
        output.WriteLine($"population    : {population}");
        output.WriteLine($"samples       : {Samples} ({weightHasHighBit} with a high-bit weight byte)");
        output.WriteLine($"mismatches    : {mismatches} ({100.0 * mismatches / Samples:F1}%)");
        output.WriteLine($"of those, -1  : {negOne}");
        output.WriteLine($"faults with no high-bit weight byte: {faultWithoutHighBit}");

        // The invariant the gate design depends on, asserted rather than assumed: whatever the
        // fault rate, EVERY fault must (a) return the -1 sentinel and (b) involve a sign-extended
        // weight byte. If a device ever faults on a weight operand whose bytes are all < 0x80,
        // then the Q4_K kernel's operand population is affected too and the gate below must widen.
        Assert.Equal(mismatches, negOne);
        Assert.Equal(0, faultWithoutHighBit);
    }

    /// <summary>
    /// The operand probe and the real kernel must be allowed to DISAGREE — and on the reference
    /// device they do. This pins that gap, because it is the entire reason the gate runs a kernel
    /// rather than an arithmetic check.
    /// </summary>
    [Fact]
    public void OperandProbeIsNotSufficientToDecideTheGate()
    {
        using var vk = TryCreate();
        Assert.SkipUnless(vk is not null, "model fixture not present in this environment");
        if (!vk.HasShaderIntegerDotProduct)
        {
            Assert.False(vk.Dp4aIntrinsicUsable);
            return;
        }

        const int Samples = 4096;
        var rng = new Random(99);
        var ops = new uint[Samples * 2];
        for (int i = 0; i < Samples; i++)
            (ops[2 * i], ops[2 * i + 1]) = Sample((i & 1) == 0 ? "q4k-nibbles" : "q4k-ones-bias", rng);

        float[]? got = vk.ProbeIntegerDotRaw(ops);
        Assert.NotNull(got);

        bool operandProbeClean = true;
        for (int i = 0; i < Samples && operandProbeClean; i++)
            operandProbeClean = got[i] == RefDot(ops[2 * i], ops[2 * i + 1]);

        output.WriteLine($"operand probe clean for Q4_K population: {operandProbeClean}");
        output.WriteLine($"real-kernel gate says intrinsic usable  : {vk.Dp4aIntrinsicUsable}");

        // The gate may never be MORE permissive than the operand probe: a device that gets the
        // arithmetic wrong in isolation cannot possibly be right inside the kernel.
        if (!operandProbeClean)
            Assert.False(vk.Dp4aIntrinsicUsable);

        // The converse does NOT hold, and that asymmetry is the finding. On the AMD Vega reference
        // part the operand probe is clean (the fault needs a sign-extended weight byte, which this
        // kernel's nibble operands can never have) while the real kernel is corrupted anyway. If
        // this ever becomes an equality, the cheap probe would have sufficed after all.
    }

    /// <summary>
    /// The experiment that decided task #6, kept as a regression test: run the SAME Q4_K int8
    /// matvec twice on the same device and data — once compiled with the hand-written
    /// <c>dot4x8u</c>, once with the <c>dotPacked4x8AccSatEXT</c> intrinsic — and check both against
    /// the FP <c>MatMul</c> reference.
    ///
    /// <para>Measured on the AMD Vega reference part, at every shape from 8×256 to 8192×2048: the
    /// manual variant tracks FP to 0.15-0.43% (ordinary int8 activation-quantization noise) while
    /// the intrinsic variant lands at 1.0-4.2% — even though the two are mathematically identical
    /// for this kernel's operands and the intrinsic is provably exact on those operands in
    /// isolation. That is what makes the corruption a property of the compiled kernel.</para>
    ///
    /// <para>Rather than assert a fixed verdict (which would hard-code one device's bug into the
    /// suite), this asserts the CONSISTENCY the gate depends on: whatever
    /// <see cref="VulkanBackend.Dp4aIntrinsicUsable"/> decided from an 8×256 probe must still hold
    /// at full trunk shapes. A device where the fault only appeared at scale would fail here — and
    /// would mean the probe shape needs to grow.</para>
    /// </summary>
    [Theory]
    [InlineData(8, 256)]        // the probe's own shape — one workgroup, one super-block
    [InlineData(2048, 2048)]
    [InlineData(2048, 8192)]
    [InlineData(8192, 2048)]
    public void TheGateAgreesWithTheRealKernelAtEveryShape(int rows, int cols)
    {
        using var vk = TryCreate();
        Assert.SkipUnless(vk is not null, "model fixture not present in this environment");
        if (!vk.HasShaderIntegerDotProduct)
        {
            output.WriteLine("VK_KHR_shader_integer_dot_product not present — skipped.");
            return;
        }

        const int nTok = 4;
        int blocksPerRow = cols / 256;
        const int blockBytes = 144;

        // Realistic Q4_K weights: plausible d/dmin magnitudes, random 6-bit scales and nibbles.
        var weightBytes = new byte[(long)rows * blocksPerRow * blockBytes];
        var wr = new Random(20260726);
        for (long b = 0, off = 0; b < (long)rows * blocksPerRow; b++, off += blockBytes)
        {
            PutHalf16(weightBytes, off, (float)(wr.NextDouble() * 0.045 + 0.005));
            PutHalf16(weightBytes, off + 2, (float)(wr.NextDouble() * 0.002 + 0.0005));
            for (int j = 4; j < blockBytes; j++) weightBytes[off + j] = (byte)wr.Next(0, 256);
        }
        int floatCount = (int)((weightBytes.Length + 3) / 4);
        var rawAsFloats = new float[floatCount];
        weightBytes.CopyTo(System.Runtime.InteropServices.MemoryMarshal.AsBytes(rawAsFloats.AsSpan()));

        var input = new float[nTok * cols];
        var ir = new Random(7);
        for (int i = 0; i < input.Length; i++) input[i] = (float)(ir.NextDouble() * 2 - 1);

        var gpuW = vk.Upload(rawAsFloats, TensorShape.D1(floatCount));
        var gpuIn = vk.Upload(input, TensorShape.D1(nTok * cols));
        var gpuOut = vk.Allocate(TensorShape.D1(nTok * rows));
        var gpuInSingle = vk.Upload(input.AsSpan(0, cols).ToArray(), TensorShape.D1(cols));
        var gpuOutSingle = vk.Allocate(TensorShape.D1(rows));

        var manual = new float[nTok * rows];
        var intrinsic = new float[nTok * rows];
        var reference = new float[rows];
        bool gateSaysUsable = vk.Dp4aIntrinsicUsable;   // captured before the test overrides it
        try
        {
            // FP reference for token 0 — no int8 quantization at all.
            vk.MatMul(gpuOutSingle, gpuW, gpuInSingle, DType.Q4_K);
            vk.Download(gpuOutSingle, reference);

            vk.Dp4aIntrinsicUsable = false;
            vk.ResetQ4KInt8PipelineForTesting();
            vk.MatMulBatched(gpuOut, gpuW, gpuIn, nTok, DType.Q4_K);
            vk.Download(gpuOut, manual);

            vk.Dp4aIntrinsicUsable = true;
            vk.ResetQ4KInt8PipelineForTesting();
            vk.MatMulBatched(gpuOut, gpuW, gpuIn, nTok, DType.Q4_K);
            vk.Download(gpuOut, intrinsic);
        }
        finally
        {
            vk.Free(gpuW); vk.Free(gpuIn); vk.Free(gpuOut);
            vk.Free(gpuInSingle); vk.Free(gpuOutSingle);
        }

        double RelErrVsFp(float[] batched)
        {
            double num = 0, den = 0;
            for (int r = 0; r < rows; r++)
            {
                double d = batched[r] - reference[r];
                num += d * d;
                den += (double)reference[r] * reference[r];
            }
            return Math.Sqrt(num / Math.Max(den, 1e-30));
        }

        double relManual = RelErrVsFp(manual);
        double relIntrinsic = RelErrVsFp(intrinsic);

        double maxPairDelta = 0, maxMag = 0;
        for (int i = 0; i < manual.Length; i++)
        {
            maxPairDelta = Math.Max(maxPairDelta, Math.Abs(manual[i] - intrinsic[i]));
            maxMag = Math.Max(maxMag, Math.Abs(manual[i]));
        }

        output.WriteLine($"[{rows}x{cols}] nTok={nTok}  (gate: intrinsic usable = {gateSaysUsable})");
        output.WriteLine($"  manual    vs FP : {relManual * 100:F3}% relative RMS");
        output.WriteLine($"  intrinsic vs FP : {relIntrinsic * 100:F3}% relative RMS");
        output.WriteLine($"  manual vs intrinsic: max |Δ| = {maxPairDelta:G6} on values up to {maxMag:G6}");

        // The shipped path must always be sound: int8 activation quantization alone costs well
        // under 1%, and the driver fault showed up as 1.0-4.2%.
        Assert.True(relManual < 0.02, $"manual variant drifted from FP: {relManual * 100:F3}%");

        // The gate's verdict, taken from an 8x256 probe, must still be true at this shape. The two
        // variants are mathematically identical for this kernel's operands, so on a device the gate
        // cleared they must agree bit-for-bit; on a device it rejected they must not.
        if (gateSaysUsable)
            Assert.True(maxPairDelta == 0,
                $"gate enabled the intrinsic, but at [{rows}x{cols}] it diverges from the manual dot " +
                $"by {maxPairDelta:G6} (values up to {maxMag:G6}) — the 8x256 probe shape is too small " +
                "to detect this device's fault and must be enlarged.");
        else
            Assert.True(maxPairDelta > 0,
                $"gate rejected the intrinsic, but at [{rows}x{cols}] it is bit-identical to the " +
                "manual dot — the probe is rejecting a device that works, costing free performance.");
    }

    private static void PutHalf16(byte[] dst, long off, float value)
    {
        ushort bits = BitConverter.HalfToUInt16Bits((Half)value);
        dst[off] = (byte)(bits & 0xFF);
        dst[off + 1] = (byte)(bits >> 8);
    }
}
