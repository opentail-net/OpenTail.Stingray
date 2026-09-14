using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion;
using OpenTail.Stingray.Vulkan;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Synthetic-weight numeric parity test for Z-Image-Turbo's GPU-residency block
/// (<see cref="ZImageGpuWeights"/>/<see cref="ZImageGpuWorkspace"/>/<see cref="ZImageDiT.
/// ApplyBlockGpu"/>, docs/075) against the existing CPU path (<see cref="ZImageDiT.ApplyBlock"/>).
/// Real weights are not available on this machine (see docs/075's own 2026-09-14 entry) -- this
/// mirrors WanGpuParityTests'/T5GpuParityTests' own primary verification method (synthetic but
/// structurally real weights, real shapes, real math), the same technique this whole session's
/// other GPU-residency work has relied on as its first correctness check.
/// </summary>
public sealed class ZImageGpuParityTests
{
    private sealed class SyntheticWeightLoader : IWeightLoader
    {
        private readonly Dictionary<string, float[]> _weights = new(StringComparer.Ordinal);
        public void Add(string name, float[] data) => _weights[name] = data;
        public bool Contains(string name) => _weights.ContainsKey(name);
        public float[] ReadF32(string name) => _weights[name];
        public DType GetDType(string name) => DType.Float32;
        public long[] GetShape(string name) => new long[] { _weights[name].Length };
        public IReadOnlyList<string> TensorNames => _weights.Keys.ToList();
        public bool TryGetRaw(string name, out nint dataPtr, out long byteLen, out DType dtype, out int rows, out int cols)
        {
            dataPtr = 0; byteLen = 0; dtype = DType.Float32; rows = 0; cols = 0;
            return false;
        }
        public void Dispose() { }
    }

    private static VulkanBackend? TryCreateVulkan()
    {
        try { return new VulkanBackend(); }
        catch { return null; }
    }

    [Fact]
    public void ApplyBlockGpu_MatchesApplyBlockCpu_Numerically()
    {
        using var vulkan = TryCreateVulkan();
        if (vulkan is null) return;

        // Small-but-real shapes: headDim MUST stay 128 (the real value) since it determines
        // which fused attention kernel MultiHeadAttentionTiled selects internally -- scale down
        // nHeads instead (dim = nHeads * headDim = 3*128 = 384) to keep the test cheap while
        // exercising the exact same code path production uses.
        const int dim = 384;
        const int nHeads = 3;
        const int headDim = 128;
        const int adalnEmbedDim = 256; // real, fixed value
        const int t = 24; // small-but-real token count

        var rng = new Random(42);
        var loader = new SyntheticWeightLoader();

        void AddWeight(string name, int size)
        {
            var arr = new float[size];
            for (int i = 0; i < size; i++) arr[i] = (float)(rng.NextDouble() * 0.04 - 0.02);
            loader.Add(name, arr);
        }
        void AddOnes(string name, int size)
        {
            var arr = new float[size];
            Array.Fill(arr, 1.0f);
            loader.Add(name, arr);
        }

        var zparams = new ZImageParams
        {
            Dim = dim,
            NHeads = nHeads,
            NLayers = 1,
            AdalnEmbedDim = adalnEmbedDim,
        };
        // FfnHidden is a derived property (≈8/3*Dim, rounded), computed from Dim automatically --
        // use ZImageParams' own real formula so CPU/GPU paths stay consistent, rather than an
        // independently-chosen placeholder.
        int realFfnHidden = zparams.FfnHidden;

        const string p = "layers.0";
        AddWeight($"{p}.adaLN_modulation.0.weight", 4 * dim * adalnEmbedDim);
        AddWeight($"{p}.adaLN_modulation.0.bias", 4 * dim);
        AddOnes($"{p}.attention_norm1.weight", dim);
        AddOnes($"{p}.attention_norm2.weight", dim);
        AddWeight($"{p}.attention.qkv.weight", 3 * dim * dim);
        AddOnes($"{p}.attention.q_norm.weight", headDim);
        AddOnes($"{p}.attention.k_norm.weight", headDim);
        AddWeight($"{p}.attention.out.weight", dim * dim);
        AddOnes($"{p}.ffn_norm1.weight", dim);
        AddOnes($"{p}.ffn_norm2.weight", dim);
        AddWeight($"{p}.feed_forward.w1.weight", realFfnHidden * dim);
        AddWeight($"{p}.feed_forward.w3.weight", realFfnHidden * dim);
        AddWeight($"{p}.feed_forward.w2.weight", dim * realFfnHidden);

        using var zdit = new ZImageDiT(loader, zparams);

        var x = new float[t * dim];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        var adaln = new float[adalnEmbedDim];
        for (int i = 0; i < adaln.Length; i++) adaln[i] = (float)(rng.NextDouble() * 0.5 - 0.25);

        // Real RoPE freqs for a simple sequential position assignment (matches ZImageRoPE.BuildFreqs's
        // own real output format: interleaved (cos,sin) pairs per token).
        var posIds = new int[t * 3];
        for (int i = 0; i < t; i++) { posIds[i * 3] = i; posIds[i * 3 + 1] = 0; posIds[i * 3 + 2] = 0; }
        var rope = new ZImageRoPE(zparams);
        var freqsInterleaved = rope.BuildFreqs(posIds, t);

        var cpuX = (float[])x.Clone();
        zdit.ApplyBlockForTest("layers.0", cpuX, t, freqsInterleaved, adaln, true);

        // De-interleave into separate cos/sin arrays for the GPU RoPE kernel.
        int halfHead = headDim / 2;
        var ropeCos = new float[t * halfHead];
        var ropeSin = new float[t * halfHead];
        for (int i = 0; i < t * halfHead; i++) { ropeCos[i] = freqsInterleaved[i * 2]; ropeSin[i] = freqsInterleaved[i * 2 + 1]; }

        using var gpuWeights = new ZImageGpuWeights(vulkan, loader.ReadF32, 1, dim, headDim, realFfnHidden, adalnEmbedDim);
        using var ws = new ZImageGpuWorkspace(vulkan, t, dim, realFfnHidden, ropeCos, ropeSin, headDim);
        using var xGpu = vulkan.Upload(x, TensorShape.D2(t, dim), exact: true);
        vulkan.ScaleInPlace(ws.X, 0f);
        vulkan.AddInPlace(ws.X, xGpu);
        using var adalnGpu = vulkan.Upload(adaln, TensorShape.D2(1, adalnEmbedDim), exact: true);

        zdit.ApplyBlockGpu(gpuWeights.Layers[0], ws, t, adalnGpu, vulkan);

        var gpuX = new float[t * dim];
        vulkan.Download(ws.X, gpuX);

        float maxDiff = 0f;
        double sumAbs = 0;
        int worstIdx = -1;
        for (int i = 0; i < cpuX.Length; i++)
        {
            float diff = MathF.Abs(cpuX[i] - gpuX[i]);
            if (diff > maxDiff) { maxDiff = diff; worstIdx = i; }
            sumAbs += MathF.Abs(cpuX[i]);
        }
        float meanAbs = (float)(sumAbs / cpuX.Length);
        Console.WriteLine($"[ZImage GPU Parity] maxDiff={maxDiff:F6} meanAbs={meanAbs:F6} worstIdx={worstIdx} CPU={cpuX[worstIdx]:F6} GPU={gpuX[worstIdx]:F6}");

        Assert.True(maxDiff < 0.1f * Math.Max(1f, meanAbs),
            $"Max difference {maxDiff} too large relative to mean magnitude {meanAbs}");
    }
}
