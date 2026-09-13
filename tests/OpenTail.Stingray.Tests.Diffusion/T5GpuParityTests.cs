using System.Diagnostics;
using System.Runtime.InteropServices;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion;
using OpenTail.Stingray.Diffusion.TextEncoders;
using OpenTail.Stingray.Vulkan;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class T5GpuParityTests
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
    public void T5AttentionWithRelBias_IntegratedVulkanBackend_MatchesCpuReference()
    {
        using var vulkan = TryCreateVulkan();
        if (vulkan is null) return;

        const int numHeads = 64;
        const int headDim = 64;
        const int dim = numHeads * headDim; // 4096
        const int seq = 32; // test sequence length

        var rng = new Random(42);
        var qHost = new float[seq * dim];
        var kHost = new float[seq * dim];
        var vHost = new float[seq * dim];
        var biasHost = new float[numHeads * seq * seq];

        for (int i = 0; i < qHost.Length; i++) qHost[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * 0.1f;
        for (int i = 0; i < kHost.Length; i++) kHost[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * 0.1f;
        for (int i = 0; i < vHost.Length; i++) vHost[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * 0.1f;
        for (int i = 0; i < biasHost.Length; i++) biasHost[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * 0.5f;

        // CPU Reference Implementation
        var refOut = new float[seq * dim];
        float scale = 1f / MathF.Sqrt(headDim);

        for (int h = 0; h < numHeads; h++)
        {
            var scores = new float[seq * seq];
            for (int i = 0; i < seq; i++)
            {
                int qOff = i * dim + h * headDim;
                int relRowOff = (h * seq + i) * seq;

                for (int j = 0; j < seq; j++)
                {
                    int kOff = j * dim + h * headDim;
                    float dot = 0f;
                    for (int d = 0; d < headDim; d++)
                        dot += qHost[qOff + d] * kHost[kOff + d];
                    scores[i * seq + j] = dot * scale + biasHost[relRowOff + j];
                }

                // Softmax
                float maxS = float.NegativeInfinity;
                for (int j = 0; j < seq; j++) maxS = MathF.Max(maxS, scores[i * seq + j]);
                float sumExp = 0f;
                for (int j = 0; j < seq; j++)
                {
                    float e = MathF.Exp(scores[i * seq + j] - maxS);
                    scores[i * seq + j] = e;
                    sumExp += e;
                }
                float invSum = 1f / sumExp;
                for (int j = 0; j < seq; j++) scores[i * seq + j] *= invSum;

                // Accumulate V
                int outOff = i * dim + h * headDim;
                for (int j = 0; j < seq; j++)
                {
                    float pScore = scores[i * seq + j];
                    int vOff = j * dim + h * headDim;
                    for (int d = 0; d < headDim; d++)
                        refOut[outOff + d] += pScore * vHost[vOff + d];
                }
            }
        }

        // GPU Test via VulkanBackend method
        var qGpu = vulkan.Upload(qHost, TensorShape.D2(seq, dim), exact: true);
        var kGpu = vulkan.Upload(kHost, TensorShape.D2(seq, dim), exact: true);
        var vGpu = vulkan.Upload(vHost, TensorShape.D2(seq, dim), exact: true);
        var biasGpu = vulkan.Upload(biasHost, TensorShape.D3(numHeads, seq, seq), exact: true);
        var oGpu = vulkan.Allocate(TensorShape.D2(seq, dim));

        vulkan.T5MultiHeadAttentionRelBias(oGpu, qGpu, kGpu, vGpu, biasGpu, seq, seq, numHeads, headDim);

        var gpuOut = new float[refOut.Length];
        vulkan.Download(oGpu, gpuOut);

        float maxDiff = 0f;
        for (int i = 0; i < refOut.Length; i++)
        {
            float diff = MathF.Abs(refOut[i] - gpuOut[i]);
            if (diff > maxDiff) maxDiff = diff;
        }

        Assert.True(maxDiff < 1e-3f, $"Max difference {maxDiff} exceeded tolerance 1e-3");
    }

    [Fact]
    public void T5Encoder_EncodeGpu_MatchesCpuReference_Numerically()
    {
        using var vulkan = TryCreateVulkan();
        if (vulkan is null) return;

        const int layers = 24;
        const int dim = 4096;
        const int ffDim = 10240;
        const int seq = 16;
        const int vocabSize = 32128;

        var rng = new Random(1234);
        var loader = new SyntheticWeightLoader();

        // Token embeddings
        var tokEmb = new float[vocabSize * dim];
        for (int i = 0; i < seq * dim; i++) tokEmb[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * 0.05f;
        loader.Add("shared.weight", tokEmb);

        // Rel pos bias weights: [32, 64]
        var rpW = new float[32 * 64];
        for (int i = 0; i < rpW.Length; i++) rpW[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * 0.1f;
        loader.Add("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight", rpW);

        // 24 Layers - reuse shared weight arrays to keep test instant and lightweight
        var ln0 = new float[dim];
        Array.Fill(ln0, 1.0f);
        var qW = new float[dim * dim];
        var kW = new float[dim * dim];
        var vW = new float[dim * dim];
        var oW = new float[dim * dim];
        for (int i = 0; i < dim * dim; i += 100)
        {
            qW[i] = 0.01f;
            kW[i] = 0.01f;
            vW[i] = 0.01f;
            oW[i] = 0.01f;
        }

        var ln1 = new float[dim];
        Array.Fill(ln1, 1.0f);
        var wi0 = new float[ffDim * dim];
        var wi1 = new float[ffDim * dim];
        var wo = new float[dim * ffDim];
        for (int i = 0; i < 1000; i++)
        {
            wi0[i] = 0.01f;
            wi1[i] = 0.01f;
            wo[i] = 0.01f;
        }

        for (int l = 0; l < layers; l++)
        {
            string p = $"encoder.block.{l}.layer";
            loader.Add($"{p}.0.layer_norm.weight", ln0);
            loader.Add($"{p}.0.SelfAttention.q.weight", qW);
            loader.Add($"{p}.0.SelfAttention.k.weight", kW);
            loader.Add($"{p}.0.SelfAttention.v.weight", vW);
            loader.Add($"{p}.0.SelfAttention.o.weight", oW);
            loader.Add($"{p}.1.layer_norm.weight", ln1);
            loader.Add($"{p}.1.DenseReluDense.wi_0.weight", wi0);
            loader.Add($"{p}.1.DenseReluDense.wi_1.weight", wi1);
            loader.Add($"{p}.1.DenseReluDense.wo.weight", wo);
        }

        var fnW = new float[dim];
        Array.Fill(fnW, 1.0f);
        loader.Add("encoder.final_layer_norm.weight", fnW);

        using var encoder = T5Encoder.FromLoader(loader);
        var tokens = new int[seq];
        for (int i = 0; i < seq; i++) tokens[i] = i;

        var cpuOut = encoder.Encode(tokens);
        var gpuOut = encoder.EncodeGpu(tokens, vulkan);

        Assert.Equal(cpuOut.Length, gpuOut.Length);
        float maxDiff = 0f;
        for (int i = 0; i < cpuOut.Length; i++)
        {
            float diff = MathF.Abs(cpuOut[i] - gpuOut[i]);
            if (diff > maxDiff) maxDiff = diff;
        }

        Assert.True(maxDiff < 0.08f, $"Max difference {maxDiff} exceeded tolerance 0.08");
    }
}
