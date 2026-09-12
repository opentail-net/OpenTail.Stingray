using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class ZImageDiTParityTests
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

    [Fact]
    public void ZImageDiT_Forward_ProducesConsistentOutputAcrossStepsWithContextCaching()
    {
        var p = new ZImageParams
        {
            Dim = 256,
            NHeads = 2,
            NLayers = 1,
            NRefinerLayers = 1,
            CapFeatDim = 64,
            InChannels = 16,
            PatchSize = 2,
            AdalnEmbedDim = 64
        };

        var loader = new SyntheticWeightLoader();
        var random = new Random(42);

        void AddWeight(string name, int size)
        {
            var arr = new float[size];
            for (int i = 0; i < size; i++) arr[i] = (float)(random.NextDouble() * 0.02 - 0.01);
            loader.Add(name, arr);
        }

        int dim = p.Dim;
        int headDim = p.HeadDim;
        int ffnHidden = p.FfnHidden;

        AddWeight("x_embedder.weight", dim * p.PatchDim);
        AddWeight("x_embedder.bias", dim);
        AddWeight("cap_embedder.0.weight", p.CapFeatDim);
        AddWeight("cap_embedder.1.weight", dim * p.CapFeatDim);
        AddWeight("cap_embedder.1.bias", dim);
        AddWeight("t_embedder.mlp.0.weight", 1024 * 256);
        AddWeight("t_embedder.mlp.0.bias", 1024);
        AddWeight("t_embedder.mlp.2.weight", p.AdalnEmbedDim * 1024);
        AddWeight("t_embedder.mlp.2.bias", p.AdalnEmbedDim);

        void AddBlock(string prefix, bool modulated)
        {
            if (modulated)
            {
                AddWeight($"{prefix}.adaLN_modulation.0.weight", 4 * dim * p.AdalnEmbedDim);
                AddWeight($"{prefix}.adaLN_modulation.0.bias", 4 * dim);
            }
            AddWeight($"{prefix}.attention_norm1.weight", dim);
            AddWeight($"{prefix}.attention.qkv.weight", 3 * dim * dim);
            AddWeight($"{prefix}.attention.q_norm.weight", headDim);
            AddWeight($"{prefix}.attention.k_norm.weight", headDim);
            AddWeight($"{prefix}.attention.out.weight", dim * dim);
            AddWeight($"{prefix}.attention_norm2.weight", dim);
            AddWeight($"{prefix}.ffn_norm1.weight", dim);
            AddWeight($"{prefix}.feed_forward.w1.weight", ffnHidden * dim);
            AddWeight($"{prefix}.feed_forward.w3.weight", ffnHidden * dim);
            AddWeight($"{prefix}.feed_forward.w2.weight", dim * ffnHidden);
            AddWeight($"{prefix}.ffn_norm2.weight", dim);
        }

        AddBlock("context_refiner.0", false);
        AddBlock("noise_refiner.0", true);
        AddBlock("layers.0", true);

        AddWeight("final_layer.adaLN_modulation.1.weight", dim * p.AdalnEmbedDim);
        AddWeight("final_layer.adaLN_modulation.1.bias", dim);
        AddWeight("final_layer.linear.weight", p.PatchDim * dim);
        AddWeight("final_layer.linear.bias", p.PatchDim);

        using var dit = new ZImageDiT(loader, p);

        int nImg = 4;
        int nTxt = 2;
        var imgPatches = new float[nImg * p.PatchDim];
        var txtEmbeds = new float[nTxt * p.CapFeatDim];
        var imgPosIds = ZImageRoPE.ImagePosIds(nTxt, 2, 2);
        var txtPosIds = ZImageRoPE.TextPosIds(nTxt);

        for (int i = 0; i < imgPatches.Length; i++) imgPatches[i] = (float)random.NextDouble();
        for (int i = 0; i < txtEmbeds.Length; i++) txtEmbeds[i] = (float)random.NextDouble();

        // Run step 1 (populates context cache)
        var outStep1 = dit.Forward(imgPatches, imgPosIds, txtEmbeds, txtPosIds, 1.0f);
        Assert.NotNull(outStep1);
        Assert.Equal(nImg * p.PatchDim, outStep1.Length);

        // Run step 2 with same txtEmbeds (uses cached refined context)
        var outStep2 = dit.Forward(imgPatches, imgPosIds, txtEmbeds, txtPosIds, 1.0f);
        Assert.Equal(outStep1.Length, outStep2.Length);

        for (int i = 0; i < outStep1.Length; i++)
        {
            Assert.True(MathF.Abs(outStep1[i] - outStep2[i]) < 1e-6f,
                $"Mismatch between step 1 and step 2 at index {i}: {outStep1[i]} vs {outStep2[i]}");
        }
    }
}
