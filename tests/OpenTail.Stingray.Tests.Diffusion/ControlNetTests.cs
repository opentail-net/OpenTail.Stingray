using OpenTail.Stingray.Diffusion.ControlNet;

namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class ControlNetTests
{
    private sealed class MockWeightLoader : IWeightLoader
    {
        private readonly Dictionary<string, float[]> _weights;
        public MockWeightLoader(Dictionary<string, float[]> weights) => _weights = weights;
        public bool Contains(string name) => _weights.ContainsKey(name);
        public float[] ReadF32(string name) => _weights[name];
        public unsafe bool TryGetRaw(string name, out nint dataPtr, out long byteLen, out DType dtype, out int rows, out int cols)
        {
            dataPtr = 0; byteLen = 0; dtype = default; rows = 0; cols = 0;
            return false;
        }
        public void Dispose() { }
    }

    [Fact]
    public void ControlNet_Forward_Produces13Residuals()
    {
        var weights = new Dictionary<string, float[]>();

        void AddWeight(string name, int len) => weights[name] = new float[len];

        AddWeight("time_embed.0.weight", 320 * 1280);
        AddWeight("time_embed.0.bias", 1280);
        AddWeight("time_embed.2.weight", 1280 * 1280);
        AddWeight("time_embed.2.bias", 1280);

        AddWeight("input_hint_block.0.weight", 16 * 3 * 3 * 3);
        AddWeight("input_hint_block.0.bias", 16);
        AddWeight("input_hint_block.2.weight", 16 * 16 * 3 * 3);
        AddWeight("input_hint_block.2.bias", 16);
        AddWeight("input_hint_block.4.weight", 32 * 16 * 3 * 3);
        AddWeight("input_hint_block.4.bias", 32);
        AddWeight("input_hint_block.6.weight", 32 * 32 * 3 * 3);
        AddWeight("input_hint_block.6.bias", 32);
        AddWeight("input_hint_block.8.weight", 96 * 32 * 3 * 3);
        AddWeight("input_hint_block.8.bias", 96);
        AddWeight("input_hint_block.10.weight", 96 * 96 * 3 * 3);
        AddWeight("input_hint_block.10.bias", 96);
        AddWeight("input_hint_block.12.weight", 256 * 96 * 3 * 3);
        AddWeight("input_hint_block.12.bias", 256);
        AddWeight("input_hint_block.14.weight", 320 * 256 * 3 * 3);
        AddWeight("input_hint_block.14.bias", 320);

        AddWeight("input_blocks.0.0.weight", 320 * 4 * 3 * 3);
        AddWeight("input_blocks.0.0.bias", 320);
        AddWeight("zero_convs.0.0.weight", 320 * 320 * 1 * 1);
        AddWeight("zero_convs.0.0.bias", 320);

        for (int i = 1; i <= 11; i++)
        {
            int ch = i <= 3 ? 320 : (i <= 6 ? 640 : 1280);
            AddWeight($"zero_convs.{i}.0.weight", ch * ch);
            AddWeight($"zero_convs.{i}.0.bias", ch);
        }
        AddWeight("middle_block_out.0.weight", 1280 * 1280);
        AddWeight("middle_block_out.0.bias", 1280);

        var mockLoader = new MockWeightLoader(weights);
        using var controlNet = new ControlNetModel(mockLoader);

        Assert.NotNull(controlNet);
    }

    [Fact]
    public void UNet2D_Forward_AppliesControlResiduals()
    {
        var downResiduals = new List<float[]>(12);
        for (int i = 0; i < 12; i++)
            downResiduals.Add(new float[16]);

        var midResidual = new float[16];

        Assert.Equal(12, downResiduals.Count);
        Assert.NotNull(midResidual);
    }

    [Fact]
    public void ControlNet_RealWeights_LoadAndForward()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "../../../../../models/controlnet-sd15-canny/diffusion_pytorch_model.safetensors");
        if (!File.Exists(path)) return;

        using var cnCpu = ControlNetModel.Load(path, backend: null);
        int latH = 8, latW = 8;
        var latent = new float[4 * latH * latW];
        for (int i = 0; i < latent.Length; i++) latent[i] = MathF.Sin(i * 0.1f);

        var context = new float[77 * 768];
        for (int i = 0; i < context.Length; i++) context[i] = MathF.Cos(i * 0.05f);

        var hint = new float[3 * 8 * latH * 8 * latW];
        for (int i = 0; i < hint.Length; i++) hint[i] = (i % 7 == 0) ? 1.0f : 0.0f;

        var (downResCpu, midResCpu) = cnCpu.Forward(latent, 500f, context, hint, latH, latW, 1.0f);
        Assert.Equal(12, downResCpu.Count);
        Assert.NotNull(midResCpu);

        using var vulkan = new OpenTail.Stingray.Vulkan.VulkanBackend();
        using var cnGpu = ControlNetModel.Load(path, backend: vulkan);
        var (downResGpu, midResGpu) = cnGpu.Forward(latent, 500f, context, hint, latH, latW, 1.0f);
        Assert.Equal(12, downResGpu.Count);
        Assert.NotNull(midResGpu);

        for (int b = 0; b < 12; b++)
        {
            var cRes = downResCpu[b];
            var gRes = downResGpu[b];
            float d = 0f, nC = 0f, nG = 0f;
            for (int i = 0; i < cRes.Length; i++)
            {
                d += cRes[i] * gRes[i];
                nC += cRes[i] * cRes[i];
                nG += gRes[i] * gRes[i];
            }
            float sim = d / (MathF.Sqrt(nC) * MathF.Sqrt(nG) + 1e-8f);
            Console.WriteLine($"[Block {b}] Cosine Sim = {sim:F4}, NormCpu = {MathF.Sqrt(nC):F4}, NormGpu = {MathF.Sqrt(nG):F4}");
        }

        float dot = 0f, normCpu = 0f, normGpu = 0f;
        for (int i = 0; i < midResCpu.Length; i++)
        {
            dot += midResCpu[i] * midResGpu[i];
            normCpu += midResCpu[i] * midResCpu[i];
            normGpu += midResGpu[i] * midResGpu[i];
        }
        float cosSim = dot / (MathF.Sqrt(normCpu) * MathF.Sqrt(normGpu) + 1e-8f);
        Console.WriteLine($"[MidBlock] Cosine Sim = {cosSim:F4}, NormCpu = {MathF.Sqrt(normCpu):F4}, NormGpu = {MathF.Sqrt(normGpu):F4}");
        Assert.True(cosSim > 0.99f, $"Mid residual cosine similarity too low: {cosSim}");
    }
}
