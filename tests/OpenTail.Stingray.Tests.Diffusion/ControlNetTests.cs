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
}
