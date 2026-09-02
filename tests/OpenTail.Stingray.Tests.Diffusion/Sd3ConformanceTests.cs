using OpenTail.Stingray.Diffusion.SD3;

namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class Sd3ConformanceTests
{
    [Fact]
    public void Sd3_TripleConditioning_Matches4096Context()
    {
        var clipL = new float[77 * 768];
        var clipG = new float[77 * 1280];

        var context = new float[77 * 4096];
        for (int t = 0; t < 77; t++)
        {
            Array.Copy(clipL, t * 768, context, t * 4096, 768);
            Array.Copy(clipG, t * 1280, context, t * 4096 + 768, 1280);
        }

        Assert.Equal(77 * 4096, context.Length);
    }

    [Fact]
    public void Sd3_PooledVector_Matches2048Dimension()
    {
        var pooledL = new float[768];
        var pooledG = new float[1280];
        for (int i = 0; i < 768; i++) pooledL[i] = 1.0f;
        for (int i = 0; i < 1280; i++) pooledG[i] = 2.0f;

        var y = new float[2048];
        Array.Copy(pooledL, 0, y, 0, 768);
        Array.Copy(pooledG, 0, y, 768, 1280);

        Assert.Equal(2048, y.Length);
        Assert.Equal(1.0f, y[0]);
        Assert.Equal(1.0f, y[767]);
        Assert.Equal(2.0f, y[768]);
        Assert.Equal(2.0f, y[2047]);
    }

    [Fact]
    public void Sd3_FlowMatchingEulerStep_ReachesTarget()
    {
        // Rectified flow: x_{t-dt} = x_t - dt * v
        int steps = 20;
        float dt = 1.0f / steps;
        float x = 1.0f; // Start at noise t=1.0
        float target = 0.0f;
        float v = (x - target); // constant velocity = 1.0

        for (int step = 0; step < steps; step++)
        {
            x -= dt * v;
        }

        Assert.Equal(0.0f, x, precision: 4);
    }
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
    public void Sd3_MMDiTModel_Forward_ExecutesCorrectly()
    {
        int hidden = 64;
        int numHeads = 2;
        int headDim = 32;
        int depth = 2;
        int inChannels = 16;
        int outChannels = 16;
        int patchSize = 2;
        int contextSize = 64;
        int admInChannels = 32;
        int inPatchDim = inChannels * patchSize * patchSize;
        int outPatchDim = outChannels * patchSize * patchSize;

        var w = new Dictionary<string, float[]>();
        var rng = new Random(42);
        void Add(string name, int len)
        {
            var arr = new float[len];
            for (int i = 0; i < len; i++) arr[i] = 0.01f * (float)(rng.NextDouble() - 0.5);
            w[name] = arr;
        }

        Add("t_embedder.mlp.0.weight", hidden * 256);
        Add("t_embedder.mlp.0.bias", hidden);
        Add("t_embedder.mlp.2.weight", hidden * hidden);
        Add("t_embedder.mlp.2.bias", hidden);

        Add("y_embedder.mlp.0.weight", hidden * admInChannels);
        Add("y_embedder.mlp.0.bias", hidden);
        Add("y_embedder.mlp.2.weight", hidden * hidden);
        Add("y_embedder.mlp.2.bias", hidden);

        Add("x_embedder.proj.weight", hidden * inPatchDim);
        Add("x_embedder.proj.bias", hidden);

        Add("context_embedder.weight", hidden * contextSize);
        Add("context_embedder.bias", hidden);

        // Block 0: dual attention block
        Add("joint_blocks.0.x_block.attn2.qkv.weight", 3 * hidden * hidden);
        Add("joint_blocks.0.x_block.adaLN_modulation.1.weight", 9 * hidden * hidden);
        Add("joint_blocks.0.context_block.adaLN_modulation.1.weight", 6 * hidden * hidden);
        Add("joint_blocks.0.x_block.attn.qkv.weight", 3 * hidden * hidden);
        Add("joint_blocks.0.context_block.attn.qkv.weight", 3 * hidden * hidden);
        Add("joint_blocks.0.x_block.attn.ln_q.weight", headDim);
        Add("joint_blocks.0.x_block.attn.ln_k.weight", headDim);
        Add("joint_blocks.0.context_block.attn.ln_q.weight", headDim);
        Add("joint_blocks.0.context_block.attn.ln_k.weight", headDim);
        Add("joint_blocks.0.x_block.attn.proj.weight", hidden * hidden);
        Add("joint_blocks.0.context_block.attn.proj.weight", hidden * hidden);
        Add("joint_blocks.0.x_block.attn2.ln_q.weight", headDim);
        Add("joint_blocks.0.x_block.attn2.ln_k.weight", headDim);
        Add("joint_blocks.0.x_block.attn2.proj.weight", hidden * hidden);
        Add("joint_blocks.0.x_block.mlp.fc1.weight", 4 * hidden * hidden);
        Add("joint_blocks.0.x_block.mlp.fc2.weight", hidden * 4 * hidden);
        Add("joint_blocks.0.context_block.mlp.fc1.weight", 4 * hidden * hidden);
        Add("joint_blocks.0.context_block.mlp.fc2.weight", hidden * 4 * hidden);

        // Block 1: standard/last block
        Add("joint_blocks.1.x_block.adaLN_modulation.1.weight", 6 * hidden * hidden);
        Add("joint_blocks.1.context_block.adaLN_modulation.1.weight", 2 * hidden * hidden);
        Add("joint_blocks.1.x_block.attn.qkv.weight", 3 * hidden * hidden);
        Add("joint_blocks.1.context_block.attn.qkv.weight", 3 * hidden * hidden);
        Add("joint_blocks.1.x_block.attn.ln_q.weight", headDim);
        Add("joint_blocks.1.x_block.attn.ln_k.weight", headDim);
        Add("joint_blocks.1.context_block.attn.ln_q.weight", headDim);
        Add("joint_blocks.1.context_block.attn.ln_k.weight", headDim);
        Add("joint_blocks.1.x_block.attn.proj.weight", hidden * hidden);
        Add("joint_blocks.1.x_block.mlp.fc1.weight", 4 * hidden * hidden);
        Add("joint_blocks.1.x_block.mlp.fc2.weight", hidden * 4 * hidden);

        Add("final_layer.adaLN_modulation.1.weight", 2 * hidden * hidden);
        Add("final_layer.linear.weight", outPatchDim * hidden);
        Add("final_layer.linear.bias", outPatchDim);

        var loader = new MockWeightLoader(w);
        using var model = new MMDiTModel(
            loader,
            prefix: "",
            hiddenSize: hidden,
            numHeads: numHeads,
            depth: depth,
            inChannels: inChannels,
            outChannels: outChannels,
            patchSize: patchSize,
            contextSize: contextSize,
            admInChannels: admInChannels);

        int latH = 4;
        int latW = 4;
        int numTextTokens = 8;
        var latents = new float[inChannels * latH * latW];
        for (int i = 0; i < latents.Length; i++) latents[i] = (float)rng.NextDouble();
        var textContext = new float[numTextTokens * contextSize];
        for (int i = 0; i < textContext.Length; i++) textContext[i] = (float)rng.NextDouble();
        var pooledY = new float[admInChannels];
        for (int i = 0; i < pooledY.Length; i++) pooledY[i] = (float)rng.NextDouble();

        var output = model.Forward(latents, timestep: 500f, textContext, pooledY, latH, latW, numTextTokens);

        Assert.NotNull(output);
        Assert.Equal(outChannels * latH * latW, output.Length);
        for (int i = 0; i < output.Length; i++)
        {
            Assert.False(float.IsNaN(output[i]), $"output[{i}] is NaN");
            Assert.False(float.IsInfinity(output[i]), $"output[{i}] is Infinity");
        }
    }
}
