using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.LTXVideo;

namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class LtxVideoTests
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

    /// <summary>Builds a tiny synthetic checkpoint with real LTX tensor names (deterministic
    /// pseudo-random weights) at a structurally-correct but small scale (2 layers, hidden=64,
    /// heads=2, headDim=32), to exercise <see cref="LtxVideoModel"/>'s shape/dispatch logic without
    /// requiring the real multi-GB checkpoint.</summary>
    private static MockWeightLoader BuildSyntheticCheckpoint(
        int numLayers, int hidden, int heads, int headDim, int inChannels, int crossDim, int captionCh)
    {
        var w = new Dictionary<string, float[]>();
        var rng = new Random(1234);
        void Add(string name, int len)
        {
            var arr = new float[len];
            for (int i = 0; i < len; i++) arr[i] = 0.02f * ((float)rng.NextDouble() - 0.5f);
            w[name] = arr;
        }

        const string p = "model.diffusion_model.";
        // Steers `LTXAVConfig::infer_attention_layout`'s head-count inference (real reference reads
        // this bias tensor's length as the preferred head count) without setting `self_attention_gated`
        // (that flag only flips on the corresponding `.weight` tensor's presence, which is
        // deliberately omitted here since v0.9.1's real checkpoint has no gating).
        Add(p + "transformer_blocks.0.attn1.to_gate_logits.bias", heads);
        Add(p + "patchify_proj.weight", hidden * inChannels);
        Add(p + "patchify_proj.bias", hidden);
        Add(p + "caption_projection.linear_1.weight", hidden * captionCh);
        Add(p + "caption_projection.linear_1.bias", hidden);
        Add(p + "caption_projection.linear_2.weight", hidden * hidden);
        Add(p + "caption_projection.linear_2.bias", hidden);
        Add(p + "proj_out.weight", inChannels * hidden);
        Add(p + "proj_out.bias", inChannels);
        Add(p + "scale_shift_table", 2 * hidden);
        Add(p + "adaln_single.emb.timestep_embedder.linear_1.weight", hidden * 256);
        Add(p + "adaln_single.emb.timestep_embedder.linear_1.bias", hidden);
        Add(p + "adaln_single.emb.timestep_embedder.linear_2.weight", hidden * hidden);
        Add(p + "adaln_single.emb.timestep_embedder.linear_2.bias", hidden);
        Add(p + "adaln_single.linear.weight", hidden * 6 * hidden);
        Add(p + "adaln_single.linear.bias", hidden * 6);

        for (int l = 0; l < numLayers; l++)
        {
            string b = $"{p}transformer_blocks.{l}.";
            Add(b + "scale_shift_table", 6 * hidden);

            Add(b + "attn1.to_q.weight", hidden * hidden);
            Add(b + "attn1.to_q.bias", hidden);
            Add(b + "attn1.to_k.weight", hidden * hidden);
            Add(b + "attn1.to_k.bias", hidden);
            Add(b + "attn1.to_v.weight", hidden * hidden);
            Add(b + "attn1.to_v.bias", hidden);
            Add(b + "attn1.to_out.0.weight", hidden * hidden);
            Add(b + "attn1.to_out.0.bias", hidden);
            Add(b + "attn1.q_norm.weight", hidden);
            Add(b + "attn1.k_norm.weight", hidden);

            Add(b + "attn2.to_q.weight", hidden * hidden);
            Add(b + "attn2.to_q.bias", hidden);
            Add(b + "attn2.to_k.weight", hidden * crossDim);
            Add(b + "attn2.to_k.bias", hidden);
            Add(b + "attn2.to_v.weight", hidden * crossDim);
            Add(b + "attn2.to_v.bias", hidden);
            Add(b + "attn2.to_out.0.weight", hidden * hidden);
            Add(b + "attn2.to_out.0.bias", hidden);
            Add(b + "attn2.q_norm.weight", hidden);
            Add(b + "attn2.k_norm.weight", hidden);

            Add(b + "ff.net.0.proj.weight", hidden * 4 * hidden);
            Add(b + "ff.net.0.proj.bias", hidden * 4);
            Add(b + "ff.net.2.weight", hidden * hidden * 4);
            Add(b + "ff.net.2.bias", hidden);
        }

        return new MockWeightLoader(w);
    }

    [Fact]
    public void LtxVideoRoPE_ComputeContinuous3DRoPE_ValidatesShapesAndRanges()
    {
        int numFrames = 3;
        int patchH = 16;
        int patchW = 16;
        int headDim = 64;

        var (cos, sin) = LtxVideoRoPE.ComputeContinuous3DRoPE(numFrames, patchH, patchW, headDim);

        int totalTokens = numFrames * patchH * patchW;
        Assert.Equal(totalTokens * headDim, cos.Length);
        Assert.Equal(totalTokens * headDim, sin.Length);

        for (int i = 0; i < cos.Length; i++)
        {
            Assert.InRange(cos[i], -1.0001f, 1.0001f);
            Assert.InRange(sin[i], -1.0001f, 1.0001f);
        }
    }

    [Fact]
    public void LtxVideoModel_DetectConfig_ReadsShapesFromCheckpoint()
    {
        int hidden = 256, heads = 8, headDim = 32, inChannels = 128, crossDim = 256, captionCh = 200;
        using var loader = BuildSyntheticCheckpoint(2, hidden, heads, headDim, inChannels, crossDim, captionCh);
        var model = new LtxVideoModel(loader);

        Assert.Equal(2, model.NumLayers);
        Assert.Equal(hidden, model.HiddenSize);
        Assert.Equal(inChannels, model.InChannels);
        Assert.Equal(inChannels, model.OutChannels);
        Assert.Equal(crossDim, model.CrossAttentionDim);
        Assert.Equal(captionCh, model.CaptionChannels);
        Assert.Equal(hidden / headDim, model.NumHeads);
    }

    [Fact]
    public void LtxVideoModel_Forward_ComputesVelocityPrediction()
    {
        int hidden = 256, headDim = 32, inChannels = 128, crossDim = 256, captionCh = 200;
        using var loader = BuildSyntheticCheckpoint(2, hidden, hidden / headDim, headDim, inChannels, crossDim, captionCh);
        var model = new LtxVideoModel(loader);

        int numFrames = 2;
        int patchH = 4;
        int patchW = 4;
        int numTokens = numFrames * patchH * patchW;

        var latents = new float[numTokens * inChannels];
        for (int i = 0; i < latents.Length; i++)
            latents[i] = 0.05f * (i % 17 - 8);

        var context = new float[16 * captionCh];
        for (int i = 0; i < context.Length; i++)
            context[i] = 0.01f * (i % 5);

        var velocity = model.Forward(latents, timestep: 500f, context, numFrames, patchH, patchW);

        Assert.Equal(numTokens * model.OutChannels, velocity.Length);
        for (int i = 0; i < velocity.Length; i++)
        {
            Assert.False(float.IsNaN(velocity[i]));
            Assert.False(float.IsInfinity(velocity[i]));
        }
    }

    [Fact]
    public void LtxVideoPipeline_GenerateVideo_ProducesValidFrames()
    {
        int hidden = 256, headDim = 32, inChannels = 128, crossDim = 256, captionCh = 200;
        using var loader = BuildSyntheticCheckpoint(2, hidden, hidden / headDim, headDim, inChannels, crossDim, captionCh);
        var model = new LtxVideoModel(loader);
        using var pipeline = new LtxVideoPipeline(model, temporalScale: 8, spatialScale: 32);

        var frames = pipeline.GenerateVideo(
            prompt: "A high speed car drifting on neon wet asphalt at night",
            width: 64,
            height: 64,
            numFrames: 1,
            steps: 2,
            seed: 42);

        Assert.NotEmpty(frames);
        Assert.Equal(3 * 64 * 64, frames[0].Length);
    }
}
