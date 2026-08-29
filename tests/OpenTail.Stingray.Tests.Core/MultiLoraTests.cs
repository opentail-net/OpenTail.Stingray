using OpenTail.Stingray.Core.Lora;

namespace OpenTail.Stingray.Tests.Core;

public sealed class MultiLoraTests
{
    [Fact]
    public void LoraLayer_ApplyDelta_ComputesAccurateLowRankDelta()
    {
        int inDim = 4;
        int outDim = 4;
        int rank = 2;
        float alpha = 2.0f; // scaling = 2.0 / 2 = 1.0

        // Down matrix A: [Rank, InDim] = [2, 4]
        // [ [1, 0, 1, 0],
        //   [0, 1, 0, 1] ]
        var down = new float[]
        {
            1f, 0f, 1f, 0f,
            0f, 1f, 0f, 1f
        };

        // Up matrix B: [OutDim, Rank] = [4, 2]
        // [ [1, 0],
        //   [0, 1],
        //   [1, 1],
        //   [0, 0] ]
        var up = new float[]
        {
            1f, 0f,
            0f, 1f,
            1f, 1f,
            0f, 0f
        };

        var layer = new LoraLayer("q_proj", 0, down, up, inDim, outDim, rank, alpha);

        // Input X = [2, 3, 4, 5]
        // X * A^T = [2*1 + 4*1, 3*1 + 5*1] = [6, 8]
        // (X * A^T) * B^T =
        // out[0] = 6*1 + 8*0 = 6
        // out[1] = 6*0 + 8*1 = 8
        // out[2] = 6*1 + 8*1 = 14
        // out[3] = 6*0 + 8*0 = 0
        var input = new float[] { 2f, 3f, 4f, 5f };
        var output = new float[4]; // starts with 0

        layer.ApplyDelta(input, output);

        Assert.Equal(6f, output[0], tolerance: 1e-5f);
        Assert.Equal(8f, output[1], tolerance: 1e-5f);
        Assert.Equal(14f, output[2], tolerance: 1e-5f);
        Assert.Equal(0f, output[3], tolerance: 1e-5f);
    }

    [Fact]
    public void LoraAdapter_ApplyDelta_RoutesByLayerAndModule()
    {
        int inDim = 2, outDim = 2, rank = 1;
        float alpha = 1.0f;
        var down = new float[] { 1f, 1f };
        var up = new float[] { 2f, 3f };

        var layer0Q = new LoraLayer("q_proj", 0, down, up, inDim, outDim, rank, alpha);
        var layer1K = new LoraLayer("k_proj", 1, down, up, inDim, outDim, rank, alpha);

        using var adapter = new LoraAdapter("test-adapter", "test.safetensors", new[] { layer0Q, layer1K });

        Assert.Equal(2, adapter.LayerCount);
        Assert.True(adapter.TryGetLayer(0, "q_proj", out _));
        Assert.True(adapter.TryGetLayer(1, "k_proj", out _));
        Assert.False(adapter.TryGetLayer(0, "k_proj", out _));

        var input = new float[] { 1f, 2f }; // sum = 3
        var output = new float[2];

        // Layer 0 Q: input * [1,1] = 3; 3 * [2,3] = [6, 9]
        adapter.ApplyDelta(0, "q_proj", input, output);
        Assert.Equal(6f, output[0]);
        Assert.Equal(9f, output[1]);

        // Unrelated layer/module should not modify output
        adapter.ApplyDelta(0, "v_proj", input, output);
        Assert.Equal(6f, output[0]);
        Assert.Equal(9f, output[1]);
    }

    [Fact]
    public void LoraRegistry_RegisterAndRetrieve_CachesByNameAndPath()
    {
        var registry = new LoraRegistry();
        using var adapter = new LoraAdapter("custom-coder", "models/loras/coder.safetensors", Array.Empty<LoraLayer>());

        registry.Register(adapter);

        Assert.True(registry.TryGet("custom-coder", out var retrieved1));
        Assert.Same(adapter, retrieved1);

        Assert.True(registry.TryGet("models/loras/coder.safetensors", out var retrieved2));
        Assert.Same(adapter, retrieved2);

        registry.Remove("custom-coder");
        Assert.False(registry.TryGet("custom-coder", out _));
    }
}
