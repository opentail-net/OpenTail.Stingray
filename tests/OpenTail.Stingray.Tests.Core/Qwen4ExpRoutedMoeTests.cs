using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Integrated routed-MoE check: router -> top-2 of 4 experts -> stacked-tensor expert slicing -> weighted sum.
/// Expected values are computed by hand in the test body from the fixture's deliberately sparse matrices.
/// </summary>
public sealed unsafe class Qwen4ExpRoutedMoeTests
{
    private const int E = 4, TopK = 2, Dim = 4, Inter = 2;

    [Fact]
    public void ExecuteMoe_RoutedExperts_ReadStackedTensorsAndWeightTheSelectedOnes()
    {
        var allocations = new List<nint>();
        Qwen4ExpTensorRef Ref(string name, long[] dims, float[] data)
        {
            nint mem = (nint)System.Runtime.InteropServices.NativeMemory.Alloc((nuint)data.Length, sizeof(float));
            allocations.Add(mem);
            for (int i = 0; i < data.Length; i++) ((float*)mem)[i] = data[i];
            var info = new GgufTensorInfo(name, dims.Length, dims, DType.Float32, 0);
            return new Qwen4ExpTensorRef(name, info, (byte*)mem);
        }

        try
        {
            // input x = [1,0,0,0]; every matvec then just reads column 0 of each row.
            // router[e] = W[e*Dim + 0]: logits [0, 5, -2, 4] -> experts 1 and 3 win, weights softmax-renormalised.
            var router = new float[E * Dim];
            float[] logits = [0f, 5f, -2f, 4f];
            for (int e = 0; e < E; e++) router[e * Dim] = logits[e];

            // stacked gate/up: dims [Dim, Inter, E]; row (e*Inter + j) is expert e's output unit j.
            var gate = new float[E * Inter * Dim];
            var up = new float[E * Inter * Dim];
            // down: dims [Inter, Dim, E]; row (e*Dim + d), columns j.
            var down = new float[E * Dim * Inter];

            // expert 1: unit 0 has gate 1, up 1; expert 3: unit 0 has gate 2, up 1. Experts 0 and 2 get loud values
            // that must NOT show up in the output (they are not selected).
            gate[(1 * Inter + 0) * Dim] = 1f; up[(1 * Inter + 0) * Dim] = 1f;
            gate[(3 * Inter + 0) * Dim] = 2f; up[(3 * Inter + 0) * Dim] = 1f;
            gate[(0 * Inter + 0) * Dim] = 9f; up[(0 * Inter + 0) * Dim] = 9f;
            gate[(2 * Inter + 0) * Dim] = 9f; up[(2 * Inter + 0) * Dim] = 9f;
            // every expert writes its unit-0 activation to output dim (e % Dim)... use dim 0 for expert 1, dim 2 for 3
            down[(1 * Dim + 0) * Inter + 0] = 1f;
            down[(3 * Dim + 2) * Inter + 0] = 1f;
            down[(0 * Dim + 1) * Inter + 0] = 1f;
            down[(2 * Dim + 3) * Inter + 0] = 1f;

            var layer = new Qwen4ExpLayerTensors
            {
                LayerIndex = 0,
                FfnGateInp = Ref("blk.0.ffn_gate_inp.weight", [Dim, E], router),
                FfnGateExps = Ref("blk.0.ffn_gate_exps.weight", [Dim, Inter, E], gate),
                FfnUpExps = Ref("blk.0.ffn_up_exps.weight", [Dim, Inter, E], up),
                FfnDownExps = Ref("blk.0.ffn_down_exps.weight", [Inter, Dim, E], down),
            };

            var hp = new Qwen4ExpHyperparams
            {
                EmbedDim = Dim, HyperConnectionCount = 1, NumLayer = 1, ExpertCount = E, ExpertUsedCount = TopK,
                RecurrentLayers = [true],
            };
            var tokEmbd = Ref("token_embd.weight", [Dim, 2], new float[Dim * 2]);
            var tensors = new Qwen4ExpTensorSet(tokEmbd, tokEmbd, tokEmbd, tokEmbd, tokEmbd, null, [layer]);
            using var fwd = new Qwen4ExpForwardPass(null!, hp, tensors);

            var output = new float[Dim];
            fwd.ExecuteMoe(layer, [1f, 0f, 0f, 0f], output);

            float w1 = MathF.Exp(5f) / (MathF.Exp(5f) + MathF.Exp(4f));
            float w3 = 1f - w1;
            float silu1 = 1f / (1f + MathF.Exp(-1f));                 // silu(1) * up(1)
            float silu2 = 2f / (1f + MathF.Exp(-2f));                 // silu(2) * up(1)

            Assert.Equal(w1 * silu1, output[0], 1e-5f);  // expert 1 -> dim 0
            Assert.Equal(0f, output[1], 1e-6f);          // expert 0 not selected
            Assert.Equal(w3 * silu2, output[2], 1e-5f);  // expert 3 -> dim 2
            Assert.Equal(0f, output[3], 1e-6f);          // expert 2 not selected
        }
        finally
        {
            foreach (var p in allocations) System.Runtime.InteropServices.NativeMemory.Free((void*)p);
        }
    }
}
