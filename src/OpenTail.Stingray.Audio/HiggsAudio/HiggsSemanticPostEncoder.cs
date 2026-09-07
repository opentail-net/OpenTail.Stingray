namespace OpenTail.Stingray.Audio.HiggsAudio;

public sealed class HiggsSemanticResidualUnitWeights
{
    public required float[] Conv1Weight { get; init; } // [768, 768, 3], no bias
    public required float[] Conv2Weight { get; init; } // [768, 768, 1], no bias
}

public sealed class HiggsSemanticEncoderBlockWeights
{
    public required HiggsSemanticResidualUnitWeights[] ResidualUnits { get; init; } // 2
    public required float[] ConvWeight { get; init; } // [768, 768, 3], WITH bias
    public required float[] ConvBias { get; init; }
}

/// <summary>
/// Real weights + forward pass for Higgs Audio TTS's `semantic_encoder` post-network -- ported
/// from `codec.cpp`'s `semantic_encoder`/`semantic_residual_unit`/`load_semantic_encoder_block`
/// (not guessed), the real separate real ELU-activated residual-conv stage that runs AFTER
/// `hubert_hidden_state_mean` (<see cref="OmniVoice.OmniVoiceSemanticEncoder.ForwardHiddenStateMean"/>)
/// and BEFORE concatenation with the acoustic encoder's output in `codec_encode`. Real, plain
/// (non-causal, symmetric-padded) `Conv1d`s -- this stage runs once over the whole reference clip
/// before generation, no streaming/causality requirement, unlike the DAC-style decode path.
/// </summary>
public static class HiggsSemanticPostEncoder
{
    public const int HiddenDim = 768;
    public const int NumBlocks = 2;
    public const int ResidualUnitsPerBlock = 2;

    public sealed class Weights
    {
        public required float[] InputConvWeight { get; init; } // [768, 768, 3], no bias
        public required HiggsSemanticEncoderBlockWeights[] Blocks { get; init; } // 2

        public static Weights Load(Func<string, float[]> get)
        {
            string codec(string name) => "tied.embedding.modality_embeddings.0.model." + name;

            var blocks = new HiggsSemanticEncoderBlockWeights[NumBlocks];
            for (int b = 0; b < NumBlocks; b++)
            {
                string bp = $"encoder_semantic.conv_blocks.{b}";
                var units = new HiggsSemanticResidualUnitWeights[ResidualUnitsPerBlock];
                for (int u = 0; u < ResidualUnitsPerBlock; u++)
                {
                    string up = $"{bp}.res_units.{u}";
                    units[u] = new HiggsSemanticResidualUnitWeights
                    {
                        Conv1Weight = get(codec($"{up}.conv1.weight")),
                        Conv2Weight = get(codec($"{up}.conv2.weight")),
                    };
                }
                blocks[b] = new HiggsSemanticEncoderBlockWeights
                {
                    ResidualUnits = units,
                    ConvWeight = get(codec($"{bp}.conv.weight")),
                    ConvBias = get(codec($"{bp}.conv.bias")),
                };
            }

            return new Weights
            {
                InputConvWeight = get(codec("encoder_semantic.conv.weight")),
                Blocks = blocks,
            };
        }
    }

    /// <summary>Frame-major `[frames][768]` in, `[frames][768]` out.</summary>
    public static float[][] Forward(Weights w, float[][] hiddenFrames)
    {
        var x = Conv1dSamePad(hiddenFrames, w.InputConvWeight, bias: null, kernel: 3);
        foreach (var block in w.Blocks)
        {
            foreach (var unit in block.ResidualUnits)
                x = ResidualUnit(x, unit);
            x = Conv1dSamePad(x, block.ConvWeight, block.ConvBias, kernel: 3);
        }
        return x;
    }

    private static float[][] ResidualUnit(float[][] x, HiggsSemanticResidualUnitWeights w)
    {
        var h = EluRows(x);
        h = Conv1dSamePad(h, w.Conv1Weight, bias: null, kernel: 3);
        h = EluRows(h);
        h = Conv1dSamePad(h, w.Conv2Weight, bias: null, kernel: 1);
        int t = x.Length;
        var output = new float[t][];
        for (int i = 0; i < t; i++)
        {
            var row = new float[HiddenDim];
            for (int d = 0; d < HiddenDim; d++) row[d] = x[i][d] + h[i][d];
            output[i] = row;
        }
        return output;
    }

    private static float[][] EluRows(float[][] x)
    {
        var output = new float[x.Length][];
        for (int i = 0; i < x.Length; i++)
        {
            var row = new float[x[i].Length];
            for (int d = 0; d < row.Length; d++)
            {
                float v = x[i][d];
                row[d] = v > 0f ? v : MathF.Exp(v) - 1f;
            }
            output[i] = row;
        }
        return output;
    }

    /// <summary>Real standard (symmetric-padded, non-causal) Conv1d over channels-last frame-major
    /// input, `[in=768,out=768,kernel]` weight, `pad=(kernel-1)/2`.</summary>
    private static float[][] Conv1dSamePad(float[][] x, float[] weight, float[]? bias, int kernel)
    {
        int t = x.Length;
        int pad = (kernel - 1) / 2;
        var output = new float[t][];
        System.Threading.Tasks.Parallel.For(0, t, ti =>
        {
            var row = new float[HiddenDim];
            for (int oc = 0; oc < HiddenDim; oc++)
            {
                float sum = bias != null ? bias[oc] : 0f;
                int wBase = oc * HiddenDim * kernel;
                for (int k = 0; k < kernel; k++)
                {
                    int src = ti - pad + k;
                    if ((uint)src >= (uint)t) continue;
                    var xs = x[src];
                    for (int ic = 0; ic < HiddenDim; ic++)
                        sum += weight[wBase + ic * kernel + k] * xs[ic];
                }
                row[oc] = sum;
            }
            output[ti] = row;
        });
        return output;
    }
}
