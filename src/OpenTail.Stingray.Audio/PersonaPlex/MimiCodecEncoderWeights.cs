namespace OpenTail.Stingray.Audio.PersonaPlex;

public sealed class MimiCodecEncoderStageWeights
{
    public required MimiCodecResidualUnitWeights ResidualBlock { get; init; }
    public required float[] DownsampleWeight { get; init; } // [outCh, inCh, kernel]
    public required float[] DownsampleBias { get; init; }
}

/// <summary>
/// Real weights for the Mimi neural codec's ENCODE path (live-duplex user-audio conditioning),
/// ported from `mimi_codec_runtime.cpp`'s `MimiEncoderPreTransformerGraph`/
/// `MimiEncoderPostTransformerGraph`/`quantize_projected` (not guessed) -- the real mirror
/// direction of the already-ported <see cref="MimiCodecDecoder"/>. Real, non-obvious detail:
/// the semantic/acoustic codebooks used for nearest-code search here are the EXACT SAME real
/// checkpoint tensors <see cref="MimiCodecDecoderWeights"/> already loads for decode (the
/// `embedding_sum/cluster_usage`-normalized centroid table works identically for lookup-by-index
/// (decode) and nearest-neighbor search (encode)) -- this class takes an already-loaded
/// <see cref="MimiCodecDecoderWeights"/> and reuses its codebooks rather than reloading them.
///
/// <para>Real pipeline (ONE-SHOT/non-streaming, same simplification convention as
/// <see cref="MimiCodecDecoder"/>): `Conv1d(1-&gt;64,k=7)` -&gt; 4x [SEANet residual unit -&gt; ELU
/// -&gt; `Conv1d`(real per-stage kernel/stride) doubling channels] -&gt; ELU -&gt;
/// `Conv1d(1024-&gt;512,k=3)` -&gt; the ALREADY-PORTED 8-layer Mimi transformer (shared class,
/// `MimiCodecDecoder`'s own `RunTransformer` logic, just this encoder's own transformer weights)
/// -&gt; `Conv1d(512-&gt;512,k=4,stride=2)` real frame-rate HALVING (the encoder's mirror of the
/// decoder's `FrameUpsampleWeight` 2x upsample) -&gt; per-frame `semantic_input_proj`/
/// `acoustic_input_proj` (Linear, hidden-&gt;latent=256, WITH bias, two SEPARATE projections from
/// the SAME downsampled latent) -&gt; real plain-Euclidean nearest-code RVQ search (semantic: 1
/// codebook, acoustic: 7 codebooks, real residual subtraction of the CHOSEN RAW codebook centroid
/// between levels -- no re-projection, simpler than MOSS-TTS-Nano's cosine-similarity RVQ).</para>
/// </summary>
public sealed class MimiCodecEncoderWeights
{
    public const int InputChannels = 1;

    // Real per-stage config (channels[stage] = OUTPUT width after that stage's downsample conv).
    public static readonly int[] StageResidualChannels = [64, 128, 256, 512]; // real "channels" in the reference
    public static readonly int[] StageHiddenChannels = [32, 64, 128, 256];
    public static readonly int[] StageDownsampleKernels = [8, 10, 12, 16];
    public static readonly int[] StageDownsampleStrides = [4, 5, 6, 8];

    public required float[] InputProjWeight { get; init; } // [64, 1, 7]
    public required float[] InputProjBias { get; init; }
    public required MimiCodecEncoderStageWeights[] Stages { get; init; } // 4 real stages
    public required float[] OutputProjWeight { get; init; } // [512, 1024, 3]
    public required float[] OutputProjBias { get; init; }

    public required MimiCodecTransformerLayerWeights[] TransformerLayers { get; init; } // 8 real layers

    public required float[] DownsampleConvWeight { get; init; } // [512, 512, 4]
    public required float[] DownsampleConvBias { get; init; }

    public required float[] SemanticInputProjWeight { get; init; } // [256, 512]
    public required float[]? SemanticInputProjBias { get; init; }
    public required float[] AcousticInputProjWeight { get; init; } // [256, 512]
    public required float[]? AcousticInputProjBias { get; init; }

    public required MimiCodecCodebookWeights SemanticCodebook { get; init; }
    public required MimiCodecCodebookWeights[] AcousticCodebooks { get; init; }

    public static MimiCodecEncoderWeights Load(Func<string, float[]> get, Func<string, bool> has, MimiCodecDecoderWeights decoderWeights)
    {
        string mimi(string name) => "mimi/" + name;

        var stages = new MimiCodecEncoderStageWeights[4];
        // Real model index scheme (confirmed via a real tensor-name dump, not guessed):
        // model.0 = input proj; model.{1,4,7,10} = residual blocks; model.{3,6,9,12} = downsamples;
        // model.14 = output proj.
        string[] residualPrefixes = ["encoder.model.1", "encoder.model.4", "encoder.model.7", "encoder.model.10"];
        string[] downsamplePrefixes = ["encoder.model.3", "encoder.model.6", "encoder.model.9", "encoder.model.12"];
        for (int s = 0; s < 4; s++)
        {
            stages[s] = new MimiCodecEncoderStageWeights
            {
                ResidualBlock = new MimiCodecResidualUnitWeights
                {
                    Conv1Weight = get(mimi(residualPrefixes[s] + ".block.1.conv.conv.weight")),
                    Conv1Bias = get(mimi(residualPrefixes[s] + ".block.1.conv.conv.bias")),
                    Conv2Weight = get(mimi(residualPrefixes[s] + ".block.3.conv.conv.weight")),
                    Conv2Bias = get(mimi(residualPrefixes[s] + ".block.3.conv.conv.bias")),
                },
                DownsampleWeight = get(mimi(downsamplePrefixes[s] + ".conv.conv.weight")),
                DownsampleBias = get(mimi(downsamplePrefixes[s] + ".conv.conv.bias")),
            };
        }

        const int hiddenDim = MimiCodecDecoderWeights.HiddenDim;
        var transformerLayers = new MimiCodecTransformerLayerWeights[MimiCodecDecoderWeights.NumTransformerLayers];
        for (int l = 0; l < transformerLayers.Length; l++)
        {
            string p = mimi($"encoder_transformer.transformer.layers.{l}.");
            var packedQkv = get(p + "self_attn.in_proj_weight"); // [3*hidden, hidden]
            var q = new float[hiddenDim * hiddenDim];
            var k = new float[hiddenDim * hiddenDim];
            var v = new float[hiddenDim * hiddenDim];
            Array.Copy(packedQkv, 0, q, 0, q.Length);
            Array.Copy(packedQkv, q.Length, k, 0, k.Length);
            Array.Copy(packedQkv, 2 * q.Length, v, 0, v.Length);
            transformerLayers[l] = new MimiCodecTransformerLayerWeights
            {
                Norm1Weight = get(p + "norm1.weight"),
                Norm1Bias = get(p + "norm1.bias"),
                QWeight = q,
                KWeight = k,
                VWeight = v,
                OutWeight = get(p + "self_attn.out_proj.weight"),
                LayerScale1 = get(p + "layer_scale_1.scale"),
                Norm2Weight = get(p + "norm2.weight"),
                Norm2Bias = get(p + "norm2.bias"),
                Linear1Weight = get(p + "linear1.weight"),
                Linear2Weight = get(p + "linear2.weight"),
                LayerScale2 = get(p + "layer_scale_2.scale"),
            };
        }

        string semBiasName = mimi("quantizer.rvq_first.input_proj.bias");
        string acBiasName = mimi("quantizer.rvq_rest.input_proj.bias");

        return new MimiCodecEncoderWeights
        {
            InputProjWeight = get(mimi("encoder.model.0.conv.conv.weight")),
            InputProjBias = get(mimi("encoder.model.0.conv.conv.bias")),
            Stages = stages,
            OutputProjWeight = get(mimi("encoder.model.14.conv.conv.weight")),
            OutputProjBias = get(mimi("encoder.model.14.conv.conv.bias")),
            TransformerLayers = transformerLayers,
            DownsampleConvWeight = get(mimi("downsample.conv.conv.conv.weight")),
            DownsampleConvBias = has(mimi("downsample.conv.conv.conv.bias")) ? get(mimi("downsample.conv.conv.conv.bias")) : new float[hiddenDim],
            SemanticInputProjWeight = get(mimi("quantizer.rvq_first.input_proj.weight")),
            SemanticInputProjBias = has(semBiasName) ? get(semBiasName) : null,
            AcousticInputProjWeight = get(mimi("quantizer.rvq_rest.input_proj.weight")),
            AcousticInputProjBias = has(acBiasName) ? get(acBiasName) : null,
            SemanticCodebook = decoderWeights.SemanticCodebook,
            AcousticCodebooks = decoderWeights.AcousticCodebooks,
        };
    }
}
