namespace OpenTail.Stingray.Audio.PersonaPlex;

/// <summary>One real Mimi quantizer level's codebook, ported from `mimi_codec_runtime.cpp`'s
/// `normalized_codebook`/`load_codebook` (not guessed): the checkpoint stores real EMA-VQ-VAE
/// accumulator buffers (`embedding_sum`, `cluster_usage`), NOT the codebook vectors directly --
/// the real per-code embedding is `embedding_sum[code] / max(cluster_usage[code], 1e-5)`. Would
/// have been a silently-wrong codebook if this division were missed.</summary>
public sealed class MimiCodecCodebookWeights
{
    public required float[] Embedding { get; init; } // [codebookSize, latentSize], already normalized
}

public sealed class MimiCodecResidualUnitWeights
{
    public required float[] Conv1Weight { get; init; }
    public required float[] Conv1Bias { get; init; }
    public required float[] Conv2Weight { get; init; }
    public required float[] Conv2Bias { get; init; }
}

public sealed class MimiCodecTransformerLayerWeights
{
    public required float[] Norm1Weight { get; init; }
    public required float[] Norm1Bias { get; init; }
    public required float[] QWeight { get; init; }
    public required float[] KWeight { get; init; }
    public required float[] VWeight { get; init; }
    public required float[] OutWeight { get; init; }
    public required float[] LayerScale1 { get; init; }
    public required float[] Norm2Weight { get; init; }
    public required float[] Norm2Bias { get; init; }
    public required float[] Linear1Weight { get; init; } // [intermediate, hidden]
    public required float[] Linear2Weight { get; init; } // [hidden, intermediate]
    public required float[] LayerScale2 { get; init; }
}

public sealed class MimiCodecDecoderStageWeights
{
    public required float[] UpsampleWeight { get; init; } // [inCh, outCh, kernel] (ConvTranspose1d layout)
    public required float[] UpsampleBias { get; init; }
    public required MimiCodecResidualUnitWeights ResidualBlock { get; init; }
}

/// <summary>
/// Real weights for the Mimi neural codec's DECODE path, ported from
/// `mimi_codec_runtime.cpp`'s `load_quantizer_weights`/`load_decoder_weights` and the default
/// `MimiCodecWeightBinding` (not guessed) -- this port targets a ONE-SHOT (non-streaming) full-
/// sequence decode, so every "stateful" conv this session's scoping pass found degenerates to a
/// plain causal (left-zero-pad) Conv1d or a plain unpadded ConvTranspose1d (see the doc entry
/// this class implements for the full derivation).
/// </summary>
public sealed class MimiCodecDecoderWeights
{
    public const int HiddenDim = 512;
    public const int LatentSize = 256;
    public const int CodebookSize = 2048;
    public const int ActiveCodebooks = 8; // real kMimiActiveCodebooks: codebook 0 = semantic, 1-7 = acoustic
    public const int Channels = 1;
    public const int NumTransformerLayers = 8;
    public const int NumHeads = 8;
    public const int HeadDim = HiddenDim / NumHeads; // 64
    public const int IntermediateSize = 2048;
    public const float NormEps = 1e-5f;
    public const float LinearScaleEps = 1e-5f; // kCodebookEps

    public required MimiCodecCodebookWeights SemanticCodebook { get; init; }
    public required MimiCodecCodebookWeights[] AcousticCodebooks { get; init; } // 7 real levels (ActiveCodebooks-1)
    public required float[] SemanticOutputProjWeight { get; init; } // [hidden, latent]
    public required float[] AcousticOutputProjWeight { get; init; } // [hidden, latent]

    // Pre-transformer: depthwise ConvTranspose1d(hidden, kernel=4, stride=2) frame-rate 2x upsample.
    public required float[] FrameUpsampleWeight { get; init; } // [hidden][4], real kernel REVERSED at load time
    public required float[] FrameUpsampleBias { get; init; }

    public required MimiCodecTransformerLayerWeights[] TransformerLayers { get; init; }

    public required float[] DecoderInputProjWeight { get; init; } // [1024, hidden], k=7
    public required float[] DecoderInputProjBias { get; init; }
    public required MimiCodecDecoderStageWeights[] Stages { get; init; } // 4 real stages
    public required float[] DecoderOutputProjWeight { get; init; } // [channels, 64], k=3
    public required float[] DecoderOutputProjBias { get; init; }

    public static readonly int[] StageInChannels = [1024, 512, 256, 128];
    public static readonly int[] StageOutChannels = [512, 256, 128, 64];
    public static readonly int[] StageHiddenChannels = [256, 128, 64, 32];
    public static readonly int[] StageKernelSizes = [16, 12, 10, 8];
    public static readonly int[] StageStrides = [8, 6, 5, 4];

    public static MimiCodecDecoderWeights Load(Func<string, float[]> get)
    {
        string mimi(string name) => "mimi/" + name;

        var semanticCodebook = LoadCodebook(get, mimi("quantizer.rvq_first.vq.layers.0._codebook."));
        var acousticCodebooks = new MimiCodecCodebookWeights[ActiveCodebooks - 1];
        for (int i = 0; i < ActiveCodebooks - 1; i++)
            acousticCodebooks[i] = LoadCodebook(get, mimi($"quantizer.rvq_rest.vq.layers.{i}._codebook."));

        var transformerLayers = new MimiCodecTransformerLayerWeights[NumTransformerLayers];
        for (int l = 0; l < NumTransformerLayers; l++)
        {
            string p = mimi($"decoder_transformer.transformer.layers.{l}.");
            var packedQkv = get(p + "self_attn.in_proj_weight"); // [3*hidden, hidden]
            var q = new float[HiddenDim * HiddenDim];
            var k = new float[HiddenDim * HiddenDim];
            var v = new float[HiddenDim * HiddenDim];
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

        var frameUpsampleRaw = get(mimi("upsample.convtr.convtr.convtr.weight")); // [hidden, 1, 4]
        var frameUpsample = ReverseKernelPerChannel(frameUpsampleRaw, HiddenDim, 4);

        var stages = new MimiCodecDecoderStageWeights[4];
        string[] residualPrefixes = ["decoder.model.3", "decoder.model.6", "decoder.model.9", "decoder.model.12"];
        string[] upsampleWeightNames = ["decoder.model.2.convtr.convtr.weight", "decoder.model.5.convtr.convtr.weight", "decoder.model.8.convtr.convtr.weight", "decoder.model.11.convtr.convtr.weight"];
        string[] upsampleBiasNames = ["decoder.model.2.convtr.convtr.bias", "decoder.model.5.convtr.convtr.bias", "decoder.model.8.convtr.convtr.bias", "decoder.model.11.convtr.convtr.bias"];
        for (int s = 0; s < 4; s++)
        {
            stages[s] = new MimiCodecDecoderStageWeights
            {
                UpsampleWeight = get(mimi(upsampleWeightNames[s])),
                UpsampleBias = get(mimi(upsampleBiasNames[s])),
                ResidualBlock = new MimiCodecResidualUnitWeights
                {
                    Conv1Weight = get(mimi(residualPrefixes[s] + ".block.1.conv.conv.weight")),
                    Conv1Bias = get(mimi(residualPrefixes[s] + ".block.1.conv.conv.bias")),
                    Conv2Weight = get(mimi(residualPrefixes[s] + ".block.3.conv.conv.weight")),
                    Conv2Bias = get(mimi(residualPrefixes[s] + ".block.3.conv.conv.bias")),
                },
            };
        }

        return new MimiCodecDecoderWeights
        {
            SemanticCodebook = semanticCodebook,
            AcousticCodebooks = acousticCodebooks,
            SemanticOutputProjWeight = get(mimi("quantizer.rvq_first.output_proj.weight")),
            AcousticOutputProjWeight = get(mimi("quantizer.rvq_rest.output_proj.weight")),
            FrameUpsampleWeight = frameUpsample,
            FrameUpsampleBias = new float[HiddenDim], // real reference: this depthwise conv has NO bias tensor (use_bias=false)
            TransformerLayers = transformerLayers,
            DecoderInputProjWeight = get(mimi("decoder.model.0.conv.conv.weight")),
            DecoderInputProjBias = get(mimi("decoder.model.0.conv.conv.bias")),
            Stages = stages,
            DecoderOutputProjWeight = get(mimi("decoder.model.14.conv.conv.weight")),
            DecoderOutputProjBias = get(mimi("decoder.model.14.conv.conv.bias")),
        };
    }

    private static MimiCodecCodebookWeights LoadCodebook(Func<string, float[]> get, string prefix)
    {
        var clusterUsage = get(prefix + "cluster_usage"); // [codebookSize]
        var embeddingSum = get(prefix + "embedding_sum"); // [codebookSize, latentSize]
        var embedding = new float[embeddingSum.Length];
        for (int code = 0; code < CodebookSize; code++)
        {
            float denom = MathF.Max(clusterUsage[code], LinearScaleEps);
            int baseIdx = code * LatentSize;
            for (int d = 0; d < LatentSize; d++) embedding[baseIdx + d] = embeddingSum[baseIdx + d] / denom;
        }
        return new MimiCodecCodebookWeights { Embedding = embedding };
    }

    // Real reference reverses the kernel axis of the depthwise frame-rate-upsample conv at load
    // time (per-channel, confirmed via load_decoder_weights's explicit std::reverse).
    private static float[] ReverseKernelPerChannel(float[] flat, int channels, int kernel)
    {
        var output = (float[])flat.Clone();
        for (int c = 0; c < channels; c++)
        {
            int baseIdx = c * kernel;
            Array.Reverse(output, baseIdx, kernel);
        }
        return output;
    }
}
