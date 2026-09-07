namespace OpenTail.Stingray.Audio.HiggsAudio;

/// <summary>Real per-quantizer weights, ported from `codec.cpp`'s `load_quantizer` (not
/// guessed): a real RVQ level -- `codebook.embed` (`[codebookSize=1024, codebookDim=64]`) plus
/// `project_out` (`[codebookDim -> hiddenSize=1024]`, with bias). `project_in`
/// (`[hiddenSize -> codebookDim]`, real ENCODE-direction weight, added 2026-09-07 for
/// `HiggsCodecEncoder.QuantizeEncode`) is also loaded now.</summary>
public sealed class HiggsCodecQuantizerWeights
{
    public required float[] CodebookEmbed { get; init; } // [1024, 64]
    public required float[] ProjectOutWeight { get; init; } // [1024, 64]
    public required float[] ProjectOutBias { get; init; } // [1024]
    public required float[] ProjectInWeight { get; init; } // [64, 1024]
    public required float[] ProjectInBias { get; init; } // [64]
}

public sealed class HiggsCodecResidualUnitWeights
{
    public required float[] Snake1Alpha { get; init; }
    public required float[] Conv1Weight { get; init; } // [channels, channels, 7]
    public required float[] Conv1Bias { get; init; }
    public required float[] Snake2Alpha { get; init; }
    public required float[] Conv2Weight { get; init; } // [channels, channels, 1]
    public required float[] Conv2Bias { get; init; }
}

public sealed class HiggsCodecDecoderBlockWeights
{
    public required float[] SnakeAlpha { get; init; }
    public required float[] ConvTransposeWeight { get; init; } // [inChannels, outChannels, 2*ratio]
    public required float[] ConvTransposeBias { get; init; }
    public required HiggsCodecResidualUnitWeights[] ResidualUnits { get; init; } // 3, dilations 1/3/9
}

/// <summary>
/// Real weights for Higgs Audio TTS's acoustic codec DECODE path (RVQ codes -> 24kHz waveform),
/// ported from `codec.cpp`'s `load_higgs_codec_decode_weights`/`quantizer_decode`/
/// `acoustic_decoder` (not guessed). Real config: `numCodebooks=8`, `codebookSize=1024`,
/// `codebookDim=64`, `codecHiddenSize=1024`, `acousticHiddenSize=256`,
/// `decoderChannels=[1024,512,256,128,64,32]`, `upsampleRatios=[8,5,4,2,3]` (total upsample
/// 8*5*4*2*3=960, matching the real `kCodecHopLength=960`), `residualDilations=[1,3,9]`. Real
/// tensor prefix `tied.embedding.modality_embeddings.0.model.*` (the SAME modality-embedding
/// table `HiggsLlmTensorSource.ModalityEmbeddingWeight` exposes -- confirms the codec and the
/// LLM's audio-token embeddings are tied). `project_in` (per-quantizer, encode-direction) and
/// `codec_project` (the real `[acousticHiddenSize+semanticHiddenSize=832 -> codecHiddenSize=1024]`
/// linear combining the acoustic+semantic encoders' outputs) are now also loaded, added 2026-09-07
/// for `HiggsCodecEncoder`'s real RVQ encode path (`quantizer_encode`, ported precisely from
/// `codec.cpp` -- not guessed). The semantic hidden-state-AVERAGING variant and the separate real
/// `semantic_encoder` post-network `codec_encode` also needs are still NOT ported (see
/// docs/audio-review-progress.md's "real scope correction" entry) -- this class's `codec_project`
/// alone does not make full voice-cloning end-to-end yet.
/// </summary>
public sealed class HiggsCodecDecoderWeights
{
    public const int NumCodebooks = 8;
    public const int CodebookSize = 1024;
    public const int CodebookDim = 64;
    public const int CodecHiddenSize = 1024;
    public const int AcousticHiddenSize = 256;
    public const int SemanticHiddenSize = 768;
    public const int CodecProjectInputSize = AcousticHiddenSize + SemanticHiddenSize; // 832
    public static readonly int[] DecoderChannels = [1024, 512, 256, 128, 64, 32];
    public static readonly int[] UpsampleRatios = [8, 5, 4, 2, 3];
    public static readonly int[] ResidualDilations = [1, 3, 9];
    public const int OutputSampleRate = 24000;

    public required HiggsCodecQuantizerWeights[] Quantizers { get; init; } // 8

    public required float[] AcousticProjectWeight { get; init; } // [256, 1024] (fc2)
    public required float[] AcousticProjectBias { get; init; }

    /// <summary>Real `codec_project`, encode-direction only, added 2026-09-07: combines the
    /// concatenated acoustic+semantic encoder outputs (`[832]`) into the codec's real
    /// `[1024]`-wide hidden space before RVQ quantization.</summary>
    public required float[] CodecProjectWeight { get; init; } // [1024, 832]
    public required float[] CodecProjectBias { get; init; }

    public required float[] DecoderInputConvWeight { get; init; } // [1024, 256, 7]
    public required float[] DecoderInputConvBias { get; init; }

    public required HiggsCodecDecoderBlockWeights[] DecoderBlocks { get; init; } // 5

    public required float[] DecoderOutputSnakeAlpha { get; init; } // [32]
    public required float[] DecoderOutputConvWeight { get; init; } // [1, 32, 7]
    public required float[] DecoderOutputConvBias { get; init; }

    public static HiggsCodecDecoderWeights Load(Func<string, float[]> get)
    {
        string codec(string name) => "tied.embedding.modality_embeddings.0.model." + name;

        var quantizers = new HiggsCodecQuantizerWeights[NumCodebooks];
        for (int i = 0; i < NumCodebooks; i++)
        {
            string p = $"quantizer.quantizers.{i}.";
            quantizers[i] = new HiggsCodecQuantizerWeights
            {
                CodebookEmbed = get(codec(p + "codebook.embed")),
                ProjectOutWeight = get(codec(p + "project_out.weight")),
                ProjectOutBias = get(codec(p + "project_out.bias")),
                ProjectInWeight = get(codec(p + "project_in.weight")),
                ProjectInBias = get(codec(p + "project_in.bias")),
            };
        }

        var decoderBlocks = new HiggsCodecDecoderBlockWeights[UpsampleRatios.Length];
        for (int block = 0; block < UpsampleRatios.Length; block++)
        {
            string p = $"acoustic_decoder.block.{block}.";
            var units = new HiggsCodecResidualUnitWeights[3];
            for (int unit = 0; unit < 3; unit++)
            {
                string up = $"{p}res_unit{unit + 1}.";
                units[unit] = new HiggsCodecResidualUnitWeights
                {
                    Snake1Alpha = get(codec(up + "snake1.alpha")),
                    Conv1Weight = get(codec(up + "conv1.weight")),
                    Conv1Bias = get(codec(up + "conv1.bias")),
                    Snake2Alpha = get(codec(up + "snake2.alpha")),
                    Conv2Weight = get(codec(up + "conv2.weight")),
                    Conv2Bias = get(codec(up + "conv2.bias")),
                };
            }
            decoderBlocks[block] = new HiggsCodecDecoderBlockWeights
            {
                SnakeAlpha = get(codec(p + "snake1.alpha")),
                ConvTransposeWeight = get(codec(p + "conv_t1.weight")),
                ConvTransposeBias = get(codec(p + "conv_t1.bias")),
                ResidualUnits = units,
            };
        }

        return new HiggsCodecDecoderWeights
        {
            Quantizers = quantizers,
            AcousticProjectWeight = get(codec("fc2.weight")),
            AcousticProjectBias = get(codec("fc2.bias")),
            DecoderInputConvWeight = get(codec("acoustic_decoder.conv1.weight")),
            DecoderInputConvBias = get(codec("acoustic_decoder.conv1.bias")),
            DecoderBlocks = decoderBlocks,
            DecoderOutputSnakeAlpha = get(codec("acoustic_decoder.snake1.alpha")),
            DecoderOutputConvWeight = get(codec("acoustic_decoder.conv2.weight")),
            DecoderOutputConvBias = get(codec("acoustic_decoder.conv2.bias")),
            CodecProjectWeight = get(codec("fc.weight")),
            CodecProjectBias = get(codec("fc.bias")),
        };
    }
}
