using OpenTail.Stingray.Audio.Rvc;

namespace OpenTail.Stingray.Audio.MossTts;

/// <summary>
/// Real weight loader for MOSS-Audio-Tokenizer-Nano's RVQ dequantizer, ported from
/// `examples/audio.cpp/src/framework/codecs/moss_audio_tokenizer_codec_runtime.cpp`'s
/// `MossAudioTokenizerQuantizer` constructor (decode-only -- this pipeline never needs `encode`,
/// since MOSS-TTS-Nano only ever DECODES generated codes into audio, never re-encodes a reference
/// clip; the reference's voice-cloning `encode` path is out of scope, see
/// `MossTtsPromptBuilder`'s doc comment). Real, non-obvious optimization ported verbatim: each
/// codebook's `codebook -&gt; out_proj -&gt; (shared) output_proj` chain is a fixed linear map per
/// codebook, so it is pre-multiplied ONCE at load time into a single `[codebookSize, CodeDim]`
/// "latent table" per codebook -- decode-time cost per frame becomes `NumQuantizers` table
/// lookups + adds instead of `NumQuantizers` small matmuls.
/// </summary>
public sealed class MossTtsAudioCodecQuantizerWeights
{
    public const int CodebookSize = 1024;
    public const int CodebookDim = 8;
    public const int RvqDim = 512;
    public const int CodeDim = 768;
    public const int NumQuantizers = 16; // real n_vq for the nano checkpoint

    /// <summary>[NumQuantizers][CodebookSize * CodeDim] row-major per-code decoded latent vectors.</summary>
    public float[][] LatentTables { get; } = new float[NumQuantizers][];
    public float[] OutputBias { get; } // [CodeDim]

    public MossTtsAudioCodecQuantizerWeights(RvcPackedTensorSource source)
    {
        float[] outputWeight = ReconstructWeightNorm(
            source.GetTensor("audio_tokenizer_weights/quantizer.output_proj.parametrizations.weight.original0"),
            source.GetTensor("audio_tokenizer_weights/quantizer.output_proj.parametrizations.weight.original1"),
            outChannels: CodeDim, inChannels: RvqDim);
        OutputBias = source.GetTensor("audio_tokenizer_weights/quantizer.output_proj.bias");

        for (int q = 0; q < NumQuantizers; q++)
        {
            string p = $"audio_tokenizer_weights/quantizer.quantizers.{q}";
            float[] codebookTable = source.GetTensor($"{p}.codebook.weight"); // [CodebookSize, CodebookDim]
            float[] outWeight = ReconstructWeightNorm(
                source.GetTensor($"{p}.out_proj.parametrizations.weight.original0"),
                source.GetTensor($"{p}.out_proj.parametrizations.weight.original1"),
                outChannels: RvqDim, inChannels: CodebookDim);
            float[] outBias = source.GetTensor($"{p}.out_proj.bias"); // [RvqDim]

            // combinedWeight[out, k] = sum_rvq outputWeight[out, rvq] * outWeight[rvq, k]  (CodeDim x CodebookDim)
            // combinedBias[out]      = sum_rvq outputWeight[out, rvq] * outBias[rvq]
            var combinedBias = new float[CodeDim];
            var combinedWeight = new float[CodeDim * CodebookDim];
            for (int o = 0; o < CodeDim; o++)
            {
                var outputRow = outputWeight.AsSpan(o * RvqDim, RvqDim);
                float biasSum = 0f;
                for (int rvq = 0; rvq < RvqDim; rvq++) biasSum += outputRow[rvq] * outBias[rvq];
                combinedBias[o] = biasSum;
                for (int k = 0; k < CodebookDim; k++)
                {
                    float sum = 0f;
                    for (int rvq = 0; rvq < RvqDim; rvq++) sum += outputRow[rvq] * outWeight[rvq * CodebookDim + k];
                    combinedWeight[o * CodebookDim + k] = sum;
                }
            }

            var latentTable = new float[CodebookSize * CodeDim];
            for (int code = 0; code < CodebookSize; code++)
            {
                var embedding = codebookTable.AsSpan(code * CodebookDim, CodebookDim);
                for (int o = 0; o < CodeDim; o++)
                {
                    float sum = combinedBias[o];
                    var row = combinedWeight.AsSpan(o * CodebookDim, CodebookDim);
                    for (int k = 0; k < CodebookDim; k++) sum += row[k] * embedding[k];
                    latentTable[code * CodeDim + o] = sum;
                }
            }
            LatentTables[q] = latentTable;
        }
    }

    /// <summary>Real PyTorch `weight_norm(dim=0)` reconstruction: `weight = v * (g / ||v||)`, norm
    /// per output channel (dim 0) -- matches `reconstruct_weight_norm` in the reference exactly
    /// (including its double-precision norm accumulation).</summary>
    internal static float[] ReconstructWeightNorm(float[] g, float[] v, int outChannels, int inChannels)
    {
        var weight = new float[outChannels * inChannels];
        for (int o = 0; o < outChannels; o++)
        {
            double normSq = 0;
            int offset = o * inChannels;
            for (int k = 0; k < inChannels; k++) normSq += (double)v[offset + k] * v[offset + k];
            float scale = (float)(g[o] / Math.Sqrt(normSq));
            for (int k = 0; k < inChannels; k++) weight[offset + k] = v[offset + k] * scale;
        }
        return weight;
    }

    /// <summary>Dequantizes one frame's codes directly into `CodeDim` latent floats.</summary>
    public float[] DecodeFrame(ReadOnlySpan<int> codesPerQuantizer)
    {
        var latent = new float[CodeDim];
        Array.Copy(OutputBias, latent, CodeDim);
        for (int q = 0; q < NumQuantizers; q++)
        {
            int code = codesPerQuantizer[q];
            var row = LatentTables[q].AsSpan(code * CodeDim, CodeDim);
            for (int o = 0; o < CodeDim; o++) latent[o] += row[o];
        }
        return latent;
    }
}

/// <summary>One MOSS-Audio-Tokenizer-Nano decoder "ProjectedTransformer" stage's weights.</summary>
public sealed class MossTtsAudioCodecTransformerLayerWeights
{
    public required float[] Norm1Weight { get; init; }
    public required float[] Norm1Bias { get; init; }
    public required float[] InProjWeight { get; init; } // [3*DModel, DModel]
    public required float[] OutProjWeight { get; init; } // [DModel, DModel]
    public required float[] Norm2Weight { get; init; }
    public required float[] Norm2Bias { get; init; }
    public required float[] Fc1Weight { get; init; } // [Intermediate, DModel]
    public required float[] Fc2Weight { get; init; } // [DModel, Intermediate]
    public required float[] LayerScale1 { get; init; } // [DModel]
    public required float[] LayerScale2 { get; init; } // [DModel]
}

public sealed class MossTtsAudioCodecTransformerStageWeights
{
    public required int InputDim { get; init; }
    public required int OutputDim { get; init; }
    public required int DModel { get; init; }
    public required int NumHeads { get; init; }
    public required int IntermediateDim { get; init; }
    public required int Context { get; init; }
    public required int Patch { get; init; }
    public required float[] InputProjWeight { get; init; } // [DModel, InputDim]
    public float[]? OutputProjWeight { get; init; } // [OutputDim, DModel], null when OutputDim==DModel (identity)
    public required MossTtsAudioCodecTransformerLayerWeights[] Layers { get; init; }
}

/// <summary>
/// Real weight loader for MOSS-Audio-Tokenizer-Nano's decoder stack: 4 "ProjectedTransformer"
/// stages (real config from `moss_audio_tokenizer_nano_config()`, confirmed against real tensor
/// names via `MossTtsCodecTensorNameDumpDebugTest`), each a causal-windowed-RoPE Transformer with
/// LayerScale and exact-erf GELU, separated by reshape-based "patch" upsamples (no conv layers at
/// all -- this codec is genuinely "CNN-free" per the reference's own comment). Real module indices
/// in the packed checkpoint are 1, 3, 5, 7 (`decoder_module_start=1, decoder_module_stride=2` for
/// the nano config -- odd slots; even slots are unrelated encoder-side modules in the same
/// ModuleList, confirmed present but skipped here since this pipeline never encodes).
/// </summary>
public sealed class MossTtsAudioCodecDecoderWeights
{
    public const int SamplingRate = 48000;
    public const int SamplesPerFrame = 3840; // per channel
    public const int Channels = 2;
    public const int DecoderInitialPatch = 4;
    public const float LayerNormEps = 1e-5f;
    public const float RopeBase = 10000f;

    // Real nano decoder_stages config (input,output,d_model,heads,layers,intermediate,context,patch).
    private static readonly (int Input, int Output, int DModel, int Heads, int Layers, int Intermediate, int Context, int Patch)[] StageSpecs =
    [
        (192, 768, 256, 4, 4, 1024, 1000, 2),
        (384, 768, 256, 4, 2, 1024, 1600, 2),
        (384, 768, 256, 4, 2, 1024, 2400, 2),
        (384, 240, 256, 4, 4, 1024, 3200, 240),
    ];
    private static readonly int[] ModuleIndices = [1, 3, 5, 7];

    public MossTtsAudioCodecTransformerStageWeights[] Stages { get; } = new MossTtsAudioCodecTransformerStageWeights[StageSpecs.Length];

    public MossTtsAudioCodecDecoderWeights(RvcPackedTensorSource source)
    {
        for (int s = 0; s < StageSpecs.Length; s++)
        {
            var spec = StageSpecs[s];
            string p = $"audio_tokenizer_weights/decoder.{ModuleIndices[s]}";
            var layers = new MossTtsAudioCodecTransformerLayerWeights[spec.Layers];
            for (int l = 0; l < spec.Layers; l++)
            {
                string lp = $"{p}.transformer.layers.{l}";
                layers[l] = new MossTtsAudioCodecTransformerLayerWeights
                {
                    Norm1Weight = source.GetTensor($"{lp}.norm1.weight"),
                    Norm1Bias = source.GetTensor($"{lp}.norm1.bias"),
                    InProjWeight = source.GetTensor($"{lp}.self_attn.in_proj.weight"),
                    OutProjWeight = source.GetTensor($"{lp}.self_attn.out_proj.weight"),
                    Norm2Weight = source.GetTensor($"{lp}.norm2.weight"),
                    Norm2Bias = source.GetTensor($"{lp}.norm2.bias"),
                    Fc1Weight = source.GetTensor($"{lp}.ffn.0.weight"),
                    Fc2Weight = source.GetTensor($"{lp}.ffn.2.weight"),
                    LayerScale1 = source.GetTensor($"{lp}.layer_scale_1.scale"),
                    LayerScale2 = source.GetTensor($"{lp}.layer_scale_2.scale"),
                };
            }

            bool hasOutputProj = source.HasTensor($"{p}.output_proj.weight");
            if (!hasOutputProj && spec.Output != spec.DModel)
                throw new InvalidDataException($"MOSS codec stage {p} changes width but carries no output projection.");

            Stages[s] = new MossTtsAudioCodecTransformerStageWeights
            {
                InputDim = spec.Input,
                OutputDim = spec.Output,
                DModel = spec.DModel,
                NumHeads = spec.Heads,
                IntermediateDim = spec.Intermediate,
                Context = spec.Context,
                Patch = spec.Patch,
                InputProjWeight = source.GetTensor($"{p}.input_proj.weight"),
                OutputProjWeight = hasOutputProj ? source.GetTensor($"{p}.output_proj.weight") : null,
                Layers = layers,
            };
        }
    }
}
