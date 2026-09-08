using OpenTail.Stingray.Audio.Rvc;

namespace OpenTail.Stingray.Audio.MossTts;

/// <summary>
/// Real weight loader for MOSS-Audio-Tokenizer-Nano's RVQ (de)quantizer, ported from
/// `examples/audio.cpp/src/framework/codecs/moss_audio_tokenizer_codec_runtime.cpp`'s
/// `MossAudioTokenizerQuantizer` constructor/`decode`/`encode` (not guessed). Real, non-obvious
/// decode-time optimization ported verbatim: each codebook's `codebook -&gt; out_proj -&gt; (shared)
/// output_proj` chain is a fixed linear map per codebook, so it is pre-multiplied ONCE at load
/// time into a single `[codebookSize, CodeDim]` "latent table" per codebook -- decode-time cost
/// per frame becomes `NumQuantizers` table lookups + adds instead of `NumQuantizers` small
/// matmuls. Encode (added for voice-cloning reference-audio conditioning) cannot use this
/// shortcut -- it needs the RAW per-codebook `in_proj`/`out_proj` weights and the global
/// `input_proj`, loaded separately below and used one quantizer at a time (real residual-RVQ
/// nearest-code search, not table lookup).
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

    /// <summary>Real `quantizer.input_proj`: encoder latent `[CodeDim]` -&gt; `[RvqDim]` (encode only).</summary>
    public float[] InputWeight { get; } // [RvqDim, CodeDim]
    public float[] InputBias { get; } // [RvqDim]

    /// <summary>Per-codebook raw tensors needed only by `Encode` (decode uses `LatentTables` instead).</summary>
    public float[][] CodebookTable { get; } = new float[NumQuantizers][]; // [CodebookSize, CodebookDim]
    public float[][] CodebookTableNormalized { get; } = new float[NumQuantizers][]; // L2-normalized rows
    public float[][] CodebookInWeight { get; } = new float[NumQuantizers][]; // [CodebookDim, RvqDim]
    public float[][] CodebookInBias { get; } = new float[NumQuantizers][]; // [CodebookDim]
    public float[][] CodebookOutWeight { get; } = new float[NumQuantizers][]; // [RvqDim, CodebookDim]
    public float[][] CodebookOutBias { get; } = new float[NumQuantizers][]; // [RvqDim]

    public MossTtsAudioCodecQuantizerWeights(RvcPackedTensorSource source)
    {
        float[] outputWeight = ReconstructWeightNorm(
            source.GetTensor("audio_tokenizer_weights/quantizer.output_proj.parametrizations.weight.original0"),
            source.GetTensor("audio_tokenizer_weights/quantizer.output_proj.parametrizations.weight.original1"),
            outChannels: CodeDim, inChannels: RvqDim);
        OutputBias = source.GetTensor("audio_tokenizer_weights/quantizer.output_proj.bias");

        InputWeight = ReconstructWeightNorm(
            source.GetTensor("audio_tokenizer_weights/quantizer.input_proj.parametrizations.weight.original0"),
            source.GetTensor("audio_tokenizer_weights/quantizer.input_proj.parametrizations.weight.original1"),
            outChannels: RvqDim, inChannels: CodeDim);
        InputBias = source.GetTensor("audio_tokenizer_weights/quantizer.input_proj.bias");

        for (int q = 0; q < NumQuantizers; q++)
        {
            string p = $"audio_tokenizer_weights/quantizer.quantizers.{q}";
            float[] codebookTable = source.GetTensor($"{p}.codebook.weight"); // [CodebookSize, CodebookDim]
            float[] outWeight = ReconstructWeightNorm(
                source.GetTensor($"{p}.out_proj.parametrizations.weight.original0"),
                source.GetTensor($"{p}.out_proj.parametrizations.weight.original1"),
                outChannels: RvqDim, inChannels: CodebookDim);
            float[] outBias = source.GetTensor($"{p}.out_proj.bias"); // [RvqDim]
            float[] inWeight = ReconstructWeightNorm(
                source.GetTensor($"{p}.in_proj.parametrizations.weight.original0"),
                source.GetTensor($"{p}.in_proj.parametrizations.weight.original1"),
                outChannels: CodebookDim, inChannels: RvqDim);
            float[] inBias = source.GetTensor($"{p}.in_proj.bias"); // [CodebookDim]

            CodebookTable[q] = codebookTable;
            CodebookOutWeight[q] = outWeight;
            CodebookOutBias[q] = outBias;
            CodebookInWeight[q] = inWeight;
            CodebookInBias[q] = inBias;

            // Pre-normalize codebook rows once (encode does L2-normalized nearest search,
            // matching the training LFQ; F.normalize uses eps=1e-12).
            var tableNormalized = new float[CodebookSize * CodebookDim];
            for (int code = 0; code < CodebookSize; code++)
            {
                var row = codebookTable.AsSpan(code * CodebookDim, CodebookDim);
                double norm = 0;
                for (int k = 0; k < CodebookDim; k++) norm += (double)row[k] * row[k];
                double scale = 1.0 / Math.Max(Math.Sqrt(norm), 1e-12);
                for (int k = 0; k < CodebookDim; k++) tableNormalized[code * CodebookDim + k] = (float)(row[k] * scale);
            }
            CodebookTableNormalized[q] = tableNormalized;

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

    /// <summary>Real `MossAudioTokenizerQuantizer::encode` (residual RVQ nearest-code search), ported
    /// verbatim including its double-precision accumulation: for each frame, `residual = input_proj
    /// (frame)`, then for each quantizer in order: L2-normalize `in_proj(residual)`, pick the
    /// codebook row with the highest cosine similarity (dot product on the unit sphere -- both sides
    /// pre-normalized), subtract `out_proj(rawCodebookRow)` from the residual (including its bias,
    /// exactly matching the reference's own `residual[out] -= sum` where `sum` already includes
    /// `out_bias`), and move to the next quantizer.</summary>
    public int[][] Encode(float[] hiddenFrameMajor, int frames)
    {
        var codes = new int[NumQuantizers][];
        for (int q = 0; q < NumQuantizers; q++) codes[q] = new int[frames];

        var residual = new double[RvqDim];
        var encoding = new double[CodebookDim];
        for (int step = 0; step < frames; step++)
        {
            var frameHidden = hiddenFrameMajor.AsSpan(step * CodeDim, CodeDim);
            for (int o = 0; o < RvqDim; o++)
            {
                double sum = InputBias[o];
                var row = InputWeight.AsSpan(o * CodeDim, CodeDim);
                for (int k = 0; k < CodeDim; k++) sum += (double)row[k] * frameHidden[k];
                residual[o] = sum;
            }

            for (int q = 0; q < NumQuantizers; q++)
            {
                var inWeight = CodebookInWeight[q];
                var inBias = CodebookInBias[q];
                double encNorm = 0;
                for (int c = 0; c < CodebookDim; c++)
                {
                    double sum = inBias[c];
                    var row = inWeight.AsSpan(c * RvqDim, RvqDim);
                    for (int k = 0; k < RvqDim; k++) sum += row[k] * residual[k];
                    encoding[c] = sum;
                    encNorm += sum * sum;
                }
                double encScale = 1.0 / Math.Max(Math.Sqrt(encNorm), 1e-12);
                for (int c = 0; c < CodebookDim; c++) encoding[c] *= encScale;

                var tableNormalized = CodebookTableNormalized[q];
                int bestCode = 0;
                double bestDot = double.NegativeInfinity;
                for (int code = 0; code < CodebookSize; code++)
                {
                    var row = tableNormalized.AsSpan(code * CodebookDim, CodebookDim);
                    double dot = 0;
                    for (int c = 0; c < CodebookDim; c++) dot += row[c] * encoding[c];
                    if (dot > bestDot) { bestDot = dot; bestCode = code; }
                }
                codes[q][step] = bestCode;

                var embedding = CodebookTable[q].AsSpan(bestCode * CodebookDim, CodebookDim);
                var outWeight = CodebookOutWeight[q];
                var outBias = CodebookOutBias[q];
                for (int o = 0; o < RvqDim; o++)
                {
                    double sum = outBias[o];
                    var row = outWeight.AsSpan(o * CodebookDim, CodebookDim);
                    for (int k = 0; k < CodebookDim; k++) sum += row[k] * embedding[k];
                    residual[o] -= sum;
                }
            }
        }
        return codes;
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
/// the nano config). **Correction of an earlier guess**: these are NOT shared with the encoder
/// stack via one interleaved `ModuleList` -- a real tensor-name dump
/// (`MossTtsCodecTensorNameDumpDebugTest`) confirms `encoder.*` and `decoder.*` are two entirely
/// separate top-level prefixes in the packed checkpoint, each independently indexed 1/3/5/7; see
/// <see cref="MossTtsAudioCodecEncoderWeights"/> for the (now real, ported) encoder side.
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
            Stages[s] = LoadStage(source, "decoder", ModuleIndices[s], StageSpecs[s]);
    }

    /// <summary>Shared stage loader for both the decoder and encoder stacks -- same real tensor
    /// naming convention under a different top-level prefix ("decoder"/"encoder"), confirmed via a
    /// real tensor-name dump (`MossTtsCodecTensorNameDumpDebugTest`).</summary>
    internal static MossTtsAudioCodecTransformerStageWeights LoadStage(
        RvcPackedTensorSource source, string stackPrefix, int moduleIndex,
        (int Input, int Output, int DModel, int Heads, int Layers, int Intermediate, int Context, int Patch) spec)
    {
        string p = $"audio_tokenizer_weights/{stackPrefix}.{moduleIndex}";
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

        return new MossTtsAudioCodecTransformerStageWeights
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

/// <summary>
/// Real weight loader for MOSS-Audio-Tokenizer-Nano's ENCODER stack (voice-cloning reference-audio
/// conditioning): the reference's own documented "structural mirror" of
/// <see cref="MossTtsAudioCodecDecoderWeights"/> -- same 4-stage causal-windowed-attention
/// Transformer design, same real module indices (1, 3, 5, 7, confirmed via
/// `MossTtsCodecTensorNameDumpDebugTest` against the real checkpoint) but under the separate
/// `encoder.` tensor-name prefix (NOT sharing indices with the decoder stack -- they are two
/// independent `ModuleList`s in the same packed checkpoint, not one interleaved list as an earlier
/// session note incorrectly guessed before this real tensor dump).
/// </summary>
public sealed class MossTtsAudioCodecEncoderWeights
{
    public const int EncoderFinalPatch = 4;

    // Real nano encoder_stages config (input,output,d_model,heads,layers,intermediate,context,patch).
    private static readonly (int Input, int Output, int DModel, int Heads, int Layers, int Intermediate, int Context, int Patch)[] StageSpecs =
    [
        (240, 384, 256, 4, 4, 1024, 1600, 240),
        (768, 384, 256, 4, 2, 1024, 1200, 2),
        (768, 384, 256, 4, 2, 1024, 800, 2),
        (768, 192, 256, 4, 4, 1024, 500, 2),
    ];
    private static readonly int[] ModuleIndices = [1, 3, 5, 7];

    public MossTtsAudioCodecTransformerStageWeights[] Stages { get; } = new MossTtsAudioCodecTransformerStageWeights[StageSpecs.Length];

    public MossTtsAudioCodecEncoderWeights(RvcPackedTensorSource source)
    {
        for (int s = 0; s < StageSpecs.Length; s++)
            Stages[s] = MossTtsAudioCodecDecoderWeights.LoadStage(source, "encoder", ModuleIndices[s], StageSpecs[s]);
    }
}
