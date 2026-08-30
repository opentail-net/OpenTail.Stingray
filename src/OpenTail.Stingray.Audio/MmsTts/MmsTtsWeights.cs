
namespace OpenTail.Stingray.Audio.MmsTts;

/// <summary>
/// Real MMS-TTS (Meta Massively Multilingual Speech TTS) weights, loaded directly from a
/// HuggingFace `transformers.VitsModel` `model.safetensors` checkpoint (e.g.
/// `facebook/mms-tts-eng`). MMS-TTS IS a VITS model -- same architecture family as
/// <see cref="OpenTail.Stingray.Audio.Piper"/>'s real, weight-driven VITS+HiFi-GAN implementation
/// (see docs/audio-review-progress.md's "MMS-TTS (VITS) and XTTS-v2" entry for the full research
/// this loader is based on) -- this class mirrors <c>PiperOnnxWeights</c>'s structure closely, but
/// reads HuggingFace's own module-path tensor names (`text_encoder.*`, `duration_predictor.*`,
/// `flow.*`, `decoder.*`) instead of Piper's ONNX-exported `enc_p.*`/`dp.*`/`flow.*`/`dec.*` names.
///
/// <para>Only the INFERENCE-relevant submodules are loaded: `text_encoder` (prior), `duration_
/// predictor` (only its main `flows`, NOT the training-only `post_flows`/`post_pre`/`post_proj`/
/// `post_convs` branch -- that branch computes the posterior u/v from ground-truth durations,
/// never used at inference/reverse time), `flow` (reverse mode), `decoder`. `posterior_encoder.*`
/// (training-only, encodes real mel spectrograms) is NOT loaded at all.</para>
///
/// <para>Weight-norm status (checked directly against the real checkpoint's safetensors header via
/// HTTP range request, not assumed): `decoder.*`, `duration_predictor.*`, and `text_encoder.*` are
/// ALREADY FUSED (plain `.weight`/`.bias`). ONLY `flow.flows.N.wavenet.{in,res_skip}_layers.N.*`
/// ship as raw `weight_g`/`weight_v` pairs (older `nn.utils.weight_norm` convention, dim=0) --
/// folded here via <see cref="FoldConvWeight"/>, same exact math as
/// `OpenTail.Stingray.Audio.Parler.DacWeights`'s private `FoldConvWeight` (not shared directly
/// since that one is private to its own file -- worth extracting to a shared primitive in this
/// port's eventual DRY pass, per CLAUDE.md rule 7).</para>
/// </summary>
public sealed class MmsTtsWeights
{
    public int HiddenDim { get; }
    public int NumHeads { get; }
    public int WindowSize { get; }
    public int FfnKernel { get; }
    public int VocabSize { get; }

    public float[] EmbeddingWeight { get; }   // text_encoder.embed_tokens.weight [VocabSize, HiddenDim]
    public float[] ProjWeight { get; }        // text_encoder.project.weight [2*HiddenDim, HiddenDim, 1]
    public float[] ProjBias { get; }

    public MmsEncoderLayerWeights[] Layers { get; }

    // --- duration_predictor.* : StochasticDurationPredictor (reverse/inference path only) ---
    public float[] DpConvPreWeight { get; }
    public float[] DpConvPreBias { get; }
    public VitsDdsConvWeights DpConvDds { get; } // duration_predictor.conv_dds
    public float[] DpConvProjWeight { get; }
    public float[] DpConvProjBias { get; }
    public float[] DpFlow0LogScale { get; }     // duration_predictor.flows.0.log_scale (ElementwiseAffine's logs)
    public float[] DpFlow0ExpNegLogScale { get; } // exp(-log_scale), folded here (HF keeps it raw, unlike Piper's constant-folded ONNX export)
    public float[] DpFlow0Translate { get; }    // duration_predictor.flows.0.translate (ElementwiseAffine's m)
    // duration_predictor.flows.1 is the "useless vflow" the real reference's reverse-mode pruning
    // drops (`flows[:-2] + [flows[-1]]` on the reversed list) -- confirmed via Piper's own real
    // ONNX-graph-inspected reverse order (same reference lineage); NOT loaded here.
    public VitsConvFlowWeights DpFlow2 { get; } // duration_predictor.flows.2
    public VitsConvFlowWeights DpFlow3 { get; } // duration_predictor.flows.3
    public VitsConvFlowWeights DpFlow4 { get; } // duration_predictor.flows.4

    // --- flow.* : ResidualCouplingBlock, 4 ResidualCouplingLayers (list-indices 0,1,2,3 in HF's
    // naming -- HF does not store the parameter-free Flip modules, unlike Piper's ONNX export
    // which numbers them into the same flat list). Reverse execution order is list-reversed with a
    // Flip between each layer: Flip -> Layer(3) -> Flip -> Layer(2) -> Flip -> Layer(1) -> Flip ->
    // Layer(0) (mean_only=True, confirmed via conv_post's real output-channel count == half_channels
    // via the safetensors header: flow.flows.0.conv_post.weight is [96,192,1], not [192,192,1]).
    public const int FlowWnLayers = 4;
    public const int FlowWnKernel = 5;
    public const int FlowWnDilation = 1;
    public int FlowHalfChannels { get; }
    public MmsCouplingLayerWeights[] FlowLayers { get; }

    // --- decoder.* : HiFi-GAN vocoder, classic ResBlock1 topology (3 conv PAIRS per resblock,
    // dilations (1,3,5) cycling regardless of the resblock's own kernel size -- SAME topology as
    // MeloTTS's `MeloGenerator`/`ResBlock1Forward`, confirmed via `resblock_kernel_sizes=[3,7,11]`/
    // `resblock_dilation_sizes=[[1,3,5],[1,3,5],[1,3,5]]` in the real config.json, matching
    // MeloTTS's own confirmed-via-ONNX-inspection topology exactly). No speaker conditioning
    // (`speaker_embedding_size=0`, `num_speakers=1` in config.json -- single-speaker checkpoint,
    // unlike MeloTTS's `gin_channels`-conditioned decoder).
    public int[] UpsampleRates { get; }
    public int[] UpsampleKernelSizes { get; }
    public int UpsampleInitialChannel { get; }
    public int[] ResblockKernelSizes { get; }
    public float[] DecConvPreWeight { get; }
    public float[] DecConvPreBias { get; }
    public float[][] DecUpsWeight { get; }
    public float[][] DecUpsBias { get; }
    public MmsResBlockWeights[] DecResblocks { get; }
    public float[] DecConvPostWeight { get; }

    public MmsTtsWeights(string safetensorsPath, MmsTtsConfig config)
    {
        using var loader = SafetensorsLoader.Open(safetensorsPath);

        HiddenDim = config.HiddenSize;
        NumHeads = config.NumAttentionHeads;
        WindowSize = config.WindowSize;
        FfnKernel = config.FfnKernelSize;

        var emb = loader.ReadF32("text_encoder.embed_tokens.weight");
        int[] embShape = loader.GetShape("text_encoder.embed_tokens.weight");
        VocabSize = embShape[0];
        EmbeddingWeight = emb;

        ProjWeight = loader.ReadF32("text_encoder.project.weight");
        ProjBias = loader.ReadF32("text_encoder.project.bias");

        Layers = new MmsEncoderLayerWeights[config.NumHiddenLayers];
        for (int i = 0; i < config.NumHiddenLayers; i++)
            Layers[i] = new MmsEncoderLayerWeights(loader, i);

        Func<string, float[]> getDds = DdsNameAdapter(loader);

        DpConvPreWeight = loader.ReadF32("duration_predictor.conv_pre.weight");
        DpConvPreBias = loader.ReadF32("duration_predictor.conv_pre.bias");
        DpConvDds = new VitsDdsConvWeights(getDds, "duration_predictor.conv_dds");
        DpConvProjWeight = loader.ReadF32("duration_predictor.conv_proj.weight");
        DpConvProjBias = loader.ReadF32("duration_predictor.conv_proj.bias");
        DpFlow0LogScale = loader.ReadF32("duration_predictor.flows.0.log_scale");
        DpFlow0Translate = loader.ReadF32("duration_predictor.flows.0.translate");
        DpFlow0ExpNegLogScale = new float[DpFlow0LogScale.Length];
        for (int i = 0; i < DpFlow0LogScale.Length; i++)
            DpFlow0ExpNegLogScale[i] = MathF.Exp(-DpFlow0LogScale[i]);
        DpFlow2 = new VitsConvFlowWeights(getDds, "duration_predictor.flows.2");
        DpFlow3 = new VitsConvFlowWeights(getDds, "duration_predictor.flows.3");
        DpFlow4 = new VitsConvFlowWeights(getDds, "duration_predictor.flows.4");

        int[] flowPostShape = loader.GetShape("flow.flows.0.conv_post.weight");
        FlowHalfChannels = flowPostShape[0];
        FlowLayers = new MmsCouplingLayerWeights[FlowWnLayers];
        for (int i = 0; i < FlowWnLayers; i++)
            FlowLayers[i] = new MmsCouplingLayerWeights(loader, $"flow.flows.{i}");

        UpsampleRates = config.UpsampleRates;
        UpsampleKernelSizes = config.UpsampleKernelSizes;
        UpsampleInitialChannel = config.UpsampleInitialChannel;
        ResblockKernelSizes = config.ResblockKernelSizes;

        DecConvPreWeight = loader.ReadF32("decoder.conv_pre.weight");
        DecConvPreBias = loader.ReadF32("decoder.conv_pre.bias");

        int numStages = UpsampleRates.Length;
        DecUpsWeight = new float[numStages][];
        DecUpsBias = new float[numStages][];
        for (int i = 0; i < numStages; i++)
        {
            DecUpsWeight[i] = loader.ReadF32($"decoder.upsampler.{i}.weight");
            DecUpsBias[i] = loader.ReadF32($"decoder.upsampler.{i}.bias");
        }

        int numResblocksPerStage = ResblockKernelSizes.Length;
        int totalResblocks = numStages * numResblocksPerStage;
        DecResblocks = new MmsResBlockWeights[totalResblocks];
        for (int i = 0; i < totalResblocks; i++)
            DecResblocks[i] = new MmsResBlockWeights(loader, $"decoder.resblocks.{i}");

        DecConvPostWeight = loader.ReadF32("decoder.conv_post.weight");
    }

    /// <summary>
    /// Adapts the shared <c>VitsDdsConvWeights</c>/<c>VitsConvFlowWeights</c> primitives (extracted
    /// from Piper's ONNX naming: `.pre.`/`.proj.`/`.convs.convs_sep.`/`.convs.convs_1x1.`/`.gamma`/
    /// `.beta`) to the real HuggingFace `transformers.VitsModel` DDS-conv/ConvFlow naming (confirmed
    /// via the real safetensors header): `.conv_pre.`/`.conv_proj.`/`.conv_dds.convs_dilated.`/
    /// `.conv_dds.convs_pointwise.`/`.weight`/`.bias`. Translating the requested tensor NAME here
    /// (rather than forking the shared weight-struct classes) keeps the real DDSConv/ConvFlow math
    /// in `VitsDurationFlowKernels` a single source of truth across Piper and MMS-TTS.
    /// </summary>
    private static Func<string, float[]> DdsNameAdapter(SafetensorsLoader loader) => name =>
    {
        string real = name
            .Replace(".convs.convs_sep.", ".conv_dds.convs_dilated.")
            .Replace(".convs.convs_1x1.", ".conv_dds.convs_pointwise.")
            .Replace(".convs.norms_1.", ".conv_dds.norms_1.")
            .Replace(".convs.norms_2.", ".conv_dds.norms_2.")
            .Replace(".convs_sep.", ".convs_dilated.")
            .Replace(".convs_1x1.", ".convs_pointwise.")
            .Replace(".pre.", ".conv_pre.")
            .Replace(".proj.", ".conv_proj.")
            .Replace(".gamma", ".weight")
            .Replace(".beta", ".bias");
        return loader.ReadF32(real);
    };

    /// <summary>Folds `weight_g` (magnitude, `[outCh,1,1]`) * `weight_v` (direction, `[outCh,inCh,K]`) / ||v[outCh,:,:]||_2 into a plain conv weight -- PyTorch's older `nn.utils.weight_norm` convention (dim=0, norm over all other dims per output channel). Same math as `OpenTail.Stingray.Audio.Parler.DacWeights.FoldConvWeight`.</summary>
    internal static float[] FoldConvWeight(SafetensorsLoader loader, string prefix)
    {
        var g = loader.ReadF32($"{prefix}.weight_g");
        var v = loader.ReadF32($"{prefix}.weight_v");
        int[] vShape = loader.GetShape($"{prefix}.weight_v");
        int outCh = vShape[0];
        int perChannel = v.Length / outCh;

        var folded = new float[v.Length];
        for (int o = 0; o < outCh; o++)
        {
            double sumSq = 0;
            int baseIdx = o * perChannel;
            for (int j = 0; j < perChannel; j++) { double vv = v[baseIdx + j]; sumSq += vv * vv; }
            float norm = (float)Math.Sqrt(sumSq);
            float scale = norm > 1e-12f ? g[o] / norm : 0f;
            for (int j = 0; j < perChannel; j++) folded[baseIdx + j] = v[baseIdx + j] * scale;
        }
        return folded;
    }
}

/// <summary>text_encoder.encoder.layers.N -- HF uses separate q/k/v/out_proj Linear layers where
/// Piper's ONNX export used kernel=1 Conv1d; mathematically identical ([out,in] row-major weight),
/// directly compatible with <see cref="VitsAttentionKernels.RelPositionSelfAttention"/>'s expected
/// conv-weight layout with no transpose needed.</summary>
public sealed class MmsEncoderLayerWeights
{
    public float[] EmbRelK { get; }
    public float[] EmbRelV { get; }
    public float[] ConvQWeight { get; }
    public float[] ConvQBias { get; }
    public float[] ConvKWeight { get; }
    public float[] ConvKBias { get; }
    public float[] ConvVWeight { get; }
    public float[] ConvVBias { get; }
    public float[] ConvOWeight { get; }
    public float[] ConvOBias { get; }

    public float[] Norm1Gamma { get; }
    public float[] Norm1Beta { get; }

    public float[] Ffn1Weight { get; }
    public float[] Ffn1Bias { get; }
    public float[] Ffn2Weight { get; }
    public float[] Ffn2Bias { get; }

    public float[] Norm2Gamma { get; }
    public float[] Norm2Beta { get; }

    public MmsEncoderLayerWeights(SafetensorsLoader loader, int i)
    {
        string p = $"text_encoder.encoder.layers.{i}";
        EmbRelK = loader.ReadF32($"{p}.attention.emb_rel_k");
        EmbRelV = loader.ReadF32($"{p}.attention.emb_rel_v");
        ConvQWeight = loader.ReadF32($"{p}.attention.q_proj.weight");
        ConvQBias = loader.ReadF32($"{p}.attention.q_proj.bias");
        ConvKWeight = loader.ReadF32($"{p}.attention.k_proj.weight");
        ConvKBias = loader.ReadF32($"{p}.attention.k_proj.bias");
        ConvVWeight = loader.ReadF32($"{p}.attention.v_proj.weight");
        ConvVBias = loader.ReadF32($"{p}.attention.v_proj.bias");
        ConvOWeight = loader.ReadF32($"{p}.attention.out_proj.weight");
        ConvOBias = loader.ReadF32($"{p}.attention.out_proj.bias");

        Norm1Gamma = loader.ReadF32($"{p}.layer_norm.weight");
        Norm1Beta = loader.ReadF32($"{p}.layer_norm.bias");

        Ffn1Weight = loader.ReadF32($"{p}.feed_forward.conv_1.weight");
        Ffn1Bias = loader.ReadF32($"{p}.feed_forward.conv_1.bias");
        Ffn2Weight = loader.ReadF32($"{p}.feed_forward.conv_2.weight");
        Ffn2Bias = loader.ReadF32($"{p}.feed_forward.conv_2.bias");

        Norm2Gamma = loader.ReadF32($"{p}.final_layer_norm.weight");
        Norm2Beta = loader.ReadF32($"{p}.final_layer_norm.bias");
    }
}

/// <summary>ResidualCouplingLayer (mean_only=True): conv_pre (1x1) -> WN (wavenet) -> conv_post (1x1, m only).</summary>
public sealed class MmsCouplingLayerWeights
{
    public float[] PreWeight { get; }
    public float[] PreBias { get; }
    public MmsWnWeights Wavenet { get; }
    public float[] PostWeight { get; }
    public float[] PostBias { get; }

    public MmsCouplingLayerWeights(SafetensorsLoader loader, string prefix)
    {
        PreWeight = loader.ReadF32($"{prefix}.conv_pre.weight");
        PreBias = loader.ReadF32($"{prefix}.conv_pre.bias");
        Wavenet = new MmsWnWeights(loader, $"{prefix}.wavenet");
        PostWeight = loader.ReadF32($"{prefix}.conv_post.weight");
        PostBias = loader.ReadF32($"{prefix}.conv_post.bias");
    }
}

/// <summary>WN (WaveNet-style dilated conv stack), gin_channels=0 (no speaker conditioning).
/// in_layers/res_skip_layers ship as raw `weight_g`/`weight_v` pairs -- folded via
/// <see cref="MmsTtsWeights.FoldConvWeight"/>.</summary>
public sealed class MmsWnWeights
{
    public float[][] InWeight { get; } = new float[MmsTtsWeights.FlowWnLayers][];
    public float[][] InBias { get; } = new float[MmsTtsWeights.FlowWnLayers][];
    public float[][] ResSkipWeight { get; } = new float[MmsTtsWeights.FlowWnLayers][];
    public float[][] ResSkipBias { get; } = new float[MmsTtsWeights.FlowWnLayers][];

    public MmsWnWeights(SafetensorsLoader loader, string prefix)
    {
        for (int i = 0; i < MmsTtsWeights.FlowWnLayers; i++)
        {
            InWeight[i] = MmsTtsWeights.FoldConvWeight(loader, $"{prefix}.in_layers.{i}");
            InBias[i] = loader.ReadF32($"{prefix}.in_layers.{i}.bias");
            ResSkipWeight[i] = MmsTtsWeights.FoldConvWeight(loader, $"{prefix}.res_skip_layers.{i}");
            ResSkipBias[i] = loader.ReadF32($"{prefix}.res_skip_layers.{i}.bias");
        }
    }
}

/// <summary>HiFi-GAN ResBlock1: 3 conv PAIRS (convs1[j] at dilation (1,3,5), convs2[j] at dilation 1),
/// same topology as MeloTTS's `MeloResBlockWeights`.</summary>
public sealed class MmsResBlockWeights
{
    public float[][] Convs1Weight { get; } = new float[3][];
    public float[][] Convs1Bias { get; } = new float[3][];
    public float[][] Convs2Weight { get; } = new float[3][];
    public float[][] Convs2Bias { get; } = new float[3][];

    public MmsResBlockWeights(SafetensorsLoader loader, string prefix)
    {
        for (int i = 0; i < 3; i++)
        {
            Convs1Weight[i] = loader.ReadF32($"{prefix}.convs1.{i}.weight");
            Convs1Bias[i] = loader.ReadF32($"{prefix}.convs1.{i}.bias");
            Convs2Weight[i] = loader.ReadF32($"{prefix}.convs2.{i}.weight");
            Convs2Bias[i] = loader.ReadF32($"{prefix}.convs2.{i}.bias");
        }
    }
}
