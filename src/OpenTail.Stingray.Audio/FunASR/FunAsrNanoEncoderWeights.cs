
namespace OpenTail.Stingray.Audio.FunASR;

/// <summary>
/// Real weight loader for Fun-ASR-Nano-2512's audio-tower SAN-M encoder (`model.audio_tower.*` in
/// the real audio.cpp-packed GGUF, e.g. `models/paraformer-q8.gguf` -- despite the filename, this
/// is the CURRENT Fun-ASR-Nano-2512 architecture, not the old CIF-Paraformer one; see
/// `docs/audio-review-progress.md`'s "FunASR-nano" section). Transcribed directly from
/// `examples/audio.cpp/src/models/fun_asr_nano/encoder.cpp`'s `load_encoder_weights` (not
/// guessed): one `stem` SAN-M block (`input_size=560 -&gt; d_model=512`), `layers-1=49` main SAN-M
/// blocks (`d_model -&gt; d_model`), a `layer_norm`, `timestamp_prediction_layers=20` more SAN-M
/// blocks, and a final `timestamp_prediction_layer_norm`. Real hard-asserted config (from
/// `assets.cpp`): `input_size=560, d_model=512, attention_heads=4, ffn_dim=2048, layers=50,
/// timestamp_prediction_layers=20, kernel_size=11, max_position_embeddings=2049`.
/// </summary>
public sealed class FunAsrNanoEncoderWeights
{
    public const int InputSize = 560;
    public const int DModel = 512;
    public const int AttentionHeads = 4;
    public const int FfnDim = 2048;
    public const int Layers = 50;
    public const int TimestampPredictionLayers = 20;
    public const int KernelSize = 11;
    public const int MaxPositionEmbeddings = 2049;
    public const float LayerNormEps = 1e-5f;

    public FunAsrNanoSanmWeights Stem { get; }
    public FunAsrNanoSanmWeights[] MainLayers { get; }
    public (float[] Weight, float[] Bias) MainNorm { get; }
    public FunAsrNanoSanmWeights[] TimestampLayers { get; }
    public (float[] Weight, float[] Bias) TimestampNorm { get; }

    public FunAsrNanoEncoderWeights(Rvc.RvcPackedTensorSource source)
    {
        const string root = "model.audio_tower.";
        Stem = LoadBlock(source, root + "stem");

        MainLayers = new FunAsrNanoSanmWeights[Layers - 1];
        for (int i = 0; i < Layers - 1; i++)
            MainLayers[i] = LoadBlock(source, $"{root}layers.{i}");
        MainNorm = (source.GetTensor(root + "layer_norm.weight"), source.GetTensor(root + "layer_norm.bias"));

        TimestampLayers = new FunAsrNanoSanmWeights[TimestampPredictionLayers];
        for (int i = 0; i < TimestampPredictionLayers; i++)
            TimestampLayers[i] = LoadBlock(source, $"{root}timestamp_prediction_layers.{i}");
        TimestampNorm = (source.GetTensor(root + "timestamp_prediction_layer_norm.weight"), source.GetTensor(root + "timestamp_prediction_layer_norm.bias"));
    }

    private static FunAsrNanoSanmWeights LoadBlock(Rvc.RvcPackedTensorSource source, string prefix) => new()
    {
        SelfAttnLayerNormWeight = source.GetTensor($"{prefix}.self_attn_layer_norm.weight"),
        SelfAttnLayerNormBias = source.GetTensor($"{prefix}.self_attn_layer_norm.bias"),
        QWeight = source.GetTensor($"{prefix}.self_attn.q_proj.weight"),
        QBias = source.GetTensor($"{prefix}.self_attn.q_proj.bias"),
        KWeight = source.GetTensor($"{prefix}.self_attn.k_proj.weight"),
        KBias = source.GetTensor($"{prefix}.self_attn.k_proj.bias"),
        VWeight = source.GetTensor($"{prefix}.self_attn.v_proj.weight"),
        VBias = source.GetTensor($"{prefix}.self_attn.v_proj.bias"),
        OutWeight = source.GetTensor($"{prefix}.self_attn.out_proj.weight"),
        OutBias = source.GetTensor($"{prefix}.self_attn.out_proj.bias"),
        FsmnConvWeight = source.GetTensor($"{prefix}.fsmn.conv.weight"),
        FinalLayerNormWeight = source.GetTensor($"{prefix}.final_layer_norm.weight"),
        FinalLayerNormBias = source.GetTensor($"{prefix}.final_layer_norm.bias"),
        Fc1Weight = source.GetTensor($"{prefix}.fc1.weight"),
        Fc1Bias = source.GetTensor($"{prefix}.fc1.bias"),
        Fc2Weight = source.GetTensor($"{prefix}.fc2.weight"),
        Fc2Bias = source.GetTensor($"{prefix}.fc2.bias"),
    };
}
