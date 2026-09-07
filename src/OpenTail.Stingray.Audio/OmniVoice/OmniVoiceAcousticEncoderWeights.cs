
namespace OpenTail.Stingray.Audio.OmniVoice;

/// <summary>
/// Real weight loader for OmniVoice's acoustic DAC-style encoder (`acoustic_encoder.*` in
/// `audio_tokenizer/model.safetensors`), transcribed directly from
/// `examples/audio.cpp/src/models/omnivoice/audio_tokenizer.cpp`'s `build_acoustic_encoder` (not
/// guessed). The mirror image of <see cref="OmniVoiceAcousticDecoderWeights"/>'s decoder: real
/// `encoder_hidden_size=64`, 5 downsampling stages with the SAME real rates
/// `downsampling_ratios=[8,5,4,2,3]` (channels DOUBLE each stage instead of halving: 64-&gt;128-&gt;
/// 256-&gt;512-&gt;1024-&gt;2048), each stage's residual units come BEFORE its strided downsample conv
/// (decoder order is reversed: upsample-then-residual). Needed only for re-encoding reference
/// audio (voice cloning) -- not needed for basic text-to-speech generation from LLM-produced
/// codes, which only needs the decoder.
/// </summary>
public sealed class OmniVoiceAcousticEncoderWeights
{
    public const int EncoderHiddenSize = 64;
    public static readonly int[] DownsamplingRatios = [8, 5, 4, 2, 3];

    public float[] Conv1Weight { get; } // [64, 1, 7]
    public float[] Conv1Bias { get; }
    public OmniVoiceDacEncoderBlockWeights[] Blocks { get; } = new OmniVoiceDacEncoderBlockWeights[DownsamplingRatios.Length];
    public float[] FinalSnakeAlpha { get; }
    public float[] Conv2Weight { get; } // [256, 2048, 3]
    public float[] Conv2Bias { get; }

    public OmniVoiceAcousticEncoderWeights(SafetensorsLoader loader) : this(loader.ReadF32)
    {
    }

    /// <summary>
    /// Real, generalized constructor, added 2026-09-07 so this same class can be reused for
    /// Higgs Audio TTS's real acoustic encoder -- confirmed identical real architecture
    /// (`kEncoderChannels=[64,128,256,512,1024,2048]`, `kUpsampleRatios=[8,5,4,2,3]` reused
    /// for the encoder's own strided downsample convs with real kernel `2*ratio`, same DAC-style
    /// residual-units-before-downsample block order), verified directly against
    /// `higgs_audio_tts/codec.cpp`'s own `load_encoder_block`, not assumed -- an earlier session
    /// note comparing `kAcousticHiddenSize=256` (Higgs's final projected dim) against this class's
    /// `EncoderHiddenSize=64` (the STARTING dim, not final) was an apples-to-oranges mistake; this
    /// class's OWN final `Conv2Weight` is ALSO `[256, 2048, 3]`, i.e. the same 256-channel final
    /// projection -- the two encoders are the same real architecture after all.
    /// </summary>
    public OmniVoiceAcousticEncoderWeights(Func<string, float[]> get)
    {
        Conv1Weight = get("acoustic_encoder.conv1.weight");
        Conv1Bias = get("acoustic_encoder.conv1.bias");

        int channels = EncoderHiddenSize;
        for (int b = 0; b < DownsamplingRatios.Length; b++)
        {
            string p = $"acoustic_encoder.block.{b}";
            int nextChannels = EncoderHiddenSize * (1 << (b + 1));
            Blocks[b] = new OmniVoiceDacEncoderBlockWeights
            {
                Res1 = LoadResidualUnit(get, $"{p}.res_unit1"),
                Res2 = LoadResidualUnit(get, $"{p}.res_unit2"),
                Res3 = LoadResidualUnit(get, $"{p}.res_unit3"),
                SnakeAlpha = get($"{p}.snake1.alpha"),
                ConvWeight = get($"{p}.conv1.weight"),
                ConvBias = get($"{p}.conv1.bias"),
                InChannels = channels,
                OutChannels = nextChannels,
            };
            channels = nextChannels;
        }

        FinalSnakeAlpha = get("acoustic_encoder.snake1.alpha");
        Conv2Weight = get("acoustic_encoder.conv2.weight");
        Conv2Bias = get("acoustic_encoder.conv2.bias");
    }

    private static OmniVoiceResidualUnitWeights LoadResidualUnit(Func<string, float[]> get, string prefix) => new()
    {
        Snake1Alpha = get($"{prefix}.snake1.alpha"),
        Conv1Weight = get($"{prefix}.conv1.weight"),
        Conv1Bias = get($"{prefix}.conv1.bias"),
        Snake2Alpha = get($"{prefix}.snake2.alpha"),
        Conv2Weight = get($"{prefix}.conv2.weight"),
        Conv2Bias = get($"{prefix}.conv2.bias"),
    };
}

public sealed class OmniVoiceDacEncoderBlockWeights
{
    public OmniVoiceResidualUnitWeights Res1 { get; set; } = new();
    public OmniVoiceResidualUnitWeights Res2 { get; set; } = new();
    public OmniVoiceResidualUnitWeights Res3 { get; set; } = new();
    public float[] SnakeAlpha { get; set; } = [];
    public float[] ConvWeight { get; set; } = []; // real PyTorch Conv1d [out,in,kernel]
    public float[] ConvBias { get; set; } = [];
    public int InChannels { get; set; }
    public int OutChannels { get; set; }
}
