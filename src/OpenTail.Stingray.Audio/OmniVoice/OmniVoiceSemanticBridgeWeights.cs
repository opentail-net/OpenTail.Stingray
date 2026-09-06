
namespace OpenTail.Stingray.Audio.OmniVoice;

/// <summary>
/// Real weight loader for OmniVoice's semantic-to-acoustic "bridge" stage
/// (`encoder_semantic.*` in `audio_tokenizer/model.safetensors`), transcribed directly from
/// `examples/audio.cpp/src/models/omnivoice/audio_tokenizer.cpp`'s `build_semantic_encoder`/
/// `build_semantic_residual_unit` (not guessed). A small, plain EnCodec-style residual conv stack
/// bridging HuBERT's semantic output into the acoustic codec's latent space -- NOT a
/// downsampling stage (the real top-level `config.strides` is `[1,1]`, i.e. both blocks use
/// stride 1; time-axis downsampling to the acoustic codec's rate happens earlier, via a plain
/// nearest-neighbor frame-index selection over HuBERT's own output, not a learned conv). Real
/// structure: `conv` (kernel=`config.kernel_size`=3, pad=1, no bias) -&gt; 2 blocks, each: 2
/// residual units (`ELU -&gt; conv1(kernel=`unit_kernel_size`=3, dilation=`block_dilations[i]`=1,
/// no bias) -&gt; ELU -&gt; conv2(kernel=1, no bias) -&gt; +residual`) followed by one more conv
/// (kernel=3 since stride=1, pad=1, WITH bias).
/// </summary>
public sealed class OmniVoiceSemanticBridgeWeights
{
    public const int HiddenDim = 768; // semantic_model.hidden_size
    public const int NumBlocks = 2; // config.strides.Length
    public const int ResUnitsPerBlock = 2; // config.block_dilations.Length
    public const int UnitKernelSize = 3; // config.unit_kernel_size
    public const int KernelSize = 3; // config.kernel_size

    public float[] ConvWeight { get; } // [768,768,3], no bias
    public OmniVoiceBridgeBlockWeights[] Blocks { get; } = new OmniVoiceBridgeBlockWeights[NumBlocks];

    public OmniVoiceSemanticBridgeWeights(SafetensorsLoader loader)
    {
        ConvWeight = loader.ReadF32("encoder_semantic.conv.weight");
        for (int b = 0; b < NumBlocks; b++)
        {
            var resConv1 = new float[ResUnitsPerBlock][];
            var resConv2 = new float[ResUnitsPerBlock][];
            for (int r = 0; r < ResUnitsPerBlock; r++)
            {
                string p = $"encoder_semantic.conv_blocks.{b}.res_units.{r}";
                resConv1[r] = loader.ReadF32($"{p}.conv1.weight");
                resConv2[r] = loader.ReadF32($"{p}.conv2.weight");
            }
            Blocks[b] = new OmniVoiceBridgeBlockWeights
            {
                ResConv1 = resConv1,
                ResConv2 = resConv2,
                ConvWeight = loader.ReadF32($"encoder_semantic.conv_blocks.{b}.conv.weight"),
                ConvBias = loader.ReadF32($"encoder_semantic.conv_blocks.{b}.conv.bias"),
            };
        }
    }
}

public sealed class OmniVoiceBridgeBlockWeights
{
    public float[][] ResConv1 { get; set; } = [];
    public float[][] ResConv2 { get; set; } = [];
    public float[] ConvWeight { get; set; } = [];
    public float[] ConvBias { get; set; } = [];
}
