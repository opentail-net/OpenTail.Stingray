
namespace OpenTail.Stingray.Audio.OmniVoice;

/// <summary>
/// Real weight loader for OmniVoice's acoustic DAC-style decoder + quantizer-decode path
/// (`acoustic_decoder.*`/`quantizer.quantizers.*` in `audio_tokenizer/model.safetensors`),
/// transcribed directly from `examples/audio.cpp/src/models/omnivoice/audio_tokenizer.cpp`'s
/// `build_acoustic_decoder`/`build_quantizer_decode_sequence` (not guessed). Same DAC lineage as
/// this codebase's `Parler.DacDecoder`/`DacWeights` (Snake activations, `ResidualUnit`
/// snake-conv-snake-conv-plus-residual, `ConvTranspose1d`-then-crop upsampling), but
/// re-parameterized: `decoder_hidden_size=1024` (not Parler's 1536), 5 upsample stages with real
/// rates `upsampling_ratios=[8,5,4,2,3]` (not Parler's 4-stage `[8,8,4,2]`), and a real top-level
/// `num_codebooks=8`/`codebook_size=1024`/`codebook_dim=64` (from `audio_tokenizer/config.json`'s
/// own top-level fields -- NOT the nested `acoustic_model_config.codebook_dim=8`, a distinct,
/// unused-for-this-purpose field with the same name -- confirmed via a real tensor dump: 8
/// `quantizer.quantizers.N` groups, each `codebook.embed` shaped `[1024,64]`). The quantizer
/// decode is real DAC-style: for each codebook, look up `codebook.embed[code]`, project through
/// `project_out` (`Linear(64-&gt;1024)`, with bias), and SUM across all 8 codebooks with no
/// per-codebook scaling -- the encoder-side-only `score` precomputation (`2*embed`/`-||embed||^2`,
/// a nearest-neighbor-as-linear-layer trick for the encoder's own codebook search) is not needed
/// for decoding and is not loaded here.
/// </summary>
public sealed class OmniVoiceAcousticDecoderWeights
{
    public const int DecoderHiddenSize = 1024; // acoustic_model_config.decoder_hidden_size
    public const int QuantizerLatentDim = 1024; // top-level audio_tokenizer config hidden_size -- the quantizer-decode sum's real width, confirmed via project_out.weight=[1024,64]
    public const int AcousticLatentDim = 256; // acoustic_model_config.hidden_size -- acoustic_decoder.conv1's real real input width
    public const int NumCodebooks = 8; // top-level audio_tokenizer config num_codebooks
    public const int CodebookSize = 1024;
    public const int CodebookDim = 64;
    public static readonly int[] UpsamplingRatios = [8, 5, 4, 2, 3];

    /// <summary>Real `fc2` projection (`[256,1024]`+bias`[256]`) applied to the quantizer-decode
    /// sum BEFORE `acoustic_decoder.conv1` -- confirmed via a real tensor dump (missed on first
    /// pass: the quantizer's summed `project_out` output is `QuantizerLatentDim`=1024-wide, but
    /// `acoustic_decoder.conv1`'s real weight shape requires a 256-wide input, so this projection
    /// is NOT optional). Real reference call site:
    /// `native_pipeline... build_linear_btc_as_conv1d(..., weights_->fc2, hidden_size,
    /// acoustic_model.hidden_size, true)` in `audio_tokenizer.cpp`.</summary>
    public float[] Fc2Weight { get; } // [256, 1024]
    public float[] Fc2Bias { get; } // [256]

    public float[] Conv1Weight { get; } // [1024, 256, 7]
    public float[] Conv1Bias { get; }
    public OmniVoiceDacDecoderBlockWeights[] Blocks { get; } = new OmniVoiceDacDecoderBlockWeights[UpsamplingRatios.Length];
    public float[] FinalSnakeAlpha { get; }
    public float[] Conv2Weight { get; } // [1, lastChannels, 7]
    public float[] Conv2Bias { get; }

    public OmniVoiceQuantizerWeights[] Quantizers { get; } = new OmniVoiceQuantizerWeights[NumCodebooks];

    public OmniVoiceAcousticDecoderWeights(SafetensorsLoader loader)
    {
        Fc2Weight = loader.ReadF32("fc2.weight");
        Fc2Bias = loader.ReadF32("fc2.bias");

        Conv1Weight = loader.ReadF32("acoustic_decoder.conv1.weight");
        Conv1Bias = loader.ReadF32("acoustic_decoder.conv1.bias");

        int channels = DecoderHiddenSize;
        for (int b = 0; b < UpsamplingRatios.Length; b++)
        {
            string p = $"acoustic_decoder.block.{b}";
            int outCh = channels / 2;
            Blocks[b] = new OmniVoiceDacDecoderBlockWeights
            {
                SnakeAlpha = loader.ReadF32($"{p}.snake1.alpha"),
                ConvTWeight = loader.ReadF32($"{p}.conv_t1.weight"),
                ConvTBias = loader.ReadF32($"{p}.conv_t1.bias"),
                InChannels = channels,
                OutChannels = outCh,
                Res1 = LoadResidualUnit(loader, $"{p}.res_unit1"),
                Res2 = LoadResidualUnit(loader, $"{p}.res_unit2"),
                Res3 = LoadResidualUnit(loader, $"{p}.res_unit3"),
            };
            channels = outCh;
        }

        FinalSnakeAlpha = loader.ReadF32("acoustic_decoder.snake1.alpha");
        Conv2Weight = loader.ReadF32("acoustic_decoder.conv2.weight");
        Conv2Bias = loader.ReadF32("acoustic_decoder.conv2.bias");

        for (int i = 0; i < NumCodebooks; i++)
        {
            string p = $"quantizer.quantizers.{i}";
            Quantizers[i] = new OmniVoiceQuantizerWeights
            {
                CodebookEmbed = loader.ReadF32($"{p}.codebook.embed"),
                ProjectOutWeight = loader.ReadF32($"{p}.project_out.weight"),
                ProjectOutBias = loader.ReadF32($"{p}.project_out.bias"),
            };
        }
    }

    private static OmniVoiceResidualUnitWeights LoadResidualUnit(SafetensorsLoader loader, string prefix) => new()
    {
        Snake1Alpha = loader.ReadF32($"{prefix}.snake1.alpha"),
        Conv1Weight = loader.ReadF32($"{prefix}.conv1.weight"),
        Conv1Bias = loader.ReadF32($"{prefix}.conv1.bias"),
        Snake2Alpha = loader.ReadF32($"{prefix}.snake2.alpha"),
        Conv2Weight = loader.ReadF32($"{prefix}.conv2.weight"),
        Conv2Bias = loader.ReadF32($"{prefix}.conv2.bias"),
    };
}

public sealed class OmniVoiceDacDecoderBlockWeights
{
    public float[] SnakeAlpha { get; set; } = [];
    public float[] ConvTWeight { get; set; } = []; // real PyTorch ConvTranspose1d [in,out,kernel]
    public float[] ConvTBias { get; set; } = [];
    public int InChannels { get; set; }
    public int OutChannels { get; set; }
    public OmniVoiceResidualUnitWeights Res1 { get; set; } = new();
    public OmniVoiceResidualUnitWeights Res2 { get; set; } = new();
    public OmniVoiceResidualUnitWeights Res3 { get; set; } = new();
}

public sealed class OmniVoiceResidualUnitWeights
{
    public float[] Snake1Alpha { get; set; } = [];
    public float[] Conv1Weight { get; set; } = []; // [C,C,7]
    public float[] Conv1Bias { get; set; } = [];
    public float[] Snake2Alpha { get; set; } = [];
    public float[] Conv2Weight { get; set; } = []; // [C,C,1]
    public float[] Conv2Bias { get; set; } = [];
}

public sealed class OmniVoiceQuantizerWeights
{
    public float[] CodebookEmbed { get; set; } = []; // [1024, 64]
    public float[] ProjectOutWeight { get; set; } = []; // [1024, 64] (out=hidden, in=codebookDim)
    public float[] ProjectOutBias { get; set; } = [];
}
