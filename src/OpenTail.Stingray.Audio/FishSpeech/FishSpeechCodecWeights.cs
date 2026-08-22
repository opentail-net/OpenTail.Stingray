using System;
using System.IO;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Audio.FishSpeech;

/// <summary>
/// Real weight loader for Fish Speech S2 Pro's codec DECODE path only (real "DownsampleResidual
/// VectorQuantize.decode" + "DAC.decoder", transcribed from the real `fishaudio/fish-speech`
/// GitHub repo's `fish_speech/models/dac/{modded_dac.py,rvq.py}` -- see docs/audio-review-
/// progress.md's Fish Speech codec section for the full derivation).
///
/// <para><b>Critical correction to an earlier scoping pass, confirmed from real tensor names,
/// not guessed</b>: the real `DownsampleResidualVectorQuantize` module has BOTH a `pre_module`
/// (the real 8-layer transformer, confirmed via `c.quantizer.pre_module.layers.{0..7}.*` real
/// tensors) AND a `post_module` (confirmed via real tensors to be just a SINGLE bare
/// `RMSNorm` -- `c.quantizer.post_module.norm.weight`, no transformer layers at all). The real
/// `decode()` method ONLY calls `post_module` (the bare norm) -- `pre_module` (the real 8-layer
/// transformer) is applied to the CONTINUOUS latent during ENCODING only, never during decode.
/// This means decode-only inference (all this codebase's TTS pipelines ever need) does NOT
/// require porting the 8-layer transformer at all -- a significant, real scope reduction from
/// an earlier entry's assumption.</para>
///
/// <para>Weight-norm is ALREADY FOLDED in this GGUF (plain `.conv.weight`/`.conv.bias` tensor
/// names throughout, confirmed via `list-tensors` -- unlike Parler-TTS's DAC safetensors, which
/// needed manual `weight_g`/`weight_v` folding). No folding step needed here.</para>
///
/// <para>Real config, confirmed from GGUF metadata + real tensor shapes: 1 semantic quantizer
/// (codebook_size=4096, codebook_dim=8) + 9 residual quantizers (codebook_size=1024,
/// codebook_dim=8), `latent_dim=1024`, 2-stage downsample/upsample (`quantizer_downsample_
/// factor=[2,2]`, each stage: causal ConvTranspose1d(k=2,stride=2) + ConvNeXtBlock), then the
/// plain `DAC.decoder`: causal conv (1024-&gt;1536, k=7) -&gt; 4x causal DecoderBlock (real
/// `decoder_rates=[8,8,4,2]`, decoder_dim=1536) -&gt; final causal conv (96-&gt;1, k=7) -&gt;
/// Tanh. All convolutions in the DECODE path are CAUSAL (left-pad only) -- confirmed from the
/// real `DAC.__init__`'s `causal=True` default -- NOT the symmetric same-padding SNAC/Parler's
/// DAC use.</para>
/// </summary>
public sealed class FishSpeechCodecWeights : IDisposable
{
    public const int LatentDim = 1024;
    public const int DecoderDim = 1536;
    public const int SemanticCodebookSize = 4096;
    public const int ResidualCodebookSize = 1024;
    public const int CodebookDim = 8;
    public const int NumResidualCodebooks = 9;
    public static readonly int[] DecoderRates = [8, 8, 4, 2];
    public static readonly int[] UpsampleFactors = [2, 2];

    public GgufModel Model { get; }

    public FishSpeechCodecQuantizerWeights SemanticQuantizer { get; }
    public FishSpeechCodecQuantizerWeights[] ResidualQuantizers { get; } = new FishSpeechCodecQuantizerWeights[NumResidualCodebooks];
    public float[] PostModuleNormWeight { get; }
    public FishSpeechCodecUpsampleStageWeights[] UpsampleStages { get; } = new FishSpeechCodecUpsampleStageWeights[2];

    public float[] DecIn0Weight { get; }
    public float[] DecIn0Bias { get; }
    public FishSpeechCodecDecoderBlockWeights[] DecBlocks { get; } = new FishSpeechCodecDecoderBlockWeights[4];
    public float[] DecOutAlpha { get; }
    public float[] DecOutWeight { get; }
    public float[] DecOutBias { get; }

    public FishSpeechCodecWeights(string ggufPath)
    {
        Model = GgufModel.Open(ggufPath);

        SemanticQuantizer = LoadQuantizer("c.quantizer.semantic_quantizer.quantizers.0");
        for (int i = 0; i < NumResidualCodebooks; i++)
            ResidualQuantizers[i] = LoadQuantizer($"c.quantizer.quantizer.quantizers.{i}");

        PostModuleNormWeight = GetTensor("c.quantizer.post_module.norm.weight");

        for (int i = 0; i < 2; i++)
        {
            string p = $"c.quantizer.upsample.{i}";
            UpsampleStages[i] = new FishSpeechCodecUpsampleStageWeights
            {
                ConvWeight = GetTensor($"{p}.0.conv.weight"),
                ConvBias = GetTensor($"{p}.0.conv.bias"),
                Block = new FishSpeechConvNeXtBlockWeights
                {
                    DwConvWeight = GetTensor($"{p}.1.dwconv.conv.weight"),
                    DwConvBias = GetTensor($"{p}.1.dwconv.conv.bias"),
                    NormWeight = GetTensor($"{p}.1.norm.weight"),
                    NormBias = GetTensor($"{p}.1.norm.bias"),
                    PwConv1Weight = GetTensor($"{p}.1.pwconv1.weight"),
                    PwConv1Bias = GetTensor($"{p}.1.pwconv1.bias"),
                    PwConv2Weight = GetTensor($"{p}.1.pwconv2.weight"),
                    PwConv2Bias = GetTensor($"{p}.1.pwconv2.bias"),
                    Gamma = GetTensor($"{p}.1.gamma"),
                },
            };
        }

        DecIn0Weight = GetTensor("c.decoder.model.0.conv.weight");
        DecIn0Bias = GetTensor("c.decoder.model.0.conv.bias");

        for (int i = 0; i < 4; i++)
        {
            string p = $"c.decoder.model.{i + 1}";
            var res = new FishSpeechCodecResidualUnitWeights[3];
            for (int r = 0; r < 3; r++)
            {
                string rp = $"{p}.block.{r + 2}.block";
                res[r] = new FishSpeechCodecResidualUnitWeights
                {
                    Alpha0 = GetTensor($"{rp}.0.alpha"),
                    Conv0Weight = GetTensor($"{rp}.1.conv.weight"),
                    Conv0Bias = GetTensor($"{rp}.1.conv.bias"),
                    Alpha1 = GetTensor($"{rp}.2.alpha"),
                    Conv1Weight = GetTensor($"{rp}.3.conv.weight"),
                    Conv1Bias = GetTensor($"{rp}.3.conv.bias"),
                };
            }
            DecBlocks[i] = new FishSpeechCodecDecoderBlockWeights
            {
                Alpha = GetTensor($"{p}.block.0.alpha"),
                UpWeight = GetTensor($"{p}.block.1.conv.weight"),
                UpBias = GetTensor($"{p}.block.1.conv.bias"),
                Res = res,
            };
        }

        DecOutAlpha = GetTensor("c.decoder.model.5.alpha");
        DecOutWeight = GetTensor("c.decoder.model.6.conv.weight");
        DecOutBias = GetTensor("c.decoder.model.6.conv.bias");
    }

    private FishSpeechCodecQuantizerWeights LoadQuantizer(string prefix) => new()
    {
        Codebook = GetTensor($"{prefix}.codebook.weight"),
        OutProjWeight = GetTensor($"{prefix}.out_proj.weight"),
        OutProjBias = GetTensor($"{prefix}.out_proj.bias"),
    };

    public float[] GetTensor(string name)
    {
        var info = Model.FindTensor(name) ?? throw new InvalidDataException($"Fish Speech codec GGUF missing required tensor '{name}'.");
        var bytes = Model.GetTensorData(info);
        var dst = new float[info.ElementCount];
        Dequantize.ToFloat32(bytes, dst, info.DType, info.ElementCount);
        return dst;
    }

    public void Dispose() => Model.Dispose();
}

public sealed class FishSpeechCodecQuantizerWeights
{
    public required float[] Codebook { get; init; } // [CodebookSize, CodebookDim]
    public required float[] OutProjWeight { get; init; } // pointwise conv, CodebookDim -> LatentDim
    public required float[] OutProjBias { get; init; }
}

public sealed class FishSpeechConvNeXtBlockWeights
{
    public required float[] DwConvWeight { get; init; } // depthwise, kernel=7, causal
    public required float[] DwConvBias { get; init; }
    public required float[] NormWeight { get; init; } // real nn.LayerNorm (channels-last)
    public required float[] NormBias { get; init; }
    public required float[] PwConv1Weight { get; init; } // Linear dim -> 4*dim
    public required float[] PwConv1Bias { get; init; }
    public required float[] PwConv2Weight { get; init; } // Linear 4*dim -> dim
    public required float[] PwConv2Bias { get; init; }
    public required float[] Gamma { get; init; } // LayerScale
}

public sealed class FishSpeechCodecUpsampleStageWeights
{
    public required float[] ConvWeight { get; init; } // causal ConvTranspose1d, kernel=2, stride=2
    public required float[] ConvBias { get; init; }
    public required FishSpeechConvNeXtBlockWeights Block { get; init; }
}

public sealed class FishSpeechCodecResidualUnitWeights
{
    public required float[] Alpha0 { get; init; }
    public required float[] Conv0Weight { get; init; } // FULL conv, causal, dilated, kernel=7
    public required float[] Conv0Bias { get; init; }
    public required float[] Alpha1 { get; init; }
    public required float[] Conv1Weight { get; init; } // FULL conv, kernel=1
    public required float[] Conv1Bias { get; init; }
}

public sealed class FishSpeechCodecDecoderBlockWeights
{
    public required float[] Alpha { get; init; }
    public required float[] UpWeight { get; init; } // causal ConvTranspose1d
    public required float[] UpBias { get; init; }
    public required FishSpeechCodecResidualUnitWeights[] Res { get; init; }
}
