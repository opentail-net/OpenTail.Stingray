using System;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Audio.Parler;

/// <summary>
/// Real weight loader for Parler-TTS's DAC (Descript Audio Codec) decoder, loaded from
/// `models/parler-tts-mini-v1.safetensors` (`audio_encoder.*` prefix). Real architecture and
/// config confirmed from the actual external `descript-audio-codec` package (`pip download
/// descript-audio-codec --no-deps`, real source in `dac/model/dac.py`/`dac/nn/quantize.py`) --
/// SNAC (already ported and golden-verified this session) is a sibling/derivative of this same
/// DAC lineage, but NOT identical: DAC's `ResidualUnit` uses FULL (non-depthwise, no `groups`
/// parameter at all in the real `WNConv1d(dim,dim,kernel=7,dilation=dilation,...)` call) 1D
/// convolutions, unlike SNAC's depthwise ones -- do not reuse `SnacDecoder`'s depthwise conv
/// kernel here.
///
/// <para>Real config, confirmed from `parler_tts/dac_wrapper/configuration_dac.py` (Parler's own
/// values) plus `dac/model/dac.py`'s `DAC.__init__` defaults (used for the parameters Parler's
/// wrapper doesn't override): `n_codebooks=9`, `codebook_size=1024`, `latent_dim=1024`,
/// `decoder_dim=1536` (DAC default), `decoder_rates=[8,8,4,2]` (DAC default), `codebook_dim=8`
/// (DAC default) -- all independently cross-checked against the real tensor shapes in
/// `models/parler-tts-mini-v1.safetensors` (e.g. `audio_encoder.model.decoder.model.0.weight_v`
/// shape `[1536,1024,7]` = out=1536(decoder_dim),in=1024(latent_dim),kernel=7, confirming the
/// first decoder conv exactly).</para>
///
/// <para>Real quantizer, confirmed different from SNAC's: `ResidualVectorQuantize.from_codes`
/// sums `out_proj(codebook[i][codes[i]])` across all 9 real quantizers with NO time-upsampling/
/// stride step -- all 9 codebooks operate at the SAME time resolution (unlike SNAC's
/// hierarchical 1/2/4-rate scheme), genuinely simpler in this one respect.</para>
///
/// <para>Weight-norm is UNFOLDED in this checkpoint (real `weight_g`/`weight_v` tensor pairs,
/// the OLDER `nn.utils.weight_norm` convention -- confirmed from the real
/// `DACModel.apply_weight_norm`'s version-conditional call) -- folded here via
/// <see cref="FoldConvWeight"/>, following the same math as
/// `CosyVoiceHiftWeights.GetFoldedConvWeight` (which uses the newer `.parametrizations.weight.
/// original0/1` naming for a different checkpoint -- same decomposition, different tensor
/// names).</para>
/// </summary>
public sealed class DacWeights
{
    public const int LatentDim = 1024;
    public const int DecoderDim = 1536;
    public const int CodebookSize = 1024;
    public const int CodebookDim = 8;
    public const int NumCodebooks = 9;
    public static readonly int[] DecoderRates = [8, 8, 4, 2];

    public float[] In0Weight { get; } // first conv: latent_dim -> decoder_dim, k=7
    public float[] In0Bias { get; }
    public DacDecoderBlockWeights[] DecBlocks { get; } = new DacDecoderBlockWeights[DecoderRates.Length];
    public float[] OutAlpha { get; }
    public float[] OutWeight { get; } // final conv: 96 -> 1, k=7
    public float[] OutBias { get; }
    public DacQuantizerWeights[] Quantizers { get; } = new DacQuantizerWeights[NumCodebooks];

    public DacWeights(SafetensorsLoader loader)
    {
        In0Weight = FoldConvWeight(loader, "audio_encoder.model.decoder.model.0");
        In0Bias = loader.ReadF32("audio_encoder.model.decoder.model.0.bias");

        for (int i = 0; i < DecoderRates.Length; i++)
        {
            string p = $"audio_encoder.model.decoder.model.{i + 1}";
            var res = new DacResidualUnitWeights[3];
            for (int r = 0; r < 3; r++)
            {
                string rp = $"{p}.block.{r + 2}.block";
                res[r] = new DacResidualUnitWeights
                {
                    Alpha0 = loader.ReadF32($"{rp}.0.alpha"),
                    Conv0Weight = FoldConvWeight(loader, $"{rp}.1"),
                    Conv0Bias = loader.ReadF32($"{rp}.1.bias"),
                    Alpha1 = loader.ReadF32($"{rp}.2.alpha"),
                    Conv1Weight = FoldConvWeight(loader, $"{rp}.3"),
                    Conv1Bias = loader.ReadF32($"{rp}.3.bias"),
                };
            }
            DecBlocks[i] = new DacDecoderBlockWeights
            {
                Alpha = loader.ReadF32($"{p}.block.0.alpha"),
                UpWeight = FoldConvWeight(loader, $"{p}.block.1"),
                UpBias = loader.ReadF32($"{p}.block.1.bias"),
                Res = res,
            };
        }

        OutAlpha = loader.ReadF32("audio_encoder.model.decoder.model.5.alpha");
        OutWeight = FoldConvWeight(loader, "audio_encoder.model.decoder.model.6");
        OutBias = loader.ReadF32("audio_encoder.model.decoder.model.6.bias");

        for (int i = 0; i < NumCodebooks; i++)
        {
            Quantizers[i] = new DacQuantizerWeights
            {
                Codebook = loader.ReadF32($"audio_encoder.model.quantizer.quantizers.{i}.codebook.weight"),
                OutProjWeight = FoldConvWeight(loader, $"audio_encoder.model.quantizer.quantizers.{i}.out_proj"),
                OutProjBias = loader.ReadF32($"audio_encoder.model.quantizer.quantizers.{i}.out_proj.bias"),
            };
        }
    }

    /// <summary>
    /// Real GGUF loader, for the same `ecyht2/parler-tts-mini-v1-GGUF` conversion
    /// <see cref="Parler.ParlerDecoderWeights(Core.GgufModel)"/> loads its decoder from. Real
    /// tensor names here are a GENUINELY DIFFERENT, flatter convention than the Safetensors
    /// checkpoint's (confirmed via `list-tensors`, not assumed): `audio_encoder.initial.*` (first
    /// conv), `audio_encoder.decoder_block.{1..4}.*` (1-based, matching `DecoderRates.Length=4`),
    /// `audio_encoder.final.*` (last conv), `audio_encoder.quantizers.{0..8}.*`. One real, initially
    /// confusing naming choice, confirmed by cross-checking real tensor SHAPES against this
    /// checkpoint's known channel progression (1536-&gt;768-&gt;384-&gt;192-&gt;96): each
    /// `decoder_block.{i}.final.*` group bundles what this project's Safetensors-derived field
    /// names split into three separate pieces -- `final.alpha` is the pre-upsample Snake
    /// activation (shape `[1, prevChannels]`, confirmed via its shape matching the INPUT channel
    /// count, not the output), `final.weight`/`final.bias` are the actual `ConvTranspose1d`
    /// upsample (kernel=2*rate) -- i.e. GGUF's `decoder_block.{i}.final` == this class's own
    /// `DecBlocks[i-1].{Alpha,UpWeight,UpBias}`, NOT a second/different "final" conv. Weight-norm
    /// is ALREADY FOLDED in this GGUF (plain `.weight`/`.bias`/`.alpha`, no `weight_g`/`weight_v`
    /// pair anywhere) -- no <see cref="FoldConvWeight"/> step needed, unlike the Safetensors path.
    /// </summary>
    public DacWeights(GgufModel model)
    {
        In0Weight = GetF32(model, "audio_encoder.initial.weight");
        In0Bias = GetF32(model, "audio_encoder.initial.bias");

        for (int i = 0; i < DecoderRates.Length; i++)
        {
            string p = $"audio_encoder.decoder_block.{i + 1}";
            var res = new DacResidualUnitWeights[3];
            for (int r = 0; r < 3; r++)
            {
                string rp = $"{p}.residual_unit.{r}.res";
                res[r] = new DacResidualUnitWeights
                {
                    Alpha0 = GetF32(model, $"{rp}.initial.alpha"),
                    Conv0Weight = GetF32(model, $"{rp}.initial.weight"),
                    Conv0Bias = GetF32(model, $"{rp}.initial.bias"),
                    Alpha1 = GetF32(model, $"{rp}.final.alpha"),
                    Conv1Weight = GetF32(model, $"{rp}.final.weight"),
                    Conv1Bias = GetF32(model, $"{rp}.final.bias"),
                };
            }
            DecBlocks[i] = new DacDecoderBlockWeights
            {
                Alpha = GetF32(model, $"{p}.final.alpha"),
                UpWeight = GetF32(model, $"{p}.final.weight"),
                UpBias = GetF32(model, $"{p}.final.bias"),
                Res = res,
            };
        }

        OutAlpha = GetF32(model, "audio_encoder.final.alpha");
        OutWeight = GetF32(model, "audio_encoder.final.weight");
        OutBias = GetF32(model, "audio_encoder.final.bias");

        for (int i = 0; i < NumCodebooks; i++)
        {
            Quantizers[i] = new DacQuantizerWeights
            {
                Codebook = GetF32(model, $"audio_encoder.quantizers.{i}.codebook.weight"),
                OutProjWeight = GetF32(model, $"audio_encoder.quantizers.{i}.out_proj.weight"),
                OutProjBias = GetF32(model, $"audio_encoder.quantizers.{i}.out_proj.bias"),
            };
        }
    }

    private static float[] GetF32(GgufModel model, string name)
    {
        var info = model.FindTensor(name) ?? throw new System.IO.InvalidDataException($"Parler DAC GGUF missing required tensor '{name}'.");
        var bytes = model.GetTensorData(info);
        var dst = new float[info.ElementCount];
        Dequantize.ToFloat32(bytes, dst, info.DType, info.ElementCount);
        return dst;
    }

    /// <summary>Folds `weight_g` (magnitude, `[outCh,1,1]`) * `weight_v` (direction, `[outCh,inCh,K]`) / ||v[outCh,:,:]||_2 into a plain conv weight -- PyTorch's older `nn.utils.weight_norm` convention (dim=0, norm over all other dims per output channel).</summary>
    private static float[] FoldConvWeight(SafetensorsLoader loader, string prefix)
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

public sealed class DacResidualUnitWeights
{
    public required float[] Alpha0 { get; init; }
    public required float[] Conv0Weight { get; init; } // FULL conv, dilated, kernel=7
    public required float[] Conv0Bias { get; init; }
    public required float[] Alpha1 { get; init; }
    public required float[] Conv1Weight { get; init; } // FULL conv, kernel=1
    public required float[] Conv1Bias { get; init; }
}

public sealed class DacDecoderBlockWeights
{
    public required float[] Alpha { get; init; }
    public required float[] UpWeight { get; init; } // ConvTranspose1d
    public required float[] UpBias { get; init; }
    public required DacResidualUnitWeights[] Res { get; init; }
}

public sealed class DacQuantizerWeights
{
    public required float[] Codebook { get; init; } // [CodebookSize, CodebookDim]
    public required float[] OutProjWeight { get; init; } // pointwise conv, CodebookDim -> LatentDim
    public required float[] OutProjBias { get; init; }
}
