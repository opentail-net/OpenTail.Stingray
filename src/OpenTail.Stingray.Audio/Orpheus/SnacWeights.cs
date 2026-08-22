using System;
using System.IO;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Audio.Orpheus;

/// <summary>
/// Real GGUF weight loader for the SNAC 24kHz codec decoder (`hubertsiuzdak/snac_24khz`, used
/// by Orpheus TTS to turn generated codec tokens into PCM). Config confirmed directly from the
/// real HF `config.json` (not assumed/defaulted from the `snac` package's class defaults, which
/// are for a different, unrelated 44.1kHz variant): `decoder_dim=1024`, `decoder_rates=[8,8,4,2]`,
/// `latent_dim=768` (= encoder_dim 48 * 2^4), `vq_strides=[4,2,1]`, `codebook_size=4096`,
/// `codebook_dim=8`, `attn_window_size=null` (no LocalMHA anywhere in this variant's decoder --
/// confirmed by the real GGUF tensor dump also having no attention tensors), `depthwise=true`,
/// `noise=true` (NoiseBlock present in weights but made a no-op at inference, see
/// <see cref="SnacDecoder"/>'s doc comment for why).
///
/// <para>Real tensor naming (`snac.*`), dumped directly from `models/snac-24khz.gguf` via
/// `list-tensors`, not guessed: `snac.dec.in0`/`in1` (Decoder's two init convs, depthwise 768ch
/// k=7 then pointwise 768-&gt;1024 k=1), `snac.dec.{0..3}.*` (the four DecoderBlocks, strides
/// 8/8/4/2), `snac.dec.out.*` (final Snake1d -&gt; conv -&gt; tanh), `snac.q.{0..2}.*` (the three
/// residual-VQ sub-quantizers -- codebook + in_proj + out_proj each, though only `codebook` and
/// `out_proj` are needed for decode-only use, matching Orpheus's use case).</para>
///
/// <para>Weight-norm is already folded into a single plain weight tensor per conv in this GGUF
/// (confirmed by the tensor dump: no separate `.original0`/`.original1` (g/v) pairs anywhere,
/// unlike e.g. `CosyVoiceHiftWeights`'s checkpoint) -- <see cref="GetTensor"/> can load each
/// conv's weight directly with no folding step.</para>
/// </summary>
public sealed class SnacWeights : IDisposable
{
    public const int DecoderDim = 1024;
    public const int LatentDim = 768;
    public const int CodebookSize = 4096;
    public const int CodebookDim = 8;
    public static readonly int[] DecoderRates = [8, 8, 4, 2];
    public static readonly int[] VqStrides = [4, 2, 1]; // per-quantizer time-upsample factor

    public GgufModel Model { get; }

    // Decoder init convs: in0 = depthwise 768ch k=7, in1 = pointwise 768->1024 k=1.
    public float[] In0Weight { get; }
    public float[] In0Bias { get; }
    public float[] In1Weight { get; }
    public float[] In1Bias { get; }

    public SnacDecoderBlockWeights[] DecBlocks { get; }

    // Final: Snake1d(64) -> conv(64->1, k=7) -> Tanh.
    public float[] OutAlpha { get; }
    public float[] OutWeight { get; }
    public float[] OutBias { get; }

    public SnacQuantizerWeights[] Quantizers { get; }

    public SnacWeights(string ggufPath)
    {
        Model = GgufModel.Open(ggufPath);

        In0Weight = GetTensor("snac.dec.in0.weight");
        In0Bias = GetTensor("snac.dec.in0.bias");
        In1Weight = GetTensor("snac.dec.in1.weight");
        In1Bias = GetTensor("snac.dec.in1.bias");

        DecBlocks = new SnacDecoderBlockWeights[DecoderRates.Length];
        for (int i = 0; i < DecBlocks.Length; i++)
            DecBlocks[i] = LoadDecoderBlock($"snac.dec.{i}");

        OutAlpha = GetTensor("snac.dec.out.alpha");
        OutWeight = GetTensor("snac.dec.out.weight");
        OutBias = GetTensor("snac.dec.out.bias");

        Quantizers = new SnacQuantizerWeights[VqStrides.Length];
        for (int i = 0; i < Quantizers.Length; i++)
        {
            Quantizers[i] = new SnacQuantizerWeights
            {
                Codebook = GetTensor($"snac.q.{i}.codebook"),
                OutProjWeight = GetTensor($"snac.q.{i}.out_proj.weight"),
                OutProjBias = GetTensor($"snac.q.{i}.out_proj.bias"),
                Stride = VqStrides[i],
            };
        }
    }

    private SnacDecoderBlockWeights LoadDecoderBlock(string prefix)
    {
        var res = new SnacResidualUnitWeights[3];
        for (int r = 0; r < 3; r++)
        {
            res[r] = new SnacResidualUnitWeights
            {
                Alpha0 = GetTensor($"{prefix}.res.{r}.alpha0"),
                Conv0Weight = GetTensor($"{prefix}.res.{r}.conv0.weight"),
                Conv0Bias = GetTensor($"{prefix}.res.{r}.conv0.bias"),
                Alpha1 = GetTensor($"{prefix}.res.{r}.alpha1"),
                Conv1Weight = GetTensor($"{prefix}.res.{r}.conv1.weight"),
                Conv1Bias = GetTensor($"{prefix}.res.{r}.conv1.bias"),
            };
        }

        return new SnacDecoderBlockWeights
        {
            Alpha = GetTensor($"{prefix}.alpha"),
            UpWeight = GetTensor($"{prefix}.up.weight"),
            UpBias = GetTensor($"{prefix}.up.bias"),
            Res = res,
        };
    }

    public float[] GetTensor(string name)
    {
        var info = Model.FindTensor(name) ?? throw new InvalidDataException($"SNAC GGUF missing required tensor '{name}'.");
        var bytes = Model.GetTensorData(info);
        var dst = new float[info.ElementCount];
        Dequantize.ToFloat32(bytes, dst, info.DType, info.ElementCount);
        return dst;
    }

    public void Dispose() => Model.Dispose();
}

public sealed class SnacResidualUnitWeights
{
    public required float[] Alpha0 { get; init; }
    public required float[] Conv0Weight { get; init; } // depthwise, dilated, kernel=7
    public required float[] Conv0Bias { get; init; }
    public required float[] Alpha1 { get; init; }
    public required float[] Conv1Weight { get; init; } // pointwise, kernel=1
    public required float[] Conv1Bias { get; init; }
}

public sealed class SnacDecoderBlockWeights
{
    public required float[] Alpha { get; init; } // Snake1d before the upsample conv
    public required float[] UpWeight { get; init; } // ConvTranspose1d
    public required float[] UpBias { get; init; }
    public required SnacResidualUnitWeights[] Res { get; init; } // dilations 1, 3, 9
}

public sealed class SnacQuantizerWeights
{
    public required float[] Codebook { get; init; } // [CodebookSize, CodebookDim]
    public required float[] OutProjWeight { get; init; } // pointwise conv, CodebookDim -> LatentDim
    public required float[] OutProjBias { get; init; }
    public required int Stride { get; init; }
}
