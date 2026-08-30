
namespace OpenTail.Stingray.Audio.Xtts;

/// <summary>
/// Real XTTS-v2 DVAE (discrete VAE audio tokenizer) DECODER-ONLY weights, loaded from
/// `dvae.safetensors` (converted from the real `coqui/XTTS-v2` `dvae.pth` via
/// `scratch-llamacpp-ref/xtts_convert_to_safetensors.py` -- a plain, no-values-altered .pth-to-
/// safetensors conversion, done so this loads through this codebase's existing
/// <see cref="SafetensorsLoader"/> instead of a from-scratch Python-pickle parser).
///
/// <para>Real architecture confirmed from the actual `coqui-ai-TTS` source
/// (`TTS/tts/layers/xtts/dvae.py`'s `DiscreteVAE`, constructed with the REAL args found in
/// `TTS/tts/layers/xtts/trainer/gpt_trainer.py`: <c>channels=80, positional_dims=1,
/// codebook_dim=512, hidden_dim=512, num_resnet_blocks=3, kernel_size=3, num_layers=2,
/// use_transposed_convs=False</c>, activation defaults to ReLU (not overridden)) -- NOT guessed
/// from tensor shapes alone, cross-checked against the real construction call.</para>
///
/// <para><b>Only the DECODER path is loaded/needed at inference</b>: the GPT2 autoregressively
/// predicts codebook INDICES directly (no audio-side DVAE encoding happens at synthesis time --
/// `DiscreteVAE.encode`/the `encoder.*` weights are training-only, used to build the GPT2's
/// training targets from real audio). `codebook.embed` IS needed (index -> 512-dim vector
/// lookup, `DiscreteVAE.decode`'s first step).</para>
/// </summary>
public sealed class XttsDvaeWeights
{
    public const int CodebookDim = 512;
    public const int NumTokens = 1024;
    public const int InnermostDim = 1024; // hidden_dim * 2^(num_layers-1) = 512*2
    public const int MelDim = 80;
    public const int UpsampleStride = 2;
    public const int UpsampleKernel = 3;

    /// <summary>[CodebookDim, NumTokens] -- column `codes[t]` is that code's 512-dim embedding vector (real `Quantize.embed`, `embed_code` selects by column since the buffer is stored transposed relative to a normal embedding table).</summary>
    public float[] CodebookEmbed { get; }

    public float[] Decoder0Weight { get; } // conv1x1, CodebookDim(512) -> InnermostDim(1024)
    public float[] Decoder0Bias { get; }

    public XttsDvaeResBlockWeights[] ResBlocks { get; } = new XttsDvaeResBlockWeights[3]; // decoder.1/2/3

    public float[] Decoder4Weight { get; } // UpsampledConv, k3, InnermostDim -> InnermostDim
    public float[] Decoder4Bias { get; }
    public float[] Decoder5Weight { get; } // UpsampledConv, k3, InnermostDim -> InnermostDim/2 (512)
    public float[] Decoder5Bias { get; }

    public float[] Decoder6Weight { get; } // conv1x1, 512 -> MelDim(80)
    public float[] Decoder6Bias { get; }

    public XttsDvaeWeights(string safetensorsPath)
    {
        using var loader = SafetensorsLoader.Open(safetensorsPath);

        CodebookEmbed = loader.ReadF32("codebook.embed");

        Decoder0Weight = loader.ReadF32("decoder.0.weight");
        Decoder0Bias = loader.ReadF32("decoder.0.bias");

        for (int i = 0; i < 3; i++)
            ResBlocks[i] = new XttsDvaeResBlockWeights(loader, $"decoder.{i + 1}");

        Decoder4Weight = loader.ReadF32("decoder.4.0.conv.weight");
        Decoder4Bias = loader.ReadF32("decoder.4.0.conv.bias");
        Decoder5Weight = loader.ReadF32("decoder.5.0.conv.weight");
        Decoder5Bias = loader.ReadF32("decoder.5.0.conv.bias");

        Decoder6Weight = loader.ReadF32("decoder.6.weight");
        Decoder6Bias = loader.ReadF32("decoder.6.bias");
    }
}

/// <summary>Real `ResBlock`: net = [conv(chan,chan,k3,pad1), ReLU, conv(chan,chan,k3,pad1), ReLU, conv(chan,chan,k1)], output = net(x) + x.</summary>
public sealed class XttsDvaeResBlockWeights
{
    public float[] Conv0Weight { get; } // net.0, k3
    public float[] Conv0Bias { get; }
    public float[] Conv2Weight { get; } // net.2, k3
    public float[] Conv2Bias { get; }
    public float[] Conv4Weight { get; } // net.4, k1
    public float[] Conv4Bias { get; }

    public XttsDvaeResBlockWeights(SafetensorsLoader loader, string prefix)
    {
        Conv0Weight = loader.ReadF32($"{prefix}.net.0.weight");
        Conv0Bias = loader.ReadF32($"{prefix}.net.0.bias");
        Conv2Weight = loader.ReadF32($"{prefix}.net.2.weight");
        Conv2Bias = loader.ReadF32($"{prefix}.net.2.bias");
        Conv4Weight = loader.ReadF32($"{prefix}.net.4.weight");
        Conv4Bias = loader.ReadF32($"{prefix}.net.4.bias");
    }
}
