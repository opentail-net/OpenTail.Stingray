
namespace OpenTail.Stingray.Audio.Rvc;

/// <summary>
/// Real weight loader for RVC's VITS-style synthesizer (text encoder with relative-position
/// self-attention, a normalizing flow, and an NSF-HiFiGAN generator) -- transcribed directly from
/// `examples/audio.cpp/src/models/rvc/synthesizer.cpp`'s `load_weights`/`fold_weight_norm` (not
/// guessed). Unlike HuBERT/RMVPE (which live under a single fixed `support_*` prefix), the packed
/// `rvc-f16.gguf` bundles MULTIPLE named voices, each under its own
/// `voice_v{1,2}_&lt;name&gt;_checkpoint/` prefix (confirmed via a real tensor-name dump -- e.g.
/// `voice_v2_default_checkpoint/enc_p.emb_phone.weight`), with a matching
/// `voice_v{1,2}_&lt;name&gt;_index_vectors/` prefix holding that voice's k-NN retrieval index. This
/// class is weight-loading plumbing only; the forward pass (RvcSynthesizer) is a separate,
/// not-yet-written next step -- see docs/audio-review-progress.md's RVC section.
/// </summary>
public sealed class RvcSynthesizerWeights
{
    public const int InterChannels = 192;
    public const int HiddenChannels = 192;
    public const int FilterChannels = 768;
    public const int Heads = 2;
    public const int HeadDim = 96;
    public const int TextLayers = 6;
    public const int RelativeWindowSize = 10;
    public const int GinChannels = 256;

    public bool V1 { get; }
    public bool HasF0 { get; }
    public int SampleRate { get; }
    public int[] UpsampleRates { get; } // 4 entries
    public int[] UpsampleKernelSizes { get; } // 4 entries
    public int HopSamples { get; }

    public float[] EmbPhoneWeight { get; } // [HiddenChannels, featureDim] (Linear)
    public float[] EmbPhoneBias { get; }
    public float[]? EmbPitchWeight { get; } // [256, HiddenChannels], null when !HasF0
    public float[] ProjWeight { get; } // [InterChannels*2, HiddenChannels, 1]
    public float[] ProjBias { get; }
    public RvcTextAttnLayer[] AttnLayers { get; } = new RvcTextAttnLayer[TextLayers];
    public RvcConv1d[] FfnConv1 { get; } = new RvcConv1d[TextLayers];
    public RvcConv1d[] FfnConv2 { get; } = new RvcConv1d[TextLayers];
    public RvcLayerNorm[] NormLayers1 { get; } = new RvcLayerNorm[TextLayers];
    public RvcLayerNorm[] NormLayers2 { get; } = new RvcLayerNorm[TextLayers];

    public float[] EmbGWeight { get; } // [numSpeakers, GinChannels]

    public RvcFlow[] Flows { get; } = new RvcFlow[4]; // real flow indices 0,2,4,6 (odd indices are pure channel-flips, no weights)

    public float[]? SourceLinearWeight { get; } // dec.m_source.l_linear, [1,1], null when !HasF0
    public float[]? SourceLinearBias { get; }
    public RvcConv1d DecConvPre { get; } // [512, InterChannels, 7]
    public RvcConv1d DecCond { get; } // [512, GinChannels, 1]
    public RvcConvTranspose1d[] DecUps { get; } = new RvcConvTranspose1d[4];
    public RvcConv1d?[] DecNoiseConvs { get; } = new RvcConv1d?[4]; // null entries when !HasF0
    public RvcResBlock1[] DecResBlocks { get; } = new RvcResBlock1[12]; // 4 upsample stages x 3 kernels
    public float[] DecConvPostWeight { get; } // [1, 32, 7], no bias

    /// <param name="voicePrefix">e.g. "voice_v2_default_checkpoint" (no trailing slash/dot) --
    /// the reference bundles multiple named voices side by side under distinct prefixes.</param>
    public RvcSynthesizerWeights(RvcPackedTensorSource source, string voicePrefix, int sampleRate, bool v1, bool hasF0)
    {
        SampleRate = sampleRate;
        V1 = v1;
        HasF0 = hasF0;
        string root = voicePrefix + "/";
        Tensor get(string name) => new(source, root, name);

        (UpsampleRates, UpsampleKernelSizes) = InferLayout(get, sampleRate);
        HopSamples = 1;
        foreach (var r in UpsampleRates) HopSamples *= r;

        EmbPhoneWeight = get("enc_p.emb_phone.weight").Get();
        EmbPhoneBias = get("enc_p.emb_phone.bias").Get();
        EmbPitchWeight = hasF0 ? get("enc_p.emb_pitch.weight").Get() : null;
        ProjWeight = get("enc_p.proj.weight").Get();
        ProjBias = get("enc_p.proj.bias").Get();

        for (int i = 0; i < TextLayers; i++)
        {
            string p = $"enc_p.encoder.attn_layers.{i}";
            AttnLayers[i] = new RvcTextAttnLayer
            {
                ConvQ = LoadConv1d(get, $"{p}.conv_q"),
                ConvK = LoadConv1d(get, $"{p}.conv_k"),
                ConvV = LoadConv1d(get, $"{p}.conv_v"),
                ConvO = LoadConv1d(get, $"{p}.conv_o"),
                EmbRelK = get($"{p}.emb_rel_k").Get(),
                EmbRelV = get($"{p}.emb_rel_v").Get(),
            };
            FfnConv1[i] = LoadConv1d(get, $"enc_p.encoder.ffn_layers.{i}.conv_1");
            FfnConv2[i] = LoadConv1d(get, $"enc_p.encoder.ffn_layers.{i}.conv_2");
            NormLayers1[i] = new RvcLayerNorm(get($"enc_p.encoder.norm_layers_1.{i}.gamma").Get(), get($"enc_p.encoder.norm_layers_1.{i}.beta").Get());
            NormLayers2[i] = new RvcLayerNorm(get($"enc_p.encoder.norm_layers_2.{i}.gamma").Get(), get($"enc_p.encoder.norm_layers_2.{i}.beta").Get());
        }

        EmbGWeight = get("emb_g.weight").Get();

        int[] flowIndices = [0, 2, 4, 6];
        for (int f = 0; f < 4; f++)
        {
            string p = $"flow.flows.{flowIndices[f]}";
            Flows[f] = new RvcFlow
            {
                Pre = LoadConv1d(get, $"{p}.pre"),
                CondLayer = LoadWeightNormConv1d(get, $"{p}.enc.cond_layer"),
                InLayers = [.. Enumerable.Range(0, 3).Select(i => LoadWeightNormConv1d(get, $"{p}.enc.in_layers.{i}"))],
                ResSkipLayers = [.. Enumerable.Range(0, 3).Select(i => LoadWeightNormConv1d(get, $"{p}.enc.res_skip_layers.{i}"))],
                Post = LoadConv1d(get, $"{p}.post"),
            };
        }

        if (hasF0)
        {
            SourceLinearWeight = get("dec.m_source.l_linear.weight").Get();
            SourceLinearBias = get("dec.m_source.l_linear.bias").Get();
        }
        DecConvPre = LoadConv1d(get, "dec.conv_pre");
        DecCond = LoadConv1d(get, "dec.cond");
        for (int up = 0; up < 4; up++)
        {
            DecUps[up] = LoadWeightNormConvTranspose1d(get, $"dec.ups.{up}");
            if (hasF0) DecNoiseConvs[up] = LoadConv1d(get, $"dec.noise_convs.{up}");
        }
        for (int up = 0; up < 4; up++)
        {
            for (int k = 0; k < 3; k++)
            {
                int idx = up * 3 + k;
                string p = $"dec.resblocks.{idx}";
                DecResBlocks[idx] = new RvcResBlock1
                {
                    Convs1 = [.. Enumerable.Range(0, 3).Select(i => LoadWeightNormConv1d(get, $"{p}.convs1.{i}"))],
                    Convs2 = [.. Enumerable.Range(0, 3).Select(i => LoadWeightNormConv1d(get, $"{p}.convs2.{i}"))],
                };
            }
        }
        DecConvPostWeight = get("dec.conv_post.weight").Get();
    }

    private readonly record struct Tensor(RvcPackedTensorSource Source, string Root, string Name)
    {
        public float[] Get() => Source.GetTensor(Root + Name);
        public bool Exists() => Source.HasTensor(Root + Name);
    }

    private static (int[] Rates, int[] Kernels) InferLayout(Func<string, Tensor> get, int sampleRate)
    {
        int[] rates = sampleRate switch
        {
            32000 => [10, 8, 2, 2],
            40000 => [10, 10, 2, 2],
            48000 => [12, 10, 2, 2],
            _ => throw new NotSupportedException($"RVC synthesizer sample rate {sampleRate} is not supported."),
        };
        int[] inChannels = [512, 256, 128, 64];
        int[] channels = [256, 128, 64, 32];
        var kernels = new int[4];
        for (int up = 0; up < 4; up++)
        {
            var plain = get($"dec.ups.{up}.weight");
            var flat = plain.Exists() ? plain.Get() : get($"dec.ups.{up}.weight_v").Get();
            // Real PyTorch ConvTranspose1d weight: [inCh, outCh, kernel].
            int kernel = flat.Length / (inChannels[up] * channels[up]);
            kernels[up] = kernel;
        }
        return (rates, kernels);
    }

    private static RvcConv1d LoadConv1d(Func<string, Tensor> get, string prefix) =>
        new(get($"{prefix}.weight").Get(), get($"{prefix}.bias").Get());

    /// <summary>Real PyTorch `weight_norm(dim=0)` reconstruction: `weight = weight_v * (weight_g /
    /// ||weight_v||)`, norm computed per output-channel (dim 0) across all remaining dims combined
    /// -- confirmed against the reference's own `fold_weight_norm` (a SIMPLER case than F5-TTS's
    /// positional-conv `weight_norm(dim=2)`, since these are ordinary conv weights).</summary>
    private static RvcConv1d LoadWeightNormConv1d(Func<string, Tensor> get, string prefix)
    {
        var g = get($"{prefix}.weight_g").Get();
        var v = get($"{prefix}.weight_v").Get();
        return new RvcConv1d(FoldWeightNormDim0(g, v), get($"{prefix}.bias").Get());
    }

    private static RvcConvTranspose1d LoadWeightNormConvTranspose1d(Func<string, Tensor> get, string prefix)
    {
        var g = get($"{prefix}.weight_g").Get();
        var v = get($"{prefix}.weight_v").Get();
        return new RvcConvTranspose1d(FoldWeightNormDim0(g, v), get($"{prefix}.bias").Get());
    }

    private static float[] FoldWeightNormDim0(float[] g, float[] v)
    {
        int dim0 = g.Length;
        int rest = v.Length / dim0;
        var output = new float[v.Length];
        for (int row = 0; row < dim0; row++)
        {
            double sumSq = 0;
            int offset = row * rest;
            for (int i = 0; i < rest; i++) sumSq += (double)v[offset + i] * v[offset + i];
            double norm = Math.Sqrt(sumSq);
            float scale = (float)(g[row] / norm);
            for (int i = 0; i < rest; i++) output[offset + i] = v[offset + i] * scale;
        }
        return output;
    }
}

public readonly record struct RvcConv1d(float[] Weight, float[] Bias);
public readonly record struct RvcConvTranspose1d(float[] Weight, float[] Bias);
public readonly record struct RvcLayerNorm(float[] Gamma, float[] Beta);

public sealed class RvcTextAttnLayer
{
    public RvcConv1d ConvQ { get; set; }
    public RvcConv1d ConvK { get; set; }
    public RvcConv1d ConvV { get; set; }
    public RvcConv1d ConvO { get; set; }
    public float[] EmbRelK { get; set; } = [];
    public float[] EmbRelV { get; set; } = [];
}

public sealed class RvcFlow
{
    public RvcConv1d Pre { get; set; }
    public RvcConv1d CondLayer { get; set; }
    public RvcConv1d[] InLayers { get; set; } = [];
    public RvcConv1d[] ResSkipLayers { get; set; } = [];
    public RvcConv1d Post { get; set; }
}

public sealed class RvcResBlock1
{
    public RvcConv1d[] Convs1 { get; set; } = []; // dilations 1,3,5
    public RvcConv1d[] Convs2 { get; set; } = []; // dilation 1 always
}
