using System;
using System.IO;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Audio.Kokoro;

/// <summary>
/// Full real-weight loader for the Kokoro-82M StyleTTS2 GGUF graph (bert.*, pred.*, dec.*
/// tensor namespaces), verified against models/kokoro-82m-q8_0.gguf's actual KV metadata
/// and tensor table (see docs/audio-review-progress.md, "REBUILD PLAN"). Architecture
/// hyperparameters below are read straight from the file's own kokoro.* KV keys rather
/// than hardcoded, but fall back to the known Kokoro-82M defaults if a key is missing.
/// </summary>
public sealed class KokoroWeights : IDisposable
{
    public GgufModel Model { get; }
    public GgufModel? VoiceModel { get; }
    public float[]? StyleVector { get; }

    // --- Architecture hyperparameters (kokoro.* KV keys) ---
    public int HiddenDim { get; }          // kokoro.hidden_dim (decoder/predictor working dim), 512
    public int StyleDim { get; }           // kokoro.style_dim, 128
    public int MaxConvDim { get; }         // kokoro.max_conv_dim, 512
    public int MaxDur { get; }             // kokoro.max_dur, 50
    public int NTokenVocab { get; }        // kokoro.n_token, 178
    public int NMels { get; }              // kokoro.n_mels, 80
    public int NStyleLayers { get; }       // kokoro.n_layer -- number of AdaIN blocks (F0/N/decode/dur_enc), 3
    public int TextEncoderKernelSize { get; } // kokoro.text_encoder_kernel_size, 5
    public int SampleRate { get; }         // kokoro.sample_rate, 24000

    public int IstftNFft { get; }          // kokoro.istft.n_fft, 20
    public int IstftHopSize { get; }       // kokoro.istft.hop_size, 5
    public int[] UpsampleRates { get; }    // kokoro.istft.upsample_rates, [10, 6]
    public int[] UpsampleKernelSizes { get; } // kokoro.istft.upsample_kernel_sizes, [20, 12]
    public int[] ResblockKernelSizes { get; } // kokoro.istft.resblock_kernel_sizes, [3, 7, 11]
    public int[] ResblockDilations { get; }   // flattened kokoro.istft.resblock_dilation_sizes, [1,3,5]*3

    public int BertEmbeddingSize { get; }  // kokoro.plbert.embedding_size, 128
    public int BertHiddenSize { get; }     // kokoro.plbert.hidden_size, 768
    public int BertNumHiddenLayers { get; } // kokoro.plbert.num_hidden_layers, 12 (ALBERT-style: one shared weight set applied this many times)
    public int BertNumHeads { get; }       // kokoro.plbert.num_attention_heads, 12
    public int BertIntermediateSize { get; } // kokoro.plbert.intermediate_size, 2048
    public int BertMaxPositionEmbeddings { get; } // kokoro.plbert.max_position_embeddings, 512

    // --- BERT (ALBERT-style, one shared layer applied BertNumHiddenLayers times) ---
    public BertWeights Bert { get; }
    public float[] BertProjWeight { get; } // [768, 512] file-order [outFeatures, inFeatures] -> maps 768 BERT hidden to 512 working dim
    public float[] BertProjBias { get; }   // [512]

    // --- Text encoder (CNN + BiLSTM over token embeddings) ---
    public TextEncoderWeights TextEncoder { get; }

    // --- Prosody predictor (duration encoder, shared/duration LSTMs, F0 & N AdaIN stacks) ---
    public ProsodyPredictorWeights Predictor { get; }

    // --- Decoder (AdaIN ResBlock stack + iSTFTNet generator) ---
    public DecoderWeights Decoder { get; }

    public KokoroWeights(string modelPath, string? voicePath = null)
    {
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"Kokoro model file not found: {modelPath}");

        Model = GgufModel.Open(modelPath);

        HiddenDim = GetInt("kokoro.hidden_dim", 512);
        StyleDim = GetInt("kokoro.style_dim", 128);
        MaxConvDim = GetInt("kokoro.max_conv_dim", 512);
        MaxDur = GetInt("kokoro.max_dur", 50);
        NTokenVocab = GetInt("kokoro.n_token", 178);
        NMels = GetInt("kokoro.n_mels", 80);
        NStyleLayers = GetInt("kokoro.n_layer", 3);
        TextEncoderKernelSize = GetInt("kokoro.text_encoder_kernel_size", 5);
        SampleRate = GetInt("kokoro.sample_rate", 24000);

        IstftNFft = GetInt("kokoro.istft.n_fft", 20);
        IstftHopSize = GetInt("kokoro.istft.hop_size", 5);
        UpsampleRates = GetIntArray("kokoro.istft.upsample_rates", [10, 6]);
        UpsampleKernelSizes = GetIntArray("kokoro.istft.upsample_kernel_sizes", [20, 12]);
        ResblockKernelSizes = GetIntArray("kokoro.istft.resblock_kernel_sizes", [3, 7, 11]);
        ResblockDilations = GetIntArray("kokoro.istft.resblock_dilation_sizes", [1, 3, 5, 1, 3, 5, 1, 3, 5]);

        BertEmbeddingSize = GetInt("kokoro.plbert.embedding_size", 128);
        BertHiddenSize = GetInt("kokoro.plbert.hidden_size", 768);
        BertNumHiddenLayers = GetInt("kokoro.plbert.num_hidden_layers", 12);
        BertNumHeads = GetInt("kokoro.plbert.num_attention_heads", 12);
        BertIntermediateSize = GetInt("kokoro.plbert.intermediate_size", 2048);
        BertMaxPositionEmbeddings = GetInt("kokoro.plbert.max_position_embeddings", 512);

        Bert = new BertWeights(this);
        BertProjWeight = GetTensor("bert_proj.weight");
        BertProjBias = GetTensor("bert_proj.bias");

        TextEncoder = new TextEncoderWeights(this);
        Predictor = new ProsodyPredictorWeights(this);
        Decoder = new DecoderWeights(this);

        if (voicePath != null && File.Exists(voicePath))
        {
            VoiceModel = GgufModel.Open(voicePath);
            if (VoiceModel.Tensors.Count > 0)
            {
                var t = VoiceModel.Tensors[0];
                StyleVector = DequantTensor(VoiceModel, t);
            }
        }
    }

    private int GetInt(string key, int fallback) =>
        Model.Metadata.TryGetValue(key, out var v) ? Convert.ToInt32(v) : fallback;

    private int[] GetIntArray(string key, int[] fallback)
    {
        if (Model.Metadata.TryGetValue(key, out var v) && v is System.Collections.IEnumerable e and not string)
        {
            var list = new System.Collections.Generic.List<int>();
            foreach (var item in e) list.Add(Convert.ToInt32(item));
            return list.Count > 0 ? list.ToArray() : fallback;
        }
        return fallback;
    }

    /// <summary>Loads and dequantizes a required tensor by exact GGUF name to a flat float[] in file storage order.</summary>
    public float[] GetTensor(string name)
    {
        var info = Model.FindTensor(name) ?? throw new InvalidDataException($"Kokoro GGUF missing required tensor '{name}'.");
        return DequantTensor(Model, info);
    }

    /// <summary>Loads and dequantizes an optional tensor by exact GGUF name; returns null if absent.</summary>
    public float[]? TryGetTensor(string name)
    {
        var info = Model.FindTensor(name);
        return info is null ? null : DequantTensor(Model, info.Value);
    }

    private static float[] DequantTensor(GgufModel model, GgufTensorInfo info)
    {
        var bytes = model.GetTensorData(info);
        var dst = new float[info.ElementCount];
        Dequantize.ToFloat32(bytes, dst, info.DType, info.ElementCount);
        return dst;
    }

    public void Dispose()
    {
        VoiceModel?.Dispose();
        Model.Dispose();
    }
}

/// <summary>
/// ALBERT-style text encoder: one shared transformer layer's weights, applied
/// <see cref="KokoroWeights.BertNumHiddenLayers"/> times (parameter sharing, per
/// the single un-indexed bert.attn_*/bert.ffn_* tensor set in the GGUF file).
/// </summary>
public sealed class BertWeights
{
    public float[] EmbdTokWeight { get; }   // [128, 178] -> vocab embedding, embedding_size cols
    public float[] EmbdPosWeight { get; }   // [128, 512] -> position embedding
    public float[] EmbdTtWeight { get; }    // [128, 2]   -> token-type embedding
    public float[] EmbdLnWeight { get; }
    public float[] EmbdLnBias { get; }
    public float[] EmbdProjWeight { get; }  // [128, 768] -> projects embedding_size up to hidden_size
    public float[] EmbdProjBias { get; }

    public float[] AttnQWeight { get; }
    public float[] AttnQBias { get; }
    public float[] AttnKWeight { get; }
    public float[] AttnKBias { get; }
    public float[] AttnVWeight { get; }
    public float[] AttnVBias { get; }
    public float[] AttnOWeight { get; }
    public float[] AttnOBias { get; }
    public float[] AttnLnWeight { get; }
    public float[] AttnLnBias { get; }

    public float[] FfnUpWeight { get; }
    public float[] FfnUpBias { get; }
    public float[] FfnDownWeight { get; }
    public float[] FfnDownBias { get; }
    public float[] FfnLnWeight { get; }
    public float[] FfnLnBias { get; }

    public float[] PoolerWeight { get; }
    public float[] PoolerBias { get; }

    public BertWeights(KokoroWeights w)
    {
        EmbdTokWeight = w.GetTensor("bert.embd.tok.weight");
        EmbdPosWeight = w.GetTensor("bert.embd.pos.weight");
        EmbdTtWeight = w.GetTensor("bert.embd.tt.weight");
        EmbdLnWeight = w.GetTensor("bert.embd.ln.weight");
        EmbdLnBias = w.GetTensor("bert.embd.ln.bias");
        EmbdProjWeight = w.GetTensor("bert.embd_proj.weight");
        EmbdProjBias = w.GetTensor("bert.embd_proj.bias");

        AttnQWeight = w.GetTensor("bert.attn_q.weight");
        AttnQBias = w.GetTensor("bert.attn_q.bias");
        AttnKWeight = w.GetTensor("bert.attn_k.weight");
        AttnKBias = w.GetTensor("bert.attn_k.bias");
        AttnVWeight = w.GetTensor("bert.attn_v.weight");
        AttnVBias = w.GetTensor("bert.attn_v.bias");
        AttnOWeight = w.GetTensor("bert.attn_o.weight");
        AttnOBias = w.GetTensor("bert.attn_o.bias");
        AttnLnWeight = w.GetTensor("bert.attn_ln.weight");
        AttnLnBias = w.GetTensor("bert.attn_ln.bias");

        FfnUpWeight = w.GetTensor("bert.ffn_up.weight");
        FfnUpBias = w.GetTensor("bert.ffn_up.bias");
        FfnDownWeight = w.GetTensor("bert.ffn_down.weight");
        FfnDownBias = w.GetTensor("bert.ffn_down.bias");
        FfnLnWeight = w.GetTensor("bert.ffn_ln.weight");
        FfnLnBias = w.GetTensor("bert.ffn_ln.bias");

        PoolerWeight = w.GetTensor("bert.pooler.weight");
        PoolerBias = w.GetTensor("bert.pooler.bias");
    }
}

/// <summary>
/// CNN + BiLSTM text encoder (StyleTTS2 TextEncoder, examples/kokoro-py/modules.py TextEncoder),
/// token embedding -> 3x(Conv1D+LayerNorm+LeakyReLU) -> BiLSTM. GGUF prefix is "text_enc" (NOT
/// "text_encoder" -- verified against models/kokoro-82m-q8_0.gguf's actual tensor table); LayerNorm
/// params are named gamma/beta (matching modules.py's custom LayerNorm), not weight/bias.
/// </summary>
public sealed class TextEncoderWeights
{
    public float[] EmbeddingWeight { get; } // text_enc.embd.weight [hidden_dim, n_token] (nn.Embedding stores [vocab, dim]; GGUF file order confirmed [512,178])

    // 3 Conv1D+LayerNorm blocks, each: conv.weight [kernel, hidden, hidden], conv.bias [hidden]
    public float[][] ConvWeight { get; }
    public float[][] ConvBias { get; }
    public float[][] ConvLnGamma { get; }
    public float[][] ConvLnBeta { get; }

    public float[] LstmWeightIhL0 { get; }
    public float[] LstmWeightHhL0 { get; }
    public float[] LstmBiasIhL0 { get; }
    public float[] LstmBiasHhL0 { get; }
    public float[] LstmWeightIhL0Rev { get; }
    public float[] LstmWeightHhL0Rev { get; }
    public float[] LstmBiasIhL0Rev { get; }
    public float[] LstmBiasHhL0Rev { get; }

    public TextEncoderWeights(KokoroWeights w)
    {
        EmbeddingWeight = w.GetTensor("text_enc.embd.weight");

        ConvWeight = new float[3][];
        ConvBias = new float[3][];
        ConvLnGamma = new float[3][];
        ConvLnBeta = new float[3][];
        for (int i = 0; i < 3; i++)
        {
            ConvWeight[i] = w.GetTensor($"text_enc.cnn.{i}.conv.weight");
            ConvBias[i] = w.GetTensor($"text_enc.cnn.{i}.conv.bias");
            ConvLnGamma[i] = w.GetTensor($"text_enc.cnn.{i}.ln.gamma");
            ConvLnBeta[i] = w.GetTensor($"text_enc.cnn.{i}.ln.beta");
        }

        LstmWeightIhL0 = w.GetTensor("text_enc.lstm.weight_ih_l0");
        LstmWeightHhL0 = w.GetTensor("text_enc.lstm.weight_hh_l0");
        LstmBiasIhL0 = w.GetTensor("text_enc.lstm.bias_ih_l0");
        LstmBiasHhL0 = w.GetTensor("text_enc.lstm.bias_hh_l0");
        LstmWeightIhL0Rev = w.GetTensor("text_enc.lstm.weight_ih_l0_reverse");
        LstmWeightHhL0Rev = w.GetTensor("text_enc.lstm.weight_hh_l0_reverse");
        LstmBiasIhL0Rev = w.GetTensor("text_enc.lstm.bias_ih_l0_reverse");
        LstmBiasHhL0Rev = w.GetTensor("text_enc.lstm.bias_hh_l0_reverse");
    }
}

public sealed class AdainResBlk1dWeights
{
    public required float[] Conv1Weight;
    public required float[] Conv1Bias;
    public required float[] Conv2Weight;
    public required float[] Conv2Bias;
    public required float[] Adain1Weight;
    public required float[] Adain1Bias;
    public required float[] Adain2Weight;
    public required float[] Adain2Bias;
    public float[]? Conv1x1Weight;   // present only on upsampling blocks (block index 1)
    public float[]? PoolWeight;      // present only on upsampling blocks (block index 1)
    public float[]? PoolBias;
}

public sealed class BiLstmWeights
{
    public required float[] WeightIhL0;
    public required float[] WeightHhL0;
    public required float[] BiasIhL0;
    public required float[] BiasHhL0;
    public required float[] WeightIhL0Rev;
    public required float[] WeightHhL0Rev;
    public required float[] BiasIhL0Rev;
    public required float[] BiasHhL0Rev;
}

public sealed class ProsodyPredictorWeights
{
    // DurationEncoder: NStyleLayers stacked (AdaLayerNorm + BiLSTM) blocks over [text;style] frames.
    public BiLstmWeights[] DurEncLstm { get; }
    public float[][] DurEncAdaLnWeight { get; }
    public float[][] DurEncAdaLnBias { get; }

    public BiLstmWeights SharedLstm { get; }   // pred.lstm -- BiLSTM over duration-encoded features, feeds dur_proj
    public float[] DurProjWeight { get; }      // pred.dur_proj.weight [512, max_dur]
    public float[] DurProjBias { get; }

    public BiLstmWeights SharedFeatureLstm { get; } // pred.shared -- BiLSTM over length-regulated features, feeds F0/N stacks

    public AdainResBlk1dWeights[] F0 { get; }
    public float[] F0ProjWeight { get; }
    public float[] F0ProjBias { get; }

    public AdainResBlk1dWeights[] N { get; }
    public float[] NProjWeight { get; }
    public float[] NProjBias { get; }

    public ProsodyPredictorWeights(KokoroWeights w)
    {
        int n = w.NStyleLayers;
        DurEncLstm = new BiLstmWeights[n];
        DurEncAdaLnWeight = new float[n][];
        DurEncAdaLnBias = new float[n][];
        for (int i = 0; i < n; i++)
        {
            DurEncLstm[i] = new BiLstmWeights
            {
                WeightIhL0 = w.GetTensor($"pred.dur_enc.{i}.lstm.weight_ih_l0"),
                WeightHhL0 = w.GetTensor($"pred.dur_enc.{i}.lstm.weight_hh_l0"),
                BiasIhL0 = w.GetTensor($"pred.dur_enc.{i}.lstm.bias_ih_l0"),
                BiasHhL0 = w.GetTensor($"pred.dur_enc.{i}.lstm.bias_hh_l0"),
                WeightIhL0Rev = w.GetTensor($"pred.dur_enc.{i}.lstm.weight_ih_l0_reverse"),
                WeightHhL0Rev = w.GetTensor($"pred.dur_enc.{i}.lstm.weight_hh_l0_reverse"),
                BiasIhL0Rev = w.GetTensor($"pred.dur_enc.{i}.lstm.bias_ih_l0_reverse"),
                BiasHhL0Rev = w.GetTensor($"pred.dur_enc.{i}.lstm.bias_hh_l0_reverse"),
            };
            DurEncAdaLnWeight[i] = w.GetTensor($"pred.dur_enc.{i}.adaln.weight");
            DurEncAdaLnBias[i] = w.GetTensor($"pred.dur_enc.{i}.adaln.bias");
        }

        SharedLstm = new BiLstmWeights
        {
            WeightIhL0 = w.GetTensor("pred.lstm.weight_ih_l0"),
            WeightHhL0 = w.GetTensor("pred.lstm.weight_hh_l0"),
            BiasIhL0 = w.GetTensor("pred.lstm.bias_ih_l0"),
            BiasHhL0 = w.GetTensor("pred.lstm.bias_hh_l0"),
            WeightIhL0Rev = w.GetTensor("pred.lstm.weight_ih_l0_reverse"),
            WeightHhL0Rev = w.GetTensor("pred.lstm.weight_hh_l0_reverse"),
            BiasIhL0Rev = w.GetTensor("pred.lstm.bias_ih_l0_reverse"),
            BiasHhL0Rev = w.GetTensor("pred.lstm.bias_hh_l0_reverse"),
        };
        DurProjWeight = w.GetTensor("pred.dur_proj.weight");
        DurProjBias = w.GetTensor("pred.dur_proj.bias");

        SharedFeatureLstm = new BiLstmWeights
        {
            WeightIhL0 = w.GetTensor("pred.shared.weight_ih_l0"),
            WeightHhL0 = w.GetTensor("pred.shared.weight_hh_l0"),
            BiasIhL0 = w.GetTensor("pred.shared.bias_ih_l0"),
            BiasHhL0 = w.GetTensor("pred.shared.bias_hh_l0"),
            WeightIhL0Rev = w.GetTensor("pred.shared.weight_ih_l0_reverse"),
            WeightHhL0Rev = w.GetTensor("pred.shared.weight_hh_l0_reverse"),
            BiasIhL0Rev = w.GetTensor("pred.shared.bias_ih_l0_reverse"),
            BiasHhL0Rev = w.GetTensor("pred.shared.bias_hh_l0_reverse"),
        };

        F0 = LoadAdainStack(w, "pred.F0", n);
        F0ProjWeight = w.GetTensor("pred.F0_proj.weight");
        F0ProjBias = w.GetTensor("pred.F0_proj.bias");

        N = LoadAdainStack(w, "pred.N", n);
        NProjWeight = w.GetTensor("pred.N_proj.weight");
        NProjBias = w.GetTensor("pred.N_proj.bias");
    }

    internal static AdainResBlk1dWeights[] LoadAdainStack(KokoroWeights w, string prefix, int n)
    {
        var blocks = new AdainResBlk1dWeights[n];
        for (int i = 0; i < n; i++)
        {
            string p = $"{prefix}.{i}";
            blocks[i] = new AdainResBlk1dWeights
            {
                Conv1Weight = w.GetTensor($"{p}.conv1.weight"),
                Conv1Bias = w.GetTensor($"{p}.conv1.bias"),
                Conv2Weight = w.GetTensor($"{p}.conv2.weight"),
                Conv2Bias = w.GetTensor($"{p}.conv2.bias"),
                Adain1Weight = w.GetTensor($"{p}.adain1.weight"),
                Adain1Bias = w.GetTensor($"{p}.adain1.bias"),
                Adain2Weight = w.GetTensor($"{p}.adain2.weight"),
                Adain2Bias = w.GetTensor($"{p}.adain2.bias"),
                Conv1x1Weight = w.TryGetTensor($"{p}.conv1x1.weight"),
                PoolWeight = w.TryGetTensor($"{p}.pool.weight"),
                PoolBias = w.TryGetTensor($"{p}.pool.bias"),
            };
        }
        return blocks;
    }
}

/// <summary>
/// StyleTTS2 Decoder (examples/kokoro-py/istftnet.py Decoder). Real GGUF prefix is "dec"; the decode
/// stack is a FIXED 4 AdainResBlk1d blocks regardless of kokoro.n_layer (that KV key governs the
/// ProsodyPredictor's DurationEncoder depth only, per modules.py/istftnet.py -- verified against
/// the file's actual dec.decode.{0,1,2,3}.* tensors, where only block 3 has pool.*, matching
/// Decoder.decode's 4th block being the only one constructed with upsample=True).
/// </summary>
public sealed class DecoderWeights
{
    public const int DecodeStackSize = 4;

    public float[] F0ConvWeight { get; }  // dec.F0_conv -- stride-2 conv halving the F0 curve's frame rate to match asr's rate
    public float[] F0ConvBias { get; }
    public float[] NConvWeight { get; }   // dec.N_conv -- same for the noise/energy curve
    public float[] NConvBias { get; }
    public float[] AsrResWeight { get; }  // dec.asr_res -- 1x1 conv residual projection of encoder features (512 -> 64)
    public float[] AsrResBias { get; }

    public AdainResBlk1dWeights Encode { get; }        // dec.encode (single block, no upsampling)
    public AdainResBlk1dWeights[] Decode { get; }       // dec.decode.{0,1,2,3} -- fixed size, last block upsamples

    public GeneratorWeights Generator { get; }

    public DecoderWeights(KokoroWeights w)
    {
        F0ConvWeight = w.GetTensor("dec.F0_conv.weight");
        F0ConvBias = w.GetTensor("dec.F0_conv.bias");
        NConvWeight = w.GetTensor("dec.N_conv.weight");
        NConvBias = w.GetTensor("dec.N_conv.bias");
        AsrResWeight = w.GetTensor("dec.asr_res.weight");
        AsrResBias = w.GetTensor("dec.asr_res.bias");

        Encode = new AdainResBlk1dWeights
        {
            Conv1Weight = w.GetTensor("dec.encode.conv1.weight"),
            Conv1Bias = w.GetTensor("dec.encode.conv1.bias"),
            Conv2Weight = w.GetTensor("dec.encode.conv2.weight"),
            Conv2Bias = w.GetTensor("dec.encode.conv2.bias"),
            Adain1Weight = w.GetTensor("dec.encode.adain1.weight"),
            Adain1Bias = w.GetTensor("dec.encode.adain1.bias"),
            Adain2Weight = w.GetTensor("dec.encode.adain2.weight"),
            Adain2Bias = w.GetTensor("dec.encode.adain2.bias"),
            Conv1x1Weight = w.TryGetTensor("dec.encode.conv1x1.weight"),
            PoolWeight = w.TryGetTensor("dec.encode.pool.weight"),
            PoolBias = w.TryGetTensor("dec.encode.pool.bias"),
        };

        Decode = ProsodyPredictorWeights.LoadAdainStack(w, "dec.decode", DecodeStackSize);

        Generator = new GeneratorWeights(w);
    }
}

/// <summary>
/// iSTFTNet + NSF (Neural Source Filter) generator (istftnet.py Generator): a learned harmonic
/// source (SourceModuleHnNSF: sine generator -> linear merge -> tanh) is injected via per-stage
/// noise_convs/noise_res alongside AdaIN+Snake-activated resblocks (istftnet.py AdainResBlock1),
/// ending in a conv_post that emits log-magnitude/phase for a learned inverse-STFT head. Real GGUF
/// prefix is "dec.gen" (NOT "dec.generator").
/// </summary>
public sealed class GeneratorWeights
{
    // NSF harmonic source module (istftnet.py SourceModuleHnNSF): merges 9 harmonics (harmonic_num=8) into one channel.
    public float[] MSourceWeight { get; }  // dec.gen.m_source.weight [9] -- nn.Linear(9,1) weight, flattened
    public float[] MSourceBias { get; }    // dec.gen.m_source.bias [1]

    // One ConvTranspose1d per upsample stage (len(UpsampleRates) == 2)
    public float[][] UpWeight { get; }
    public float[][] UpBias { get; }

    // Per-upsample-stage noise injection: conv projecting the harmonic mag/phase spectrum into
    // the current channel width, then an AdainResBlock1 refining it before it's added to `x`.
    public float[][] NoiseConvWeight { get; }
    public float[][] NoiseConvBias { get; }
    public ResBlockWeights[] NoiseRes { get; }

    // len(UpsampleRates) * len(ResblockKernelSizes) == 6 AdaIN+Snake HiFi-GAN MRF resblocks.
    public ResBlockWeights[] ResBlocks { get; }

    public float[] ConvPostWeight { get; }
    public float[] ConvPostBias { get; }

    public GeneratorWeights(KokoroWeights w)
    {
        MSourceWeight = w.GetTensor("dec.gen.m_source.weight");
        MSourceBias = w.GetTensor("dec.gen.m_source.bias");

        int numUp = w.UpsampleRates.Length;
        UpWeight = new float[numUp][];
        UpBias = new float[numUp][];
        NoiseConvWeight = new float[numUp][];
        NoiseConvBias = new float[numUp][];
        NoiseRes = new ResBlockWeights[numUp];
        for (int i = 0; i < numUp; i++)
        {
            UpWeight[i] = w.GetTensor($"dec.gen.ups.{i}.weight");
            UpBias[i] = w.GetTensor($"dec.gen.ups.{i}.bias");
            NoiseConvWeight[i] = w.GetTensor($"dec.gen.noise_convs.{i}.weight");
            NoiseConvBias[i] = w.GetTensor($"dec.gen.noise_convs.{i}.bias");
            NoiseRes[i] = new ResBlockWeights(w, $"dec.gen.noise_res.{i}");
        }

        int numKernels = w.ResblockKernelSizes.Length;
        int numResBlocks = numUp * numKernels;
        ResBlocks = new ResBlockWeights[numResBlocks];
        for (int i = 0; i < numResBlocks; i++)
        {
            ResBlocks[i] = new ResBlockWeights(w, $"dec.gen.resblocks.{i}");
        }

        ConvPostWeight = w.GetTensor("dec.gen.conv_post.weight");
        ConvPostBias = w.GetTensor("dec.gen.conv_post.bias");
    }
}

/// <summary>istftnet.py AdainResBlock1: 3 dilated conv pairs, each preceded by AdaIN1d + Snake1D activation.</summary>
public sealed class ResBlockWeights
{
    public float[][] Convs1Weight { get; } // 3 dilated convs
    public float[][] Convs1Bias { get; }
    public float[][] Convs2Weight { get; } // 3 non-dilated convs
    public float[][] Convs2Bias { get; }
    public float[][] Adain1Weight { get; } // per-conv AdaIN modulation (3 each): nn.Linear(style_dim, channels*2)
    public float[][] Adain1Bias { get; }
    public float[][] Adain2Weight { get; }
    public float[][] Adain2Bias { get; }
    public float[][] Alpha1 { get; }       // Snake1D per-channel frequency param before conv1, shape [1, channels, 1]
    public float[][] Alpha2 { get; }       // Snake1D per-channel frequency param before conv2

    public ResBlockWeights(KokoroWeights w, string prefix)
    {
        Convs1Weight = new float[3][];
        Convs1Bias = new float[3][];
        Convs2Weight = new float[3][];
        Convs2Bias = new float[3][];
        Adain1Weight = new float[3][];
        Adain1Bias = new float[3][];
        Adain2Weight = new float[3][];
        Adain2Bias = new float[3][];
        Alpha1 = new float[3][];
        Alpha2 = new float[3][];
        for (int i = 0; i < 3; i++)
        {
            Convs1Weight[i] = w.GetTensor($"{prefix}.convs1.{i}.weight");
            Convs1Bias[i] = w.GetTensor($"{prefix}.convs1.{i}.bias");
            Convs2Weight[i] = w.GetTensor($"{prefix}.convs2.{i}.weight");
            Convs2Bias[i] = w.GetTensor($"{prefix}.convs2.{i}.bias");
            Adain1Weight[i] = w.GetTensor($"{prefix}.adain1.{i}.weight");
            Adain1Bias[i] = w.GetTensor($"{prefix}.adain1.{i}.bias");
            Adain2Weight[i] = w.GetTensor($"{prefix}.adain2.{i}.weight");
            Adain2Bias[i] = w.GetTensor($"{prefix}.adain2.{i}.bias");
            Alpha1[i] = w.GetTensor($"{prefix}.alpha1.{i}");
            Alpha2[i] = w.GetTensor($"{prefix}.alpha2.{i}");
        }
    }
}
