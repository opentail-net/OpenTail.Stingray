using System;
using System.IO;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Audio.QwenASR;

/// <summary>
/// Container for Alibaba Qwen3-ASR (AuT audio encoder + Qwen3 LLM text decoder) GGUF weights.
/// </summary>
public sealed class QwenAsrWeights : IDisposable
{
    public GgufModel Model { get; }

    public int AudioLayers { get; } = 18;
    public int AudioDim { get; } = 896;
    public int AudioHeads { get; } = 14;
    public int AudioHeadDim { get; } = 64;
    public int AudioFfDim { get; } = 3584;
    public int AudioConvChannels { get; } = 480;
    public int AudioProjDim { get; } = 1024;
    public int AudioMaxSourcePositions { get; } = 1500;
    public int LlmLayers { get; } = 28;
    public int LlmDim { get; } = 1024;
    public int LlmHeads { get; } = 16;
    public int LlmKvHeads { get; } = 8;
    public int LlmHeadDim { get; } = 128;
    public int LlmFfDim { get; } = 3072;
    public int LlmVocabSize { get; } = 151936;
    public float LlmRmsNormEps { get; } = 1e-6f;
    public float LlmRopeTheta { get; } = 1_000_000f;

    public int NMels { get; } = 128;
    public int NFft { get; } = 400;
    public int WinLength { get; } = 400;
    public int HopLength { get; } = 160;
    public int SampleRate { get; } = 16000;

    // Real special-token ids, read from the checkpoint's own metadata (verified directly
    // against models/qwen3-asr-0.6b-q4_k.gguf's tokenizer.ggml.tokens array -- see
    // docs/audio-review-progress.md's QwenASR section: the previous hardcoded ids/strings in
    // QwenAsrTokenizer.cs were entirely fictional, including an assumed dedicated timestamp-
    // token range that does not exist anywhere in this vocabulary).
    public int AudioStartTokenId { get; } = 151669; // "<|audio_start|>"
    public int AudioEndTokenId { get; } = 151670;   // "<|audio_end|>"
    public int AudioPadTokenId { get; } = 151676;   // "<|audio_pad|>"
    public int EosTokenId { get; } = 151645;        // "<|im_end|>"
    public int PadTokenId { get; } = 151643;        // "<|endoftext|>"

    public GgufTokenizer Tokenizer { get; }

    public QwenAsrWeights(string ggufPath)
    {
        if (!File.Exists(ggufPath))
            throw new FileNotFoundException($"Qwen3-ASR model file not found: {ggufPath}");

        Model = GgufModel.Open(ggufPath);

        if (Model.Metadata.TryGetValue("qwen3asr.audio.n_layers", out var anl) && anl is int anli)
            AudioLayers = anli;
        if (Model.Metadata.TryGetValue("qwen3asr.audio.d_model", out var adm) && adm is int admi)
            AudioDim = admi;
        if (Model.Metadata.TryGetValue("qwen3asr.audio.n_heads", out var anh) && anh is int anhi)
            AudioHeads = anhi;
        if (Model.Metadata.TryGetValue("qwen3asr.audio.head_dim", out var ahd) && ahd is int ahdi)
            AudioHeadDim = ahdi;
        if (Model.Metadata.TryGetValue("qwen3asr.audio.ff_dim", out var aff) && aff is int affi)
            AudioFfDim = affi;
        if (Model.Metadata.TryGetValue("qwen3asr.audio.conv_channels", out var acc) && acc is int acci)
            AudioConvChannels = acci;
        if (Model.Metadata.TryGetValue("qwen3asr.audio.proj_dim", out var apd) && apd is int apdi)
            AudioProjDim = apdi;
        if (Model.Metadata.TryGetValue("qwen3asr.audio.max_source_pos", out var amsp) && amsp is int amspi)
            AudioMaxSourcePositions = amspi;

        if (Model.Metadata.TryGetValue("qwen3asr.llm.n_layers", out var lnl) && lnl is int lnli)
            LlmLayers = lnli;
        if (Model.Metadata.TryGetValue("qwen3asr.llm.d_model", out var ldm) && ldm is int ldmi)
            LlmDim = ldmi;
        if (Model.Metadata.TryGetValue("qwen3asr.llm.n_heads", out var lnh) && lnh is int lnhi)
            LlmHeads = lnhi;
        if (Model.Metadata.TryGetValue("qwen3asr.llm.n_kv_heads", out var lnkv) && lnkv is int lnkvi)
            LlmKvHeads = lnkvi;
        if (Model.Metadata.TryGetValue("qwen3asr.llm.head_dim", out var lhd) && lhd is int lhdi)
            LlmHeadDim = lhdi;
        if (Model.Metadata.TryGetValue("qwen3asr.llm.ff_dim", out var lff) && lff is int lffi)
            LlmFfDim = lffi;
        if (Model.Metadata.TryGetValue("qwen3asr.llm.vocab_size", out var lvs) && lvs is int lvsi)
            LlmVocabSize = lvsi;
        if (Model.Metadata.TryGetValue("qwen3asr.llm.rms_norm_eps", out var lre) && lre is float lrei)
            LlmRmsNormEps = lrei;
        if (Model.Metadata.TryGetValue("qwen3asr.llm.rope_theta", out var lrt) && lrt is float lrti)
            LlmRopeTheta = lrti;

        if (Model.Metadata.TryGetValue("qwen3asr.n_mels", out var nm) && nm is int nmi)
            NMels = nmi;
        if (Model.Metadata.TryGetValue("qwen3asr.n_fft", out var nf) && nf is int nfi)
            NFft = nfi;
        if (Model.Metadata.TryGetValue("qwen3asr.win_length", out var wl) && wl is int wli)
            WinLength = wli;
        if (Model.Metadata.TryGetValue("qwen3asr.hop_length", out var hl) && hl is int hli)
            HopLength = hli;
        if (Model.Metadata.TryGetValue("qwen3asr.sample_rate", out var sr) && sr is int sri)
            SampleRate = sri;

        if (Model.Metadata.TryGetValue("qwen3asr.audio_start_token_id", out var ast) && ast is int asti)
            AudioStartTokenId = asti;
        if (Model.Metadata.TryGetValue("qwen3asr.audio_end_token_id", out var aet) && aet is int aeti)
            AudioEndTokenId = aeti;
        if (Model.Metadata.TryGetValue("qwen3asr.audio_pad_token_id", out var apt) && apt is int apti)
            AudioPadTokenId = apti;
        if (Model.Metadata.TryGetValue("qwen3asr.eos_token_id", out var eos) && eos is int eosi)
            EosTokenId = eosi;
        if (Model.Metadata.TryGetValue("qwen3asr.pad_token_id", out var pad) && pad is int padi)
            PadTokenId = padi;

        Tokenizer = BuildTokenizer(Model, AudioStartTokenId, AudioEndTokenId, AudioPadTokenId, EosTokenId, PadTokenId);

        Conv1Weight = GetTensor("audio.conv.1.weight");
        Conv1Bias = GetTensor("audio.conv.1.bias");
        Conv2Weight = GetTensor("audio.conv.2.weight");
        Conv2Bias = GetTensor("audio.conv.2.bias");
        Conv3Weight = GetTensor("audio.conv.3.weight");
        Conv3Bias = GetTensor("audio.conv.3.bias");
        ConvOutWeight = GetTensor("audio.conv_out.weight"); // no bias tensor for this one
        LnPostWeight = GetTensor("audio.ln_post.weight");
        LnPostBias = GetTensor("audio.ln_post.bias");
        MelFilters = GetTensor("audio.mel_filters"); // [n_mels, n_freqs]
        MelWindow = GetTensor("audio.mel_window");
        Proj1Weight = GetTensor("audio.proj1.weight");
        Proj1Bias = GetTensor("audio.proj1.bias");
        Proj2Weight = GetTensor("audio.proj2.weight");
        Proj2Bias = GetTensor("audio.proj2.bias");

        AudioLayerWeights = new QwenAsrAudioLayerWeights[AudioLayers];
        for (int i = 0; i < AudioLayers; i++)
            AudioLayerWeights[i] = new QwenAsrAudioLayerWeights(this, $"audio.blk.{i}");
    }

    // --- AuT audio encoder weights (Whisper-style: conv2d stem + absolute-pos transformer) ---
    public float[] Conv1Weight { get; }
    public float[] Conv1Bias { get; }
    public float[] Conv2Weight { get; }
    public float[] Conv2Bias { get; }
    public float[] Conv3Weight { get; }
    public float[] Conv3Bias { get; }
    public float[] ConvOutWeight { get; } // [7680, 896], no bias
    public float[] LnPostWeight { get; }
    public float[] LnPostBias { get; }
    public float[] MelFilters { get; }
    public float[] MelWindow { get; }
    public float[] Proj1Weight { get; }
    public float[] Proj1Bias { get; }
    public float[] Proj2Weight { get; }
    public float[] Proj2Bias { get; }
    public QwenAsrAudioLayerWeights[] AudioLayerWeights { get; }

    /// <summary>Loads and dequantizes a required tensor by exact GGUF name to a flat float[] in file storage order.</summary>
    public float[] GetTensor(string name)
    {
        var info = Model.FindTensor(name) ?? throw new InvalidDataException($"Qwen3-ASR GGUF missing required tensor '{name}'.");
        var bytes = Model.GetTensorData(info);
        var dst = new float[info.ElementCount];
        Dequantize.ToFloat32(bytes, dst, info.DType, info.ElementCount);
        return dst;
    }

    /// <summary>
    /// <see cref="GgufTokenizer.FromGgufModel"/> only recognizes special tokens via a
    /// <c>tokenizer.ggml.token_type</c> array, which this checkpoint doesn't ship -- without
    /// it, the audio-start/end/pad and chat-marker special tokens would get silently BPE-
    /// merged character-by-character instead of recognized as single tokens. Builds the
    /// <see cref="TokenizerSource"/> by hand instead (mirroring
    /// <see cref="GgufTokenizer.FromGgufModel"/>'s own logic) so
    /// <see cref="TokenizerSource.AdditionalSpecialTokens"/> can be supplied explicitly from
    /// this checkpoint's real ids.
    /// </summary>
    private static GgufTokenizer BuildTokenizer(GgufModel model, int audioStart, int audioEnd, int audioPad, int eos, int pad)
    {
        var tokensArray = (object[])model.Metadata["tokenizer.ggml.tokens"];
        var tokens = new string[tokensArray.Length];
        for (int i = 0; i < tokensArray.Length; i++) tokens[i] = (string)tokensArray[i];

        var mergesArray = model.Metadata.TryGetValue("tokenizer.ggml.merges", out var mergesObj) ? (object[])mergesObj : [];
        var merges = new string[mergesArray.Length];
        for (int i = 0; i < mergesArray.Length; i++) merges[i] = (string)mergesArray[i];

        var additionalSpecial = new Dictionary<string, int>(StringComparer.Ordinal)
        {
            ["<|im_start|>"] = 151644,
            ["<|im_end|>"] = eos,
            [tokens[audioStart]] = audioStart,
            [tokens[audioEnd]] = audioEnd,
            [tokens[audioPad]] = audioPad,
            [tokens[pad]] = pad,
        };

        var source = new TokenizerSource
        {
            Tokens = tokens,
            Merges = merges,
            AdditionalSpecialTokens = additionalSpecial,
            BosTokenId = pad,
            EosTokenId = eos,
            UnknownTokenId = pad,
            PadTokenId = pad,
            AddBosToken = false,
            ModelFamily = model.Metadata.TryGetValue("tokenizer.ggml.model", out var tm) ? (string)tm : "gpt2",
        };
        return GgufTokenizer.FromSource(source);
    }

    public void Dispose()
    {
        Model.Dispose();
    }
}

/// <summary>
/// One AuT (Whisper-style) audio-encoder block: pre-LN self-attention (with bias on q/k/v/out,
/// unlike Whisper's own encoder which has no key bias) + pre-LN 2-layer GELU FFN, both plain
/// LayerNorm (with bias, not RMSNorm) -- no rel-pos bias tensors, no conv module, no macaron
/// step; this is architecturally simpler than Parakeet's FastConformer and close to
/// `Whisper/WhisperEncoderLayerWeights`, see `Whisper/WhisperEncoder.cs` for the block math
/// this was modeled on.
/// </summary>
public sealed class QwenAsrAudioLayerWeights
{
    public float[] AttnNormWeight { get; }
    public float[] AttnNormBias { get; }
    public float[] AttnQWeight { get; }
    public float[] AttnQBias { get; }
    public float[] AttnKWeight { get; }
    public float[] AttnKBias { get; }
    public float[] AttnVWeight { get; }
    public float[] AttnVBias { get; }
    public float[] AttnOutWeight { get; }
    public float[] AttnOutBias { get; }
    public float[] FfnNormWeight { get; }
    public float[] FfnNormBias { get; }
    public float[] FfnUpWeight { get; }
    public float[] FfnUpBias { get; }
    public float[] FfnDownWeight { get; }
    public float[] FfnDownBias { get; }

    public QwenAsrAudioLayerWeights(QwenAsrWeights w, string prefix)
    {
        AttnNormWeight = w.GetTensor($"{prefix}.attn_norm.weight");
        AttnNormBias = w.GetTensor($"{prefix}.attn_norm.bias");
        AttnQWeight = w.GetTensor($"{prefix}.attn_q.weight");
        AttnQBias = w.GetTensor($"{prefix}.attn_q.bias");
        AttnKWeight = w.GetTensor($"{prefix}.attn_k.weight");
        AttnKBias = w.GetTensor($"{prefix}.attn_k.bias");
        AttnVWeight = w.GetTensor($"{prefix}.attn_v.weight");
        AttnVBias = w.GetTensor($"{prefix}.attn_v.bias");
        AttnOutWeight = w.GetTensor($"{prefix}.attn_out.weight");
        AttnOutBias = w.GetTensor($"{prefix}.attn_out.bias");
        FfnNormWeight = w.GetTensor($"{prefix}.ffn_norm.weight");
        FfnNormBias = w.GetTensor($"{prefix}.ffn_norm.bias");
        FfnUpWeight = w.GetTensor($"{prefix}.ffn_up.weight");
        FfnUpBias = w.GetTensor($"{prefix}.ffn_up.bias");
        FfnDownWeight = w.GetTensor($"{prefix}.ffn_down.weight");
        FfnDownBias = w.GetTensor($"{prefix}.ffn_down.bias");
    }
}
