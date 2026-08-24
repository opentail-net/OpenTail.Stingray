using System;
using System.IO;
using System.Text.Json;
using OpenTail.Stingray.Audio.Primitives;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Audio.QwenASR;

/// <summary>
/// Container for Alibaba Qwen3-ASR (AuT audio encoder + Qwen3 LLM text decoder) weights --
/// either the original GGUF (`Model` populated) or the real, canonical Hugging Face
/// `Qwen/Qwen3-ASR-0.6B` Safetensors checkpoint (`Model` null, real
/// `thinker.audio_tower.*`/`thinker.model.*` tensors read through a real
/// <see cref="SafetensorsLoader"/> instead). Both paths populate the exact same public
/// properties, so <see cref="QwenAsrAudioEncoder"/>/<see cref="QwenAsrTokenizer"/> work
/// unchanged regardless of which loader was used -- same DRY pattern as
/// <see cref="Whisper.WhisperGgmlModel.LoadFromSafetensors"/>.
/// </summary>
public sealed class QwenAsrWeights : IDisposable
{
    /// <summary>Null when constructed via <see cref="LoadFromSafetensors"/> -- the Safetensors path never needed a `GgufModel` (the LLM decoder's real Safetensors generation loop uses <see cref="QwenAsrLlmSafetensorsTensorSource"/> directly, not `Model`).</summary>
    public GgufModel? Model { get; }
    private readonly SafetensorsLoader? _stLoader;
    private readonly System.Collections.Generic.Dictionary<string, string> _stRename = new(StringComparer.Ordinal);

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
        var convOutF32 = GetTensor("audio.conv_out.weight"); // no bias tensor for this one
        ConvOutWeight = CfmLinearWeight.FromF32(convOutF32, outDim: AudioDim, inDim: convOutF32.Length / AudioDim);
        LnPostWeight = GetTensor("audio.ln_post.weight");
        LnPostBias = GetTensor("audio.ln_post.bias");
        MelFilters = GetTensor("audio.mel_filters"); // [n_mels, n_freqs]
        MelWindow = GetTensor("audio.mel_window");
        var proj1F32 = GetTensor("audio.proj1.weight");
        Proj1Bias = GetTensor("audio.proj1.bias");
        Proj1Weight = CfmLinearWeight.FromF32(proj1F32, outDim: Proj1Bias.Length, inDim: proj1F32.Length / Proj1Bias.Length);
        var proj2F32 = GetTensor("audio.proj2.weight");
        Proj2Bias = GetTensor("audio.proj2.bias");
        Proj2Weight = CfmLinearWeight.FromF32(proj2F32, outDim: Proj2Bias.Length, inDim: proj2F32.Length / Proj2Bias.Length);

        AudioLayerWeights = new QwenAsrAudioLayerWeights[AudioLayers];
        for (int i = 0; i < AudioLayers; i++)
            AudioLayerWeights[i] = new QwenAsrAudioLayerWeights(this, $"audio.blk.{i}");
    }

    /// <summary>
    /// Real Safetensors constructor (used only by <see cref="LoadFromSafetensors"/>): populates
    /// the exact same properties as the GGUF constructor, reading through
    /// <see cref="SafetensorsLoader"/> instead, with a real name-remap table
    /// (`audio.*` canonical -&gt; real `thinker.audio_tower.*` HF names) built up front so every
    /// existing `GetTensor(name)` call site below works completely unchanged.
    /// </summary>
    private QwenAsrWeights(string checkpointDir, JsonElement audioConfig, JsonElement textConfig, JsonElement rootConfig)
    {
        _stLoader = SafetensorsLoader.Open(Path.Combine(checkpointDir, "model.safetensors"));

        AudioLayers = audioConfig.GetProperty("encoder_layers").GetInt32();
        AudioDim = audioConfig.GetProperty("d_model").GetInt32();
        AudioHeads = audioConfig.GetProperty("encoder_attention_heads").GetInt32();
        AudioHeadDim = AudioDim / AudioHeads;
        AudioFfDim = audioConfig.GetProperty("encoder_ffn_dim").GetInt32();
        AudioConvChannels = audioConfig.GetProperty("downsample_hidden_size").GetInt32();
        AudioProjDim = audioConfig.GetProperty("output_dim").GetInt32();
        AudioMaxSourcePositions = audioConfig.GetProperty("max_source_positions").GetInt32();

        LlmLayers = textConfig.GetProperty("num_hidden_layers").GetInt32();
        LlmDim = textConfig.GetProperty("hidden_size").GetInt32();
        LlmHeads = textConfig.GetProperty("num_attention_heads").GetInt32();
        LlmKvHeads = textConfig.GetProperty("num_key_value_heads").GetInt32();
        LlmHeadDim = textConfig.GetProperty("head_dim").GetInt32();
        LlmFfDim = textConfig.GetProperty("intermediate_size").GetInt32();
        LlmVocabSize = textConfig.GetProperty("vocab_size").GetInt32();
        LlmRmsNormEps = textConfig.GetProperty("rms_norm_eps").GetSingle();
        LlmRopeTheta = textConfig.GetProperty("rope_theta").GetSingle();

        NMels = AudioDim > 0 ? audioConfig.GetProperty("num_mel_bins").GetInt32() : NMels;

        // Real special ids confirmed identical to the GGUF checkpoint's own metadata (a real
        // cross-check, not assumed): audio_token_id (=audio_pad) matches AudioPadTokenId=151676
        // exactly.
        AudioStartTokenId = rootConfig.GetProperty("audio_start_token_id").GetInt32();
        AudioEndTokenId = rootConfig.GetProperty("audio_end_token_id").GetInt32();
        AudioPadTokenId = rootConfig.GetProperty("audio_token_id").GetInt32();

        Tokenizer = BuildTokenizerFromHfFiles(checkpointDir, AudioStartTokenId, AudioEndTokenId, AudioPadTokenId, EosTokenId, PadTokenId);

        // Real name remap: canonical `audio.*` (what GetTensor/AudioLayerWeights already call)
        // -> real HF `thinker.audio_tower.*` names, confirmed via a direct `safe_open` tensor
        // dump of the real checkpoint before writing this (not guessed from the module tree).
        _stRename["audio.conv.1.weight"] = "thinker.audio_tower.conv2d1.weight";
        _stRename["audio.conv.1.bias"] = "thinker.audio_tower.conv2d1.bias";
        _stRename["audio.conv.2.weight"] = "thinker.audio_tower.conv2d2.weight";
        _stRename["audio.conv.2.bias"] = "thinker.audio_tower.conv2d2.bias";
        _stRename["audio.conv.3.weight"] = "thinker.audio_tower.conv2d3.weight";
        _stRename["audio.conv.3.bias"] = "thinker.audio_tower.conv2d3.bias";
        _stRename["audio.conv_out.weight"] = "thinker.audio_tower.conv_out.weight";
        _stRename["audio.ln_post.weight"] = "thinker.audio_tower.ln_post.weight";
        _stRename["audio.ln_post.bias"] = "thinker.audio_tower.ln_post.bias";
        _stRename["audio.proj1.weight"] = "thinker.audio_tower.proj1.weight";
        _stRename["audio.proj1.bias"] = "thinker.audio_tower.proj1.bias";
        _stRename["audio.proj2.weight"] = "thinker.audio_tower.proj2.weight";
        _stRename["audio.proj2.bias"] = "thinker.audio_tower.proj2.bias";
        for (int i = 0; i < AudioLayers; i++)
        {
            string canon = $"audio.blk.{i}";
            string real = $"thinker.audio_tower.layers.{i}";
            _stRename[$"{canon}.attn_norm.weight"] = $"{real}.self_attn_layer_norm.weight";
            _stRename[$"{canon}.attn_norm.bias"] = $"{real}.self_attn_layer_norm.bias";
            _stRename[$"{canon}.attn_q.weight"] = $"{real}.self_attn.q_proj.weight";
            _stRename[$"{canon}.attn_q.bias"] = $"{real}.self_attn.q_proj.bias";
            _stRename[$"{canon}.attn_k.weight"] = $"{real}.self_attn.k_proj.weight";
            _stRename[$"{canon}.attn_k.bias"] = $"{real}.self_attn.k_proj.bias";
            _stRename[$"{canon}.attn_v.weight"] = $"{real}.self_attn.v_proj.weight";
            _stRename[$"{canon}.attn_v.bias"] = $"{real}.self_attn.v_proj.bias";
            _stRename[$"{canon}.attn_out.weight"] = $"{real}.self_attn.out_proj.weight";
            _stRename[$"{canon}.attn_out.bias"] = $"{real}.self_attn.out_proj.bias";
            _stRename[$"{canon}.ffn_norm.weight"] = $"{real}.final_layer_norm.weight";
            _stRename[$"{canon}.ffn_norm.bias"] = $"{real}.final_layer_norm.bias";
            _stRename[$"{canon}.ffn_up.weight"] = $"{real}.fc1.weight";
            _stRename[$"{canon}.ffn_up.bias"] = $"{real}.fc1.bias";
            _stRename[$"{canon}.ffn_down.weight"] = $"{real}.fc2.weight";
            _stRename[$"{canon}.ffn_down.bias"] = $"{real}.fc2.bias";
        }

        Conv1Weight = GetTensor("audio.conv.1.weight");
        Conv1Bias = GetTensor("audio.conv.1.bias");
        Conv2Weight = GetTensor("audio.conv.2.weight");
        Conv2Bias = GetTensor("audio.conv.2.bias");
        Conv3Weight = GetTensor("audio.conv.3.weight");
        Conv3Bias = GetTensor("audio.conv.3.bias");
        var convOutF32 = GetTensor("audio.conv_out.weight");
        ConvOutWeight = CfmLinearWeight.FromF32(convOutF32, outDim: AudioDim, inDim: convOutF32.Length / AudioDim);
        LnPostWeight = GetTensor("audio.ln_post.weight");
        LnPostBias = GetTensor("audio.ln_post.bias");
        MelFilters = []; // real HF checkpoint doesn't ship these -- confirmed unused anywhere in this codebase's own mel extraction (QwenAsrMelExtractor computes its own filterbank independently)
        MelWindow = [];
        var proj1F32 = GetTensor("audio.proj1.weight");
        Proj1Bias = GetTensor("audio.proj1.bias");
        Proj1Weight = CfmLinearWeight.FromF32(proj1F32, outDim: Proj1Bias.Length, inDim: proj1F32.Length / Proj1Bias.Length);
        var proj2F32 = GetTensor("audio.proj2.weight");
        Proj2Bias = GetTensor("audio.proj2.bias");
        Proj2Weight = CfmLinearWeight.FromF32(proj2F32, outDim: Proj2Bias.Length, inDim: proj2F32.Length / Proj2Bias.Length);

        AudioLayerWeights = new QwenAsrAudioLayerWeights[AudioLayers];
        for (int i = 0; i < AudioLayers; i++)
            AudioLayerWeights[i] = new QwenAsrAudioLayerWeights(this, $"audio.blk.{i}");
    }

    /// <summary>
    /// Loads a real Qwen3-ASR pipeline directly from the canonical Hugging Face
    /// `Qwen/Qwen3-ASR-0.6B` Safetensors checkpoint directory (`config.json`/`model.safetensors`/
    /// `vocab.json`/`merges.txt`).
    /// </summary>
    public static QwenAsrWeights LoadFromSafetensors(string checkpointDir)
    {
        if (!Directory.Exists(checkpointDir))
            throw new DirectoryNotFoundException($"Qwen3-ASR Safetensors checkpoint directory not found: {checkpointDir}");

        using var configDoc = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(checkpointDir, "config.json")));
        var root = configDoc.RootElement;
        var thinker = root.GetProperty("thinker_config");
        var audioConfig = thinker.GetProperty("audio_config");
        var textConfig = thinker.GetProperty("text_config");

        return new QwenAsrWeights(checkpointDir, audioConfig, textConfig, thinker);
    }

    /// <summary>
    /// Real HF tokenizer construction from `vocab.json`/`merges.txt` (byte-level BPE, GPT-2
    /// family -- same real convention as this codebase's other GGUF/HF-tokenizer loaders).
    /// Mirrors <see cref="BuildTokenizer"/>'s real special-token handling (audio start/end/pad
    /// ids must be explicit, or they'd get silently BPE-merged character-by-character).
    /// </summary>
    /// <summary>
    /// Real HF convention (confirmed directly against this checkpoint's own
    /// tokenizer_config.json, not assumed): `vocab.json` only holds the BASE byte-level BPE
    /// vocab (~151643 entries) -- every special/"added" token (audio_start/end/pad,
    /// im_start/im_end, etc., 62 real entries here) lives separately in
    /// `tokenizer_config.json`'s `added_tokens_decoder` (id -> {content, ...}). Treating
    /// vocab.json as the complete vocab left `tokens[audioPad]` etc. as an empty string,
    /// which made `AdditionalSpecialTokens` map the EMPTY string to a real id -- a real bug
    /// this pass found (via an OutOfMemoryException in the real end-to-end pipeline test,
    /// not by inspection) that made every encode call match a zero-length "special token"
    /// pathologically. Real fix, now shared with CosyVoice2's tokenizer builder (DRY pass) via
    /// <see cref="Primitives.HfBpeTokenizerLoader"/>: read `added_tokens_decoder` too and let it
    /// fill in exactly the ids `vocab.json` doesn't cover.
    /// </summary>
    private static GgufTokenizer BuildTokenizerFromHfFiles(string checkpointDir, int audioStart, int audioEnd, int audioPad, int eos, int pad)
    {
        var (loadedTokens, merges, _) = Primitives.HfBpeTokenizerLoader.Load(checkpointDir);
        var tokens = Primitives.HfBpeTokenizerLoader.EnsureCovers(loadedTokens, audioStart, audioEnd, audioPad, pad);

        var additionalSpecial = new System.Collections.Generic.Dictionary<string, int>(StringComparer.Ordinal)
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
            ModelFamily = "gpt2",
        };
        return GgufTokenizer.FromSource(source);
    }

    // --- AuT audio encoder weights (Whisper-style: conv2d stem + absolute-pos transformer) ---
    public float[] Conv1Weight { get; }
    public float[] Conv1Bias { get; }
    public float[] Conv2Weight { get; }
    public float[] Conv2Bias { get; }
    public float[] Conv3Weight { get; }
    public float[] Conv3Bias { get; }
    public CfmLinearWeight ConvOutWeight { get; } // [7680, 896], no bias
    public float[] LnPostWeight { get; }
    public float[] LnPostBias { get; }
    public float[] MelFilters { get; }
    public float[] MelWindow { get; }
    public CfmLinearWeight Proj1Weight { get; }
    public float[] Proj1Bias { get; }
    public CfmLinearWeight Proj2Weight { get; }
    public float[] Proj2Bias { get; }
    public QwenAsrAudioLayerWeights[] AudioLayerWeights { get; }

    /// <summary>Loads and dequantizes a required tensor by exact canonical (`audio.*`) name to a flat float[] in file storage order -- transparently reads through whichever real backing store (GGUF or Safetensors) this instance was constructed from.</summary>
    public float[] GetTensor(string name)
    {
        if (Model is not null)
        {
            var info = Model.FindTensor(name) ?? throw new InvalidDataException($"Qwen3-ASR GGUF missing required tensor '{name}'.");
            var bytes = Model.GetTensorData(info);
            var dst = new float[info.ElementCount];
            Dequantize.ToFloat32(bytes, dst, info.DType, info.ElementCount);
            return dst;
        }

        string realName = _stRename.TryGetValue(name, out var mapped) ? mapped : name;
        return _stLoader!.ReadF32(realName);
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
        Model?.Dispose();
        _stLoader?.Dispose();
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
    public CfmLinearWeight AttnQWeight { get; }
    public float[] AttnQBias { get; }
    public CfmLinearWeight AttnKWeight { get; }
    public float[] AttnKBias { get; }
    public CfmLinearWeight AttnVWeight { get; }
    public float[] AttnVBias { get; }
    public CfmLinearWeight AttnOutWeight { get; }
    public float[] AttnOutBias { get; }
    public float[] FfnNormWeight { get; }
    public float[] FfnNormBias { get; }
    public CfmLinearWeight FfnUpWeight { get; }
    public float[] FfnUpBias { get; }
    public CfmLinearWeight FfnDownWeight { get; }
    public float[] FfnDownBias { get; }

    public QwenAsrAudioLayerWeights(QwenAsrWeights w, string prefix)
    {
        AttnNormWeight = w.GetTensor($"{prefix}.attn_norm.weight");
        AttnNormBias = w.GetTensor($"{prefix}.attn_norm.bias");
        var qF32 = w.GetTensor($"{prefix}.attn_q.weight");
        AttnQBias = w.GetTensor($"{prefix}.attn_q.bias");
        AttnQWeight = CfmLinearWeight.FromF32(qF32, outDim: AttnQBias.Length, inDim: qF32.Length / AttnQBias.Length);
        var kF32 = w.GetTensor($"{prefix}.attn_k.weight");
        AttnKBias = w.GetTensor($"{prefix}.attn_k.bias");
        AttnKWeight = CfmLinearWeight.FromF32(kF32, outDim: AttnKBias.Length, inDim: kF32.Length / AttnKBias.Length);
        var vF32 = w.GetTensor($"{prefix}.attn_v.weight");
        AttnVBias = w.GetTensor($"{prefix}.attn_v.bias");
        AttnVWeight = CfmLinearWeight.FromF32(vF32, outDim: AttnVBias.Length, inDim: vF32.Length / AttnVBias.Length);
        var outF32 = w.GetTensor($"{prefix}.attn_out.weight");
        AttnOutBias = w.GetTensor($"{prefix}.attn_out.bias");
        AttnOutWeight = CfmLinearWeight.FromF32(outF32, outDim: AttnOutBias.Length, inDim: outF32.Length / AttnOutBias.Length);
        FfnNormWeight = w.GetTensor($"{prefix}.ffn_norm.weight");
        FfnNormBias = w.GetTensor($"{prefix}.ffn_norm.bias");
        var ffnUpF32 = w.GetTensor($"{prefix}.ffn_up.weight");
        FfnUpBias = w.GetTensor($"{prefix}.ffn_up.bias");
        FfnUpWeight = CfmLinearWeight.FromF32(ffnUpF32, outDim: FfnUpBias.Length, inDim: ffnUpF32.Length / FfnUpBias.Length);
        var ffnDownF32 = w.GetTensor($"{prefix}.ffn_down.weight");
        FfnDownBias = w.GetTensor($"{prefix}.ffn_down.bias");
        FfnDownWeight = CfmLinearWeight.FromF32(ffnDownF32, outDim: FfnDownBias.Length, inDim: ffnDownF32.Length / FfnDownBias.Length);
    }
}
