using System.Runtime.InteropServices;
using System.Text.Json;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Audio.Whisper;

/// <summary>
/// Parses the legacy whisper.cpp "ggml" binary model format (magic 0x67676d6c) — a flat,
/// self-contained file: header hparams, baked mel filterbank, inline vocab, then a stream of
/// named tensors (n_dims, name, ggml type, dims, raw data) read until EOF.
/// This is NOT GGUF and NOT safetensors; it predates both and is whisper.cpp's own format,
/// which is what OpenAI's community-converted ggml-*.bin releases (e.g. ggml-large-v3.bin) use.
/// Reference: examples/whisper.cpp/src/whisper.cpp:whisper_model_load.
/// </summary>
public sealed class WhisperGgmlModel
{
    private const uint GgmlMagic = 0x67676d6c;

    public int VocabSize { get; private set; }
    public int AudioCtx { get; private set; }
    public int AudioState { get; private set; }
    public int AudioHead { get; private set; }
    public int AudioLayer { get; private set; }
    public int TextCtx { get; private set; }
    public int TextState { get; private set; }
    public int TextHead { get; private set; }
    public int TextLayer { get; private set; }
    public int NumMels { get; private set; }
    public int FType { get; private set; }

    public string[] TokenById { get; private set; } = [];

    private readonly Dictionary<string, WhisperGgmlTensor> _tensors = new(StringComparer.Ordinal);

    public WhisperConfig ToConfig() => new()
    {
        VocabSize = VocabSize,
        AudioCtx = AudioCtx,
        AudioState = AudioState,
        AudioHead = AudioHead,
        AudioLayer = AudioLayer,
        TextCtx = TextCtx,
        TextState = TextState,
        TextHead = TextHead,
        TextLayer = TextLayer,
        NumMels = NumMels,
        IsV3 = VocabSize == 51866
    };

    /// <summary>
    /// Looks up a tensor by its ggml name (e.g. "encoder.blocks.0.attn.query.weight") and returns
    /// its data dequantized to float32, plus its ne[] shape (ggml column-major: ne[0] is innermost/fastest).
    /// </summary>
    public bool TryGetTensor(string name, out float[] data, out int[] shape)
    {
        if (_tensors.TryGetValue(name, out var t))
        {
            data = t.Data;
            shape = t.Shape;
            return true;
        }
        data = [];
        shape = [];
        return false;
    }

    public float[] GetTensor(string name)
    {
        if (!TryGetTensor(name, out var data, out _))
            throw new KeyNotFoundException($"Tensor '{name}' not found in ggml model file.");
        return data;
    }

    /// <summary>
    /// Raw F16 bit patterns for a tensor whose on-disk dtype was actually F16 (true for every big
    /// matrix in the legacy ggml/.bin format and its GGUF repackaging -- Whisper releases only ever
    /// use F16 or F32; false for the Safetensors loader, which always supplies plain F32 from HF).
    /// Lets a caller use a real hardware F16C-based kernel directly against the on-disk bytes
    /// instead of the eagerly-expanded <see cref="GetTensor"/> float32 copy -- see
    /// docs/audio-review-progress.md's ggml/F16C investigation for why this matters (ggml's own
    /// F16 dot-product kernel never expands weights to F32 at all, for exactly this bandwidth
    /// reason). Returns false (not an exception) when the tensor's real dtype wasn't F16, so
    /// callers can fall back to <see cref="GetTensor"/> uniformly.
    /// </summary>
    public bool TryGetTensorRawF16(string name, out short[] f16Bits)
    {
        if (_tensors.TryGetValue(name, out var t) && t.RawF16Bits is { } bits)
        {
            f16Bits = bits;
            return true;
        }
        f16Bits = [];
        return false;
    }

    public static WhisperGgmlModel Load(string path)
    {
        using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, bufferSize: 1 << 20);
        using var br = new BinaryReader(fs);

        var model = new WhisperGgmlModel();

        uint magic = br.ReadUInt32();
        if (magic != GgmlMagic)
            throw new InvalidDataException($"Not a whisper.cpp ggml model file (bad magic 0x{magic:x8}).");

        model.VocabSize = br.ReadInt32();
        model.AudioCtx = br.ReadInt32();
        model.AudioState = br.ReadInt32();
        model.AudioHead = br.ReadInt32();
        model.AudioLayer = br.ReadInt32();
        model.TextCtx = br.ReadInt32();
        model.TextState = br.ReadInt32();
        model.TextHead = br.ReadInt32();
        model.TextLayer = br.ReadInt32();
        model.NumMels = br.ReadInt32();
        model.FType = br.ReadInt32();

        // Baked mel filterbank (unused — we compute our own Slaney filterbank in WhisperMelExtractor).
        int filterMels = br.ReadInt32();
        int filterFft = br.ReadInt32();
        long filterBytes = (long)filterMels * filterFft * sizeof(float);
        fs.Seek(filterBytes, SeekOrigin.Current);

        // Vocab: n_vocab entries of (uint32 len, len bytes utf8).
        int vocabCount = br.ReadInt32();
        var tokens = new string[vocabCount];
        for (int i = 0; i < vocabCount; i++)
        {
            uint len = br.ReadUInt32();
            tokens[i] = len == 0 ? string.Empty : System.Text.Encoding.UTF8.GetString(br.ReadBytes((int)len));
        }
        model.TokenById = tokens;

        // Tensor stream: (n_dims, name_len, ggml_type) then n_dims x int32 ne[], then name bytes, then raw data.
        while (fs.Position < fs.Length)
        {
            if (fs.Position + 12 > fs.Length) break;

            int nDims = br.ReadInt32();
            int nameLen = br.ReadInt32();
            int ggmlType = br.ReadInt32();

            if (nDims < 0 || nDims > 4)
                throw new InvalidDataException($"Invalid tensor n_dims {nDims} at offset {fs.Position}.");

            var ne = new int[4] { 1, 1, 1, 1 };
            long nElements = 1;
            for (int d = 0; d < nDims; d++)
            {
                ne[d] = br.ReadInt32();
                nElements *= ne[d];
            }

            string name = System.Text.Encoding.UTF8.GetString(br.ReadBytes(nameLen));

            float[] data = ReadTensorData(br, ggmlType, nElements, out short[]? rawF16);
            model._tensors[name] = new WhisperGgmlTensor(ne[..nDims], data, rawF16);
        }

        return model;
    }

    /// <summary>
    /// Loads from a real GGUF file produced by <c>scratch-llamacpp-ref/whisper_ggml_to_gguf.py</c>
    /// (a genuine GGUF container -- magic "GGUF", real KV-metadata block -- repackaging the same
    /// tensor values as the legacy ggml file, bit-exact). Real finding, confirmed directly against
    /// the upstream ggml-org/whisper.cpp README rather than assumed: whisper.cpp's own model files
    /// are still the legacy custom ggml format as of the current master branch, so there is no
    /// canonical/official GGUF-format Whisper release to download -- this loader consumes our own
    /// self-converted GGUF file instead of a third-party one. Populates the exact same
    /// hparams/tensor state as <see cref="Load"/> so every downstream consumer
    /// (<see cref="WhisperEncoderWeights"/>, <see cref="WhisperDecoderWeights"/>, etc.) works
    /// unchanged regardless of which loader was used.
    /// </summary>
    public static WhisperGgmlModel LoadFromGguf(string path)
    {
        using var gguf = GgufModel.Open(path);
        var model = new WhisperGgmlModel();

        model.VocabSize = gguf.GetMetadata("whisper.hparam.n_vocab", 0);
        model.AudioCtx = gguf.GetMetadata("whisper.hparam.n_audio_ctx", 0);
        model.AudioState = gguf.GetMetadata("whisper.hparam.n_audio_state", 0);
        model.AudioHead = gguf.GetMetadata("whisper.hparam.n_audio_head", 0);
        model.AudioLayer = gguf.GetMetadata("whisper.hparam.n_audio_layer", 0);
        model.TextCtx = gguf.GetMetadata("whisper.hparam.n_text_ctx", 0);
        model.TextState = gguf.GetMetadata("whisper.hparam.n_text_state", 0);
        model.TextHead = gguf.GetMetadata("whisper.hparam.n_text_head", 0);
        model.TextLayer = gguf.GetMetadata("whisper.hparam.n_text_layer", 0);
        model.NumMels = gguf.GetMetadata("whisper.hparam.n_mels", 0);
        model.FType = gguf.GetMetadata("whisper.hparam.ftype", 0);

        if (gguf.Metadata.TryGetValue("tokenizer.ggml.tokens", out var tokensRaw) && tokensRaw is object[] tokensArr)
            model.TokenById = Array.ConvertAll(tokensArr, o => (string)o);

        foreach (var info in gguf.Tensors)
        {
            var bytes = gguf.GetTensorData(info);
            var data = new float[info.ElementCount];
            Dequantize.ToFloat32(bytes, data, info.DType, info.ElementCount);

            // Preserve raw F16 bits too when the on-disk dtype really is F16 (see
            // TryGetTensorRawF16's doc comment) -- GGUF F16 storage is bit-identical IEEE754
            // half-floats, so the tensor's raw byte span can be reinterpreted directly, no
            // reconversion needed.
            short[]? rawF16 = info.DType == DType.Float16
                ? MemoryMarshal.Cast<byte, short>(bytes[..checked((int)(info.ElementCount * 2))]).ToArray()
                : null;

            // GgufModel reports shape in reversed/displayed (GGUF) order; WhisperGgmlTensor's
            // Shape field is only used for byte-layout bookkeeping elsewhere, not indexed by
            // dimension order, so storing it as-is here is consistent with WhisperGgmlModel's
            // other real usages (all downstream consumers only call GetTensor by name).
            var shape = new int[info.NDimensions];
            for (int d = 0; d < info.NDimensions; d++) shape[d] = checked((int)info.Dimensions[d]);
            model._tensors[info.Name] = new WhisperGgmlTensor(shape, data, rawF16);
        }

        return model;
    }

    /// <summary>
    /// Loads from the real, canonical OpenAI Whisper Hugging Face distribution (`openai/
    /// whisper-{tiny,base,small,medium,large-v3}`) -- a genuine `model.safetensors` +
    /// `config.json` + `vocab.json` checkpoint, confirmed to be the actual format
    /// `transformers`' `WhisperForConditionalGeneration.from_pretrained()` loads (not a
    /// community conversion), verified directly against real HF repo file listings and the
    /// real `model.safetensors.index.fp32.json` tensor-name index before writing this loader.
    /// <c>dir</c> is the checkpoint directory containing `config.json`/`model.safetensors`/
    /// `vocab.json`.
    ///
    /// <para>Real tensor namespace (top-level `model.encoder.*`/`model.decoder.*`, no extra
    /// wrapper prefix): `model.encoder.conv1/conv2.{weight,bias}`, `model.encoder.
    /// embed_positions.weight`, `model.encoder.layers.{i}.self_attn.{q,k,v}_proj.weight`
    /// (`k_proj` has NO bias -- matches OpenAI's own weights exactly, same real fact
    /// `WhisperEncoderWeights`'s doc comment already notes for the legacy ggml format),
    /// `self_attn.out_proj`, `self_attn_layer_norm`, `fc1`/`fc2`, `final_layer_norm`,
    /// `model.encoder.layer_norm` (final). Decoder mirrors this plus `encoder_attn.*`
    /// (cross-attention) and `model.decoder.embed_tokens.weight`.</para>
    /// </summary>
    public static WhisperGgmlModel LoadFromSafetensors(string dir)
    {
        string configPath = Path.Combine(dir, "config.json");
        using var configDoc = JsonDocument.Parse(File.ReadAllBytes(configPath));
        var root = configDoc.RootElement;

        var model = new WhisperGgmlModel();
        model.VocabSize = root.GetProperty("vocab_size").GetInt32();
        model.AudioCtx = root.GetProperty("max_source_positions").GetInt32();
        model.AudioState = root.GetProperty("d_model").GetInt32();
        model.AudioHead = root.GetProperty("encoder_attention_heads").GetInt32();
        model.AudioLayer = root.GetProperty("encoder_layers").GetInt32();
        model.TextCtx = root.GetProperty("max_target_positions").GetInt32();
        model.TextState = root.GetProperty("d_model").GetInt32();
        model.TextHead = root.GetProperty("decoder_attention_heads").GetInt32();
        model.TextLayer = root.GetProperty("decoder_layers").GetInt32();
        model.NumMels = root.GetProperty("num_mel_bins").GetInt32();
        model.FType = 0;

        string vocabPath = Path.Combine(dir, "vocab.json");
        if (File.Exists(vocabPath))
        {
            using var vocabDoc = JsonDocument.Parse(File.ReadAllBytes(vocabPath));
            var tokens = new string[model.VocabSize];
            foreach (var prop in vocabDoc.RootElement.EnumerateObject())
            {
                int id = prop.Value.GetInt32();
                if ((uint)id < (uint)tokens.Length) tokens[id] = prop.Name;
            }
            for (int i = 0; i < tokens.Length; i++) tokens[i] ??= string.Empty;
            model.TokenById = tokens;
        }

        string safetensorsPath = Path.Combine(dir, "model.safetensors");
        using var loader = File.Exists(Path.Combine(dir, "model.safetensors.index.json"))
            ? SafetensorsLoader.OpenDirectory(dir)
            : SafetensorsLoader.Open(safetensorsPath);

        void MapTensor(string realName, string canonicalName)
        {
            if (!loader.Contains(realName)) return;
            var data = loader.ReadF32(realName);
            var shape = loader.GetShape(realName);
            model._tensors[canonicalName] = new WhisperGgmlTensor(shape, data);
        }

        MapTensor("model.encoder.conv1.weight", "encoder.conv1.weight");
        MapTensor("model.encoder.conv1.bias", "encoder.conv1.bias");
        MapTensor("model.encoder.conv2.weight", "encoder.conv2.weight");
        MapTensor("model.encoder.conv2.bias", "encoder.conv2.bias");
        MapTensor("model.encoder.embed_positions.weight", "encoder.positional_embedding");
        MapTensor("model.encoder.layer_norm.weight", "encoder.ln_post.weight");
        MapTensor("model.encoder.layer_norm.bias", "encoder.ln_post.bias");

        for (int i = 0; i < model.AudioLayer; i++)
        {
            string p = $"model.encoder.layers.{i}";
            string b = $"encoder.blocks.{i}";
            MapTensor($"{p}.self_attn_layer_norm.weight", $"{b}.attn_ln.weight");
            MapTensor($"{p}.self_attn_layer_norm.bias", $"{b}.attn_ln.bias");
            MapTensor($"{p}.self_attn.q_proj.weight", $"{b}.attn.query.weight");
            MapTensor($"{p}.self_attn.q_proj.bias", $"{b}.attn.query.bias");
            MapTensor($"{p}.self_attn.k_proj.weight", $"{b}.attn.key.weight"); // real: no bias
            MapTensor($"{p}.self_attn.v_proj.weight", $"{b}.attn.value.weight");
            MapTensor($"{p}.self_attn.v_proj.bias", $"{b}.attn.value.bias");
            MapTensor($"{p}.self_attn.out_proj.weight", $"{b}.attn.out.weight");
            MapTensor($"{p}.self_attn.out_proj.bias", $"{b}.attn.out.bias");
            MapTensor($"{p}.final_layer_norm.weight", $"{b}.mlp_ln.weight");
            MapTensor($"{p}.final_layer_norm.bias", $"{b}.mlp_ln.bias");
            MapTensor($"{p}.fc1.weight", $"{b}.mlp.0.weight");
            MapTensor($"{p}.fc1.bias", $"{b}.mlp.0.bias");
            MapTensor($"{p}.fc2.weight", $"{b}.mlp.2.weight");
            MapTensor($"{p}.fc2.bias", $"{b}.mlp.2.bias");
        }

        MapTensor("model.decoder.embed_tokens.weight", "decoder.token_embedding.weight");
        MapTensor("model.decoder.embed_positions.weight", "decoder.positional_embedding");
        MapTensor("model.decoder.layer_norm.weight", "decoder.ln.weight");
        MapTensor("model.decoder.layer_norm.bias", "decoder.ln.bias");

        for (int i = 0; i < model.TextLayer; i++)
        {
            string p = $"model.decoder.layers.{i}";
            string b = $"decoder.blocks.{i}";
            MapTensor($"{p}.self_attn_layer_norm.weight", $"{b}.attn_ln.weight");
            MapTensor($"{p}.self_attn_layer_norm.bias", $"{b}.attn_ln.bias");
            MapTensor($"{p}.self_attn.q_proj.weight", $"{b}.attn.query.weight");
            MapTensor($"{p}.self_attn.q_proj.bias", $"{b}.attn.query.bias");
            MapTensor($"{p}.self_attn.k_proj.weight", $"{b}.attn.key.weight");
            MapTensor($"{p}.self_attn.v_proj.weight", $"{b}.attn.value.weight");
            MapTensor($"{p}.self_attn.v_proj.bias", $"{b}.attn.value.bias");
            MapTensor($"{p}.self_attn.out_proj.weight", $"{b}.attn.out.weight");
            MapTensor($"{p}.self_attn.out_proj.bias", $"{b}.attn.out.bias");
            MapTensor($"{p}.encoder_attn_layer_norm.weight", $"{b}.cross_attn_ln.weight");
            MapTensor($"{p}.encoder_attn_layer_norm.bias", $"{b}.cross_attn_ln.bias");
            MapTensor($"{p}.encoder_attn.q_proj.weight", $"{b}.cross_attn.query.weight");
            MapTensor($"{p}.encoder_attn.q_proj.bias", $"{b}.cross_attn.query.bias");
            MapTensor($"{p}.encoder_attn.k_proj.weight", $"{b}.cross_attn.key.weight");
            MapTensor($"{p}.encoder_attn.v_proj.weight", $"{b}.cross_attn.value.weight");
            MapTensor($"{p}.encoder_attn.v_proj.bias", $"{b}.cross_attn.value.bias");
            MapTensor($"{p}.encoder_attn.out_proj.weight", $"{b}.cross_attn.out.weight");
            MapTensor($"{p}.encoder_attn.out_proj.bias", $"{b}.cross_attn.out.bias");
            MapTensor($"{p}.final_layer_norm.weight", $"{b}.mlp_ln.weight");
            MapTensor($"{p}.final_layer_norm.bias", $"{b}.mlp_ln.bias");
            MapTensor($"{p}.fc1.weight", $"{b}.mlp.0.weight");
            MapTensor($"{p}.fc1.bias", $"{b}.mlp.0.bias");
            MapTensor($"{p}.fc2.weight", $"{b}.mlp.2.weight");
            MapTensor($"{p}.fc2.bias", $"{b}.mlp.2.bias");
        }

        return model;
    }

    private static float[] ReadTensorData(BinaryReader br, int ggmlType, long nElements, out short[]? rawF16)
    {
        // ggml_type enum: 0=F32, 1=F16, 6=Q5_0, 7=Q5_1, 8=Q8_0, ... (Whisper ggml releases use F16 or F32 only).
        switch (ggmlType)
        {
            case 0: // F32
            {
                var raw = br.ReadBytes(checked((int)(nElements * sizeof(float))));
                var f32 = new float[nElements];
                Buffer.BlockCopy(raw, 0, f32, 0, raw.Length);
                rawF16 = null;
                return f32;
            }
            case 1: // F16
            {
                var raw = br.ReadBytes(checked((int)(nElements * 2)));
                var f32 = new float[nElements];
                var halves = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, Half>(raw);
                for (int i = 0; i < f32.Length; i++) f32[i] = (float)halves[i];
                rawF16 = new short[nElements];
                Buffer.BlockCopy(raw, 0, rawF16, 0, raw.Length);
                return f32;
            }
            default:
                throw new NotSupportedException(
                    $"ggml tensor type {ggmlType} is not supported (expected F32=0 or F16=1). " +
                    "Quantized whisper.cpp model files are not yet supported.");
        }
    }

    private sealed record WhisperGgmlTensor(int[] Shape, float[] Data, short[]? RawF16Bits = null);
}
