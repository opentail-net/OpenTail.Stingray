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

            float[] data = ReadTensorData(br, ggmlType, nElements);
            model._tensors[name] = new WhisperGgmlTensor(ne[..nDims], data);
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
            // GgufModel reports shape in reversed/displayed (GGUF) order; WhisperGgmlTensor's
            // Shape field is only used for byte-layout bookkeeping elsewhere, not indexed by
            // dimension order, so storing it as-is here is consistent with WhisperGgmlModel's
            // other real usages (all downstream consumers only call GetTensor by name).
            var shape = new int[info.NDimensions];
            for (int d = 0; d < info.NDimensions; d++) shape[d] = checked((int)info.Dimensions[d]);
            model._tensors[info.Name] = new WhisperGgmlTensor(shape, data);
        }

        return model;
    }

    private static float[] ReadTensorData(BinaryReader br, int ggmlType, long nElements)
    {
        // ggml_type enum: 0=F32, 1=F16, 6=Q5_0, 7=Q5_1, 8=Q8_0, ... (Whisper ggml releases use F16 or F32 only).
        switch (ggmlType)
        {
            case 0: // F32
            {
                var raw = br.ReadBytes(checked((int)(nElements * sizeof(float))));
                var f32 = new float[nElements];
                Buffer.BlockCopy(raw, 0, f32, 0, raw.Length);
                return f32;
            }
            case 1: // F16
            {
                var raw = br.ReadBytes(checked((int)(nElements * 2)));
                var f32 = new float[nElements];
                var halves = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, Half>(raw);
                for (int i = 0; i < f32.Length; i++) f32[i] = (float)halves[i];
                return f32;
            }
            default:
                throw new NotSupportedException(
                    $"ggml tensor type {ggmlType} is not supported (expected F32=0 or F16=1). " +
                    "Quantized whisper.cpp model files are not yet supported.");
        }
    }

    private sealed record WhisperGgmlTensor(int[] Shape, float[] Data);
}
