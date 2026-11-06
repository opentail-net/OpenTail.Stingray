using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Mlx;

/// <summary>
/// Builds an <see cref="MlxDenseModel"/> from a real GGUF checkpoint: reads dimensions from the
/// GGUF's own metadata keys (<c>{arch}.embedding_length</c> etc. -- the llama.cpp/GGUF convention,
/// not reusing <c>ArchitectureModelResolver</c>'s hyperparameter resolution, which handles many
/// architecture-specific overrides this dense-only port doesn't need), and dequantizes every
/// quantized tensor to F32 via Stingray's existing <see cref="Dequantize.ToFloat32"/> -- no new
/// quantization logic here. Never run on real hardware; see
/// docs/4-performance/gpu/2026-11-06-mlx-metal-backend-plan.md.
/// </summary>
public static class MlxGgufDenseModelLoader
{
    /// <summary>
    /// Opt-in flag (same shape as <c>STINGRAY_TP_DEGREE</c> for the CUDA tensor-parallelism plan):
    /// <see cref="Load"/> refuses unless <c>STINGRAY_MLX_DENSE=1</c> is set, so nothing can reach this
    /// path by accident. There is no engine/planner integration yet (no <c>ForwardPassKind</c> case,
    /// no CLI <c>--backend mlx</c> option) -- that is real additional wiring work, not attempted here;
    /// this flag only gates direct calls to this loader.
    /// </summary>
    public static void CheckEnabled()
    {
        string? raw = Environment.GetEnvironmentVariable("STINGRAY_MLX_DENSE");
        if (raw != "1")
            throw new NotSupportedException(
                "MlxGgufDenseModelLoader.Load is gated behind STINGRAY_MLX_DENSE=1 -- unset or any " +
                "other value refuses it. Never run on real hardware (no Mac on this development " +
                "machine); see docs/4-performance/gpu/2026-11-06-mlx-metal-backend-plan.md.");
    }

    public static MlxDenseModel Load(GgufModel model)
    {
        CheckEnabled();
        string arch = model.GetMetadata("general.architecture", "");
        if (string.IsNullOrEmpty(arch))
            throw new InvalidOperationException("GGUF has no general.architecture key.");
        if (!string.Equals(arch, "llama", StringComparison.OrdinalIgnoreCase))
            throw new NotSupportedException($"The MLX dense loader supports only the verified 'llama' architecture; GGUF declares '{arch}'.");

        int hiddenDim = model.GetMetadata($"{arch}.embedding_length", 0);
        int numHeads = model.GetMetadata($"{arch}.attention.head_count", 0);
        int numKvHeads = model.GetMetadata($"{arch}.attention.head_count_kv", numHeads);
        int blockCount = model.GetMetadata($"{arch}.block_count", 0);
        int intermDim = model.GetMetadata($"{arch}.feed_forward_length", 0);
        float eps = model.GetMetadata($"{arch}.attention.layer_norm_rms_epsilon", 1e-5f);
        float ropeBase = model.GetMetadata($"{arch}.rope.freq_base", 10000f);

        if (hiddenDim <= 0 || numHeads <= 0 || blockCount <= 0 || intermDim <= 0)
            throw new InvalidOperationException(
                $"GGUF metadata is missing or zero for required dimensions (arch={arch}, hiddenDim={hiddenDim}, numHeads={numHeads}, blockCount={blockCount}, intermDim={intermDim}).");
        if (hiddenDim % numHeads != 0)
            throw new InvalidOperationException($"GGUF hidden dimension ({hiddenDim}) must be divisible by the attention head count ({numHeads}).");
        int headDim = hiddenDim / numHeads;
        ValidateLlamaConfiguration(model, arch, numHeads, numKvHeads, headDim, eps, ropeBase);
        int queryDim = checked(numHeads * headDim);
        int kvDim = checked(numKvHeads * headDim);

        var layers = new MlxDenseLayer[blockCount];
        for (int i = 0; i < blockCount; i++)
        {
            string prefix = $"blk.{i}.";
            layers[i] = new MlxDenseLayer
            {
                AttnNormWeight = LoadVectorWeight(model, prefix + "attn_norm.weight", hiddenDim),
                Wq = LoadLinearWeight(model, prefix + "attn_q.weight", hiddenDim, queryDim),
                Wk = LoadLinearWeight(model, prefix + "attn_k.weight", hiddenDim, kvDim),
                Wv = LoadLinearWeight(model, prefix + "attn_v.weight", hiddenDim, kvDim),
                Wo = LoadLinearWeight(model, prefix + "attn_output.weight", queryDim, hiddenDim),
                FfnNormWeight = LoadVectorWeight(model, prefix + "ffn_norm.weight", hiddenDim),
                WGate = LoadLinearWeight(model, prefix + "ffn_gate.weight", hiddenDim, intermDim),
                WUp = LoadLinearWeight(model, prefix + "ffn_up.weight", hiddenDim, intermDim),
                WDown = LoadLinearWeight(model, prefix + "ffn_down.weight", intermDim, hiddenDim),
                HiddenDim = hiddenDim,
                NumHeads = numHeads,
                NumKvHeads = numKvHeads,
                HeadDim = headDim,
                IntermDim = intermDim,
                Eps = eps,
                RopeBase = ropeBase,
            };
        }

        var embeddingInfo = model.FindTensor("token_embd.weight") ?? throw new InvalidOperationException("Missing tensor: token_embd.weight");
        if (embeddingInfo.NDimensions != 2 || embeddingInfo.Dimensions[0] != hiddenDim || embeddingInfo.Dimensions[1] <= 0 || embeddingInfo.Dimensions[1] > int.MaxValue)
            throw new InvalidOperationException($"Tensor 'token_embd.weight' must have GGUF dimensions [hiddenDim, vocabSize] = [{hiddenDim}, vocabSize]; found [{string.Join(", ", embeddingInfo.Dimensions)}].");
        int vocabSize = (int)embeddingInfo.Dimensions[1];
        float[] tokenEmbedding = DequantizeTensor(model, embeddingInfo);
        float[] outputNorm = LoadVectorWeight(model, "output_norm.weight", hiddenDim);
        // Tied embeddings (output.weight absent): the output head reuses the embedding table, as
        // llama.cpp does for architectures that declare no separate output.weight tensor. The
        // embedding table is [vocab, hidden], while MatMul consumes [hidden, vocab].
        float[] outputWeight = model.FindTensor("output.weight") is not null
            ? LoadLinearWeight(model, "output.weight", hiddenDim, vocabSize)
            : TransposeGgufLinearWeight(tokenEmbedding, hiddenDim, vocabSize);

        return new MlxDenseModel
        {
            TokenEmbedding = tokenEmbedding,
            Layers = layers,
            OutputNormWeight = outputNorm,
            OutputWeight = outputWeight,
            HiddenDim = hiddenDim,
            VocabSize = vocabSize,
            Eps = eps,
        };
    }

    private static float[] LoadVectorWeight(GgufModel model, string name, int length)
    {
        var info = model.FindTensor(name) ?? throw new InvalidOperationException($"Missing tensor: {name}");
        if (info.NDimensions != 1 || info.Dimensions[0] != length)
            throw new InvalidOperationException($"Tensor '{name}' must have length {length}; found dimensions [{string.Join(", ", info.Dimensions)}].");
        return DequantizeTensor(model, info);
    }

    private static float[] LoadLinearWeight(GgufModel model, string name, int inputDim, int outputDim)
    {
        var info = model.FindTensor(name) ?? throw new InvalidOperationException($"Missing tensor: {name}");
        if (info.NDimensions != 2 || info.Dimensions[0] != inputDim || info.Dimensions[1] != outputDim)
            throw new InvalidOperationException($"Tensor '{name}' must have GGUF dimensions [input, output] = [{inputDim}, {outputDim}]; found [{string.Join(", ", info.Dimensions)}].");
        return TransposeGgufLinearWeight(DequantizeTensor(model, info), inputDim, outputDim);
    }

    private static float[] TransposeGgufLinearWeight(ReadOnlySpan<float> source, int inputDim, int outputDim)
    {
        int expectedLength = checked(inputDim * outputDim);
        if (source.Length != expectedLength)
            throw new ArgumentException($"Weight has {source.Length} elements, expected inputDim*outputDim = {expectedLength}.", nameof(source));

        var transposed = new float[source.Length];
        for (int output = 0; output < outputDim; output++)
            for (int input = 0; input < inputDim; input++)
                transposed[input * outputDim + output] = source[output * inputDim + input];
        return transposed;
    }

    private static float[] DequantizeTensor(GgufModel model, GgufTensorInfo info)
    {
        var raw = model.GetTensorData(info);
        var output = new float[info.ElementCount];
        if (info.DType == DType.Float32)
            System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(raw).CopyTo(output);
        else
            Dequantize.ToFloat32(raw, output, info.DType, info.ElementCount);
        return output;
    }

    private static void ValidateLlamaConfiguration(GgufModel model, string arch, int numHeads, int numKvHeads,
        int headDim, float eps, float ropeBase)
    {
        if (numKvHeads <= 0 || numHeads % numKvHeads != 0)
            throw new InvalidOperationException($"GGUF key '{arch}.attention.head_count_kv' must be a positive divisor of head_count ({numHeads}); found {numKvHeads}.");
        if (headDim <= 0 || (headDim & 1) != 0)
            throw new NotSupportedException($"The MLX Llama path requires a positive even head dimension for full-dimension RoPE; found {headDim}.");
        if (!float.IsFinite(eps) || eps <= 0)
            throw new InvalidOperationException($"GGUF key '{arch}.attention.layer_norm_rms_epsilon' must be finite and positive; found {eps}.");
        if (!float.IsFinite(ropeBase) || ropeBase <= 0)
            throw new InvalidOperationException($"GGUF key '{arch}.rope.freq_base' must be finite and positive; found {ropeBase}.");

        int ropeDimensionCount = model.GetMetadata($"{arch}.rope.dimension_count", headDim);
        if (ropeDimensionCount != headDim)
            throw new NotSupportedException($"The MLX Llama path supports full-head RoPE only ({headDim} dimensions); GGUF '{arch}.rope.dimension_count' is {ropeDimensionCount}.");

        string ropePrefix = arch + ".rope.";
        foreach (string key in model.Metadata.Keys.Where(key => key.StartsWith(ropePrefix, StringComparison.OrdinalIgnoreCase)))
        {
            string setting = key[ropePrefix.Length..].ToLowerInvariant();
            if (setting is "freq_base" or "dimension_count")
                continue;
            if (setting == "scaling.type")
            {
                string scalingType = model.GetMetadata(key, "");
                if (scalingType.Trim().ToLowerInvariant() is "none" or "default")
                    continue;
            }
            throw new NotSupportedException($"The MLX Llama path does not support GGUF RoPE setting '{key}'.");
        }

        string[] unsupportedSuffixes =
        [
            "attn_q.bias", "attn_k.bias", "attn_v.bias", "attn_output.bias",
            "ffn_gate.bias", "ffn_up.bias", "ffn_down.bias", "attn_norm.bias", "ffn_norm.bias",
            "output_norm.bias", "output.bias",
            "attn_q_norm.weight", "attn_k_norm.weight", "post_attention_norm.weight", "post_ffn_norm.weight",
        ];
        foreach (var tensor in model.Tensors)
        {
            if (unsupportedSuffixes.Any(suffix => tensor.Name.EndsWith(suffix, StringComparison.Ordinal)))
                throw new NotSupportedException($"The MLX Llama path does not support tensor '{tensor.Name}'.");
        }
    }
}
