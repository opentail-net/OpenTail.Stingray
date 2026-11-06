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
        int headDim = hiddenDim / numHeads;

        var layers = new MlxDenseLayer[blockCount];
        for (int i = 0; i < blockCount; i++)
        {
            layers[i] = new MlxDenseLayer
            {
                AttnNormWeight = DequantizeTensor(model, $"blk.{i}.attn_norm.weight"),
                Wq = DequantizeTensor(model, $"blk.{i}.attn_q.weight"),
                Wk = DequantizeTensor(model, $"blk.{i}.attn_k.weight"),
                Wv = DequantizeTensor(model, $"blk.{i}.attn_v.weight"),
                Wo = DequantizeTensor(model, $"blk.{i}.attn_output.weight"),
                FfnNormWeight = DequantizeTensor(model, $"blk.{i}.ffn_norm.weight"),
                WGate = DequantizeTensor(model, $"blk.{i}.ffn_gate.weight"),
                WUp = DequantizeTensor(model, $"blk.{i}.ffn_up.weight"),
                WDown = DequantizeTensor(model, $"blk.{i}.ffn_down.weight"),
                HiddenDim = hiddenDim,
                NumHeads = numHeads,
                NumKvHeads = numKvHeads,
                HeadDim = headDim,
                IntermDim = intermDim,
                Eps = eps,
                RopeBase = ropeBase,
            };
        }

        float[] tokenEmbedding = DequantizeTensor(model, "token_embd.weight");
        int vocabSize = tokenEmbedding.Length / hiddenDim;
        float[] outputNorm = DequantizeTensor(model, "output_norm.weight");
        // Tied embeddings (output.weight absent): the output head reuses the embedding table, as
        // llama.cpp does for architectures that declare no separate output.weight tensor.
        float[] outputWeight = model.FindTensor("output.weight") is not null
            ? DequantizeTensor(model, "output.weight")
            : tokenEmbedding;

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

    private static float[] DequantizeTensor(GgufModel model, string name)
    {
        var info = model.FindTensor(name) ?? throw new InvalidOperationException($"Missing tensor: {name}");
        var raw = model.GetTensorData(info);
        var output = new float[info.ElementCount];
        if (info.DType == DType.Float32)
            System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(raw).CopyTo(output);
        else
            Dequantize.ToFloat32(raw, output, info.DType, info.ElementCount);
        return output;
    }
}
