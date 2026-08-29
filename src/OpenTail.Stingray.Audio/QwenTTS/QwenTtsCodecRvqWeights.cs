
namespace OpenTail.Stingray.Audio.QwenTTS;

/// <summary>
/// Real weight loader for the Qwen3-TTS 12Hz codec's split RVQ decode stage, loaded from
/// `models/qwen-tokenizer-12hz-Q8_0.gguf` (the real `Serveurperso/Qwen3-TTS-GGUF` conversion of
/// the official `Qwen/Qwen3-TTS-Tokenizer-12Hz` checkpoint).
///
/// <para><b>Real, confirmed architecture (NOT one shared 512-dim codebook space summed across all
/// 16 codebooks -- a real correction to an earlier rough assumption)</b>: 1 semantic + 15 acoustic
/// RVQ groups, each operating in 256 INTERNAL dimensions (`codebook_dim_internal=256`,
/// `codebook_dim=512` split in half) with its OWN learned 256-&gt;512 projection
/// (`vector_quantization_hidden_dim=512`), summed only AFTER projection:
/// <c>z = P_semantic(E_semantic[c_0]) + P_acoustic(Σ_{k=1..15} E_acoustic[k][c_k])</c>. Confirmed
/// directly against the real `examples/qwentts.cpp/src/quantizer-decode.h` (`rvq_group_decode`)
/// and cross-checked against real tensor shapes in this GGUF via `list-tensors`.</para>
///
/// <para>Real tensor names: `tok_dec.vq_first.output_proj.weight` ([1,256,512], real Conv1d
/// kernel=1 reshaped to a plain [in=256,out=512] matrix) + `tok_dec.vq_first.0.codebook`
/// ([256,2048] = [dim,size]) for the 1 semantic quantizer; `tok_dec.vq_rest.output_proj.weight`
/// (same shape) + `tok_dec.vq_rest.{0..14}.codebook` for the 15 acoustic quantizers.</para>
/// </summary>
public sealed class QwenTtsCodecRvqWeights
{
    public const int CodebookDimInternal = 256;
    public const int Hidden = 512;
    public const int CodebookSize = 2048;
    public const int NumSemanticQuantizers = 1;
    public const int NumAcousticQuantizers = 15;

    /// <summary>[NumSemanticQuantizers][CodebookSize, CodebookDimInternal] flat row-major.</summary>
    public float[][] SemanticCodebooks { get; } = new float[NumSemanticQuantizers][];
    /// <summary>[CodebookDimInternal, Hidden] -- real Conv1d(k=1) weight, plain matvec.</summary>
    public float[] SemanticOutProjWeight { get; }

    /// <summary>[NumAcousticQuantizers][CodebookSize, CodebookDimInternal] flat row-major.</summary>
    public float[][] AcousticCodebooks { get; } = new float[NumAcousticQuantizers][];
    public float[] AcousticOutProjWeight { get; }

    public QwenTtsCodecRvqWeights(GgufModel model)
    {
        for (int k = 0; k < NumSemanticQuantizers; k++)
            SemanticCodebooks[k] = GetF32(model, $"tok_dec.vq_first.{k}.codebook");
        SemanticOutProjWeight = GetF32(model, "tok_dec.vq_first.output_proj.weight");

        for (int k = 0; k < NumAcousticQuantizers; k++)
            AcousticCodebooks[k] = GetF32(model, $"tok_dec.vq_rest.{k}.codebook");
        AcousticOutProjWeight = GetF32(model, "tok_dec.vq_rest.output_proj.weight");
    }

    private static float[] GetF32(GgufModel model, string name)
    {
        var info = model.FindTensor(name) ?? throw new InvalidDataException($"QwenTTS codec GGUF missing required tensor '{name}'.");
        var bytes = model.GetTensorData(info);
        var dst = new float[info.ElementCount];
        Dequantize.ToFloat32(bytes, dst, info.DType, info.ElementCount);
        return dst;
    }
}
