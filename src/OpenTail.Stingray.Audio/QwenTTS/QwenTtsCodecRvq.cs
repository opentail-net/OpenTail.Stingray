
namespace OpenTail.Stingray.Audio.QwenTTS;

/// <summary>
/// Real split RVQ decode for the Qwen3-TTS 12Hz codec, transcribed directly from the real
/// `examples/qwentts.cpp/src/quantizer-decode.h`'s `rvq_group_decode`/`quant_decode` (itself
/// ported from the official `SplitResidualVectorQuantizer`/`ResidualVectorQuantizer` in
/// `qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py`) -- see
/// <see cref="QwenTtsCodecRvqWeights"/>'s doc comment for the full architecture derivation.
/// </summary>
public static class QwenTtsCodecRvq
{
    /// <summary>
    /// Real split-RVQ decode: codes[16][T] (real layout: codes[0] = semantic, codes[1..15] =
    /// acoustic) -&gt; hidden[T][512]. Each group's codebook lookups are summed in the group's own
    /// 256-dim internal space, THEN projected to 512 via that group's own Conv1d(k=1) weight;
    /// finally the two groups' projected 512-dim vectors are summed.
    /// </summary>
    public static float[][] Decode(QwenTtsCodecRvqWeights w, int[][] codes)
    {
        int t = codes[0].Length;
        var semanticCodes = new int[QwenTtsCodecRvqWeights.NumSemanticQuantizers][];
        for (int k = 0; k < QwenTtsCodecRvqWeights.NumSemanticQuantizers; k++) semanticCodes[k] = codes[k];
        var acousticCodes = new int[QwenTtsCodecRvqWeights.NumAcousticQuantizers][];
        for (int k = 0; k < QwenTtsCodecRvqWeights.NumAcousticQuantizers; k++)
            acousticCodes[k] = codes[QwenTtsCodecRvqWeights.NumSemanticQuantizers + k];

        var semanticOut = GroupDecode(w.SemanticCodebooks, w.SemanticOutProjWeight, semanticCodes, t);
        var acousticOut = GroupDecode(w.AcousticCodebooks, w.AcousticOutProjWeight, acousticCodes, t);

        var output = new float[t][];
        for (int i = 0; i < t; i++)
        {
            var row = new float[QwenTtsCodecRvqWeights.Hidden];
            for (int d = 0; d < QwenTtsCodecRvqWeights.Hidden; d++) row[d] = semanticOut[i][d] + acousticOut[i][d];
            output[i] = row;
        }
        return output;
    }

    /// <summary>Real `rvq_group_decode`: sum this group's codebook lookups in internal-dim space per timestep, then project once to Hidden via the group's own Conv1d(k=1) weight.</summary>
    private static float[][] GroupDecode(float[][] codebooks, float[] outProjWeight, int[][] codes, int t)
    {
        int internalDim = QwenTtsCodecRvqWeights.CodebookDimInternal;
        int numCodebooks = codebooks.Length;

        var output = new float[t][];
        for (int i = 0; i < t; i++)
        {
            var sum = new float[internalDim];
            for (int k = 0; k < numCodebooks; k++)
            {
                int code = codes[k][i];
                long rowBase = (long)code * internalDim;
                var cb = codebooks[k];
                for (int d = 0; d < internalDim; d++) sum[d] += cb[rowBase + d];
            }
            output[i] = LinearNoBias(sum, outProjWeight, internalDim, QwenTtsCodecRvqWeights.Hidden);
        }
        return output;
    }

    private static unsafe float[] LinearNoBias(float[] input, float[] weight, int inDim, int outDim)
    {
        var output = new float[outDim];
        fixed (float* wp = weight, xp = input, op = output)
        {
            SimdKernels.MatVecF32(op, wp, xp, outDim, inDim);
        }
        return output;
    }
}
