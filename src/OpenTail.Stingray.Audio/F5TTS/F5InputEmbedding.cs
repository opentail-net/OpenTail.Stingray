namespace OpenTail.Stingray.Audio.F5TTS;

/// <summary>
/// F5-TTS's `InputEmbedding` (dit.py): proj(cat[x, cond, text_embed]) + ConvPositionEmbedding(dim=1024, kernel=31, groups=16).
/// </summary>
public static class F5InputEmbedding
{
    /// <summary>x/cond are [numFrames, MelDim] (channel-last), textEmbed is [numFrames, TextDim]. Returns [numFrames, HiddenDim].</summary>
    public static float[] Forward(F5TtsWeights w, float[] x, float[] cond, float[] textEmbed, int numFrames)
    {
        int melDim = F5TtsWeights.MelDim;
        int textDim = F5TtsWeights.TextDim;
        int hidden = F5TtsWeights.HiddenDim;
        int concatDim = melDim * 2 + textDim;

        var concat = new float[numFrames * concatDim];
        for (int i = 0; i < numFrames; i++)
        {
            int outOff = i * concatDim;
            int melOff = i * melDim;
            int textOff = i * textDim;
            System.Array.Copy(x, melOff, concat, outOff, melDim);
            System.Array.Copy(cond, melOff, concat, outOff + melDim, melDim);
            System.Array.Copy(textEmbed, textOff, concat, outOff + 2 * melDim, textDim);
        }

        var h = F5Kernels.Linear(concat, numFrames, concatDim, w.InputProjWeight, w.InputProjBias, hidden);

        var pos = ConvPositionEmbedding(w, h, numFrames, hidden);
        var output = new float[h.Length];
        for (int i = 0; i < output.Length; i++) output[i] = h[i] + pos[i];
        return output;
    }

    /// <summary>ConvPositionEmbedding: Conv1d(groups=16,k=31)->Mish->Conv1d(groups=16,k=31)->Mish.</summary>
    private static float[] ConvPositionEmbedding(F5TtsWeights w, float[] x, int t, int dim)
    {
        const int kernel = 31, groups = 16;
        var h = F5Kernels.GroupedConv1dSamePad(x, t, dim, w.ConvPos1Weight, w.ConvPos1Bias, kernel, groups);
        for (int i = 0; i < h.Length; i++) h[i] = F5Kernels.Mish(h[i]);
        h = F5Kernels.GroupedConv1dSamePad(h, t, dim, w.ConvPos2Weight, w.ConvPos2Bias, kernel, groups);
        for (int i = 0; i < h.Length; i++) h[i] = F5Kernels.Mish(h[i]);
        return h;
    }
}
