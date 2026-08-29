
namespace OpenTail.Stingray.Audio.CosyVoice;

/// <summary>
/// CosyVoice2's flow "UpsampleConformerEncoder": turns concatenated [prompt_token;
/// speech_tokens] into the CFM decoder's `mu` conditioning tensor (channel-first
/// [MelChannels, 2*T]) and projects the reference x-vector into the CFM's speaker-embedding
/// space. Thin wrapper over <see cref="S3GenConformerKernels"/> -- the actual block math is
/// shared with `Chatterbox/ChatterboxFlowEncoder.cs` (confirmed architecturally identical by
/// real tensor shapes during this session's CosyVoice audit; extracted into the shared kernel
/// after both pipelines had independently-verified implementations to check against each
/// other -- see docs/audio-review-progress.md's CosyVoice section).
///
/// NOT YET golden-verified against a real oracle -- structurally complete, same caveat as
/// every pipeline in this doc before its golden-verification pass.
/// </summary>
public static class CosyVoiceFlowEncoder
{
    public static (float[] Mu, int TotalFrames) Forward(
        CosyVoiceFlowWeights w, int[] promptTokens, int[] speechTokens)
    {
        int t = promptTokens.Length + speechTokens.Length;
        var tokenEmb = new float[t][];
        for (int i = 0; i < promptTokens.Length; i++)
            tokenEmb[i] = S3GenConformerKernels.EmbedRow(w.InputEmbeddingWeight, promptTokens[i], w.HiddenDim);
        for (int i = 0; i < speechTokens.Length; i++)
            tokenEmb[promptTokens.Length + i] = S3GenConformerKernels.EmbedRow(w.InputEmbeddingWeight, speechTokens[i], w.HiddenDim);

        return S3GenConformerKernels.Forward(w, tokenEmb);
    }

    public static float[] ProjectSpeakerEmbedding(CosyVoiceFlowWeights w, float[] xvector) =>
        S3GenConformerKernels.ProjectSpeakerEmbedding(w.SpkEmbedAffineWeight, w.SpkEmbedAffineBias, w.SpkEmbedDim, w.MelChannels, xvector);
}
