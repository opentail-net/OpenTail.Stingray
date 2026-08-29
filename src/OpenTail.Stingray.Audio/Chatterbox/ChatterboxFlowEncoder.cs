
namespace OpenTail.Stingray.Audio.Chatterbox;

/// <summary>
/// S3Gen stage 1: UpsampleConformerEncoder (examples/chatterbox-tts-py/chatterbox/models/s3gen/
/// transformer/upsample_encoder.py), ported for the non-streaming/non-chunked, no-padding-mask
/// case this codebase needs (single sequence, batch size 1, offset=0). Turns concatenated
/// [prompt_token; speech_tokens] S3 speech-token ids into the CFM decoder's `mu` conditioning
/// tensor (channel-first [MelChannels, 2*T]) and projects the reference x-vector into the CFM's
/// speaker-embedding space.
///
/// Thin wrapper over <see cref="S3GenConformerKernels"/> -- the block math (rel-pos self-
/// attention + Swish FFN Conformer layers, PreLookaheadLayer, Upsample1D) is shared with
/// `CosyVoice/CosyVoiceFlowEncoder.cs`, confirmed architecturally identical by real tensor
/// shapes during this session's CosyVoice audit and extracted into the shared kernel once
/// both pipelines had independently-verified implementations to check against each other --
/// see docs/audio-review-progress.md's CosyVoice section. Golden-verified against real
/// PyTorch output (see `Tests.Audio/ChatterboxFlowEncoderTests.cs`); re-run after this
/// extraction to confirm the refactor is behavior-preserving.
///
/// Pipeline (s3gen.py's UpsampleConformerEncoder config: macaron_style=False, use_cnn_module=
/// False, so each ConformerEncoderLayer is just rel-pos self-attention + swish FFN, no macaron
/// FFN and no depthwise conv branch -- see encoder_layer.py):
///   tokenEmb = input_embedding[token]                              [T, 512]
///   x = LayerNorm(Linear(tokenEmb)); x *= sqrt(512); pos_emb = sinusoid[0:T]   (LinearNoSubsampling + RelPositionalEncoding)
///   x = PreLookaheadLayer(x)                                        (conv1 k=4 rightpad3 + leakyrelu, conv2 k=3 leftpad2, + residual)
///   x = 6x ConformerEncoderLayer(x, pos_emb)                        (enc.0..5)
///   x = Upsample1D(x)                                               (nearest x2, leftpad4, conv k=5)  -> [2T, 512]
///   x = LayerNorm(Linear(x)); x *= sqrt(512); pos_emb2 = sinusoid[0:2T]        (up_embed)
///   x = 4x ConformerEncoderLayer(x, pos_emb2)                       (ue.0..3)
///   x = LayerNorm(x)                                                (after_norm)
///   mu = Linear(x) -> [2T, 80], transposed to channel-first [80, 2T]
/// </summary>
public static class ChatterboxFlowEncoder
{
    public static (float[] Mu, int TotalFrames) Forward(
        ChatterboxS3GenWeights w, int[] promptTokens, int[] speechTokens)
    {
        int t = promptTokens.Length + speechTokens.Length;
        var tokenEmb = new float[t][];
        for (int i = 0; i < promptTokens.Length; i++)
            tokenEmb[i] = S3GenConformerKernels.EmbedRow(w.InputEmbeddingWeight, promptTokens[i], w.EncHidden);
        for (int i = 0; i < speechTokens.Length; i++)
            tokenEmb[promptTokens.Length + i] = S3GenConformerKernels.EmbedRow(w.InputEmbeddingWeight, speechTokens[i], w.EncHidden);

        return S3GenConformerKernels.Forward(w, tokenEmb);
    }

    /// <summary>spk_embed_affine_layer: F.normalize(embedding, dim=1) then Linear(SpkEncDim, MelChannels).</summary>
    public static float[] ProjectSpeakerEmbedding(ChatterboxS3GenWeights w, float[] xvector) =>
        S3GenConformerKernels.ProjectSpeakerEmbedding(w.SpkEmbedAffineWeight, w.SpkEmbedAffineBias, w.SpkEncDim, w.MelChannels, xvector);
}
