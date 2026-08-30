
namespace OpenTail.Stingray.Audio.Xtts;

/// <summary>
/// Real XTTS-v2 GPT token/positional embeddings and output heads (`gpt.text_embedding`/
/// `gpt.text_pos_embedding`/`gpt.mel_embedding`/`gpt.mel_pos_embedding`/`gpt.text_head`/
/// `gpt.mel_head`/`gpt.final_norm`) -- real, plain `nn.Embedding`/`nn.Linear`/`nn.LayerNorm`
/// lookups; `text_pos_embedding`/`mel_pos_embedding` are `LearnedPositionEmbeddings` (absolute,
/// `relative=False` -- confirmed from `TTS/tts/layers/tortoise/autoregressive.py`), simple
/// `emb[position]` lookup for position `0..seqLen-1` within each modality's OWN subsequence
/// (text positions are independent of mel positions, and vice versa).
///
/// <para>Real special token ids (confirmed from `TTS/tts/layers/xtts/gpt.py`'s `GPT.__init__`
/// defaults, cross-checked against which ones the real checkpoint's `config.json` actually
/// overrides): <c>start_text_token=261, stop_text_token=0</c> (class defaults, NOT overridden by
/// this checkpoint's config), <c>start_audio_token=1024, stop_audio_token=1025</c> (this
/// checkpoint's config.json DOES override the class defaults of 8192/8193 -- matches
/// `gpt_start_audio_token`/`gpt_stop_audio_token` in the real config.json).</para>
/// </summary>
public sealed class XttsGptEmbeddings
{
    public const int TextStartToken = 261;
    public const int TextStopToken = 0;
    public const int AudioStartToken = 1024;
    public const int AudioStopToken = 1025;

    public const int NumTextTokens = 6681;
    public const int NumAudioTokens = 1026;

    public float[] TextEmbeddingWeight { get; } // [NumTextTokens, ModelDim]
    public float[] TextPosEmbeddingWeight { get; } // [404, ModelDim]
    public float[] MelEmbeddingWeight { get; } // [NumAudioTokens, ModelDim]
    public float[] MelPosEmbeddingWeight { get; } // [608, ModelDim]

    public float[] TextHeadWeight { get; }
    public float[] TextHeadBias { get; }
    public float[] MelHeadWeight { get; }
    public float[] MelHeadBias { get; }

    public float[] FinalNormWeight { get; }
    public float[] FinalNormBias { get; }

    public XttsGptEmbeddings(SafetensorsLoader loader)
    {
        TextEmbeddingWeight = loader.ReadF32("gpt.text_embedding.weight");
        TextPosEmbeddingWeight = loader.ReadF32("gpt.text_pos_embedding.emb.weight");
        MelEmbeddingWeight = loader.ReadF32("gpt.mel_embedding.weight");
        MelPosEmbeddingWeight = loader.ReadF32("gpt.mel_pos_embedding.emb.weight");

        TextHeadWeight = loader.ReadF32("gpt.text_head.weight");
        TextHeadBias = loader.ReadF32("gpt.text_head.bias");
        MelHeadWeight = loader.ReadF32("gpt.mel_head.weight");
        MelHeadBias = loader.ReadF32("gpt.mel_head.bias");

        FinalNormWeight = loader.ReadF32("gpt.final_norm.weight");
        FinalNormBias = loader.ReadF32("gpt.final_norm.bias");
    }

    /// <summary>Real `text_embedding(ids) + text_pos_embedding(ids)` (absolute positions 0..ids.Length-1). Returns token-major [T, ModelDim].</summary>
    public float[] EmbedText(ReadOnlySpan<int> ids) => EmbedTokenMajor(ids, TextEmbeddingWeight, TextPosEmbeddingWeight);

    /// <summary>Real `mel_embedding(ids) + mel_pos_embedding(ids)` (absolute positions 0..ids.Length-1, independent of any text-side position count). Returns token-major [T, ModelDim].</summary>
    public float[] EmbedMel(ReadOnlySpan<int> ids) => EmbedTokenMajor(ids, MelEmbeddingWeight, MelPosEmbeddingWeight);

    private static float[] EmbedTokenMajor(ReadOnlySpan<int> ids, float[] tokenTable, float[] posTable)
    {
        int dim = XttsGptWeights.ModelDim;
        var output = new float[ids.Length * dim];
        for (int ti = 0; ti < ids.Length; ti++)
        {
            int tokBase = ids[ti] * dim;
            int posBase = ti * dim;
            int outBase = ti * dim;
            for (int d = 0; d < dim; d++)
                output[outBase + d] = tokenTable[tokBase + d] + posTable[posBase + d];
        }
        return output;
    }

    /// <summary>Real `mel_head(final_norm(lastHidden))`. lastHidden is a single [ModelDim] vector (the trunk's final position). Returns [NumAudioTokens] logits.</summary>
    public float[] MelLogits(float[] lastHidden)
    {
        var normed = LayerNorm(lastHidden, FinalNormWeight, FinalNormBias);
        return Linear(normed, MelHeadWeight, MelHeadBias, NumAudioTokens);
    }

    /// <summary>Real `text_head(final_norm(lastHidden))`.</summary>
    public float[] TextLogits(float[] lastHidden)
    {
        var normed = LayerNorm(lastHidden, FinalNormWeight, FinalNormBias);
        return Linear(normed, TextHeadWeight, TextHeadBias, NumTextTokens);
    }

    /// <summary>The real separate `gpt.final_norm` alone (no head projection) -- used by <see cref="XttsGptLatents"/> to extract real vocoder-input hidden states.</summary>
    public float[] FinalNormOnly(float[] hidden) => LayerNorm(hidden, FinalNormWeight, FinalNormBias);

    private static float[] LayerNorm(float[] x, float[] gamma, float[] beta, float eps = 1e-5f)
    {
        int dim = x.Length;
        double mean = 0;
        for (int i = 0; i < dim; i++) mean += x[i];
        mean /= dim;
        double var = 0;
        for (int i = 0; i < dim; i++) { double d = x[i] - mean; var += d * d; }
        var /= dim;
        float invStd = (float)(1.0 / Math.Sqrt(var + eps));
        var output = new float[dim];
        for (int i = 0; i < dim; i++)
            output[i] = (float)((x[i] - mean) * invStd) * gamma[i] + beta[i];
        return output;
    }

    private static float[] Linear(float[] x, float[] weight, float[] bias, int outDim)
    {
        int inDim = x.Length;
        var output = new float[outDim];
        for (int o = 0; o < outDim; o++)
        {
            float sum = bias[o];
            int wBase = o * inDim;
            for (int i = 0; i < inDim; i++) sum += weight[wBase + i] * x[i];
            output[o] = sum;
        }
        return output;
    }
}
