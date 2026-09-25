using System.Numerics.Tensors;

namespace OpenTail.Stingray.Engine.Encoders;

/// <summary>
/// ELECTRA discriminator (<c>ElectraForPreTraining</c>): the BERT-shaped encoder plus HF
/// <c>ElectraDiscriminatorPredictions</c> per token, <c>dense_prediction(gelu(dense(h)))</c>. A positive logit
/// means the token is predicted to be replaced (<c>round(sigmoid(logit))</c> in the model card's example).
/// Needs <c>model.safetensors</c>: google/electra-base-discriminator's main branch only ships
/// pytorch/tf/flax/rust weights; the Hugging Face safetensors-convert bot's PR (<c>refs/pr/6</c>) provides it.
/// With <c>embedding_size == hidden_size</c> HF builds no <c>embeddings_project</c>, so that stray tensor in the
/// converted file is ignored here too.
/// </summary>
public sealed class ElectraDiscriminator : IDisposable
{
    private readonly TransformerEncoder _encoder;
    private readonly PackedLinearF32 _dense;
    private readonly float[] _predW;
    private readonly float _predB;

    public EncoderTokenizer Tokenizer { get; }

    private ElectraDiscriminator(TransformerEncoder encoder, EncoderTokenizer tokenizer, PackedLinearF32 dense, float[] predW, float predB)
    {
        (_encoder, Tokenizer, _dense, _predW, _predB) = (encoder, tokenizer, dense, predW, predB);
    }

    public static ElectraDiscriminator Load(string modelDir)
    {
        var config = EncoderConfig.FromFile(Path.Combine(modelDir, "config.json"));
        if (config.ModelType != "electra")
            throw new InvalidDataException($"'{modelDir}' is not an ELECTRA checkpoint.");
        using var st = SafetensorsLoader.OpenDirectory(modelDir);
        var encoder = TransformerEncoder.Load(config, st);
        int h = config.HiddenSize;
        var dense = new PackedLinearF32(st.ReadF32("discriminator_predictions.dense.weight"), st.ReadF32("discriminator_predictions.dense.bias"), h, h);
        var predW = st.ReadF32("discriminator_predictions.dense_prediction.weight");
        float predB = st.ReadF32("discriminator_predictions.dense_prediction.bias")[0];
        var tokenizer = EncoderTokenizer.FromTokenizerJson(Path.Combine(modelDir, "tokenizer.json"));
        return new ElectraDiscriminator(encoder, tokenizer, dense, predW, predB);
    }

    /// <summary>Replaced-token logit per token of <paramref name="input"/> (special tokens included).</summary>
    public float[] ReplacedTokenLogits(EncodedInput input)
    {
        int h = _encoder.Config.HiddenSize, n = input.Ids.Length;
        var hidden = _encoder.Encode(input);
        var mid = new float[n * h];
        _dense.Forward(hidden, mid, n);
        var logits = new float[n];
        for (int t = 0; t < n; t++)
        {
            var row = mid.AsSpan(t * h, h);
            ErfGelu.InPlace(row);
            logits[t] = TensorPrimitives.Dot(row, _predW) + _predB;
        }
        return logits;
    }

    public void Dispose()
    {
        _encoder.Dispose();
        _dense.Dispose();
    }
}
