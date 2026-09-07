using OpenTail.Stingray.Audio.Rvc;

namespace OpenTail.Stingray.Audio.MossTts;

/// <summary>
/// Real weight loader for MOSS-TTS-Nano's "local transformer" -- a single-layer GPT2+RoPE block
/// (same architecture family as <see cref="MossTtsGlobalTransformerWeights"/>, real config:
/// `local_transformer_layers=1` in the checkpoint's `config.json`, same hidden/heads/rope_base as
/// the global transformer) that autoregressively decodes each output frame's 16 RVQ audio-codebook
/// tokens, conditioned on that frame's global-transformer hidden state -- ported from
/// `examples/audio.cpp/src/models/moss/moss_tts_nano/local_frame_decoder.cpp` (not guessed). Real
/// prefix: `model_weights/local_transformer.*`. The text/audio embedding tables and LM heads are
/// SHARED with the global transformer's tensors (`model_weights/transformer.wte.weight`,
/// `model_weights/text_lm_head.weight`, `model_weights/audio_embeddings.{q}.weight`) -- only the
/// per-codebook `model_weights/audio_lm_heads.{q}.weight` output heads are new here.
/// </summary>
public sealed class MossTtsLocalTransformerWeights
{
    public const int NumLayers = 1;

    public MossTtsGlobalTransformerLayerWeights[] Layers { get; } = new MossTtsGlobalTransformerLayerWeights[NumLayers];
    public float[] FinalNormWeight { get; }
    public float[] FinalNormBias { get; }
    public float[][] AudioLmHeads { get; } // [NumCodebooks][AudioCodebookSize, HiddenDim]

    public MossTtsLocalTransformerWeights(RvcPackedTensorSource source) : this(source.GetTensor)
    {
    }

    /// <summary>Test-friendly overload taking a raw name-&gt;tensor getter.</summary>
    public MossTtsLocalTransformerWeights(Func<string, float[]> get)
    {
        for (int l = 0; l < NumLayers; l++)
        {
            string p = $"model_weights/local_transformer.h.{l}";
            Layers[l] = new MossTtsGlobalTransformerLayerWeights
            {
                Ln1Weight = get($"{p}.ln_1.weight"),
                Ln1Bias = get($"{p}.ln_1.bias"),
                CAttnWeight = get($"{p}.attn.c_attn.weight"),
                CAttnBias = get($"{p}.attn.c_attn.bias"),
                CProjWeight = get($"{p}.attn.c_proj.weight"),
                CProjBias = get($"{p}.attn.c_proj.bias"),
                Ln2Weight = get($"{p}.ln_2.weight"),
                Ln2Bias = get($"{p}.ln_2.bias"),
                FcInWeight = get($"{p}.mlp.fc_in.weight"),
                FcInBias = get($"{p}.mlp.fc_in.bias"),
                FcOutWeight = get($"{p}.mlp.fc_out.weight"),
                FcOutBias = get($"{p}.mlp.fc_out.bias"),
            };
        }

        FinalNormWeight = get("model_weights/local_transformer.ln_f.weight");
        FinalNormBias = get("model_weights/local_transformer.ln_f.bias");

        AudioLmHeads = new float[MossTtsGlobalTransformerWeights.NumCodebooks][];
        for (int q = 0; q < MossTtsGlobalTransformerWeights.NumCodebooks; q++)
            AudioLmHeads[q] = get($"model_weights/audio_lm_heads.{q}.weight");
    }
}
