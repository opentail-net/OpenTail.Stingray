namespace OpenTail.Stingray.Audio.PersonaPlex;

/// <summary>One Depformer layer's real weights, ported from `depformer.cpp`'s
/// `load_depformer_layer` (not guessed): a real PER-STEP-VARYING packed-QKV/gate-up decoder
/// layer -- unlike every other transformer this session, this layer's weight matrices differ
/// at EACH of the 16 real depformer steps within a frame (not shared/reused across positions).
/// `InProjWeight`/`OutProjWeight` are the FULL packed tensors (`[steps*3*hidden, hidden]`/
/// `[steps*hidden, hidden]`); callers slice out one step's `3*hidden`/`hidden` row range at
/// forward time (real reference: `view_linear_rows` at `step*3*hidden`/`step*hidden`
/// offsets). `GateUpWeights`/`DownWeights` are ALREADY separate real per-step tensors in the
/// checkpoint (`gating.{step}.linear_in`/`linear_out`), no slicing needed.</summary>
public sealed class PersonaPlexDepformerLayerWeights
{
    public required float[] Norm1Alpha { get; init; } // [hidden]
    public required float[] InProjWeight { get; init; } // [steps*3*hidden, hidden] row-major
    public required float[] OutProjWeight { get; init; } // [steps*hidden, hidden] row-major
    public required float[] Norm2Alpha { get; init; } // [hidden]
    public required float[][] GateUpWeights { get; init; } // [step][2*intermediate, hidden]
    public required float[][] DownWeights { get; init; } // [step][hidden, intermediate]
}

/// <summary>
/// Real weights for PersonaPlex's Depformer -- a small (6-layer, real config
/// `hidden_size=1024`/`num_attention_heads=16`/`head_dim=64`/`intermediate_size=2816`,
/// `position_encoding=None`) per-frame autoregressive decoder that generates one frame's 16
/// real audio-codebook tokens (`lm.depformer_steps=16=lm.lm_codebooks`), conditioned on the
/// temporal LM's current-frame hidden state via a real per-step `depformer_in.{step}` Linear
/// (`[depformerHidden, lmHidden]`, confirmed real, no bias). Ported from
/// `load_personaplex_depformer_weights` (not guessed): step 0's token input is the CURRENT
/// frame's sampled TEXT token (embedded via `text_embedding`, `[textVocab+1, depformerHidden]`
/// -- the Depformer's OWN text-embedding table, distinct from the main LM's `text_emb.weight`);
/// steps 1-15's token input is the PREVIOUS step's sampled audio-codebook token, each embedded
/// via its OWN real per-step-index table (`AudioEmbeddings[step-1]`, real 15 separate tables --
/// again distinct from the main LM's own 16 `emb.{cb}` tables, this is the Depformer's private
/// set). Each step's output projects through its OWN real per-step LM head (`lm/linears.{step}`,
/// `[depformerHidden -> audioCodebookSize]`, no bias) to get that codebook's logits.
/// </summary>
public sealed class PersonaPlexDepformerWeights
{
    public const int HiddenDim = 1024;
    public const int NumHeads = 16;
    public const int HeadDim = 64;
    public const int FfnDim = 2816;
    public const int NumLayers = 6;
    public const int NumSteps = 16; // == lm.depformer_steps == lm.lm_codebooks
    public const int LmHiddenDim = 4096;
    public const float RmsNormEps = 1e-8f; // shared with the main LM (config.lm.rms_norm_eps)

    public required float[][] InputFromLmWeight { get; init; } // [step][depformerHidden, lmHiddenDim]
    public required float[] TextEmbedding { get; init; } // [textVocab+1, depformerHidden]
    public required float[][] AudioEmbeddings { get; init; } // [step 0..14][audioCodebookSize+1, depformerHidden]
    public required PersonaPlexDepformerLayerWeights[] Layers { get; init; } // [layer]
    public required float[][] Heads { get; init; } // [step][audioCodebookSize, depformerHidden]

    public static PersonaPlexDepformerWeights Load(int textVocabSize, int audioCodebookSize, Func<string, float[]> get)
    {
        var inputFromLm = new float[NumSteps][];
        for (int s = 0; s < NumSteps; s++) inputFromLm[s] = get($"lm/depformer_in.{s}.weight");

        var textEmbedding = get("lm/depformer_text_emb.weight");

        var audioEmbeddings = new float[NumSteps - 1][];
        for (int s = 0; s < NumSteps - 1; s++) audioEmbeddings[s] = get($"lm/depformer_emb.{s}.weight");

        var layers = new PersonaPlexDepformerLayerWeights[NumLayers];
        for (int l = 0; l < NumLayers; l++)
        {
            string p = $"lm/depformer.layers.{l}.";
            var gateUp = new float[NumSteps][];
            var down = new float[NumSteps][];
            for (int s = 0; s < NumSteps; s++)
            {
                gateUp[s] = get($"{p}gating.{s}.linear_in.weight");
                down[s] = get($"{p}gating.{s}.linear_out.weight");
            }
            layers[l] = new PersonaPlexDepformerLayerWeights
            {
                Norm1Alpha = get(p + "norm1.alpha"),
                InProjWeight = get(p + "self_attn.in_proj_weight"),
                OutProjWeight = get(p + "self_attn.out_proj.weight"),
                Norm2Alpha = get(p + "norm2.alpha"),
                GateUpWeights = gateUp,
                DownWeights = down,
            };
        }

        var heads = new float[NumSteps][];
        for (int s = 0; s < NumSteps; s++) heads[s] = get($"lm/linears.{s}.weight");

        return new PersonaPlexDepformerWeights
        {
            InputFromLmWeight = inputFromLm,
            TextEmbedding = textEmbedding,
            AudioEmbeddings = audioEmbeddings,
            Layers = layers,
            Heads = heads,
        };
    }
}
