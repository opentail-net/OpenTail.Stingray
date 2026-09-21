
namespace OpenTail.Stingray.Diffusion.TextEncoders;

/// <summary>
/// CLIP-L (ViT-L/14) text encoder.
/// Produces:
///   - penultimate_hidden_state [seq, 768] — real SD3/3.5 joint-context conditioning (see below)
///   - pooled_output [768]                 — raw, UNPROJECTED [EOS] token embedding
///
/// Loads weights from clip_l.safetensors or unified SD 1.5 checkpoints.
/// Architecture: 12 transformer encoder layers, 768-dim, 12 heads, head_dim=64.
/// Activation: QuickGELU (x * sigmoid(1.702*x)).
///
/// <para><b>Real bug found and fixed 2026-09-21</b> (SD3.5 composition/left-shift investigation):
/// this class previously returned the FINAL hidden state (post-final-LayerNorm, after all 12
/// layers) as its "last_hidden_state" output, but the real SD3 reference pipeline (`encode_prompt`
/// in `pipeline_stable_diffusion_3.py`, confirmed independently against `examples/
/// stable-diffusion.cpp`'s own `CLIPEncoder::forward` -- `clip_skip` defaults to 2, and
/// `layer_idx = n_layer - clip_skip` with a loop that breaks at `i == layer_idx + 1`, i.e.
/// processes layers 0..10 and stops, never reaching layer 11 or a final LayerNorm) requests the
/// PENULTIMATE hidden state -- exactly the convention <see cref="OpenClipGEncoder"/> already
/// implemented correctly for CLIP-G, creating a silent asymmetry between the two encoders feeding
/// the same joint context tensor. Fixed to match.</para>
///
/// <para><b>A second suspected bug was found, then RETRACTED same day</b> after checking the real
/// reference source directly (CLAUDE.md rule 8): this checkpoint file DOES physically contain a
/// `text_projection.weight` tensor, which looked like it should parallel CLIP-G's own (correct)
/// projected-pooled-output pattern. But `examples/stable-diffusion.cpp/src/model/te/clip.hpp`'s
/// `CLIPTextModel::init_params` only ever registers that param `if (version ==
/// OPEN_CLIP_VIT_BIGG_14)` -- CLIP-G ONLY -- so the reference's CLIP-L graph never wires that
/// tensor in regardless of whether a given checkpoint file happens to contain it, and always takes
/// the explicit `LOG_DEBUG("identity projection")` fallback. Applying it anyway was measurably
/// wrong on this session's own noise-injection regression test (this pooled vector feeds AdaLN
/// modulation in every block, so an incorrect transform here has whole-model impact) -- reverted;
/// this class's pooled output stays the raw, unprojected EOS vector, matching the reference.</para>
/// </summary>
public sealed class ClipLEncoder : IDisposable
{
    private const int Layers    = 12;
    private const int Dim       = 768;
    private const int Heads     = 12;
    private const int HeadDim   = 64;
    private const int MlpDim    = 3072;
    private const int MaxSeqLen = 77;
    private const int VocabSize = 49408;

    private readonly IWeightLoader _st;
    // Perf (2026-09-11): _st.ReadF32 does a real file.Seek+ReadExactly disk read under a lock on
    // EVERY call, no caching -- found while chasing the ~6.2x text-encode gap vs a real
    // stable-diffusion.cpp reference (SdxlPipeline calls Encode() twice per generation, cond+
    // uncond, re-reading the exact same ~120 weight tensors from disk both times with zero reuse).
    // Same pattern as CachedWeightReader used elsewhere in this codebase.
    private readonly Dictionary<string, float[]> _weightCache = new(StringComparer.Ordinal);
    private float[] Wt(string name)
    {
        if (_weightCache.TryGetValue(name, out var w)) return w;
        w = _st.ReadF32(name);
        _weightCache[name] = w;
        return w;
    }

    public ClipLEncoder(string path) => _st = SafetensorsLoader.Open(path);
    public ClipLEncoder(IWeightLoader st) => _st = st;

    /// <summary>
    /// Encode a list of token ids (length ≤ 77, padded to 77 with 0).
    /// Returns (finalHidden [77,768], penultimateHidden [77,768], projectedPooledOutput [768]).
    /// Both hidden states are real, distinct consumer requirements -- NOT a redundant pair: SD1.5's
    /// plain <c>CLIPTextModel</c> cross-attention conditioning uses the standard FINAL (post-final-
    /// LayerNorm) hidden state, while SD3/3.5's joint context tensor specifically requires the
    /// PENULTIMATE state (see this class's own doc comment). Callers pick whichever their real
    /// reference pipeline uses -- do not assume one is "the" hidden state for every caller.
    /// </summary>
    public (float[] finalHidden, float[] penultimateHidden, float[] pooled) Encode(int[] tokens)
    {
        int seq = MaxSeqLen;
        // Pad/truncate to MaxSeqLen
        var ids = new int[seq];
        int copy = Math.Min(tokens.Length, seq);
        Array.Copy(tokens, ids, copy);

        // Token + position embeddings
        var tokEmb = Wt("text_model.embeddings.token_embedding.weight");
        var posEmb = Wt("text_model.embeddings.position_embedding.weight");

        var x = new float[seq * Dim];
        for (int t = 0; t < seq; t++)
        {
            int tokOff = ids[t] * Dim;
            int posOff = t * Dim;
            int xOff   = t * Dim;
            for (int d = 0; d < Dim; d++)
                x[xOff + d] = tokEmb[tokOff + d] + posEmb[posOff + d];
        }

        // Build causal mask (CLIP uses causal masking like GPT)
        var mask = BuildCausalMask(seq);

        // 12 encoder layers -- capture the PENULTIMATE state (after layer Layers-2, i.e. skipping
        // only the final layer and the final LayerNorm) the same way OpenClipGEncoder already does,
        // matching the real reference's `hidden_states[-2]` convention.
        float[]? penultimateState = null;
        for (int i = 0; i < Layers; i++)
        {
            x = EncoderLayer(x, mask, seq, i);
            if (i == Layers - 2)
                penultimateState = (float[])x.Clone();
        }

        // Pooled = EOS token (last non-pad token, or last position) after the FULL stack including
        // the final layer + final LayerNorm -- CLIPTextModel's pooler_output is always taken from
        // the final hidden state, unlike the penultimate convention used for last_hidden_state above.
        var lnW = Wt("text_model.final_layer_norm.weight");
        var lnB = Wt("text_model.final_layer_norm.bias");
        DiffusionOps.LayerNorm(x, lnW, lnB, Dim);

        // CLIP uses the EOS token embedding at position of the EOT (49407)
        int eosPos = copy - 1;
        for (int t = 0; t < copy; t++)
            if (ids[t] == 49407) { eosPos = t; break; }

        // Real bug found 2026-09-21, then RETRACTED same day after checking the real reference
        // directly (`examples/stable-diffusion.cpp/src/model/te/clip.hpp`'s `CLIPTextModel::
        // init_params`/`forward`, per CLAUDE.md rule 8): a `text_projection.weight` tensor IS
        // physically present in this checkpoint file, and it first looked like CLIP-G's own
        // already-correct projection pattern should apply here too. But the reference's own
        // `init_params` only ever registers `params["text_projection"]` `if (version ==
        // OPEN_CLIP_VIT_BIGG_14)` -- i.e. CLIP-G ONLY. For CLIP-L (`OPENAI_CLIP_VIT_L_14`), that
        // param is never wired into the graph at all, so `forward`'s own `if (text_projection !=
        // nullptr)` check is always false and it falls through to the explicit `LOG_DEBUG("identity
        // projection")` branch -- the reference NEVER projects CLIP-L's pooled output, regardless of
        // what tensors happen to exist in a given checkpoint file. Applying the projection here was
        // measurably wrong: it visibly corrupted output on this session's own noise-injection
        // regression test (this pooled vector feeds AdaLN modulation in every block, so an
        // unintended transform here has whole-model impact). Left as the raw, unprojected EOS
        // vector, matching the reference exactly.
        var pooled = x.AsSpan(eosPos * Dim, Dim).ToArray();

        return (x, penultimateState ?? x, pooled);
    }

    private float[] EncoderLayer(float[] x, float[] mask, int seq, int layerIdx)
    {
        string p = $"text_model.encoder.layers.{layerIdx}";

        // Self-attention with residual
        var lnW1 = Wt($"{p}.layer_norm1.weight");
        var lnB1 = Wt($"{p}.layer_norm1.bias");
        var xNorm = x.ToArray();
        DiffusionOps.LayerNorm(xNorm, lnW1, lnB1, Dim);

        var attn = SelfAttention(xNorm, mask, seq, p);
        for (int i = 0; i < x.Length; i++) x[i] += attn[i];

        // MLP with residual
        var lnW2 = Wt($"{p}.layer_norm2.weight");
        var lnB2 = Wt($"{p}.layer_norm2.bias");
        var xNorm2 = x.ToArray();
        DiffusionOps.LayerNorm(xNorm2, lnW2, lnB2, Dim);

        var mlp = Mlp(xNorm2, seq, p);
        for (int i = 0; i < x.Length; i++) x[i] += mlp[i];

        return x;
    }

    private float[] SelfAttention(float[] x, float[] mask, int seq, string p)
    {
        var qW = Wt($"{p}.self_attn.q_proj.weight");
        var kW = Wt($"{p}.self_attn.k_proj.weight");
        var vW = Wt($"{p}.self_attn.v_proj.weight");
        var qB = Wt($"{p}.self_attn.q_proj.bias");
        var kB = Wt($"{p}.self_attn.k_proj.bias");
        var vB = Wt($"{p}.self_attn.v_proj.bias");
        var oW = Wt($"{p}.self_attn.out_proj.weight");
        var oB = Wt($"{p}.self_attn.out_proj.bias");

        var q = DiffusionOps.Linear(x, qW, qB, seq, Dim, Dim);  // [seq, Dim]
        var k = DiffusionOps.Linear(x, kW, kB, seq, Dim, Dim);
        var v = DiffusionOps.Linear(x, vW, vB, seq, Dim, Dim);

        float scale = 1f / MathF.Sqrt(HeadDim);
        var attnOut = new float[seq * Dim];

        for (int h = 0; h < Heads; h++)
        {
            var scores = new float[seq * seq];
            for (int i = 0; i < seq; i++)
            {
                for (int j = 0; j < seq; j++)
                {
                    float dot = 0f;
                    int qOff = i * Dim + h * HeadDim;
                    int kOff = j * Dim + h * HeadDim;
                    for (int d = 0; d < HeadDim; d++) dot += q[qOff + d] * k[kOff + d];
                    scores[i * seq + j] = dot * scale + mask[i * seq + j];
                }
            }
            DiffusionOps.Softmax(scores, seq);

            for (int i = 0; i < seq; i++)
            {
                int outOff = i * Dim + h * HeadDim;
                for (int j = 0; j < seq; j++)
                {
                    float w = scores[i * seq + j];
                    int vOff = j * Dim + h * HeadDim;
                    for (int d = 0; d < HeadDim; d++) attnOut[outOff + d] += w * v[vOff + d];
                }
            }
        }

        return DiffusionOps.Linear(attnOut, oW, oB, seq, Dim, Dim);
    }

    private float[] Mlp(float[] x, int seq, string p)
    {
        var fc1W = Wt($"{p}.mlp.fc1.weight");
        var fc1B = Wt($"{p}.mlp.fc1.bias");
        var fc2W = Wt($"{p}.mlp.fc2.weight");
        var fc2B = Wt($"{p}.mlp.fc2.bias");

        var h = DiffusionOps.Linear(x, fc1W, fc1B, seq, Dim, MlpDim);
        // QuickGELU activation: x * sigmoid(1.702 * x)
        for (int i = 0; i < h.Length; i++)
            h[i] = h[i] * (1f / (1f + MathF.Exp(-1.702f * h[i])));
        return DiffusionOps.Linear(h, fc2W, fc2B, seq, MlpDim, Dim);
    }

    private static float[] BuildCausalMask(int seq)
    {
        var mask = new float[seq * seq];
        for (int i = 0; i < seq; i++)
            for (int j = i + 1; j < seq; j++)
                mask[i * seq + j] = float.NegativeInfinity;
        return mask;
    }

    public void Dispose() => _st.Dispose();
}
