namespace OpenTail.Stingray.Audio.VoxCpm2;

/// <summary>
/// Native C# port of VoxCPM2's "residual LM" -- a real, SEPARATE small (8-layer) MiniCPM-style
/// CAUSAL autoregressive decoder, from `minicpm.cpp`'s `residual_lm_config`/`minicpm_layer`
/// (`is_causal=true` for this model, unlike the local encoder/DiT decoder's bidirectional use of
/// the same per-layer math) and `minicpm_blocks.h` (not guessed). Real config: same
/// `hidden_size=2048`/`num_attention_heads=16`/`num_key_value_heads=2`/`head_dim=128`/
/// `intermediate_size=6144` as `base_lm` (inherited via `residual_lm_config`), but only 8 layers
/// and `no_rope=true` (RoPE completely disabled -- confirmed via `apply_minicpm_rope`'s real
/// `if (config.no_rope) return input;` early-out). No token embedding or LM head at all (real
/// tensor dump confirms neither `embed_tokens` nor `lm_head` exist for `residual_lm` -- this
/// model NEVER does a vocabulary lookup or computes logits, it only ever consumes precomputed
/// embeddings and returns hidden states, exactly like `base_lm`'s own per-step
/// `ForwardEmbedding` calls -- but unlike `base_lm`, `ForwardPass`'s generic `minicpm`-
/// architecture graph cannot be reused here because it has no way to disable RoPE per
/// checkpoint, so this is a genuinely bespoke, from-scratch causal decoder with its own real
/// persistent per-step KV cache (`use_mup=false` inherited too, so no residual/embedding
/// scaling anywhere, same as `base_lm`).
/// </summary>
public sealed class VoxCpm2ResidualLm
{
    public const int HiddenDim = 2048;
    public const int NumHeads = 16;
    public const int NumKvHeads = 2;
    public const int HeadDim = 128;
    public const int FfnDim = 6144;
    public const int NumLayers = 8;
    public const float RmsNormEps = 1e-5f;

    private readonly VoxCpm2MiniCpmLayerWeights[] _layers;
    private readonly float[] _finalNorm;
    private readonly List<float[]>[] _keyCache;
    private readonly List<float[]>[] _valueCache;

    public VoxCpm2ResidualLm(VoxCpm2MiniCpmLayerWeights[] layers, float[] finalNorm)
    {
        if (layers.Length != NumLayers) throw new ArgumentException($"Expected {NumLayers} layers.", nameof(layers));
        _layers = layers;
        _finalNorm = finalNorm;
        _keyCache = new List<float[]>[NumLayers];
        _valueCache = new List<float[]>[NumLayers];
        Reset();
    }

    public static VoxCpm2ResidualLm Load(Func<string, float[]> get)
    {
        var layers = new VoxCpm2MiniCpmLayerWeights[NumLayers];
        for (int i = 0; i < NumLayers; i++)
        {
            string p = $"weights/residual_lm.layers.{i}.";
            layers[i] = new VoxCpm2MiniCpmLayerWeights
            {
                InputNorm = get(p + "input_layernorm.weight"),
                QProjWeight = get(p + "self_attn.q_proj.weight"),
                KProjWeight = get(p + "self_attn.k_proj.weight"),
                VProjWeight = get(p + "self_attn.v_proj.weight"),
                OProjWeight = get(p + "self_attn.o_proj.weight"),
                PostNorm = get(p + "post_attention_layernorm.weight"),
                GateProjWeight = get(p + "mlp.gate_proj.weight"),
                UpProjWeight = get(p + "mlp.up_proj.weight"),
                DownProjWeight = get(p + "mlp.down_proj.weight"),
            };
        }
        var finalNorm = get("weights/residual_lm.norm.weight");
        return new VoxCpm2ResidualLm(layers, finalNorm);
    }

    /// <summary>Clears the KV cache (real `.reset()` before a fresh generation).</summary>
    public void Reset()
    {
        for (int l = 0; l < NumLayers; l++)
        {
            _keyCache[l] = [];
            _valueCache[l] = [];
        }
    }

    /// <summary>Appends one embedding as the next causal position and returns the post-final-norm
    /// hidden state at that position (real `run_step(...).hidden`). No RoPE applied anywhere
    /// (`no_rope=true` for this model).</summary>
    public float[] Step(float[] embedding)
    {
        if (embedding.Length != HiddenDim) throw new ArgumentException($"Expected length {HiddenDim}.", nameof(embedding));

        var hidden = embedding;
        int kvRepeats = NumHeads / NumKvHeads;
        float scale = 1f / MathF.Sqrt(HeadDim);
        int qOut = NumHeads * HeadDim;
        int kvOut = NumKvHeads * HeadDim;

        for (int l = 0; l < NumLayers; l++)
        {
            var layer = _layers[l];
            var normed = RmsNorm(hidden, layer.InputNorm);
            var q = Linear(normed, layer.QProjWeight, qOut);
            var k = Linear(normed, layer.KProjWeight, kvOut);
            var v = Linear(normed, layer.VProjWeight, kvOut);

            _keyCache[l].Add(k);
            _valueCache[l].Add(v);
            int availableKeys = _keyCache[l].Count;

            var context = new float[qOut];
            for (int h = 0; h < NumHeads; h++)
            {
                int hOff = h * HeadDim;
                int kvHOff = (h / kvRepeats) * HeadDim;
                var scores = new float[availableKeys];
                for (int t = 0; t < availableKeys; t++)
                {
                    float dot = 0f;
                    var kt = _keyCache[l][t];
                    for (int d = 0; d < HeadDim; d++) dot += q[hOff + d] * kt[kvHOff + d];
                    scores[t] = dot * scale;
                }
                Softmax(scores);
                for (int t = 0; t < availableKeys; t++)
                {
                    float p = scores[t];
                    var vt = _valueCache[l][t];
                    for (int d = 0; d < HeadDim; d++) context[hOff + d] += p * vt[kvHOff + d];
                }
            }

            var attnOut = Linear(context, layer.OProjWeight, HiddenDim);
            var x = Add(hidden, attnOut);

            var ffnNormed = RmsNorm(x, layer.PostNorm);
            var gate = Linear(ffnNormed, layer.GateProjWeight, FfnDim);
            SiluInPlace(gate);
            var up = Linear(ffnNormed, layer.UpProjWeight, FfnDim);
            for (int i = 0; i < gate.Length; i++) gate[i] *= up[i];
            var down = Linear(gate, layer.DownProjWeight, HiddenDim);
            hidden = Add(x, down);
        }

        return RmsNorm(hidden, _finalNorm);
    }

    private static float[] RmsNorm(float[] x, float[] weight)
    {
        double sumSq = 0;
        for (int i = 0; i < x.Length; i++) sumSq += (double)x[i] * x[i];
        float invRms = (float)(1.0 / Math.Sqrt(sumSq / x.Length + RmsNormEps));
        var output = new float[x.Length];
        for (int i = 0; i < x.Length; i++) output[i] = x[i] * invRms * weight[i];
        return output;
    }

    private static float[] Linear(float[] input, float[] weight, int outDim)
    {
        int inDim = input.Length;
        var output = new float[outDim];
        for (int o = 0; o < outDim; o++)
        {
            float sum = 0f;
            int wBase = o * inDim;
            for (int i = 0; i < inDim; i++) sum += weight[wBase + i] * input[i];
            output[o] = sum;
        }
        return output;
    }

    private static float[] Add(float[] a, float[] b)
    {
        var output = new float[a.Length];
        for (int i = 0; i < a.Length; i++) output[i] = a[i] + b[i];
        return output;
    }

    private static void Softmax(float[] scores)
    {
        float max = float.NegativeInfinity;
        for (int i = 0; i < scores.Length; i++) if (scores[i] > max) max = scores[i];
        float sum = 0f;
        for (int i = 0; i < scores.Length; i++)
        {
            float e = MathF.Exp(scores[i] - max);
            scores[i] = e;
            sum += e;
        }
        float invSum = 1f / sum;
        for (int i = 0; i < scores.Length; i++) scores[i] *= invSum;
    }

    private static void SiluInPlace(float[] x)
    {
        for (int i = 0; i < x.Length; i++)
        {
            float v = x[i];
            x[i] = v / (1f + MathF.Exp(-v));
        }
    }
}
