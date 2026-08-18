using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Diffusion.TextEncoders;

/// <summary>
/// OpenCLIP (ViT-bigG/14) text encoder for SDXL.
/// Supports both native OpenCLIP naming (transformer.resblocks) and HuggingFace naming (text_model.encoder.layers).
/// </summary>
public sealed class OpenClipGEncoder : IDisposable
{
    private const int Layers    = 32;
    private const int Dim       = 1280;
    private const int Heads     = 16;
    private const int HeadDim   = 80;
    private const int MlpDim    = 5120;
    private const int MaxSeqLen = 77;

    private readonly IWeightLoader _st;
    private readonly string _prefix;
    private readonly bool _isOpenClipFormat;

    public OpenClipGEncoder(IWeightLoader st, string prefix = "")
    {
        _st = st;
        _prefix = prefix;
        _isOpenClipFormat = st.Contains($"{prefix}token_embedding.weight") || st.Contains($"{prefix}transformer.resblocks.0.ln_1.weight");
    }

    public (float[] penultimateHidden, float[] pooled) Encode(int[] tokens)
    {
        int seq = MaxSeqLen;
        var ids = new int[seq];
        int copy = Math.Min(tokens.Length, seq);
        Array.Copy(tokens, ids, copy);

        float[] tokEmb, posEmb;
        if (_isOpenClipFormat)
        {
            tokEmb = _st.ReadF32($"{_prefix}token_embedding.weight");
            posEmb = _st.ReadF32($"{_prefix}positional_embedding");
        }
        else
        {
            tokEmb = _st.ReadF32($"{_prefix}embeddings.token_embedding.weight");
            posEmb = _st.ReadF32($"{_prefix}embeddings.position_embedding.weight");
        }

        var x = new float[seq * Dim];
        for (int t = 0; t < seq; t++)
        {
            int tokOff = ids[t] * Dim;
            int posOff = t * Dim;
            int xOff   = t * Dim;
            for (int d = 0; d < Dim; d++)
                x[xOff + d] = tokEmb[tokOff + d] + posEmb[posOff + d];
        }

        var mask = BuildCausalMask(seq);
        float[]? penultimateState = null;

        // 32 encoder layers
        for (int i = 0; i < Layers; i++)
        {
            x = _isOpenClipFormat ? OpenClipLayer(x, mask, seq, i) : HuggingFaceLayer(x, mask, seq, i);
            if (i == Layers - 2) // Layer 30 (penultimate)
            {
                penultimateState = (float[])x.Clone();
            }
        }

        // Final layer norm
        string lnFinalKey = _isOpenClipFormat ? $"{_prefix}ln_final.weight" : $"{_prefix}final_layer_norm.weight";
        string lnFinalBiasKey = _isOpenClipFormat ? $"{_prefix}ln_final.bias" : $"{_prefix}final_layer_norm.bias";
        var lnW = _st.ReadF32(lnFinalKey);
        var lnB = _st.ReadF32(lnFinalBiasKey);
        DiffusionOps.LayerNorm(x, lnW, lnB, Dim);

        // Find EOS token position (49407)
        int eosPos = copy - 1;
        for (int t = 0; t < copy; t++)
        {
            if (ids[t] == 49407) { eosPos = t; break; }
        }

        var eosVector = x.AsSpan(eosPos * Dim, Dim).ToArray();

        // text_projection [1280, 1280]
        float[] pooled;
        string projKey = $"{_prefix}text_projection";
        if (!_st.Contains(projKey)) projKey = $"{_prefix}text_projection.weight";

        if (_st.Contains(projKey))
        {
            var projW = _st.ReadF32(projKey);
            pooled = DiffusionOps.Linear(eosVector, projW, null, 1, Dim, Dim);
        }
        else
        {
            pooled = eosVector;
        }

        return (penultimateState ?? x, pooled);
    }

    private float[] OpenClipLayer(float[] x, float[] mask, int seq, int layerIdx)
    {
        string p = $"{_prefix}transformer.resblocks.{layerIdx}";

        // LN1 + Attention
        var lnW1 = _st.ReadF32($"{p}.ln_1.weight");
        var lnB1 = _st.ReadF32($"{p}.ln_1.bias");
        var xNorm = (float[])x.Clone();
        DiffusionOps.LayerNorm(xNorm, lnW1, lnB1, Dim);

        var inProjW = _st.ReadF32($"{p}.attn.in_proj_weight");
        var inProjB = _st.ReadF32($"{p}.attn.in_proj_bias");
        var outProjW = _st.ReadF32($"{p}.attn.out_proj.weight");
        var outProjB = _st.ReadF32($"{p}.attn.out_proj.bias");

        // in_proj projects to 3 * Dim (Q, K, V)
        var qkv = DiffusionOps.Linear(xNorm, inProjW, inProjB, seq, Dim, 3 * Dim);
        var q = new float[seq * Dim];
        var k = new float[seq * Dim];
        var v = new float[seq * Dim];
        for (int t = 0; t < seq; t++)
        {
            int tOff = t * 3 * Dim;
            Array.Copy(qkv, tOff, q, t * Dim, Dim);
            Array.Copy(qkv, tOff + Dim, k, t * Dim, Dim);
            Array.Copy(qkv, tOff + 2 * Dim, v, t * Dim, Dim);
        }

        float scale = 1f / MathF.Sqrt(HeadDim);
        var attnOut = new float[seq * Dim];

        Parallel.For(0, Heads, h =>
        {
            int hOff = h * HeadDim;
            var scores = new float[seq];

            for (int qi = 0; qi < seq; qi++)
            {
                int qBase = qi * Dim + hOff;
                for (int kj = 0; kj < seq; kj++)
                {
                    int kBase = kj * Dim + hOff;
                    float dot = 0f;
                    for (int d = 0; d < HeadDim; d++)
                        dot += q[qBase + d] * k[kBase + d];
                    scores[kj] = dot * scale + mask[qi * seq + kj];
                }

                DiffusionOps.Softmax(scores, 0, seq);

                int outBase = qi * Dim + hOff;
                for (int d = 0; d < HeadDim; d++)
                {
                    float sum = 0f;
                    for (int kj = 0; kj < seq; kj++)
                        sum += scores[kj] * v[kj * Dim + hOff + d];
                    attnOut[outBase + d] = sum;
                }
            }
        });

        var attnProj = DiffusionOps.Linear(attnOut, outProjW, outProjB, seq, Dim, Dim);
        for (int i = 0; i < x.Length; i++) x[i] += attnProj[i];

        // LN2 + MLP
        var lnW2 = _st.ReadF32($"{p}.ln_2.weight");
        var lnB2 = _st.ReadF32($"{p}.ln_2.bias");
        var xNorm2 = (float[])x.Clone();
        DiffusionOps.LayerNorm(xNorm2, lnW2, lnB2, Dim);

        var cFcW = _st.ReadF32($"{p}.mlp.c_fc.weight");
        var cFcB = _st.ReadF32($"{p}.mlp.c_fc.bias");
        var cProjW = _st.ReadF32($"{p}.mlp.c_proj.weight");
        var cProjB = _st.ReadF32($"{p}.mlp.c_proj.bias");

        var mlpH = DiffusionOps.Linear(xNorm2, cFcW, cFcB, seq, Dim, MlpDim);
        DiffusionOps.GeluInPlace(mlpH);
        var mlpOut = DiffusionOps.Linear(mlpH, cProjW, cProjB, seq, MlpDim, Dim);

        for (int i = 0; i < x.Length; i++) x[i] += mlpOut[i];

        return x;
    }

    private float[] HuggingFaceLayer(float[] x, float[] mask, int seq, int layerIdx)
    {
        string p = $"{_prefix}encoder.layers.{layerIdx}";

        var lnW1 = _st.ReadF32($"{p}.layer_norm1.weight");
        var lnB1 = _st.ReadF32($"{p}.layer_norm1.bias");
        var xNorm = (float[])x.Clone();
        DiffusionOps.LayerNorm(xNorm, lnW1, lnB1, Dim);

        var qW = _st.ReadF32($"{p}.self_attn.q_proj.weight");
        var qB = _st.ReadF32($"{p}.self_attn.q_proj.bias");
        var kW = _st.ReadF32($"{p}.self_attn.k_proj.weight");
        var kB = _st.ReadF32($"{p}.self_attn.k_proj.bias");
        var vW = _st.ReadF32($"{p}.self_attn.v_proj.weight");
        var vB = _st.ReadF32($"{p}.self_attn.v_proj.bias");
        var oW = _st.ReadF32($"{p}.self_attn.out_proj.weight");
        var oB = _st.ReadF32($"{p}.self_attn.out_proj.bias");

        var q = DiffusionOps.Linear(xNorm, qW, qB, seq, Dim, Dim);
        var k = DiffusionOps.Linear(xNorm, kW, kB, seq, Dim, Dim);
        var v = DiffusionOps.Linear(xNorm, vW, vB, seq, Dim, Dim);

        float scale = 1f / MathF.Sqrt(HeadDim);
        var attnOut = new float[seq * Dim];

        Parallel.For(0, Heads, h =>
        {
            int hOff = h * HeadDim;
            var scores = new float[seq];

            for (int qi = 0; qi < seq; qi++)
            {
                int qBase = qi * Dim + hOff;
                for (int kj = 0; kj < seq; kj++)
                {
                    int kBase = kj * Dim + hOff;
                    float dot = 0f;
                    for (int d = 0; d < HeadDim; d++)
                        dot += q[qBase + d] * k[kBase + d];
                    scores[kj] = dot * scale + mask[qi * seq + kj];
                }

                DiffusionOps.Softmax(scores, 0, seq);

                int outBase = qi * Dim + hOff;
                for (int d = 0; d < HeadDim; d++)
                {
                    float sum = 0f;
                    for (int kj = 0; kj < seq; kj++)
                        sum += scores[kj] * v[kj * Dim + hOff + d];
                    attnOut[outBase + d] = sum;
                }
            }
        });

        var attnProj = DiffusionOps.Linear(attnOut, oW, oB, seq, Dim, Dim);
        for (int i = 0; i < x.Length; i++) x[i] += attnProj[i];

        var lnW2 = _st.ReadF32($"{p}.layer_norm2.weight");
        var lnB2 = _st.ReadF32($"{p}.layer_norm2.bias");
        var xNorm2 = (float[])x.Clone();
        DiffusionOps.LayerNorm(xNorm2, lnW2, lnB2, Dim);

        var fc1W = _st.ReadF32($"{p}.mlp.fc1.weight");
        var fc1B = _st.ReadF32($"{p}.mlp.fc1.bias");
        var fc2W = _st.ReadF32($"{p}.mlp.fc2.weight");
        var fc2B = _st.ReadF32($"{p}.mlp.fc2.bias");

        var mlpH = DiffusionOps.Linear(xNorm2, fc1W, fc1B, seq, Dim, MlpDim);
        DiffusionOps.GeluInPlace(mlpH);
        var mlpOut = DiffusionOps.Linear(mlpH, fc2W, fc2B, seq, MlpDim, Dim);

        for (int i = 0; i < x.Length; i++) x[i] += mlpOut[i];

        return x;
    }

    private static float[] BuildCausalMask(int seq)
    {
        var mask = new float[seq * seq];
        for (int i = 0; i < seq; i++)
        for (int j = 0; j < seq; j++)
            mask[i * seq + j] = j > i ? float.NegativeInfinity : 0f;
        return mask;
    }

    public void Dispose() => _st.Dispose();
}
