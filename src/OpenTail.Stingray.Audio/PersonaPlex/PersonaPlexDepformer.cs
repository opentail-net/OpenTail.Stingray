using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Audio.PersonaPlex;

/// <summary>
/// Real per-frame Depformer forward pass for PersonaPlex, ported from `depformer.cpp`'s
/// `PersonaPlexDepformerRuntime::Impl` step-graph construction (not guessed): generates one
/// frame's real 16 audio-codebook tokens autoregressively. Step 0's input is
/// `depformer_in[0](temporalLmHidden) + textEmbedding[promptTextToken]`; steps 1-15's input is
/// `depformer_in[step](temporalLmHidden) + audioEmbeddings[step-1][previousStepToken]` (real
/// additive fusion, SAME `temporalLmHidden` re-projected fresh at every step via that step's OWN
/// `depformer_in` weight -- not carried/accumulated across steps). Each step runs through
/// `NumLayers=6` causal, NO-RoPE (`position_encoding=None`) decoder layers using that step's own
/// per-step-sliced packed QKV/gate-up weight bank, with a real KV cache spanning all PRIOR steps
/// within this same frame (freshly reset at the start of each new frame -- a real distinct scope
/// from the temporal LM's own persistent across-frame cache). Each step's final hidden state
/// projects through its OWN real per-step LM head to that codebook's logits, argmax-sampled
/// (real sampling beyond argmax not ported, matching this session's other codebook-sampler
/// stand-ins).
/// </summary>
public sealed class PersonaPlexDepformer
{
    private readonly PersonaPlexDepformerWeights _w;
    private readonly List<float[]>[] _keyCache;
    private readonly List<float[]>[] _valueCache;

    public PersonaPlexDepformer(PersonaPlexDepformerWeights weights)
    {
        _w = weights;
        _keyCache = new List<float[]>[PersonaPlexDepformerWeights.NumLayers];
        _valueCache = new List<float[]>[PersonaPlexDepformerWeights.NumLayers];
    }

    /// <summary>Resets the per-frame KV cache. Call once before generating each new frame's 16
    /// codebook tokens.</summary>
    public void ResetFrame()
    {
        for (int l = 0; l < PersonaPlexDepformerWeights.NumLayers; l++)
        {
            _keyCache[l] = [];
            _valueCache[l] = [];
        }
    }

    /// <summary>Generates all 16 real codebook tokens for one frame. Real reference feeds
    /// `promptTextToken` (the frame's own sampled TEXT token from the temporal LM) as step 0's
    /// token input.
    ///
    /// <para><b>Update, 2026-09-07</b>: real temperature/top-k sampling is now wired via an
    /// optional <see cref="SamplingParams"/>, reusing `OpenTail.Stingray.Engine.Sampler` -- the
    /// reference's own `sampler.cpp` uses `sampling::HfSampler` with `top_p` FIXED at `1.0`
    /// (confirmed via `depformer.cpp`'s real `hf_options.top_p = 1.0F`, not guessed), so a caller
    /// wanting to match the reference exactly should leave `SamplingParams.TopP` at its default
    /// `1.0f`. `options: null` (the default) preserves the exact previous argmax-only behavior
    /// byte-for-byte.</para>
    /// </summary>
    public int[] GenerateFrame(float[] temporalLmHidden, int promptTextToken, int audioCodebookSize,
        SamplingParams? options = null, Random? rng = null)
    {
        ResetFrame();
        var codes = new int[PersonaPlexDepformerWeights.NumSteps];
        int prevToken = promptTextToken;

        for (int step = 0; step < PersonaPlexDepformerWeights.NumSteps; step++)
        {
            var projected = Linear(temporalLmHidden, _w.InputFromLmWeight[step], PersonaPlexDepformerWeights.LmHiddenDim, PersonaPlexDepformerWeights.HiddenDim);

            float[] tokenEmbedding = step == 0
                ? EmbedRow(_w.TextEmbedding, prevToken, PersonaPlexDepformerWeights.HiddenDim)
                : EmbedRow(_w.AudioEmbeddings[step - 1], prevToken, PersonaPlexDepformerWeights.HiddenDim);

            var hidden = new float[PersonaPlexDepformerWeights.HiddenDim];
            for (int i = 0; i < hidden.Length; i++) hidden[i] = projected[i] + tokenEmbedding[i];

            hidden = RunLayers(hidden, step);

            var logits = Linear(hidden, _w.Heads[step], PersonaPlexDepformerWeights.HiddenDim, audioCodebookSize);
            int best;
            if (options is null)
            {
                best = 0;
                float bestScore = float.NegativeInfinity;
                for (int v = 0; v < audioCodebookSize; v++)
                    if (logits[v] > bestScore) { bestScore = logits[v]; best = v; }
            }
            else
            {
                best = Sampler.Sample(logits, options, rng);
            }

            codes[step] = best;
            prevToken = best;
        }

        return codes;
    }

    private float[] RunLayers(float[] input, int step)
    {
        const int hidden = PersonaPlexDepformerWeights.HiddenDim;
        const int numHeads = PersonaPlexDepformerWeights.NumHeads;
        const int headDim = PersonaPlexDepformerWeights.HeadDim;
        const int ffn = PersonaPlexDepformerWeights.FfnDim;
        int qkvOut = numHeads * headDim; // == hidden, plain MHA
        float scale = 1f / MathF.Sqrt(headDim);

        var x = input;
        for (int l = 0; l < PersonaPlexDepformerWeights.NumLayers; l++)
        {
            var layer = _w.Layers[l];
            var normed = RmsNorm(x, layer.Norm1Alpha, PersonaPlexDepformerWeights.RmsNormEps);

            // Real per-step row-slice of the packed [NumSteps*3*hidden, hidden] tensor.
            var q = LinearRows(normed, layer.InProjWeight, hidden, step * 3 * qkvOut, qkvOut);
            var k = LinearRows(normed, layer.InProjWeight, hidden, step * 3 * qkvOut + qkvOut, qkvOut);
            var v = LinearRows(normed, layer.InProjWeight, hidden, step * 3 * qkvOut + 2 * qkvOut, qkvOut);

            _keyCache[l].Add(k);
            _valueCache[l].Add(v);
            int availableKeys = _keyCache[l].Count;

            var context = new float[qkvOut];
            for (int h = 0; h < numHeads; h++)
            {
                int hOff = h * headDim;
                var scores = new float[availableKeys];
                for (int t = 0; t < availableKeys; t++)
                {
                    float dot = 0f;
                    var kt = _keyCache[l][t];
                    for (int d = 0; d < headDim; d++) dot += q[hOff + d] * kt[hOff + d];
                    scores[t] = dot * scale;
                }
                Softmax(scores);
                for (int t = 0; t < availableKeys; t++)
                {
                    float p = scores[t];
                    var vt = _valueCache[l][t];
                    for (int d = 0; d < headDim; d++) context[hOff + d] += p * vt[hOff + d];
                }
            }

            var attnOut = LinearRows(context, layer.OutProjWeight, qkvOut, step * hidden, hidden);
            var afterAttn = Add(x, attnOut);

            var ffnNormed = RmsNorm(afterAttn, layer.Norm2Alpha, PersonaPlexDepformerWeights.RmsNormEps);
            var gateUp = Linear(ffnNormed, layer.GateUpWeights[step], hidden, 2 * ffn);
            var gate = gateUp.AsSpan(0, ffn).ToArray();
            var up = gateUp.AsSpan(ffn, ffn).ToArray();
            SiluInPlace(gate);
            for (int i = 0; i < ffn; i++) gate[i] *= up[i];
            var down = Linear(gate, layer.DownWeights[step], ffn, hidden);
            x = Add(afterAttn, down);
        }
        return x;
    }

    private static float[] EmbedRow(float[] table, int id, int dim)
    {
        var row = new float[dim];
        Array.Copy(table, (long)id * dim, row, 0, dim);
        return row;
    }

    private static float[] Linear(float[] input, float[] weight, int inDim, int outDim)
    {
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

    /// <summary>Linear against a row-sliced sub-range of a larger packed weight matrix: rows
    /// `[rowOffset, rowOffset+outDim)` of the `[totalRows, inDim]` packed tensor.</summary>
    private static float[] LinearRows(float[] input, float[] packedWeight, int inDim, int rowOffset, int outDim)
    {
        var output = new float[outDim];
        for (int o = 0; o < outDim; o++)
        {
            float sum = 0f;
            int wBase = (rowOffset + o) * inDim;
            for (int i = 0; i < inDim; i++) sum += packedWeight[wBase + i] * input[i];
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

    private static float[] RmsNorm(float[] x, float[] weight, float eps)
    {
        double sumSq = 0;
        for (int i = 0; i < x.Length; i++) sumSq += (double)x[i] * x[i];
        float invRms = (float)(1.0 / Math.Sqrt(sumSq / x.Length + eps));
        var output = new float[x.Length];
        for (int i = 0; i < x.Length; i++) output[i] = x[i] * invRms * weight[i];
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
