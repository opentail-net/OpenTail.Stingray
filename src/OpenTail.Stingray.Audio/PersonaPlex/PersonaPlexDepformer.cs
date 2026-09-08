using System.Numerics.Tensors;
using OpenTail.Stingray.Audio.Primitives;
using OpenTail.Stingray.Cpu;
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
            _keyCache[l] ??= new List<float[]>(PersonaPlexDepformerWeights.NumSteps);
            _keyCache[l].Clear();
            _valueCache[l] ??= new List<float[]>(PersonaPlexDepformerWeights.NumSteps);
            _valueCache[l].Clear();
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
        SamplingParams? options = null, Random? rng = null, int[]? audioTarget = null, byte[]? audioProvided = null)
        => GenerateFrame((ReadOnlySpan<float>)temporalLmHidden, promptTextToken, audioCodebookSize, options, rng, audioTarget, audioProvided);

    public int[] GenerateFrame(ReadOnlySpan<float> temporalLmHidden, int promptTextToken, int audioCodebookSize,
        SamplingParams? options = null, Random? rng = null, int[]? audioTarget = null, byte[]? audioProvided = null)
    {
        ResetFrame();
        var codes = new int[PersonaPlexDepformerWeights.NumSteps];

        int lastUnprovidedStep = -1;
        if (audioProvided != null)
        {
            for (int step = PersonaPlexDepformerWeights.NumSteps - 1; step >= 0; step--)
            {
                if (audioProvided[step] == 0)
                {
                    lastUnprovidedStep = step;
                    break;
                }
            }
            if (lastUnprovidedStep < 0)
            {
                if (audioTarget != null)
                    Array.Copy(audioTarget, codes, PersonaPlexDepformerWeights.NumSteps);
                return codes;
            }
        }
        else
        {
            lastUnprovidedStep = PersonaPlexDepformerWeights.NumSteps - 1;
        }

        int prevToken = promptTextToken;

        for (int step = 0; step <= lastUnprovidedStep; step++)
        {
            var projected = DenseKernels.LinearNoBias(temporalLmHidden, _w.InputFromLmWeight[step], PersonaPlexDepformerWeights.LmHiddenDim, PersonaPlexDepformerWeights.HiddenDim);

            ReadOnlySpan<float> tokenEmbedding = step == 0
                ? _w.TextEmbedding.AsSpan((int)((long)prevToken * PersonaPlexDepformerWeights.HiddenDim), PersonaPlexDepformerWeights.HiddenDim)
                : _w.AudioEmbeddings[step - 1].AsSpan((int)((long)prevToken * PersonaPlexDepformerWeights.HiddenDim), PersonaPlexDepformerWeights.HiddenDim);

            TensorPrimitives.Add((ReadOnlySpan<float>)projected, tokenEmbedding, projected);

            var hidden = RunLayers(projected, step);

            var logits = DenseKernels.LinearNoBias(hidden, _w.Heads[step], PersonaPlexDepformerWeights.HiddenDim, audioCodebookSize);
            int best = options is null
                ? TensorPrimitives.IndexOfMax((ReadOnlySpan<float>)logits.AsSpan(0, audioCodebookSize))
                : Sampler.Sample(logits, options, rng);

            codes[step] = best;
            prevToken = (audioProvided != null && audioProvided[step] != 0 && audioTarget != null)
                ? audioTarget[step]
                : best;
        }

        for (int step = lastUnprovidedStep + 1; step < PersonaPlexDepformerWeights.NumSteps; step++)
        {
            codes[step] = audioTarget != null ? audioTarget[step] : 0;
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
        Span<float> scoresBuf = stackalloc float[PersonaPlexDepformerWeights.NumSteps];

        var x = input;
        for (int l = 0; l < PersonaPlexDepformerWeights.NumLayers; l++)
        {
            var layer = _w.Layers[l];
            var normed = RmsNorm(x, layer.Norm1Alpha, PersonaPlexDepformerWeights.RmsNormEps);

            // Real per-step row-slice of the packed [NumSteps*3*hidden, hidden] tensor: Q, K, V combined in 1 call.
            var qkv = LinearRows(normed, layer.InProjWeight, hidden, step * 3 * qkvOut, 3 * qkvOut);
            var q = qkv.AsSpan(0, qkvOut);
            var k = qkv[qkvOut..(2 * qkvOut)];
            var v = qkv[(2 * qkvOut)..(3 * qkvOut)];

            _keyCache[l].Add(k);
            _valueCache[l].Add(v);
            int availableKeys = _keyCache[l].Count;

            var context = new float[qkvOut];
            for (int h = 0; h < numHeads; h++)
            {
                int hOff = h * headDim;
                var qHead = (ReadOnlySpan<float>)q.Slice(hOff, headDim);
                var scores = scoresBuf[..availableKeys];
                for (int t = 0; t < availableKeys; t++)
                {
                    scores[t] = TensorPrimitives.Dot(qHead, _keyCache[l][t].AsSpan(hOff, headDim)) * scale;
                }
                DenseKernels.SoftmaxInPlace(scores);
                var ctxHead = context.AsSpan(hOff, headDim);
                for (int t = 0; t < availableKeys; t++)
                {
                    float p = scores[t];
                    if (p != 0f)
                        TensorPrimitives.MultiplyAdd((ReadOnlySpan<float>)_valueCache[l][t].AsSpan(hOff, headDim), p, ctxHead, ctxHead);
                }
            }

            var attnOut = LinearRows(context, layer.OutProjWeight, qkvOut, step * hidden, hidden);
            var afterAttn = Add(x, attnOut);

            var ffnNormed = RmsNorm(afterAttn, layer.Norm2Alpha, PersonaPlexDepformerWeights.RmsNormEps);
            var gateUp = DenseKernels.LinearNoBias(ffnNormed, layer.GateUpWeights[step], hidden, 2 * ffn);
            var gate = gateUp.AsSpan(0, ffn);
            var up = gateUp.AsSpan(ffn, ffn);
            DenseKernels.SiluInPlace(gate);
            TensorPrimitives.Multiply(gate, up, gate);
            var down = DenseKernels.LinearNoBias(gate, layer.DownWeights[step], ffn, hidden);
            x = Add(afterAttn, down);
        }
        return x;
    }

    /// <summary>Linear against a row-sliced sub-range of a larger packed weight matrix: rows
    /// `[rowOffset, rowOffset+outDim)` of the `[totalRows, inDim]` packed tensor.</summary>
    private static unsafe float[] LinearRows(float[] input, float[] packedWeight, int inDim, int rowOffset, int outDim)
    {
        var output = new float[outDim];
        fixed (float* outP = output, wP = packedWeight, inP = input)
        {
            SimdKernels.MatVecF32(outP, wP + (long)rowOffset * inDim, null, inP, outDim, inDim);
        }
        return output;
    }

    private static float[] Add(float[] a, float[] b)
    {
        var output = new float[a.Length];
        TensorPrimitives.Add((ReadOnlySpan<float>)a, b, output);
        return output;
    }

    private static float[] RmsNorm(float[] x, float[] weight, float eps)
    {
        double sumSq = TensorPrimitives.SumOfSquares((ReadOnlySpan<float>)x);
        float invRms = (float)(1.0 / Math.Sqrt(sumSq / x.Length + eps));
        var output = new float[x.Length];
        TensorPrimitives.Multiply((ReadOnlySpan<float>)x, invRms, output);
        TensorPrimitives.Multiply((ReadOnlySpan<float>)output, weight, output);
        return output;
    }
}
