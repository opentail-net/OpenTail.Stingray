using OpenTail.Stingray.Audio.Primitives;

namespace OpenTail.Stingray.Audio.Parakeet;

/// <summary>
/// Greedy TDT (token-and-duration transducer) decoding for Parakeet TDT checkpoints. Port of CrispASR
/// src/parakeet.cpp <c>parakeet_tdt_decode</c>, which follows NeMo <c>GreedyTDTInfer._greedy_decode</c>:
/// <list type="bullet">
/// <item>The prediction network starts from the blank token (SOS) and only advances when a token is emitted.</item>
/// <item>Per step, the joint gives token logits [vocab + blank] and duration logits; each is argmaxed separately.</item>
/// <item>A duration &gt; 0 moves to frame t + duration and ends the frame's inner loop. Duration 0 stays on the
/// frame; after <see cref="MaxSymbolsPerFrame"/> such steps the frame advances by one.</item>
/// </list>
/// </summary>
public sealed class ParakeetTdtDecoder
{
    /// <summary>Per-frame step budget (CrispASR <c>max_per_step</c>, NeMo <c>max_symbols</c>).</summary>
    public const int MaxSymbolsPerFrame = 10;

    private readonly ParakeetTokenizer _tokenizer;
    private readonly ParakeetWeights _weights;
    private readonly ParakeetTdtWeights _tdt;

    public ParakeetTdtDecoder(ParakeetTokenizer tokenizer, ParakeetWeights weights)
    {
        _tokenizer = tokenizer;
        _weights = weights;
        _tdt = weights.Tdt ?? throw new InvalidOperationException("This Parakeet checkpoint has no TDT head (joint.out.weight).");
    }

    /// <summary>Emitted token ids with the encoder frame each was emitted at.</summary>
    public List<(int Token, int Frame, int Duration)> DecodeTokens(IReadOnlyList<float[]> encoderFrames, int numFrames)
    {
        var tdt = _tdt;
        int blank = _weights.BlankTokenId, tokenCount = blank + 1, nDur = tdt.Durations.Length, h = tdt.PredHidden;
        if (tdt.OutputDim != tokenCount + nDur)
            throw new InvalidDataException($"TDT joint output {tdt.OutputDim} != vocab+blank {tokenCount} + {nDur} durations.");

        // Encoder projection for every frame, once.
        var encProj = new float[numFrames][];
        for (int t = 0; t < numFrames; t++)
            encProj[t] = DenseKernels.Linear(encoderFrames[t], tdt.JointEncWeight, tdt.JointEncBias, _weights.HiddenDim, tdt.JointHidden);

        var hState = new float[tdt.PredLayers][];
        var cState = new float[tdt.PredLayers][];
        for (int l = 0; l < tdt.PredLayers; l++) { hState[l] = new float[h]; cState[l] = new float[h]; }
        float[] g = PredictorStep(blank, hState, cState);
        float[] predProj = DenseKernels.Linear(g, tdt.JointPredWeight, tdt.JointPredBias, h, tdt.JointHidden);

        var emitted = new List<(int, int, int)>();
        var mid = new float[tdt.JointHidden];
        int frame = 0;
        while (frame < numFrames)
        {
            int inner = 0;
            while (inner < MaxSymbolsPerFrame)
            {
                var e = encProj[frame];
                for (int i = 0; i < mid.Length; i++) mid[i] = MathF.Max(0f, e[i] + predProj[i]);
                var logits = DenseKernels.Linear(mid, tdt.JointOutWeight, tdt.JointOutBias, tdt.JointHidden, tdt.OutputDim);

                int tok = ArgMax(logits, 0, tokenCount);
                int skip = tdt.Durations[ArgMax(logits, tokenCount, nDur)];

                if (tok != blank)
                {
                    emitted.Add((tok, frame, skip));
                    g = PredictorStep(tok, hState, cState);
                    predProj = DenseKernels.Linear(g, tdt.JointPredWeight, tdt.JointPredBias, h, tdt.JointHidden);
                }
                if (skip > 0) { frame += skip; break; }
                inner++;
            }
            if (inner >= MaxSymbolsPerFrame) frame++;
        }
        return emitted;
    }

    public (string FullText, int[] Tokens, List<SpeechSegment> Segments) DecodeGreedy(
        IReadOnlyList<float[]> encoderFrames, int numFrames, TimeSpan timeOffset, float frameDurationSeconds = 0.08f)
    {
        var emitted = DecodeTokens(encoderFrames, numFrames);
        int[] tokens = emitted.Select(x => x.Token).ToArray();

        // One segment per word: a new word starts at a piece with the SentencePiece word marker.
        var segments = new List<SpeechSegment>();
        var word = new List<int>();
        int wordStart = 0, wordEnd = 0;
        void Flush()
        {
            if (word.Count == 0) return;
            string text = _tokenizer.Decode(word.ToArray());
            if (!string.IsNullOrWhiteSpace(text))
                segments.Add(new SpeechSegment
                {
                    Id = segments.Count,
                    Start = timeOffset + TimeSpan.FromSeconds(wordStart * frameDurationSeconds),
                    End = timeOffset + TimeSpan.FromSeconds(Math.Min(numFrames, wordEnd) * frameDurationSeconds),
                    Text = text,
                    Tokens = word.ToArray(),
                });
            word.Clear();
        }
        foreach (var (tok, frame, dur) in emitted)
        {
            if (word.Count > 0 && _tokenizer.GetToken(tok).StartsWith('▁')) Flush();
            if (word.Count == 0) wordStart = frame;
            word.Add(tok);
            wordEnd = frame + Math.Max(1, dur);
        }
        Flush();

        return (_tokenizer.Decode(tokens).Trim(), tokens, segments);
    }

    /// <summary>One step of the LSTM stack on <paramref name="token"/>'s embedding (PyTorch LSTM, gates i, f, g, o).
    /// Updates the state in place and returns the top layer's h.</summary>
    private float[] PredictorStep(int token, float[][] hState, float[][] cState)
    {
        var tdt = _tdt;
        int h = tdt.PredHidden;
        float[] x = tdt.Embed.AsSpan(token * h, h).ToArray();
        for (int l = 0; l < tdt.PredLayers; l++)
        {
            var z = DenseKernels.Linear(x, tdt.LstmWih[l], tdt.LstmBih[l], x.Length, 4 * h);
            var zh = DenseKernels.Linear(hState[l], tdt.LstmWhh[l], tdt.LstmBhh[l], h, 4 * h);
            var hNew = new float[h];
            var c = cState[l];
            for (int k = 0; k < h; k++)
            {
                float i = Sigmoid(z[k] + zh[k]);
                float f = Sigmoid(z[h + k] + zh[h + k]);
                float gg = MathF.Tanh(z[2 * h + k] + zh[2 * h + k]);
                float o = Sigmoid(z[3 * h + k] + zh[3 * h + k]);
                c[k] = f * c[k] + i * gg;
                hNew[k] = o * MathF.Tanh(c[k]);
            }
            hState[l] = hNew;
            x = hNew;
        }
        return x;
    }

    private static float Sigmoid(float v) => 1f / (1f + MathF.Exp(-v));

    private static int ArgMax(float[] v, int start, int count)
    {
        int best = 0;
        for (int i = 1; i < count; i++)
            if (v[start + i] > v[start + best]) best = i;
        return best;
    }
}
