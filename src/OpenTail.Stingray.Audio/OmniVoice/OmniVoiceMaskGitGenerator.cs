using OpenTail.Stingray.Audio.Primitives;

namespace OpenTail.Stingray.Audio.OmniVoice;

/// <summary>
/// Real MaskGIT/SoundStorm-style iterative parallel decoding for OmniVoice's own audio generation,
/// ported from `generator.cpp`'s `generate()`/`pack_initial_inputs`/`update_generated_tokens` (not
/// guessed) -- see docs/audio-review-progress.md's "MaskGIT" entries for the full real algorithm
/// derivation this class implements. Real per-step flow: build a CONDITIONAL sequence (real
/// style+text+optional-reference-audio+target-frames-as-MASK) and an UNCONDITIONAL sequence (just
/// the target frames, always MASK-conditioned, no prompt at all), run BOTH through
/// <see cref="OmniVoiceMaskGitForward"/> (shared weights, no causal masking), project each
/// branch's target-frame hidden states through the real `audio_head`, real classifier-free-
/// guidance combine (`combined = cond + guidanceScale*(cond-uncond)`), score every still-masked
/// (frame,codebook) cell by its best non-mask log-probability minus a real per-codebook penalty
/// (optionally Gumbel-perturbed), accept the real schedule-determined top-K cells this step, write
/// them back into both sequences, and repeat until every cell is real.
/// </summary>
public static class OmniVoiceMaskGitGenerator
{
    public readonly struct Options(
        int numInferenceSteps = 32, float guidanceScale = 2.0f, float tShift = 0.1f,
        float layerPenaltyFactor = 5.0f, float positionTemperature = 5.0f)
    {
        public int NumInferenceSteps { get; } = numInferenceSteps;
        public float GuidanceScale { get; } = guidanceScale;
        public float TShift { get; } = tShift;
        public float LayerPenaltyFactor { get; } = layerPenaltyFactor;
        public float PositionTemperature { get; } = positionTemperature;
        // Real class_temperature>0 (Top-K stochastic class sampling) path is a real, accepted gap
        // -- this port only implements the real default class_temperature=0 (greedy-per-cell) path.
    }

    /// <summary>Real per-cell accepted codes, `[targetFrames*8]` flat (frame-major:
    /// `frame*8+codebook`, matching the reference's own real `out.token_ids` layout).</summary>
    public static int[] Generate(
        OmniVoiceMaskGitWeights w,
        int[] styleTokenIds, int[] textTokenIds,
        (int[] TokenIds, int Frames)? referenceAudioTokens, // TokenIds flat [frame*8+codebook], real codes in [0,1024)
        int targetFrames, Options options, Random rng)
    {
        const int codebooks = OmniVoiceMaskGitWeights.NumCodebooks;
        const int vocab = OmniVoiceMaskGitWeights.AudioVocabSize;
        const int maskId = OmniVoiceMaskGitWeights.AudioMaskId;
        int hidden = OmniVoiceMaskGitWeights.HiddenDim;

        int styleLen = styleTokenIds.Length, textLen = textTokenIds.Length;
        int referenceFrames = referenceAudioTokens?.Frames ?? 0;
        int totalTokens = styleLen + textLen + referenceFrames + targetFrames;
        int conditionalAudioStart = styleLen + textLen;
        int conditionalTargetStart = styleLen + textLen + referenceFrames;

        // Real conditional sequence's fixed TEXT part (style+text ids; never changes across steps).
        var conditionalTextIds = new int[conditionalAudioStart];
        Array.Copy(styleTokenIds, conditionalTextIds, styleLen);
        Array.Copy(textTokenIds, 0, conditionalTextIds, styleLen, textLen);

        // Real conditional sequence's AUDIO part (reference frames + target frames), per codebook,
        // each entry already offset by `codebook*vocab` (combined embedding-table row id).
        var conditionalAudioIds = new int[codebooks][];
        var unconditionalAudioIds = new int[codebooks][];
        for (int cb = 0; cb < codebooks; cb++)
        {
            conditionalAudioIds[cb] = new int[referenceFrames + targetFrames];
            unconditionalAudioIds[cb] = new int[targetFrames];
            int offset = cb * vocab;
            for (int f = 0; f < referenceFrames + targetFrames; f++) conditionalAudioIds[cb][f] = maskId + offset;
            for (int f = 0; f < targetFrames; f++) unconditionalAudioIds[cb][f] = maskId + offset;
        }
        if (referenceAudioTokens is { } refTokens)
        {
            for (int f = 0; f < referenceFrames; f++)
                for (int cb = 0; cb < codebooks; cb++)
                    conditionalAudioIds[cb][f] = refTokens.TokenIds[f * codebooks + cb] + cb * vocab;
        }

        var accepted = new int[codebooks * targetFrames];
        Array.Fill(accepted, maskId);
        var active = new List<int>(accepted.Length);
        for (int i = 0; i < accepted.Length; i++) active.Add(i);

        int totalMaskedTokens = codebooks * targetFrames;
        var schedule = MakeSchedule(totalMaskedTokens, options.NumInferenceSteps, options.TShift);

        void ApplyAccepted()
        {
            for (int cb = 0; cb < codebooks; cb++)
            {
                int offset = cb * vocab;
                for (int f = 0; f < targetFrames; f++)
                {
                    int code = accepted[cb * targetFrames + f];
                    conditionalAudioIds[cb][referenceFrames + f] = code + offset;
                    unconditionalAudioIds[cb][f] = code + offset;
                }
            }
        }
        ApplyAccepted();

        for (int step = 0; step < options.NumInferenceSteps; step++)
        {
            int fillCount = schedule[step];
            if (fillCount <= 0) continue;
            if (active.Count == 0) break;

            // Real conditional forward: [style+text (text embed)] + [reference+target audio (audio embed)].
            var condEmbeddings = new float[totalTokens][];
            for (int t = 0; t < conditionalAudioStart; t++)
                condEmbeddings[t] = EmbedRow(w.TextEmbedding, conditionalTextIds[t], hidden);
            for (int f = 0; f < referenceFrames + targetFrames; f++)
                condEmbeddings[conditionalAudioStart + f] = SumCodebookEmbeddings(w.AudioEmbedding, conditionalAudioIds, f, codebooks, hidden);
            var condHidden = OmniVoiceMaskGitForward.Forward(w, condEmbeddings);

            // Real unconditional forward: target frames only, all audio-embedded.
            var uncondEmbeddings = new float[targetFrames][];
            for (int f = 0; f < targetFrames; f++)
                uncondEmbeddings[f] = SumCodebookEmbeddings(w.AudioEmbedding, unconditionalAudioIds, f, codebooks, hidden);
            var uncondHidden = OmniVoiceMaskGitForward.Forward(w, uncondEmbeddings);

            // Real CFG-combined per-frame [8*vocab] logits: combined = cond + guidance*(cond-uncond).
            var combinedLogits = new float[targetFrames][];
            for (int f = 0; f < targetFrames; f++)
            {
                var condRow = DenseKernels.LinearNoBias(condHidden[conditionalTargetStart + f], w.AudioHead, hidden, codebooks * vocab);
                var uncondRow = DenseKernels.LinearNoBias(uncondHidden[f], w.AudioHead, hidden, codebooks * vocab);
                var row = new float[codebooks * vocab];
                for (int i = 0; i < row.Length; i++) row[i] = condRow[i] + options.GuidanceScale * (condRow[i] - uncondRow[i]);
                combinedLogits[f] = row;
            }

            // Real per-cell candidate scoring (class_temperature=0 greedy path).
            var candidates = new List<(float Score, int Predicted, int Index)>(active.Count);
            foreach (int flatIndex in active)
            {
                int cb = flatIndex / targetFrames;
                int frame = flatIndex % targetFrames;
                var logits = combinedLogits[frame];
                int cellOffset = cb * vocab;
                var (bestToken, bestLogProb) = BestLogProbExcludingMask(logits, cellOffset, vocab, maskId);
                float score = bestLogProb - cb * options.LayerPenaltyFactor;
                if (options.PositionTemperature > 0f) score = GumbelSampleScalar(score, options.PositionTemperature, rng);
                candidates.Add((score, bestToken, flatIndex));
            }

            candidates.Sort((a, b) => a.Score != b.Score ? b.Score.CompareTo(a.Score) : a.Index.CompareTo(b.Index));
            int actualFill = Math.Min(fillCount, candidates.Count);
            for (int i = 0; i < actualFill; i++)
                accepted[candidates[i].Index] = candidates[i].Predicted;

            ApplyAccepted();
            active.RemoveAll(idx => accepted[idx] != maskId);
        }

        foreach (int token in accepted)
            if (token == maskId)
                throw new InvalidOperationException("OmniVoice MaskGIT generator left masked tokens after iterative decoding.");

        // Real reference output layout: [frame*codebooks+codebook] (this class's internal
        // `accepted` array is codebook-major [cb*targetFrames+frame], matching generator.cpp's
        // own internal layout before its final frame-major transpose).
        var output = new int[targetFrames * codebooks];
        for (int frame = 0; frame < targetFrames; frame++)
            for (int cb = 0; cb < codebooks; cb++)
                output[frame * codebooks + cb] = accepted[cb * targetFrames + frame];
        return output;
    }

    private static float[] EmbedRow(float[] table, int id, int hidden)
    {
        var row = new float[hidden];
        Array.Copy(table, (long)id * hidden, row, 0, hidden);
        return row;
    }

    private static float[] SumCodebookEmbeddings(float[] audioEmbedding, int[][] audioIdsPerCodebook, int frame, int codebooks, int hidden)
    {
        var row = new float[hidden];
        for (int cb = 0; cb < codebooks; cb++)
        {
            long id = audioIdsPerCodebook[cb][frame];
            long baseOff = id * hidden;
            for (int d = 0; d < hidden; d++) row[d] += audioEmbedding[baseOff + d];
        }
        return row;
    }

    /// <summary>Real `best_log_prob_excluding_mask`: best (non-mask) logit index/log-softmax-value
    /// within a `[cellOffset, cellOffset+vocab)` slice of a real per-frame `[8*vocab]` logits row,
    /// where `localMaskId` (the mask token's LOCAL index within this codebook's own `vocab`-wide
    /// slice) is excluded from the argmax but INCLUDED in the softmax normalization (matches the
    /// reference's own real logsumexp-over-all-including-mask formula exactly).</summary>
    private static (int Token, float LogProb) BestLogProbExcludingMask(float[] logits, int cellOffset, int vocab, int localMaskId)
    {
        int bestIndex = -1;
        float bestLogit = float.NegativeInfinity;
        float maxLogit = logits[cellOffset];
        for (int i = 0; i < vocab; i++)
        {
            float value = logits[cellOffset + i];
            if (value > maxLogit) maxLogit = value;
            if (i == localMaskId) continue;
            if (value > bestLogit) { bestLogit = value; bestIndex = i; }
        }
        double sum = 0;
        for (int i = 0; i < vocab; i++) sum += Math.Exp(logits[cellOffset + i] - maxLogit);
        float logSum = maxLogit + (float)Math.Log(sum);
        return (bestIndex, bestLogit - logSum);
    }

    private static float GumbelSampleScalar(float logit, float temperature, Random rng)
    {
        float scaled = logit / temperature;
        double u = 1.0e-6 + rng.NextDouble() * (1.0 - 2.0e-6);
        float noise = -(float)Math.Log(-Math.Log(u + 1.0e-10) + 1.0e-10);
        return scaled + noise;
    }

    /// <summary>Real `time_steps`/`make_schedule`: a cosine-shifted per-step unmask-count curve.</summary>
    private static int[] MakeSchedule(int totalMaskedTokens, int numInferenceSteps, float tShift)
    {
        var steps = new double[numInferenceSteps + 1];
        for (int i = 0; i <= numInferenceSteps; i++)
        {
            double t = (double)i / numInferenceSteps;
            steps[i] = tShift * t / (1.0 + (tShift - 1.0) * t);
        }
        var schedule = new int[numInferenceSteps];
        int remaining = totalMaskedTokens;
        for (int step = 0; step < numInferenceSteps; step++)
        {
            int num;
            if (step == numInferenceSteps - 1) num = remaining;
            else
            {
                double fraction = steps[step + 1] - steps[step];
                num = Math.Min((int)Math.Ceiling(totalMaskedTokens * fraction), remaining);
            }
            schedule[step] = num;
            remaining -= num;
        }
        return schedule;
    }
}
