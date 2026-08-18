namespace OpenTail.Stingray.Audio.Vad;

/// <summary>
/// Time-domain speech segment aggregator and silence pruner matching whisper_vad_segments_from_probs in whisper.cpp.
/// </summary>
public static class VadSegmenter
{
    public static IReadOnlyList<VadSpeechSegment> BuildSegments(
        ReadOnlySpan<float> probs,
        VadParams parameters,
        int frameSize = 512)
    {
        if (probs.IsEmpty) return [];

        int sampleRate = parameters.SampleRate > 0 ? parameters.SampleRate : 16000;
        float msPerFrame = (frameSize * 1000.0f) / sampleRate;

        int minSpeechFrames = Math.Max(1, (int)MathF.Ceiling(parameters.MinSpeechDurationMs / msPerFrame));
        int minSilenceFrames = Math.Max(1, (int)MathF.Ceiling(parameters.MinSilenceDurationMs / msPerFrame));
        int padFrames = Math.Max(0, (int)MathF.Round(parameters.SpeechPadMs / msPerFrame));

        var rawSegments = new List<(int StartFrame, int EndFrame, float AvgProb)>();

        bool isSpeaking = false;
        int speechStartFrame = 0;
        int silenceCount = 0;
        float probSum = 0f;
        int frameCountInSegment = 0;

        for (int i = 0; i < probs.Length; i++)
        {
            float p = probs[i];

            if (p >= parameters.Threshold)
            {
                if (!isSpeaking)
                {
                    isSpeaking = true;
                    speechStartFrame = i;
                    probSum = 0f;
                    frameCountInSegment = 0;
                }
                silenceCount = 0;
                probSum += p;
                frameCountInSegment++;
            }
            else
            {
                if (isSpeaking)
                {
                    silenceCount++;
                    probSum += p;
                    frameCountInSegment++;

                    if (silenceCount >= minSilenceFrames)
                    {
                        int speechEndFrame = i - silenceCount + 1;
                        int durationFrames = speechEndFrame - speechStartFrame;

                        if (durationFrames >= minSpeechFrames)
                        {
                            float avg = frameCountInSegment > 0 ? probSum / frameCountInSegment : p;
                            rawSegments.Add((speechStartFrame, speechEndFrame, avg));
                        }

                        isSpeaking = false;
                        silenceCount = 0;
                        probSum = 0f;
                        frameCountInSegment = 0;
                    }
                }
            }
        }

        // Flush trailing speech if active at end of audio
        if (isSpeaking)
        {
            int speechEndFrame = probs.Length;
            int durationFrames = speechEndFrame - speechStartFrame;
            if (durationFrames >= minSpeechFrames)
            {
                float avg = frameCountInSegment > 0 ? probSum / frameCountInSegment : parameters.Threshold;
                rawSegments.Add((speechStartFrame, speechEndFrame, avg));
            }
        }

        if (rawSegments.Count == 0) return [];

        // Apply padding and merge overlapping intervals
        var merged = new List<VadSpeechSegment>();

        int curStart = Math.Max(0, rawSegments[0].StartFrame - padFrames);
        int curEnd = Math.Min(probs.Length, rawSegments[0].EndFrame + padFrames);
        float curProb = rawSegments[0].AvgProb;

        for (int i = 1; i < rawSegments.Count; i++)
        {
            int nextStart = Math.Max(0, rawSegments[i].StartFrame - padFrames);
            int nextEnd = Math.Min(probs.Length, rawSegments[i].EndFrame + padFrames);

            if (nextStart <= curEnd)
            {
                // Overlapping or adjacent: merge
                curEnd = Math.Max(curEnd, nextEnd);
                curProb = (curProb + rawSegments[i].AvgProb) * 0.5f;
            }
            else
            {
                // Push previous
                int startSample = curStart * frameSize;
                int endSample = curEnd * frameSize;
                merged.Add(new VadSpeechSegment
                {
                    StartSample = startSample,
                    EndSample = endSample,
                    StartSeconds = startSample / (float)sampleRate,
                    EndSeconds = endSample / (float)sampleRate,
                    AvgProbability = curProb
                });

                curStart = nextStart;
                curEnd = nextEnd;
                curProb = rawSegments[i].AvgProb;
            }
        }

        // Push final segment
        int finalStartSample = curStart * frameSize;
        int finalEndSample = curEnd * frameSize;
        merged.Add(new VadSpeechSegment
        {
            StartSample = finalStartSample,
            EndSample = finalEndSample,
            StartSeconds = finalStartSample / (float)sampleRate,
            EndSeconds = finalEndSample / (float)sampleRate,
            AvgProbability = curProb
        });

        return merged;
    }
}
