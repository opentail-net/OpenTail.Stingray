namespace OpenTail.Stingray.Audio.Rvc;

/// <summary>Inference options, defaults copied from the reference's <c>RvcInferenceConfig</c>
/// (`include/engine/models/rvc/native_pipeline.h`).</summary>
public sealed record RvcInferenceOptions
{
    public int SemitoneShift { get; init; }
    public float RetrievalBlend { get; init; }
    public int PitchFilterRadius { get; init; } = 3;
    public float RmsMixRate { get; init; } = 0.25f;
    public float UnvoicedProtection { get; init; } = 0.33f;
    public int SpeakerId { get; init; }
    public int AudioPadDurationSec { get; init; } = 1;
    public int SplitQuerySec { get; init; } = 5;
    public int SplitCenterSec { get; init; } = 30;
    public int SplitThresholdSec { get; init; } = 32;
}

/// <summary>
/// End-to-end RVC voice conversion: the orchestration layer of the reference's
/// `src/models/rvc/native_pipeline.cpp` (`RvcNativePipeline::infer`), wiring the already
/// golden-verified stages (48 Hz high-pass, reflect pad, RMVPE pitch, HuBERT content, retrieval
/// blend, VITS/NSF synthesizer) together with the reference's segmenting at quiet points,
/// synthesizer-input assembly (2x feature upsample, coarse pitch bins, unvoiced protection, NSF sine
/// source), per-segment pad crop and RMS mix.
///
/// <para>Scope: RVC v2 voices (768-dim HuBERT layer-12 content, which is what
/// <see cref="RvcHubertEncoder"/> implements); v1 voices need the layer-9 tap + `final_proj` and are
/// rejected. Only the RMVPE pitch extractor. Known deviation: the sine-source noise comes from
/// <paramref name="rng"/>, not the reference's Philox CUDA RNG, so output is not bit-identical.</para>
/// </summary>
public static class RvcPipeline
{
    private const int ContentSampleRate = 16000;
    private const int HopSamples16k = 160;
    private const float RmvpeThreshold = 0.03f;
    private const int RmvpeClasses = 360;

    /// <summary>Converts mono 16 kHz audio to the voice in <paramref name="synth"/>; returns audio at
    /// the voice's own sample rate.</summary>
    public static float[] Convert(
        RvcHubertWeights hubert, RvcRmvpeWeights rmvpe, RvcSynthesizerWeights synth, RvcRetrievalIndex? retrieval,
        ReadOnlySpan<float> mono16k, RvcInferenceOptions options, Random rng)
    {
        if (synth.V1)
            throw new NotSupportedException("RVC v1 voices need HuBERT layer-9 + final_proj content features, which are not ported.");
        if (options.RetrievalBlend is < 0f or > 1f) throw new ArgumentException("RVC retrieval blend must be in [0, 1].");
        if (options.UnvoicedProtection is < 0f or > 1f) throw new ArgumentException("RVC unvoiced protection must be in [0, 1].");
        if (options.RetrievalBlend != 0f && retrieval is null) throw new ArgumentException("RVC retrieval blend needs a retrieval index.");

        var content = mono16k.ToArray();
        RvcAudioPreprocessing.HighPass48HzInPlace(content);
        long contentPad = (long)ContentSampleRate * options.AudioPadDurationSec;
        var padded = RvcAudioPreprocessing.ReflectPad(content, contentPad, contentPad);

        bool profile = Environment.GetEnvironmentVariable("STINGRAY_RVC_PROFILE") == "1";
        var sw = System.Diagnostics.Stopwatch.StartNew();
        void Mark(string stage) { if (profile) Console.WriteLine($"[RvcPipeline] {stage}: {sw.Elapsed.TotalSeconds:F2}s"); sw.Restart(); }

        float[] f0 = [];
        if (synth.HasF0)
        {
            f0 = ExtractF0(rmvpe, padded);
            if (f0.Length == 0) throw new InvalidOperationException("RVC RMVPE produced no f0 frames.");
            if (options.PitchFilterRadius > 2) f0 = MedianFilter(f0, 1);
        }
        Mark("rmvpe f0");

        var splits = QuietSplitPoints(content, options.SplitQuerySec, options.SplitCenterSec, options.SplitThresholdSec);
        long targetPad = (long)synth.SampleRate * options.AudioPadDurationSec;
        long tPad2 = 2 * contentPad;
        var converted = new List<float>();
        long segmentStart = 0;

        void RunSegment(long splitPoint, bool finalSegment)
        {
            long segmentEnd = finalSegment ? padded.Length : Math.Min(padded.Length, splitPoint + tPad2 + HopSamples16k);
            if (segmentStart >= segmentEnd) throw new InvalidOperationException("RVC chunk produced an empty audio segment.");
            var segmentAudio = padded.AsSpan((int)segmentStart, (int)(segmentEnd - segmentStart));

            float[] segmentF0 = [];
            long targetFrames = segmentAudio.Length / HopSamples16k;
            if (synth.HasF0)
            {
                long f0Begin = segmentStart / HopSamples16k;
                long f0End = finalSegment ? f0.Length : Math.Min(f0.Length, (splitPoint + tPad2) / HopSamples16k);
                if (f0Begin >= f0End) throw new InvalidOperationException("RVC chunk produced an empty F0 segment.");
                segmentF0 = f0.AsSpan((int)f0Begin, (int)(f0End - f0Begin)).ToArray();
                targetFrames = segmentF0.Length;
            }

            sw.Restart();
            var contentFrames = RvcHubertEncoder.Forward(hubert, segmentAudio);
            Mark("hubert");
            int frames = contentFrames.Length, dim = contentFrames[0].Length;
            var original = Flatten(contentFrames, dim);
            var blended = original;
            if (retrieval is not null && options.RetrievalBlend != 0f)
            {
                blended = (float[])original.Clone();
                retrieval.ApplyBlend(blended, original, frames, dim, options.RetrievalBlend);
            }
            bool protect = synth.HasF0 && options.UnvoicedProtection < 0.5f;
            var input = BuildSynthesizerInput(blended, protect ? original : null, frames, dim, segmentF0, options, synth, (int)targetFrames, rng);

            Mark("synth input");
            var audio = RvcSynthesizerEncoder.Forward(synth, input.Features, input.Pitch, input.SineSource, options.SpeakerId, rng).Audio;
            Mark("synthesizer");
            if (audio.Length <= 2 * targetPad) throw new InvalidOperationException("RVC synthesized audio is too short for the padding crop.");
            converted.AddRange(audio.AsSpan((int)targetPad, (int)(audio.Length - 2 * targetPad)));
        }

        foreach (long split in splits)
        {
            long aligned = split / HopSamples16k * HopSamples16k;
            RunSegment(aligned, finalSegment: false);
            segmentStart = aligned;
        }
        RunSegment(content.Length, finalSegment: true);

        var output = converted.ToArray();
        sw.Restart();
        ApplyRmsMix(content, output, synth.SampleRate, options.RmsMixRate);
        Mark("rms mix");
        return output;
    }

    /// <summary>RMVPE salience over the padded 16 kHz audio, decoded to Hz per 10 ms frame
    /// (`decode_rmvpe_salience`: argmax class, threshold 0.03, ±4-class local average in cents).</summary>
    public static float[] ExtractF0(RvcRmvpeWeights rmvpe, ReadOnlySpan<float> padded16k)
    {
        var (mel, realFrames) = RvcRmvpeMelExtractor.ExtractPadded(padded16k);
        var salience = RvcRmvpeEncoder.Forward(rmvpe, mel);
        var f0 = new float[realFrames];
        for (int frame = 0; frame < realFrames; frame++)
        {
            var row = salience[frame];
            int center = 0;
            float best = row[0];
            for (int c = 1; c < RmvpeClasses; c++)
                if (row[c] > best) { best = row[c]; center = c; }
            if (best <= RmvpeThreshold) continue;
            float weighted = 0f, weightSum = 0f;
            for (int off = -4; off <= 4; off++)
            {
                int c = center + off;
                if (c < 0 || c >= RmvpeClasses) continue;
                float cents = 20f * c + 1997.3794084376191f;
                weighted += row[c] * cents;
                weightSum += row[c];
            }
            if (weightSum > 0f) f0[frame] = 10f * MathF.Pow(2f, weighted / weightSum / 1200f);
        }
        return f0;
    }

    private readonly record struct SynthesizerInput(float[][] Features, int[] Pitch, float[] SineSource);

    /// <summary>`make_synthesizer_input`: repeat each HuBERT frame twice (50 Hz → 100 Hz), clip to
    /// the pitch/target frame count, semitone shift, coarse pitch bins, unvoiced protection, NSF sine source.</summary>
    private static SynthesizerInput BuildSynthesizerInput(
        float[] content, float[]? original, int contentFrames, int dim, float[] f0,
        RvcInferenceOptions options, RvcSynthesizerWeights synth, int targetFrames, Random rng)
    {
        int doubled = contentFrames * 2;
        int frames = Math.Min(targetFrames, synth.HasF0 ? Math.Min(doubled, f0.Length) : doubled);
        if (frames <= 0) throw new InvalidOperationException("RVC synthesizer input has no aligned frames.");

        var features = new float[frames][];
        for (int t = 0; t < frames; t++)
            features[t] = content.AsSpan(Math.Min(contentFrames - 1, t / 2) * dim, dim).ToArray();
        if (!synth.HasF0) return new SynthesizerInput(features, [], []);

        float semitone = MathF.Pow(2f, options.SemitoneShift / 12f);
        var pitch = new int[frames];
        var pitchf = new float[frames];
        for (int t = 0; t < frames; t++)
        {
            pitchf[t] = f0[t] * semitone;
            pitch[t] = CoarsePitchBin(pitchf[t]);
        }

        if (original is not null)
        {
            for (int t = 0; t < frames; t++)
            {
                float keep = pitchf[t] < 1f ? options.UnvoicedProtection : 1f;
                var src = original.AsSpan(Math.Min(contentFrames - 1, t / 2) * dim, dim);
                var dst = features[t];
                for (int i = 0; i < dim; i++) dst[i] = dst[i] * keep + src[i] * (1f - keep);
            }
        }

        return new SynthesizerInput(features, pitch, SineSource(pitchf, frames, synth.SampleRate, synth.HopSamples, rng));
    }

    /// <summary>Reference NSF sine excitation: per-frame phase increments linearly interpolated
    /// to sample rate, with wrap detection, amplitude 0.1, plus noise (0.003 voiced, 0.1/3 unvoiced).</summary>
    private static float[] SineSource(float[] pitchf, int frames, int sampleRate, int upsample, Random rng)
    {
        var cumulative = new float[frames];
        double running = 0;
        for (int t = 0; t < frames; t++)
        {
            float rad = MathF.Max(0f, pitchf[t]) / sampleRate % 1f;
            running += rad;
            cumulative[t] = (float)running * upsample;
        }

        int outputFrames = frames * upsample;
        var sine = new float[outputFrames];
        double sineCumsum = 0;
        float prevTmpMod = 0f;
        for (int i = 0; i < outputFrames; i++)
        {
            int sourceFrame = Math.Min(frames - 1, i / upsample);
            float baseRad = MathF.Max(0f, pitchf[sourceFrame]) / sampleRate % 1f;
            float interpolated = cumulative[0];
            if (outputFrames > 1 && frames > 1)
            {
                float pos = (float)i * (frames - 1) / (outputFrames - 1);
                int left = (int)MathF.Floor(pos);
                int right = Math.Min(frames - 1, left + 1);
                float frac = pos - left;
                interpolated = cumulative[left] * (1f - frac) + cumulative[right] * frac;
            }
            float tmpMod = interpolated - MathF.Floor(interpolated);
            bool wrapped = i > 0 && tmpMod - prevTmpMod < 0f;
            prevTmpMod = tmpMod;
            sineCumsum += baseRad + (wrapped ? -1f : 0f);
            bool voiced = pitchf[sourceFrame] > 0f;
            float noiseAmp = voiced ? 0.003f : 0.1f / 3f;
            float noise = StandardNormal(rng);
            float s = MathF.Sin((float)sineCumsum * (float)(2.0 * Math.PI)) * 0.1f;
            sine[i] = voiced ? s + noiseAmp * noise : noiseAmp * noise;
        }
        return sine;
    }

    /// <summary>`coarse_pitch_bin`: mel-scaled f0 mapped to 1..255 (1 = unvoiced).</summary>
    public static int CoarsePitchBin(float f0Hz)
    {
        if (f0Hz <= 0f) return 1;
        float mel = 1127f * MathF.Log(1f + f0Hz / 700f);
        float melMin = 1127f * MathF.Log(1f + 50f / 700f);
        float melMax = 1127f * MathF.Log(1f + 1100f / 700f);
        float scaled = (mel - melMin) * 254f / (melMax - melMin) + 1f;
        return (int)MathF.Round(Math.Clamp(scaled, 1f, 255f), MidpointRounding.ToEven);
    }

    /// <summary>`median_filter_f0` (window 2r+1, truncated at the edges, upper median for even windows).</summary>
    public static float[] MedianFilter(float[] f0, int radius)
    {
        if (radius <= 0 || f0.Length == 0) return f0;
        var filtered = new float[f0.Length];
        var window = new float[2 * radius + 1];
        for (int i = 0; i < f0.Length; i++)
        {
            int begin = Math.Max(0, i - radius), end = Math.Min(f0.Length, i + radius + 1);
            int n = end - begin;
            Array.Copy(f0, begin, window, 0, n);
            Array.Sort(window, 0, n);
            filtered[i] = window[n / 2];
        }
        return filtered;
    }

    /// <summary>`quiet_split_points`: for inputs longer than the threshold, split near every
    /// `center` seconds at the quietest 10 ms window within ±`query` seconds.</summary>
    public static List<long> QuietSplitPoints(float[] audio, int querySec, int centerSec, int thresholdSec)
    {
        const int window = 160;
        long tQuery = (long)querySec * ContentSampleRate, tCenter = (long)centerSec * ContentSampleRate, tMax = (long)thresholdSec * ContentSampleRate;
        var splits = new List<long>();
        if (audio.Length <= tMax) return splits;
        if (tQuery <= 0 || tCenter <= 0) throw new ArgumentException("RVC chunk timing options must be positive.");

        var pad = RvcAudioPreprocessing.ReflectPad(audio, window / 2, window / 2);
        var sum = new float[audio.Length];
        for (int i = 0; i < window; i++)
            for (int t = 0; t < audio.Length; t++)
                sum[t] += MathF.Abs(pad[i + t]);

        for (long t = tCenter; t < audio.Length; t += tCenter)
        {
            long begin = Math.Max(0, t - tQuery), end = Math.Min(sum.Length, t + tQuery);
            if (begin >= end) continue;
            long argMin = begin;
            for (long j = begin + 1; j < end; j++) if (sum[j] < sum[argMin]) argMin = j;
            splits.Add(t - tQuery + (argMin - begin));
        }
        return splits;
    }

    /// <summary>`apply_rms_mix`: scale the output by rms_src^(1-rate) · rms_out^(rate-1), both RMS
    /// envelopes (0.5 s frames) linearly interpolated to per-sample resolution.</summary>
    public static void ApplyRmsMix(float[] source16k, float[] converted, int targetSampleRate, float rmsMixRate)
    {
        if (rmsMixRate == 1f || converted.Length == 0) return;
        var rms1 = FrameRmsLinear(source16k, ContentSampleRate, converted.Length);
        var rms2 = FrameRmsLinear(converted, targetSampleRate, converted.Length);
        for (int i = 0; i < converted.Length; i++)
        {
            float r2 = MathF.Max(rms2[i], 1e-6f);
            converted[i] *= MathF.Pow(rms1[i], 1f - rmsMixRate) * MathF.Pow(r2, rmsMixRate - 1f);
        }
    }

    private static float[] FrameRmsLinear(float[] audio, int sampleRate, int outputFrames)
    {
        int frameLength = sampleRate / 2 * 2, hop = sampleRate / 2;
        long frameCount = Math.Max(1, (audio.Length + hop - 1) / hop);
        var rms = new float[frameCount];
        Parallel.For(0, frameCount, frame =>
        {
            long begin = frame * hop - frameLength / 2;
            double s = 0;
            for (int i = 0; i < frameLength; i++)
            {
                float v = audio[Math.Clamp(begin + i, 0, audio.Length - 1)];
                s += (double)v * v;
            }
            rms[frame] = (float)Math.Sqrt(s / frameLength);
        });
        var output = new float[outputFrames];
        if (outputFrames == 1 || frameCount == 1) { Array.Fill(output, rms[0]); return output; }
        for (int i = 0; i < outputFrames; i++)
        {
            float pos = (float)i * (frameCount - 1) / (outputFrames - 1);
            long left = (long)MathF.Floor(pos);
            long right = Math.Min(frameCount - 1, left + 1);
            float frac = pos - left;
            output[i] = rms[left] * (1f - frac) + rms[right] * frac;
        }
        return output;
    }

    private static float[] Flatten(float[][] rows, int dim)
    {
        var flat = new float[rows.Length * dim];
        for (int t = 0; t < rows.Length; t++) rows[t].CopyTo(flat, t * dim);
        return flat;
    }

    private static float StandardNormal(Random rng)
    {
        double u1 = 1.0 - rng.NextDouble(), u2 = rng.NextDouble();
        return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
    }
}
