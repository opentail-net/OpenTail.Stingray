using System;
using OpenTail.Stingray.Audio.Primitives;

namespace OpenTail.Stingray.Audio.FunASR;

/// <summary>
/// Real Kaldi-compatible mel filterbank + LFR splice + CMVN feature extractor for Paraformer,
/// transcribed directly from `torchaudio.compliance.kaldi.fbank`'s real source (installed
/// `torchaudio` package, `compliance/kaldi.py`) plus the real `apply_lfr`/`apply_cmvn`
/// functions from the real `funasr` package (`funasr/frontends/wav_frontend.py`). Real config
/// confirmed from the checkpoint family's published config (see docs/audio-review-progress.md's
/// FunASR section): fs=16000, window=hamming, n_mels=80, frame_length=25ms, frame_shift=10ms,
/// lfr_m=7, lfr_n=6. Named <c>FunAsrRealMelExtractor</c> (not <c>FunAsrMelExtractor</c>) to
/// avoid colliding with the pre-existing fake class in <c>FunAsrPipeline.cs</c> until that's
/// rewired.
///
/// <para><b>Exact real algorithm per frame</b> (window_size=400, window_shift=160,
/// padded_window_size=512=next_pow2(400)): raw frame -&gt; subtract row mean (DC removal) -&gt;
/// preemphasis (`x[0] = 0.03*x[0]`, `x[j] = x[j] - 0.97*x[j-1]` for j&gt;=1 -- note `x[0]`'s
/// coefficient is `1-0.97=0.03`, NOT skipped, since Kaldi edge-replicates the "previous" sample
/// for the first frame position) -&gt; Hamming window (`0.54 - 0.46*cos(2*pi*n/(N-1))`,
/// PyTorch's `periodic=False` convention) -&gt; zero-pad to 512 -&gt; power spectrum (real DFT,
/// `|X[k]|^2`) -&gt; 80 triangular mel filters (Kaldi's own construction: mel-scale
/// `1127*ln(1+f/700)`, filters spread via `(num_bins+1)`-way division between `low_freq=20` and
/// `high_freq=nyquist=8000`, evaluated directly in the mel domain against each FFT bin's own
/// mel frequency, NOT a standard librosa-style filterbank -- confirmed different by reading the
/// real `get_mel_banks` source, not assumed) -&gt; `log(max(mel_energy, eps))`.
/// </para>
///
/// <para><b>dither is this port's own deterministic choice of 0</b> (matching
/// `scratch-llamacpp-ref/funasr_golden_frontend.py`'s documented choice) -- the real
/// `WavFrontend` class default is `dither=1.0` (non-deterministic, adds Gaussian noise per
/// frame), and the actual production inference-time override could not be verified from the
/// published config alone. If a future golden comparison against genuinely-fixed reference
/// audio fails to clear the usual >0.99 cosine bar, revisit this choice first.</para>
/// </summary>
public sealed class FunAsrRealMelExtractor
{
    public const int SampleRate = 16000;
    public const int NumMels = 80;
    public const int WindowSize = 400;   // 25ms @ 16kHz
    public const int WindowShift = 160;  // 10ms @ 16kHz
    public const int PaddedWindowSize = 512; // next_pow2(400)
    public const int LfrM = 7;
    public const int LfrN = 6;
    public const float PreemphasisCoefficient = 0.97f;
    public const float LowFreq = 20f;

    private readonly float[] _hammingWindow;
    private readonly float[][] _melFilters; // [NumMels][PaddedWindowSize/2+1]

    public FunAsrRealMelExtractor()
    {
        _hammingWindow = CreateHammingWindow(WindowSize);
        _melFilters = CreateMelFilterBank(NumMels, PaddedWindowSize, SampleRate, LowFreq, highFreq: 0f);
    }

    /// <summary>Real Hamming window, PyTorch's `periodic=False` convention (denominator N-1).</summary>
    private static float[] CreateHammingWindow(int n)
    {
        var w = new float[n];
        for (int i = 0; i < n; i++)
            w[i] = 0.54f - 0.46f * MathF.Cos(2f * MathF.PI * i / (n - 1));
        return w;
    }

    /// <summary>Real Kaldi triangular mel filterbank construction (`get_mel_banks`) -- NOT a standard librosa-style filterbank, see class doc comment.</summary>
    private static float[][] CreateMelFilterBank(int numBins, int paddedWindowSize, float sampleFreq, float lowFreq, float highFreq)
    {
        int numFftBins = paddedWindowSize / 2; // 256
        float nyquist = 0.5f * sampleFreq;
        if (highFreq <= 0f) highFreq += nyquist;

        float fftBinWidth = sampleFreq / paddedWindowSize;
        float melLowFreq = MelScale(lowFreq);
        float melHighFreq = MelScale(highFreq);
        float melFreqDelta = (melHighFreq - melLowFreq) / (numBins + 1);

        var filters = new float[numBins][];
        for (int bin = 0; bin < numBins; bin++)
        {
            float leftMel = melLowFreq + bin * melFreqDelta;
            float centerMel = melLowFreq + (bin + 1f) * melFreqDelta;
            float rightMel = melLowFreq + (bin + 2f) * melFreqDelta;

            // filters[bin] has numFftBins+1 entries: the real code zero-pads one extra column
            // (the Nyquist bin) after building the numFftBins-wide filter.
            var row = new float[numFftBins + 1];
            for (int k = 0; k < numFftBins; k++)
            {
                float mel = MelScale(fftBinWidth * k);
                float upSlope = (mel - leftMel) / (centerMel - leftMel);
                float downSlope = (rightMel - mel) / (rightMel - centerMel);
                float v = MathF.Min(upSlope, downSlope);
                row[k] = MathF.Max(0f, v);
            }
            row[numFftBins] = 0f;
            filters[bin] = row;
        }
        return filters;
    }

    private static float MelScale(float freq) => 1127f * MathF.Log(1f + freq / 700f);

    /// <summary>Extracts real mel features: frame-major [T, 80] log-mel, BEFORE LFR splice/CMVN. `pcm16k` is expected in [-1,1] float range (scaled to int16 range internally, matching Kaldi's convention).</summary>
    public float[][] ExtractLogMel(ReadOnlySpan<float> pcm16k)
    {
        if (pcm16k.Length < WindowSize) return [];
        int numFrames = 1 + (pcm16k.Length - WindowSize) / WindowShift;
        var output = new float[numFrames][];

        var frame = new float[WindowSize];
        var padded = new float[PaddedWindowSize];
        var powerSpectrum = new float[PaddedWindowSize / 2 + 1];

        for (int f = 0; f < numFrames; f++)
        {
            int start = f * WindowShift;
            for (int i = 0; i < WindowSize; i++) frame[i] = pcm16k[start + i] * 32768f;

            // Remove DC offset (subtract row mean).
            float mean = 0f;
            for (int i = 0; i < WindowSize; i++) mean += frame[i];
            mean /= WindowSize;
            for (int i = 0; i < WindowSize; i++) frame[i] -= mean;

            // Preemphasis with edge-replicated "previous sample" for index 0.
            float prev0 = frame[0];
            for (int i = WindowSize - 1; i >= 1; i--)
                frame[i] -= PreemphasisCoefficient * frame[i - 1];
            frame[0] -= PreemphasisCoefficient * prev0;

            // Hamming window, then zero-pad to PaddedWindowSize.
            for (int i = 0; i < WindowSize; i++) padded[i] = frame[i] * _hammingWindow[i];
            Array.Clear(padded, WindowSize, PaddedWindowSize - WindowSize);

            SpectralKernels.ComputePowerSpectrum(padded, powerSpectrum);

            var melRow = new float[NumMels];
            for (int m = 0; m < NumMels; m++)
            {
                float sum = 0f;
                var filt = _melFilters[m];
                for (int k = 0; k < powerSpectrum.Length; k++) sum += powerSpectrum[k] * filt[k];
                melRow[m] = MathF.Log(MathF.Max(sum, 1.1920929e-7f)); // float32 epsilon, matches EPSILON in kaldi.py
            }
            output[f] = melRow;
        }
        return output;
    }

    /// <summary>Real `apply_lfr` (Low Frame Rate splice): concatenates lfr_m consecutive frames every lfr_n frames, left-padded by repeating frame 0. Verbatim algorithm from `funasr/frontends/wav_frontend.py`.</summary>
    public static float[][] ApplyLfr(float[][] logMel, int lfrM = LfrM, int lfrN = LfrN)
    {
        int t = logMel.Length;
        int featDim = logMel[0].Length;
        int leftPad = (lfrM - 1) / 2;

        var padded = new float[t + leftPad][];
        for (int i = 0; i < leftPad; i++) padded[i] = logMel[0];
        Array.Copy(logMel, 0, padded, leftPad, t);
        int paddedT = padded.Length;

        int tLfr = (int)MathF.Ceiling((float)t / lfrN);
        int lastIdx = (paddedT - lfrM) / lfrN + 1;
        int numPaddingRaw = lfrM - (paddedT - lastIdx * lfrN);
        if (numPaddingRaw > 0)
        {
            int numPadding = (int)((2f * lfrM - 2f * paddedT + (tLfr - 1 + lastIdx) * lfrN) / 2f * (tLfr - lastIdx));
            if (numPadding > 0)
            {
                var extended = new float[paddedT + numPadding][];
                Array.Copy(padded, extended, paddedT);
                for (int i = 0; i < numPadding; i++) extended[paddedT + i] = padded[paddedT - 1];
                padded = extended;
            }
        }

        var output = new float[tLfr][];
        for (int i = 0; i < tLfr; i++)
        {
            var row = new float[lfrM * featDim];
            int srcBase = i * lfrN;
            for (int m = 0; m < lfrM; m++)
                Array.Copy(padded[srcBase + m], 0, row, m * featDim, featDim);
            output[i] = row;
        }
        return output;
    }

    /// <summary>Real `apply_cmvn`: `(x + shift) * scale` (an add then multiply -- Kaldi stores the NEGATED mean and INVERTED std specifically so this suffices, no subtract/divide).</summary>
    public static float[][] ApplyCmvn(float[][] features, float[] cmvnShift, float[] cmvnScale)
    {
        int t = features.Length;
        int dim = features[0].Length;
        var output = new float[t][];
        for (int i = 0; i < t; i++)
        {
            var row = new float[dim];
            for (int d = 0; d < dim; d++) row[d] = (features[i][d] + cmvnShift[d]) * cmvnScale[d];
            output[i] = row;
        }
        return output;
    }

    /// <summary>Full pipeline: raw PCM -> log-mel -> LFR splice -> CMVN. Returns frame-major [T, 560].</summary>
    public float[][] Extract(ReadOnlySpan<float> pcm16k, float[] cmvnShift, float[] cmvnScale)
    {
        var logMel = ExtractLogMel(pcm16k);
        if (logMel.Length == 0) return [];
        var spliced = ApplyLfr(logMel);
        return ApplyCmvn(spliced, cmvnShift, cmvnScale);
    }
}
