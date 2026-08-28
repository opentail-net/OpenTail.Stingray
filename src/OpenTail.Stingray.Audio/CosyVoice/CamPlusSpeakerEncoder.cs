using System.Linq;
using OpenTail.Stingray.Audio.Primitives;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Audio.CosyVoice;

/// <summary>
/// Real x-vector speaker embedding extraction for CosyVoice's zero-shot voice cloning, via the
/// checkpoint's own `campplus.onnx` (found locally at `models/campplus.onnx` -- no download
/// needed, and NOT re-implemented as a native neural net: the real reference
/// (`examples/cosyvoice.cpp/src/cosyvoice-frontend.cpp`'s `extract_spk_embedding`) doesn't
/// reimplement CAM++ either, it just runs the same pre-exported ONNX graph via ONNX Runtime --
/// this class does the same, using this codebase's existing <see cref="OnnxModelSession"/>.
///
/// <para>What IS ported natively here is the Kaldi-compatible fbank feature pipeline CAM++
/// expects, ported tensor-for-tensor from that same reference's `extract_spk_embedding`
/// (16kHz mono, 25ms/10ms Povey-windowed frames, 512-point FFT, an 80-bin mel filterbank with
/// low_freq=20Hz/high_freq=8000Hz, log energy, per-utterance per-mel-bin mean subtraction) --
/// confirmed exact by reading its real filter-construction code (mel_basis_spk / povey_window
/// built in `cosyvoice_frontend_context`'s constructor), not guessed from a generic recipe.</para>
///
/// <para>Fixes CosyVoice3Pipeline's previously all-zero 192-dim speaker-embedding placeholder
/// (see its own doc comment) when a real reference audio file is supplied -- CosyVoice3 is a
/// zero-shot voice-cloning model with no baked-in voice presets (confirmed via `list-metadata`),
/// so a non-zero, real per-speaker x-vector is the single highest-leverage piece of real
/// conditioning available without also porting the reference-mel/CFG path (still not done, see
/// CosyVoice3Pipeline's own doc comment).</para>
/// </summary>
public static class CamPlusSpeakerEncoder
{
    public const int SampleRate = 16000;
    public const int WinSize = 400;   // 25ms @ 16kHz
    public const int HopSize = 160;   // 10ms @ 16kHz
    public const int NFft = 512;
    public const int NumMelBins = 80;
    private const float Preemph = 0.97f;
    private const float LowFreqHz = 20f;
    private const float HighFreqHz = 8000f;

    private static readonly float[] PoveyWindow = BuildPoveyWindow();
    private static readonly float[] MelFilterbank = BuildMelFilterbank();

    /// <summary>
    /// Extracts a real 192-dim speaker (x-vector) embedding from 16kHz mono reference audio, by
    /// running CAM++'s real ONNX graph. Returns null if `campplus.onnx` isn't available at the
    /// given path (caller should fall back to CosyVoice3Pipeline's existing zero-vector behavior).
    /// </summary>
    public static float[]? Extract(string campplusOnnxPath, ReadOnlySpan<float> pcm16k)
    {
        using var session = OnnxModelSession.TryLoad(campplusOnnxPath);
        if (session is null) return null;

        var feat = ExtractFbank(pcm16k);
        if (feat.Length == 0) return null;

        int numFrames = feat.Length / NumMelBins;
        string inputName = session.InputNames.First();
        var result = session.Run((inputName, feat, [1, numFrames, NumMelBins]));
        return result.Count > 0 ? result.Values.First() : null;
    }

    /// <summary>
    /// Real Kaldi-compatible 80-bin log-mel filterbank feature extraction, matching
    /// `cosyvoice-frontend.cpp`'s `extract_spk_embedding` exactly (see this class's doc comment).
    /// Returns frame-major [numFrames * NumMelBins] (feat[f * NumMelBins + m]).
    /// </summary>
    internal static float[] ExtractFbank(ReadOnlySpan<float> pcm)
    {
        if (pcm.Length < WinSize) return [];

        int numFrames = (pcm.Length - WinSize) / HopSize + 1;
        var feat = new float[numFrames * NumMelBins];

        var frame = new float[NFft];
        var powerSpectrum = new float[NFft / 2 + 1];

        for (int f = 0; f < numFrames; f++)
        {
            int start = f * HopSize;

            // 1. Remove-DC-offset (subtract the raw frame's own mean) -- BEFORE pre-emphasis,
            // matching the reference's `mean = sum(frame)/win_size; frame -= mean`.
            float mean = 0f;
            for (int i = 0; i < WinSize; i++) mean += pcm[start + i];
            mean /= WinSize;

            for (int i = 0; i < WinSize; i++) frame[i] = pcm[start + i] - mean;

            // 2. Pre-emphasis (0.97) against the DC-removed frame shifted by one sample, with
            // frame[0]'s "previous" sample defined as itself (matches `prev(i,0) = frame(i,0)`).
            float prev0 = frame[0];
            for (int i = WinSize - 1; i > 0; i--)
                frame[i] -= Preemph * frame[i - 1];
            frame[0] -= Preemph * prev0;

            // 3. Povey window, then zero-pad 400 -> 512 for the FFT.
            for (int i = 0; i < WinSize; i++) frame[i] *= PoveyWindow[i];
            Array.Clear(frame, WinSize, NFft - WinSize);

            SpectralKernels.ComputePowerSpectrum(frame, powerSpectrum);

            // 4. Mel-filterbank projection + log energy (floor 1e-10, matching the reference).
            for (int m = 0; m < NumMelBins; m++)
            {
                float energy = 0f;
                int fbBase = m * powerSpectrum.Length;
                for (int k = 0; k < powerSpectrum.Length; k++)
                    energy += powerSpectrum[k] * MelFilterbank[fbBase + k];

                feat[f * NumMelBins + m] = MathF.Log(Math.Max(1e-10f, energy));
            }
        }

        // 5. Per-utterance, per-mel-bin mean subtraction across all frames (cepstral mean norm).
        for (int m = 0; m < NumMelBins; m++)
        {
            double sum = 0;
            for (int f = 0; f < numFrames; f++) sum += feat[f * NumMelBins + m];
            float mean = (float)(sum / numFrames);
            for (int f = 0; f < numFrames; f++) feat[f * NumMelBins + m] -= mean;
        }

        return feat;
    }

    /// <summary>Kaldi Povey window: (0.5 - 0.5*cos(2*pi*n/(N-1)))^0.85, N=400, non-periodic.</summary>
    private static float[] BuildPoveyWindow()
    {
        var w = new float[WinSize];
        for (int i = 0; i < WinSize; i++)
        {
            float hann = 0.5f - 0.5f * MathF.Cos(2f * MathF.PI * i / (WinSize - 1));
            w[i] = MathF.Pow(hann, 0.85f);
        }
        return w;
    }

    /// <summary>
    /// Kaldi-style mel filterbank: 80 triangular bins over 256 FFT bins (Nyquist bin 256 forced
    /// to 0, matching the reference exactly), mel(hz) = 1127*ln(1 + hz/700), low=20Hz, high=8000Hz.
    /// Row-major [NumMelBins, NFft/2+1].
    /// </summary>
    private static float[] BuildMelFilterbank()
    {
        int numFftBins = NFft / 2; // 256, the reference's own `num_fft_bins`
        int numFreqBins = NFft / 2 + 1; // 257
        var fb = new float[NumMelBins * numFreqBins];

        static float HzToMel(float hz) => 1127.0f * MathF.Log(1.0f + hz / 700.0f);

        float melLow = HzToMel(LowFreqHz);
        float melHigh = HzToMel(HighFreqHz);
        float melDelta = (melHigh - melLow) / (NumMelBins + 1);

        var leftMel = new float[NumMelBins];
        var centerMel = new float[NumMelBins];
        var rightMel = new float[NumMelBins];
        for (int i = 0; i < NumMelBins; i++)
        {
            leftMel[i] = melLow + melDelta * i;
            centerMel[i] = melLow + melDelta * (i + 1);
            rightMel[i] = melLow + melDelta * (i + 2);
        }

        float fftBinWidth = (float)SampleRate / NFft;
        var mel = new float[numFftBins];
        for (int i = 0; i < numFftBins; i++)
            mel[i] = HzToMel(i * fftBinWidth);

        for (int i = 0; i < NumMelBins; i++)
        {
            float upScaling = centerMel[i] - leftMel[i];
            float downScaling = rightMel[i] - centerMel[i];
            int fbBase = i * numFreqBins;
            for (int j = 0; j < numFftBins; j++)
            {
                float up = (mel[j] - leftMel[i]) / upScaling;
                float down = (rightMel[i] - mel[j]) / downScaling;
                fb[fbBase + j] = Math.Max(0f, Math.Min(up, down));
            }
            fb[fbBase + numFftBins] = 0f; // Nyquist bin, matches the reference forcing index 256 to 0
        }

        return fb;
    }
}
