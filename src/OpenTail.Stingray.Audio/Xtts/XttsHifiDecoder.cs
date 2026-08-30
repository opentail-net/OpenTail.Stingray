using OpenTail.Stingray.Audio.Primitives;

namespace OpenTail.Stingray.Audio.Xtts;

/// <summary>
/// Real XTTS-v2 `HifiDecoder.forward` (`TTS/tts/layers/xtts/hifigan_decoder.py`): upsamples the
/// GPT trunk's real latent output (channel-first `[1024,T]`, from
/// <see cref="XttsGptLatents.ComputeLatents"/>) to the vocoder's real input time resolution via two
/// chained `torch.nn.functional.interpolate(..., mode="linear")` stages -- first by
/// `ar_mel_length_compression(1024)/output_hop_length(256)=4x`, then by
/// `output_sample_rate(24000)/input_sample_rate(22050)` -- then runs <see cref="XttsVocoder"/>.
/// </summary>
public static class XttsHifiDecoder
{
    private const double CompressionRatio = 1024.0 / 256.0;
    private const double SampleRateRatio = 24000.0 / 22050.0;

    /// <summary>latents is channel-first [1024, tIn] (real GPT trunk latents). speakerEmbedding is the real 512-dim d-vector. Returns mono waveform samples.</summary>
    public static float[] Forward(XttsVocoderWeights vocoderWeights, float[] latents, int tIn, float[] speakerEmbedding)
    {
        var z1 = HifiGanKernels.LinearInterpolate1d(latents, XttsVocoderWeights.InChannels, tIn, CompressionRatio, out int t1);
        var z2 = HifiGanKernels.LinearInterpolate1d(z1, XttsVocoderWeights.InChannels, t1, SampleRateRatio, out int t2);
        return XttsVocoder.Forward(vocoderWeights, z2, t2, speakerEmbedding);
    }
}
