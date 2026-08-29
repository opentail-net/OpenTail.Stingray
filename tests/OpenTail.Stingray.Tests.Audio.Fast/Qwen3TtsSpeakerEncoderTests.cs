using OpenTail.Stingray.Audio.QwenTTS;

namespace OpenTail.Stingray.Tests.Audio;

public sealed class Qwen3TtsSpeakerEncoderTests
{
    [Fact]
    public void SpeakerEncoder_Extracts192DimNormalizedEmbedding()
    {
        var encoder = new Qwen3TtsSpeakerEncoder();
        int numFrames = 50; // 50 time frames
        var mel = new float[numFrames * 128];
        var rng = new Random(42);
        for (int i = 0; i < mel.Length; i++)
        {
            mel[i] = (float)(rng.NextDouble() * 5.0 - 2.5);
        }

        var emb = encoder.ExtractSpeakerEmbedding(mel, numFrames);
        Assert.NotNull(emb);
        Assert.Equal(192, emb.Length);

        // Verify L2 norm is ~1.0
        float sumSq = 0f;
        for (int i = 0; i < emb.Length; i++)
        {
            Assert.False(float.IsNaN(emb[i]));
            Assert.False(float.IsInfinity(emb[i]));
            sumSq += emb[i] * emb[i];
        }
        Assert.InRange(MathF.Sqrt(sumSq), 0.99f, 1.01f);
    }
}
