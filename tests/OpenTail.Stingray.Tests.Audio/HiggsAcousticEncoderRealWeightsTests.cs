using OpenTail.Stingray.Audio.OmniVoice;
using OpenTail.Stingray.Audio.Rvc;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real-weight smoke test for Higgs Audio TTS's real acoustic (DAC-style) encoder -- confirmed
/// identical real architecture to OmniVoice's own acoustic encoder (same
/// `kEncoderChannels=[64,128,256,512,1024,2048]` progression, same real
/// `kUpsampleRatios=[8,5,4,2,3]` reused for the encoder's strided downsample convs, same 256-
/// channel final projection -- verified directly against `higgs_audio_tts/codec.cpp`'s own
/// `load_encoder_block`, correcting an earlier session mix-up that compared different pipeline
/// stages and wrongly concluded these encoders were NOT reusable). Reuses
/// <see cref="OmniVoiceAcousticEncoder"/>/<see cref="OmniVoiceAcousticEncoderWeights"/> unchanged
/// via the new generalized constructor, loading Higgs's own real
/// `tied.embedding.modality_embeddings.0.model.acoustic_encoder.*` tensors.
/// </summary>
public sealed class HiggsAcousticEncoderRealWeightsTests : HeavyTestBase
{
    private static string? FindRepoFile(string relPath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relPath);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void Encode_OnRealHiggsCheckpoint_ProducesFiniteNonDegenerateLatent()
    {
        string? path = FindRepoFile("models/_models/higgs_audio_tts/Higgs-Audio-v3-TTS-4B-GGUF/higgs-audio-v3-tts-4b-q8_0.gguf");
        Assert.SkipUnless(path != null, "higgs-audio-v3-tts-4b-q8_0.gguf not found");

        using var model = GgufModel.Open(path!);
        var source = new RvcPackedTensorSource(model);
        string Codec(string name) => "tied.embedding.modality_embeddings.0.model." + name;

        var weights = new OmniVoiceAcousticEncoderWeights(name => source.GetTensor(Codec(name)));

        int totalStride = OmniVoiceAcousticEncoderWeights.DownsamplingRatios.Aggregate(1, (a, b) => a * b);
        int samples = totalStride * 20;
        var rng = new Random(9);
        var waveform = new float[samples];
        for (int i = 0; i < samples; i++) waveform[i] = (float)(rng.NextDouble() * 0.2 - 0.1);

        var (latent, frames) = OmniVoiceAcousticEncoder.Encode(weights, waveform);

        Assert.True(frames > 0);
        Assert.Equal(frames * 256, latent.Length);
        Assert.All(latent, v => Assert.True(float.IsFinite(v)));
        Assert.Contains(latent, v => v != 0f);
    }
}
