using OpenTail.Stingray.Audio.OmniVoice;
using OpenTail.Stingray.Audio.Rvc;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real-weight smoke test for Higgs Audio TTS's real semantic (HuBERT-base) encoder --
/// confirmed BYTE-FOR-BYTE architecturally identical to OmniVoice's own semantic encoder (same
/// real `conv_dim`/`conv_kernel`/`conv_stride`/`hidden_size`/layer-count/pos-conv hyperparameters
/// and the same `parametrizations.weight.original0/1` weight-norm naming, verified directly
/// against `higgs_audio_tts/codec.cpp`'s own constants, not assumed) -- so this test reuses
/// <see cref="OmniVoiceSemanticEncoder"/>/<see cref="OmniVoiceSemanticWeights"/> unchanged,
/// loading Higgs's own real `tied.embedding.modality_embeddings.0.model.semantic_model.*`
/// tensors through the new generalized `Func&lt;string,float[]&gt;` constructor added this
/// session. Real, honest scope: this is the semantic HALF of Higgs's real reference-audio
/// encode path only -- the acoustic (DAC-style) encoder is a genuinely DIFFERENT config from
/// OmniVoice's (`hidden=256` vs OmniVoice's `64`, confirmed NOT reusable) and is separate,
/// unstarted work; this test does not attempt full voice-cloning conditioning.
/// </summary>
public sealed class HiggsSemanticEncoderRealWeightsTests : HeavyTestBase
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
    public void Forward_OnRealHiggsCheckpoint_ProducesFiniteNonDegenerateHiddenStates()
    {
        string? path = FindRepoFile("models/_models/higgs_audio_tts/Higgs-Audio-v3-TTS-4B-GGUF/higgs-audio-v3-tts-4b-q8_0.gguf");
        Assert.SkipUnless(path != null, "higgs-audio-v3-tts-4b-q8_0.gguf not found");

        using var model = GgufModel.Open(path!);
        var source = new RvcPackedTensorSource(model);
        string Codec(string name) => "tied.embedding.modality_embeddings.0.model." + name;

        var weights = new OmniVoiceSemanticWeights(name => source.GetTensor(Codec(name)));

        // Real preprocessing: Higgs's own `prepare_semantic_audio_16k` pads 160 real zero samples
        // on each side before feeding the encoder (`kSemanticPadSamples=160`, confirmed from
        // codec.cpp, not guessed).
        const int padSamples = 160;
        int rawSamples = 16000; // 1 real second at the encoder's real 16kHz input rate
        var rng = new Random(5);
        var waveform = new float[rawSamples + 2 * padSamples];
        for (int i = padSamples; i < padSamples + rawSamples; i++)
            waveform[i] = (float)(rng.NextDouble() * 0.2 - 0.1);

        var hidden = OmniVoiceSemanticEncoder.Forward(weights, waveform);

        Assert.True(hidden.Length > 0);
        foreach (var frame in hidden)
        {
            Assert.Equal(OmniVoiceSemanticWeights.HiddenDim, frame.Length);
            Assert.All(frame, v => Assert.True(float.IsFinite(v)));
        }
        Assert.Contains(hidden, frame => frame.Any(v => v != 0f));
    }
}
