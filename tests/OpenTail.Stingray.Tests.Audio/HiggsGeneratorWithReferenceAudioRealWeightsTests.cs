using OpenTail.Stingray.Audio.HiggsAudio;
using OpenTail.Stingray.Audio.OmniVoice;
using OpenTail.Stingray.Audio.Rvc;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real, live end-to-end smoke test for Higgs Audio TTS's reference-audio-conditioned (voice-
/// cloning) generation loop (<see cref="HiggsGenerator.GenerateWithReferenceAudio"/>): encodes a
/// synthetic reference waveform through the real codec encoder (already verified this session),
/// fuses it into the real prompt, prefills position-by-position, then runs the same real AR decode
/// loop as the zero-shot path. First real reference-audio-conditioned generation run for this
/// model.
/// </summary>
public sealed class HiggsGeneratorWithReferenceAudioRealWeightsTests : HeavyTestBase
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

    private const int NumLayers = 36, HiddenDim = 2560, NumHeads = 32, NumKvHeads = 8, HeadDim = 128;
    private const int FfDim = 9728, VocabSize = 151936, NumCodebooks = 8, AudioVocabSize = 1026;
    private const float RopeTheta = 1_000_000f, RmsNormEps = 1e-6f;
    private const int AudioTokenId = -100; // real config.audio_token_id, confirmed via assets.cpp's own validation

    [Fact]
    public void GenerateWithReferenceAudio_OnRealCheckpoint_ProducesFiniteNonSilentAudio()
    {
        string? path = FindRepoFile("models/_models/higgs_audio_tts/Higgs-Audio-v3-TTS-4B-GGUF/higgs-audio-v3-tts-4b-q8_0.gguf");
        Assert.SkipUnless(path != null, "higgs-audio-v3-tts-4b-q8_0.gguf not found");

        using var model = GgufModel.Open(path!);
        var source = new RvcPackedTensorSource(model);
        string Codec(string name) => "tied.embedding.modality_embeddings.0.model." + name;

        using var llm = new HiggsLlmTensorSource(source, NumLayers, HiddenDim, NumHeads, NumKvHeads, HeadDim, FfDim, VocabSize, RopeTheta, RmsNormEps, NumCodebooks, AudioVocabSize);
        var hp = ModelHyperparams.FromGgufMetadata(llm.Metadata);
        using var backend = new CpuBackend();
        using var fwd = new ForwardPass(llm, backend, hp);

        var tokenizer = HiggsTtsTextTokenizer.LoadFromPackedGguf(model, AudioTokenId);
        var codecWeights = HiggsCodecDecoderWeights.Load(source.GetTensor);
        var textEmbeddingTable = source.GetTensor("tied.embedding.text_embedding.weight");

        // Real reference-audio encode chain (already real-weight verified this session).
        var acousticWeights = new OmniVoiceAcousticEncoderWeights(name => source.GetTensor(Codec(name)));
        var semanticWeights = new OmniVoiceSemanticWeights(name => source.GetTensor(Codec(name)));
        var semanticPostWeights = HiggsSemanticPostEncoder.Weights.Load(source.GetTensor, "tied.embedding.modality_embeddings.0.model.");

        int totalAcousticStride = OmniVoiceAcousticEncoderWeights.DownsamplingRatios.Aggregate(1, (a, b) => a * b);
        int acousticFrameCount = 4;
        int samples24k = totalAcousticStride * acousticFrameCount;
        int samples16k = samples24k;

        var rng = new Random(53);
        var waveform24k = new float[samples24k];
        for (int i = 0; i < samples24k; i++) waveform24k[i] = (float)(rng.NextDouble() * 0.2 - 0.1);
        var waveform16k = new float[samples16k];
        for (int i = 0; i < samples16k; i++) waveform16k[i] = (float)(rng.NextDouble() * 0.2 - 0.1);

        var codesByCodebook = HiggsCodecEncoder.Encode(acousticWeights, semanticWeights, semanticPostWeights, codecWeights, waveform24k, waveform16k);
        Assert.Equal(NumCodebooks, codesByCodebook.Length);
        int refFrames = codesByCodebook[0].Length;
        var referenceRawCodes = new int[refFrames][];
        for (int f = 0; f < refFrames; f++)
        {
            var row = new int[NumCodebooks];
            for (int cb = 0; cb < NumCodebooks; cb++) row[cb] = codesByCodebook[cb][f];
            referenceRawCodes[f] = row;
        }

        var result = HiggsGenerator.GenerateWithReferenceAudio(
            fwd, llm, tokenizer, codecWeights, textEmbeddingTable,
            "Hi.", "This is the reference speaker.", referenceRawCodes,
            NumCodebooks, AudioVocabSize, AudioTokenId,
            maxTokens: 200);

        Assert.NotEmpty(result.RawCodes);
        Assert.All(result.RawCodes, frame =>
        {
            Assert.Equal(NumCodebooks, frame.Length);
            Assert.All(frame, c => Assert.InRange(c, 0, HiggsCodecDecoderWeights.CodebookSize - 1));
        });
        Assert.NotEmpty(result.AudioSamples);
        Assert.All(result.AudioSamples, v => Assert.True(float.IsFinite(v)));
        Assert.Contains(result.AudioSamples, v => v != 0f);
    }
}
