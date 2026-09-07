using OpenTail.Stingray.Audio.HiggsAudio;
using OpenTail.Stingray.Audio.Rvc;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real, live end-to-end smoke test for Higgs Audio TTS: real text prompt prefill
/// (via the real tokenizer + Qwen3 ForwardPass bridge), several real AR decode steps producing
/// RVQ codes, then decoding those codes through the real acoustic codec into a waveform --
/// chaining every piece real-weight verified individually this session into one live run.</summary>
public sealed class HiggsArStepperRealWeightsTests : HeavyTestBase
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

    [Fact]
    public void TextPrefillThenArSteps_OnRealCheckpoint_ProducesFiniteWaveform()
    {
        string? path = FindRepoFile("models/_models/higgs_audio_tts/Higgs-Audio-v3-TTS-4B-GGUF/higgs-audio-v3-tts-4b-q8_0.gguf");
        Assert.SkipUnless(path != null, "higgs-audio-v3-tts-4b-q8_0.gguf not found");

        using var model = GgufModel.Open(path!);
        var source = new RvcPackedTensorSource(model);
        using var llm = new HiggsLlmTensorSource(source, NumLayers, HiddenDim, NumHeads, NumKvHeads, HeadDim, FfDim, VocabSize, RopeTheta, RmsNormEps, NumCodebooks, AudioVocabSize);

        var hp = ModelHyperparams.FromGgufMetadata(llm.Metadata);
        using var backend = new CpuBackend();
        using var fwd = new ForwardPass(llm, backend, hp);

        var tokenizer = HiggsTtsTextTokenizer.LoadFromPackedGguf(model, audioTokenId: -100);
        var prompt = tokenizer.EncodePrompt("Hello there, this is a real end to end test.", "", 0);
        Assert.NotEmpty(prompt.TokenIds);

        var promptLogits = fwd.Prefill(prompt.TokenIds).ToArray();
        Assert.All(promptLogits, v => Assert.True(float.IsFinite(v)));

        var codec = HiggsCodecDecoderWeights.Load(source.GetTensor);

        // Real reference flow (generator.cpp ~line 436-448): the FIRST sampled codes come
        // directly from the prefill's own last-position hidden state (the real prompt ends in a
        // literal <|audio|> token) projected through the modality-embedding table -- BEFORE any
        // decode-step ForwardEmbedding call. Only subsequent codes come from feeding the
        // previous REAL (delay-masked) sampled codes forward. Real HiggsCodebookSampler delay
        // masking (sampler.cpp) applied each step: only codebooks [0, step] carry a real code
        // until step >= numCodebooks-1, everything after that point is BocId until unlocked.
        var sampler = new HiggsCodebookSampler(NumCodebooks);
        const int steps = NumCodebooks; // enough steps to unlock every codebook at least once
        var frames = new int[steps][];

        var raw0 = HiggsArStepper.SampleFromHidden(fwd.LastHidden, llm, NumCodebooks, AudioVocabSize);
        Assert.Equal(NumCodebooks, raw0.Length);
        Assert.All(raw0, c => Assert.InRange(c, 0, AudioVocabSize - 1));
        frames[0] = sampler.Step(raw0);

        int position = prompt.TokenIds.Length;
        for (int step = 1; step < steps; step++)
        {
            var raw = HiggsArStepper.Step(fwd, llm, sampler.LastCodes, position, NumCodebooks, AudioVocabSize);
            Assert.Equal(NumCodebooks, raw.Length);
            frames[step] = sampler.Step(raw);
            position++;
        }
        // After numCodebooks steps the delay window has fully opened (delay_count reaches
        // numCodebooks), so the LAST frame should carry a real sample in every codebook -- no
        // BocId placeholders left.
        Assert.DoesNotContain(frames[^1], c => c == HiggsCodebookSampler.BocId);

        // The AR's audio vocab (1026) can include reserved/special ids (BocId/EocId/StopCode)
        // that are never valid codec codes -- a real downstream consumer keeps generating until
        // every codebook is unlocked and gates on EOC for stopping (not ported this session);
        // for this structural smoke test, clamp any still-reserved id to 0 before decoding.
        for (int t = 0; t < steps; t++)
            for (int cb = 0; cb < NumCodebooks; cb++)
                if (frames[t][cb] < 0 || frames[t][cb] >= HiggsCodecDecoderWeights.CodebookSize)
                    frames[t][cb] = 0;

        var waveform = HiggsCodecDecoder.Decode(codec, frames);
        Assert.All(waveform, v => Assert.True(float.IsFinite(v)));
        Assert.Contains(waveform, v => v != 0f);
    }
}
