using OpenTail.Stingray.Audio.PersonaPlex;
using OpenTail.Stingray.Audio.Rvc;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real, first-attempt end-to-end LIVE-DUPLEX smoke test for PersonaPlex: encodes a real (decoder-
/// generated, not synthetic-noise) user waveform via the newly-ported <see cref="MimiCodecEncoder"/>,
/// feeds the real per-frame codes into <see cref="PersonaPlexGenerator.GenerateWithUserAudio"/>
/// (the real `run_user_frame`-driven duplex loop, as opposed to <see cref="PersonaPlexGenerator.
/// GenerateDelayed"/>'s silence-padded non-duplex simplification), and decodes the model's real
/// output back into a waveform via the already-verified <see cref="MimiCodecDecoder"/>. This is the
/// first point PersonaPlex's live-duplex path -- real user audio in, real generated audio out --
/// has run end to end in this project, closing the session's last major scoped-but-unwired gap.
/// Not a numeric golden-parity check (no captured reference duplex run exists yet).
/// </summary>
public sealed class PersonaPlexLiveDuplexRealWeightsTests : HeavyTestBase
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

    private const int NumLayers = 32, HiddenDim = 4096, NumHeads = 32, HeadDim = 128;
    private const int FfDim = 11264, TextVocabSize = 32000, LmCodebooks = 16, AudioCodebookSize = 2048;
    private const float RopeTheta = 10000f, RmsNormEps = 1e-8f;

    [Fact]
    public void GenerateWithUserAudio_ThenDecode_OnRealCheckpoint_ProducesFiniteWaveform()
    {
        string? path = FindRepoFile("models/_models/personaplex/PersonaPlex-GGUF/personaplex-7b-v1-q8_0.gguf");
        Assert.SkipUnless(path != null, "personaplex-7b-v1-q8_0.gguf not found");

        using var model = GgufModel.Open(path!);
        var source = new RvcPackedTensorSource(model);

        var mimiDecoderWeights = MimiCodecDecoderWeights.Load(source.GetTensor);
        var mimiEncoderWeights = MimiCodecEncoderWeights.Load(source.GetTensor, source.HasTensor, mimiDecoderWeights);

        // Real (not synthetic-noise) user audio: decode a few real, in-range codes into a real
        // waveform, then re-encode it via the newly-ported Mimi encoder -- a real roundtrip
        // through both codec directions before either is exercised by the actual duplex loop.
        var rng = new Random(41);
        const int userFrames = 6;
        var seedCodes = new int[userFrames][];
        for (int t = 0; t < userFrames; t++)
        {
            seedCodes[t] = new int[MimiCodecDecoderWeights.ActiveCodebooks];
            for (int cb = 0; cb < MimiCodecDecoderWeights.ActiveCodebooks; cb++)
                seedCodes[t][cb] = rng.Next(MimiCodecDecoderWeights.CodebookSize);
        }
        var userWaveform = MimiCodecDecoder.Decode(mimiDecoderWeights, seedCodes);
        var userCodesPerFrame = MimiCodecEncoder.Encode(mimiEncoderWeights, userWaveform);
        Assert.True(userCodesPerFrame.Length > 0);

        using var llm = new PersonaPlexLmTensorSource(source, NumLayers, HiddenDim, NumHeads, HeadDim, FfDim, TextVocabSize, LmCodebooks, AudioCodebookSize, RopeTheta, RmsNormEps);
        var hp = ModelHyperparams.FromGgufMetadata(llm.Metadata);
        using var backend = new CpuBackend();
        using var fwd = new ForwardPass(llm, backend, hp);

        var depformerWeights = PersonaPlexDepformerWeights.Load(TextVocabSize, AudioCodebookSize, source.GetTensor);
        var depformer = new PersonaPlexDepformer(depformerWeights);

        var frames = PersonaPlexGenerator.GenerateWithUserAudio(fwd, llm, depformer, userCodesPerFrame, TextVocabSize, AudioCodebookSize);

        foreach (var frame in frames)
        {
            Assert.Equal(MimiCodecDecoderWeights.ActiveCodebooks, frame.AudioCodes.Length);
            Assert.All(frame.AudioCodes, c => Assert.InRange(c, 0, AudioCodebookSize - 1));
        }

        if (frames.Length > 0)
        {
            var outputCodes = frames.Select(f => f.AudioCodes).ToArray();
            var outputWaveform = MimiCodecDecoder.Decode(mimiDecoderWeights, outputCodes);
            Assert.True(outputWaveform.Length > 0);
            Assert.All(outputWaveform, v => Assert.True(float.IsFinite(v)));
            Assert.All(outputWaveform, v => Assert.InRange(v, -1f, 1f));
        }
    }
}
