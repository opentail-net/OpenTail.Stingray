using OpenTail.Stingray.Audio.Rvc;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real-weight smoke test for Higgs Audio TTS's real `codec_project` + RVQ `quantizer_encode`
/// tail (ported this session) -- loads the real `project_in`/`fc` tensors (confirmed to exist in
/// the real checkpoint, previously unused by the decode-only path) and runs them on a synthetic
/// pre-encoder `[832]` concatenated frame, confirming finite, in-range codes.
/// </summary>
public sealed class HiggsCodecEncoderRealWeightsTests : HeavyTestBase
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
    public void ProjectThenQuantize_OnRealCheckpoint_ProducesInRangeCodes()
    {
        string? path = FindRepoFile("models/_models/higgs_audio_tts/Higgs-Audio-v3-TTS-4B-GGUF/higgs-audio-v3-tts-4b-q8_0.gguf");
        Assert.SkipUnless(path != null, "higgs-audio-v3-tts-4b-q8_0.gguf not found");

        using var model = GgufModel.Open(path!);
        var source = new RvcPackedTensorSource(model);
        var weights = OpenTail.Stingray.Audio.HiggsAudio.HiggsCodecDecoderWeights.Load(source.GetTensor);

        var rng = new Random(3);
        var concatenatedFrame = new float[OpenTail.Stingray.Audio.HiggsAudio.HiggsCodecDecoderWeights.CodecProjectInputSize];
        for (int i = 0; i < concatenatedFrame.Length; i++) concatenatedFrame[i] = (float)(rng.NextDouble() * 2 - 1);

        var hidden = OpenTail.Stingray.Audio.HiggsAudio.HiggsCodecEncoder.Project(weights, concatenatedFrame);
        Assert.Equal(OpenTail.Stingray.Audio.HiggsAudio.HiggsCodecDecoderWeights.CodecHiddenSize, hidden.Length);
        Assert.All(hidden, v => Assert.True(float.IsFinite(v)));

        var codes = OpenTail.Stingray.Audio.HiggsAudio.HiggsCodecEncoder.QuantizeFrame(weights, hidden);
        Assert.Equal(OpenTail.Stingray.Audio.HiggsAudio.HiggsCodecDecoderWeights.NumCodebooks, codes.Length);
        Assert.All(codes, c => Assert.InRange(c, 0, OpenTail.Stingray.Audio.HiggsAudio.HiggsCodecDecoderWeights.CodebookSize - 1));
    }
}
