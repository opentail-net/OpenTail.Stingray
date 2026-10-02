using OpenTail.Stingray.Audio.HiggsAudio;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real-weight smoke test for <see cref="HiggsTtsTextTokenizer"/>.</summary>
public sealed class HiggsTtsTextTokenizerRealWeightsTests : HeavyTestBase
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
    public void LoadFromPackedGguf_BuildsRealPrompt_WithAndWithoutReference()
    {
        string? path = FindRepoFile("models/_models/higgs_audio_tts/Higgs-Audio-v3-TTS-4B-GGUF/higgs-audio-v3-tts-4b-q8_0.gguf");
        Assert.SkipUnless(path != null, "higgs-audio-v3-tts-4b-q8_0.gguf not found");

        using var model = OpenTail.Stingray.Core.GgufModel.Open(path!);

        // Real reference default is a sentinel -100 (masked-out id, never a real vocab entry) --
        // confirmed via the real dumped config.json, which carries exactly that value for this
        // checkpoint. Not necessarily positive, so track "found" separately from the value.
        int audioTokenId = 0;
        bool foundAudioTokenId = false;
        if (OpenTail.Stingray.Audio.AudioCppEmbeddedFiles.TryGet(model, "config.json", out var configBytes))
        {
            string content = System.Text.Encoding.UTF8.GetString(configBytes);
            using var doc = System.Text.Json.JsonDocument.Parse(content);
            if (doc.RootElement.TryGetProperty("audio_token_id", out var idEl))
            {
                audioTokenId = idEl.GetInt32();
                foundAudioTokenId = true;
            }
        }
        Assert.True(foundAudioTokenId, "config.json must carry a real audio_token_id.");

        var tokenizer = HiggsTtsTextTokenizer.LoadFromPackedGguf(model, audioTokenId);

        var noRef = tokenizer.EncodePrompt("Hello, this is a real Higgs Audio TTS test.", "", 0);
        Assert.NotEmpty(noRef.TokenIds);
        Assert.Equal(0, noRef.DelayedReferenceTokens);

        var withRef = tokenizer.EncodePrompt("Hello again.", "This is the reference voice speaking.", 8);
        Assert.NotEmpty(withRef.TokenIds);
        Assert.Equal(8, withRef.DelayedReferenceTokens);
        Assert.True(withRef.TokenIds.Length > noRef.TokenIds.Length);

        int placeholderCount = withRef.TokenIds.Count(id => id == audioTokenId);
        Assert.Equal(8, placeholderCount);
    }
}
