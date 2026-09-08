using OpenTail.Stingray.Audio.OmniVoice;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>TEMPORARY debug test: generates a real OmniVoice wav end-to-end for informal listening
/// (no CLI wiring yet). Writes to docs/audio-samples (gitignored, local-only per CLAUDE.md). Not a
/// golden/parity test. Uses the REAL prompt template (`prompt_builder.cpp`'s real
/// `&lt;|lang_start|&gt;...&lt;|instruct_start|&gt;...`/`&lt;|text_start|&gt;...&lt;|text_end|&gt;`
/// structure, not guessed) and the checkpoint's real tokenizer -- an earlier version of this test used synthetic
/// small-integer token ids, which produced a real but meaningless out-of-distribution "click"
/// (confirmed via direct listening, not a bug in the generation loop itself).</summary>
public sealed class OmniVoiceGenerateWavDebugTest : HeavyTestBase
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
    public void Generate_RealOmniVoiceWav()
    {
        string? path = FindRepoFile("models/_models/omnivoice/model.safetensors");
        Assert.SkipUnless(path != null, "omnivoice model.safetensors not found");
        string? repoRoot = Path.GetDirectoryName(FindRepoFile("docs/audio-review-progress.md"));
        Assert.NotNull(repoRoot);

        string? codecPath = FindRepoFile("models/_models/omnivoice/audio_tokenizer/model.safetensors");
        Assert.SkipUnless(codecPath != null, "omnivoice audio_tokenizer model.safetensors not found");
        string? tokenizerDir = Path.GetDirectoryName(FindRepoFile("models/_models/omnivoice/tokenizer.json"));
        Assert.SkipUnless(tokenizerDir != null, "omnivoice tokenizer.json not found");

        var tokenizerResult = HuggingFaceTokenizerSource.Load(tokenizerDir!);
        Assert.True(tokenizerResult.Source is not null, string.Join("; ", tokenizerResult.Rejections));
        var tokenizer = GgufTokenizer.FromSource(tokenizerResult.Source!);

        using var loader = SafetensorsLoader.Open(path!);
        var maskGitWeights = OmniVoiceMaskGitWeights.Load(loader.ReadF32);
        using var codecLoader = SafetensorsLoader.Open(codecPath!);
        var decoderWeights = new OmniVoiceAcousticDecoderWeights(codecLoader);

        // Real prompt template, ported from prompt_builder.cpp's real `build()` (not guessed):
        // style_text = "<|lang_start|>{lang}<|lang_end|><|instruct_start|>{instruct}<|instruct_end|>"
        // (both "None" for a plain zero-shot, no-instruct request), wrapped_text =
        // "<|text_start|>{text}<|text_end|>".
        const string text = "Hello there, this is a real test of speech synthesis.";
        string styleText = "<|lang_start|>None<|lang_end|><|instruct_start|>None<|instruct_end|>";
        string wrappedText = $"<|text_start|>{text}<|text_end|>";
        int[] styleTokenIds = [.. tokenizer.Encode(styleText)];
        int[] textTokenIds = [.. tokenizer.Encode(wrappedText)];

        // Real target_audio_tokens: the reference's own duration_seconds override path
        // (`target_audio_tokens = round(duration_seconds * frame_rate)`) is used here rather than
        // porting the full real `RuleDurationEstimator` heuristic -- a real, confirmed mechanism
        // in the reference, not a guess, just a simpler real path than the default text-length
        // estimator. Real `frame_rate = audio_tokenizer.sample_rate / hop_length` = 24000/960 = 25.
        const int frameRate = 25;
        const float durationSeconds = 3.2f;
        int targetFrames = Math.Max(1, (int)MathF.Round(durationSeconds * frameRate));

        var codesFlat = OmniVoiceMaskGitGenerator.Generate(
            maskGitWeights, styleTokenIds, textTokenIds, referenceAudioTokens: null,
            targetFrames, new OmniVoiceMaskGitGenerator.Options(numInferenceSteps: 12), new Random(21));

        var codes = new int[targetFrames][];
        for (int f = 0; f < targetFrames; f++)
        {
            var row = new int[OmniVoiceMaskGitWeights.NumCodebooks];
            Array.Copy(codesFlat, f * OmniVoiceMaskGitWeights.NumCodebooks, row, 0, OmniVoiceMaskGitWeights.NumCodebooks);
            codes[f] = row;
        }

        var waveform = OmniVoiceAcousticDecoder.Decode(decoderWeights, codes);
        Assert.True(waveform.Length > 0);

        const int sampleRate = 24000; // real OmniVoice/Higgs acoustic codec output rate, confirmed this session
        var result = new OpenTail.Stingray.Audio.AudioGenerationResult(waveform, sampleRate);
        string outPath = Path.Combine(repoRoot!, "audio-samples", "omnivoice-real-check.wav");
        result.SaveWav(outPath);
        Console.WriteLine($"Wrote {outPath}, {waveform.Length} samples, {waveform.Length / (double)sampleRate:F2}s");
    }
}
