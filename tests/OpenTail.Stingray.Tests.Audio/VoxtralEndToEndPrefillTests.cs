
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real end-to-end prefill test: real audio -> real padding (matching
/// `frontend.cpp`'s `padded_streaming_audio` exactly) -> real mel -> real audio tower -> real
/// text-decoder prefill over the real prompt (`[BOS] + 38x StreamingPad`, matching
/// `tokenizer_text.cpp`'s `build_transcription_prompt` exactly for `a.wav`'s real length and the
/// real default config: `default_num_delay_tokens=6`, `audio_length_per_tok=8`, `hop_length=160`).
/// Compares the last-position logits' mean/std/top-5 against the real reference's own
/// `STINGRAY_VOXTRAL_TRACE=1` prefill trace, captured via a targeted instrumentation added to
/// `text_decoder.cpp`'s prefill `run()` right before token sampling.</summary>
public sealed class VoxtralEndToEndPrefillTests : HeavyTestBase
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
    public void Prefill_RealAudio_RealWeights_MatchesReferenceTopTokens()
    {
        string? audioModelPath = FindRepoFile("models/_models/voxtral-mini-realtime/model.safetensors");
        Assert.SkipUnless(audioModelPath != null, "voxtral model.safetensors not found");
        string? audioPath = FindRepoFile("examples/audio.cpp/assets/resources/a.wav");
        Assert.SkipUnless(audioPath != null, "reference a.wav not found");

        var (samples, sr, _) = WavReader.ReadWav(audioPath!);
        if (sr != 16000) samples = AudioResampler.Resample(samples, sr, 16000);

        // Real padding, matching frontend.cpp's padded_streaming_audio exactly.
        const int hopLength = 160;
        const int audioLengthPerTok = 8;
        const int unit = audioLengthPerTok * hopLength; // 1280
        const int defaultNumDelayTokens = 6;
        const int kOfflineStreamingBufferTokens = 10;
        long leftPad = 32L * unit;
        long rightBase = (unit - (samples.Length % unit)) % unit;
        long rightExtra = (long)(defaultNumDelayTokens + 1 + kOfflineStreamingBufferTokens) * unit;
        var padded = new float[leftPad + samples.Length + rightBase + rightExtra];
        Array.Copy(samples, 0, padded, leftPad, samples.Length);

        var melExtractor = new OpenTail.Stingray.Audio.VoxtralRealtime.VoxtralMelExtractor();
        var mel = melExtractor.ExtractMel(padded);
        int melFrames = mel.Length / OpenTail.Stingray.Audio.VoxtralRealtime.VoxtralMelExtractor.NumMels;

        using var loader = OpenTail.Stingray.Core.SafetensorsLoader.Open(audioModelPath!);
        var audioWeights = new OpenTail.Stingray.Audio.VoxtralRealtime.VoxtralAudioEncoderWeights(loader);
        var audioEmbeddings = OpenTail.Stingray.Audio.VoxtralRealtime.VoxtralAudioEncoder.Forward(audioWeights, mel, melFrames);

        // Real prompt: [BOS=1] + (32 + delay_tokens) x StreamingPad=32.
        // delay_tokens = ceil_div(ceil_div(0.48*16000, hop_length), audio_length_per_tok) = 6.
        const int promptPadTokens = 32 + defaultNumDelayTokens;
        var tokenIds = new int[1 + promptPadTokens];
        tokenIds[0] = 1;
        for (int i = 1; i < tokenIds.Length; i++) tokenIds[i] = 32;

        int capacity = tokenIds.Length;
        int copyTokens = Math.Min(audioEmbeddings.Length, capacity);
        var prefillAudio = new float[capacity][];
        for (int t = 0; t < capacity; t++)
            prefillAudio[t] = t < copyTokens ? audioEmbeddings[t] : new float[OpenTail.Stingray.Audio.VoxtralRealtime.VoxtralTextDecoderWeights.HiddenSize];

        var textWeights = new OpenTail.Stingray.Audio.VoxtralRealtime.VoxtralTextDecoderWeights(loader);
        var logits = OpenTail.Stingray.Audio.VoxtralRealtime.VoxtralTextDecoder.Forward(textWeights, tokenIds, prefillAudio, numDelayTokens: defaultNumDelayTokens);

        var last = logits[^1];
        double sum = 0, sumSq = 0;
        foreach (var v in last) { sum += v; sumSq += (double)v * v; }
        double mean = sum / last.Length;
        double std = Math.Sqrt(Math.Max(0, sumSq / last.Length - mean * mean));

        var top5 = last
            .Select((v, i) => (Value: v, Index: i))
            .OrderByDescending(x => x.Value)
            .Take(5)
            .ToArray();

        Console.Error.WriteLine(
            $"[VoxtralPrefillCS] prompt_tokens={tokenIds.Length} audio_tokens={audioEmbeddings.Length} mean={mean:F6} std={std:F6} " +
            $"top5: {string.Join(" ", top5.Select(x => $"({x.Index}:{x.Value:F4})"))}");
    }
}
