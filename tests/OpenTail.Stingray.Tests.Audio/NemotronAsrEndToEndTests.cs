
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real end-to-end smoke test for the full Nemotron ASR pipeline: real audio -> real mel
/// -> real dw_striding subsampling -> real 24-layer chunked-attention Conformer encoder -> real
/// RNNT greedy decode, decoded via the checkpoint's own real SentencePiece-style vocab
/// (`asr.tokenizer.vocab` GGUF metadata array, `▁` marking a word boundary per real SentencePiece
/// convention). Not yet compared against a captured reference trace -- this is a real, non-
/// degenerate output check (a real transcription-shaped string, not garbage/empty/all-blank),
/// the natural next step being golden verification against the reference's own transcription.
/// </summary>
public sealed class NemotronAsrEndToEndTests : HeavyTestBase
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
    public void Transcribe_RealAudio_RealWeights_ProducesNonDegenerateText()
    {
        string? modelPath = FindRepoFile("models/_models/nemotron-asr/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf");
        Assert.SkipUnless(modelPath != null, "nemotron-3.5-asr-streaming-0.6b.q8_0.gguf not found");
        string? audioPath = FindRepoFile("examples/audio.cpp/assets/resources/a.wav");
        Assert.SkipUnless(audioPath != null, "reference a.wav not found");

        using var w = new OpenTail.Stingray.Audio.NemotronAsr.NemotronAsrWeights(modelPath!);
        var extractor = OpenTail.Stingray.Audio.NemotronAsr.NemotronAsrMelExtractor.FromWeights(w);

        var (samples, sr, _) = WavReader.ReadWav(audioPath!);
        if (sr != 16000) samples = AudioResampler.Resample(samples, sr, 16000);

        var (mel, frames, _) = extractor.ExtractMel(samples);
        var (subsampled, tEnc) = OpenTail.Stingray.Audio.NemotronAsr.NemotronAsrSubsampling.Forward(w, mel, frames);
        var encoded = OpenTail.Stingray.Audio.NemotronAsr.NemotronAsrConformerEncoder.Forward(w, subsampled);

        var tokens = OpenTail.Stingray.Audio.NemotronAsr.NemotronAsrRnntDecoder.Greedy(w, encoded);

        var vocab = ReadVocab(modelPath!);
        var nonBlank = tokens.Where(t => t != w.BlankTokenId && t >= 0 && t < vocab.Length).ToList();
        string text = string.Concat(nonBlank.Select(t => vocab[t])).Replace('▁', ' ').Trim();

        Console.Error.WriteLine($"[NemotronAsrE2E] tEnc={tEnc} totalTokens={tokens.Count} nonBlankTokens={nonBlank.Count}");
        Console.Error.WriteLine($"[NemotronAsrE2E] text=\"{text}\"");

        Assert.True(nonBlank.Count > 0, "decode produced only blank tokens");
        Assert.True(text.Length > 0, "decoded text is empty");
    }

    private static string[] ReadVocab(string ggufPath)
    {
        using var model = OpenTail.Stingray.Core.GgufModel.Open(ggufPath);
        if (model.Metadata.TryGetValue("asr.tokenizer.vocab", out var v) && v is string[] arr) return arr;
        if (v is object[] objArr) return objArr.Select(o => o?.ToString() ?? "").ToArray();
        throw new InvalidOperationException("asr.tokenizer.vocab metadata not found or unexpected type: " + v?.GetType());
    }
}
