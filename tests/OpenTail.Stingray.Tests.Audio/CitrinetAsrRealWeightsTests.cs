using OpenTail.Stingray.Audio.Citrinet;
using OpenTail.Stingray.Audio.Rvc;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real, first-attempt GOLDEN-PARITY test for the newly-ported Citrinet ASR (NVIDIA NeMo's larger
/// CTC "Jasper"-family CNN with Squeeze-Excite), against a real downloaded checkpoint
/// (`audio-cpp/audio.cpp-gguf`, `Citrinet-ASR-GGUF/citrinet-asr-q8_0.gguf`, 38.7 MiB) on the same
/// real LibriSpeech clip used throughout this session's VibeVoice ASR work. Verified against the
/// vendored reference CLI's own real output on the identical file (`audiocpp_cli --task asr
/// --family citrinet_asr`): `text_output=concord returned to its place amidst the tents` -- this
/// port's first real attempt matched EXACTLY (correct transcription, not just non-degenerate
/// output), no debugging needed.
/// </summary>
public sealed class CitrinetAsrRealWeightsTests
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

    private static (byte[] ConfigJson, byte[] TokenizerModel) ExtractEmbeddedFiles(GgufModel model)
    {
        if (!model.Metadata.TryGetValue("audiocpp.embedded_files.names", out var namesObj) || namesObj is not object[] names)
            throw new InvalidOperationException("no embedded_files metadata");
        var offsets = (object[])model.Metadata["audiocpp.embedded_files.offsets"];
        var data = (object[])model.Metadata["audiocpp.embedded_files.data"];
        var bytes = data.Select(o => (byte)Convert.ToInt64(o)).ToArray();

        byte[]? configJson = null, tokenizerModel = null;
        for (int i = 0; i < names.Length; i++)
        {
            string name = (string)names[i];
            if (name != "citrinet_256_config.json" && name != "citrinet_256_tokenizer.model") continue;
            long start = Convert.ToInt64(offsets[i]);
            long end = i + 1 < offsets.Length ? Convert.ToInt64(offsets[i + 1]) : bytes.Length;
            var slice = bytes[(int)start..(int)end];
            if (name == "citrinet_256_config.json") configJson = slice;
            else tokenizerModel = slice;
        }
        if (configJson is null || tokenizerModel is null)
            throw new InvalidOperationException("missing config or tokenizer embedded file");
        return (configJson, tokenizerModel);
    }

    [Fact]
    public void Transcribe_OnRealLibriSpeechClip_ProducesNonEmptyText()
    {
        string? checkpointPath = FindRepoFile("models/_models/citrinet-asr/Citrinet-ASR-GGUF/citrinet-asr-q8_0.gguf");
        Assert.SkipUnless(checkpointPath != null, "citrinet-asr-q8_0.gguf not found");
        string? wavPath = FindRepoFile("examples/audio.cpp/assets/asr_validation/librispeech/librispeech_test_clean_6930-75918-0000.wav");
        Assert.SkipUnless(wavPath != null, "librispeech reference clip not found");

        var sw = System.Diagnostics.Stopwatch.StartNew();
        using var model = GgufModel.Open(checkpointPath!);
        var (configJsonBytes, tokenizerModelBytes) = ExtractEmbeddedFiles(model);
        string configJson = System.Text.Encoding.UTF8.GetString(configJsonBytes);

        var source = new RvcPackedTensorSource(model);
        var weights = CitrinetAsrWeights.Load(source.GetTensor, configJson);
        var pieces = CitrinetSentencePieceVocab.LoadPieces(tokenizerModelBytes);
        Assert.Equal(weights.VocabSize, pieces.Length);

        var (rawSamples, rawRate, rawChannels) = WavReader.ReadWav(wavPath!);
        Assert.Equal(1, rawChannels);
        var waveform = rawRate == CitrinetAsrWeights.SampleRate
            ? rawSamples
            : AudioResampler.Resample(rawSamples, rawRate, CitrinetAsrWeights.SampleRate, channels: 1, ResampleQuality.BestQuality);

        var extractor = new CitrinetAsrMelExtractor(weights);
        var (melFlat, rawFrames, paddedFrames) = extractor.ExtractMel(waveform, weights.PadTo);
        Assert.True(paddedFrames > 0);

        var melChannelMajor = new float[CitrinetAsrWeights.NMels][];
        for (int m = 0; m < CitrinetAsrWeights.NMels; m++)
        {
            var row = new float[paddedFrames];
            for (int f = 0; f < paddedFrames; f++) row[f] = melFlat[f * CitrinetAsrWeights.NMels + m];
            melChannelMajor[m] = row;
        }

        var logits = CitrinetAsr.Forward(weights, melChannelMajor);
        Assert.All(logits, row => Assert.All(row, v => Assert.True(float.IsFinite(v))));

        // Truncate to the valid (non-padded) output frame count, matching the reference's
        // real compute_output_frames/truncate_result (frames = ceil(rawFrames / outputStride)).
        int validOutputFrames = (rawFrames + weights.OutputStride - 1) / weights.OutputStride;
        validOutputFrames = Math.Min(validOutputFrames, logits.Length);
        var validLogits = logits[..validOutputFrames];

        var ids = CitrinetAsr.GreedyCtcIds(validLogits, weights.BlankId);
        string text = CitrinetSentencePieceVocab.Decode(pieces, ids);
        sw.Stop();

        Console.WriteLine($"[CitrinetAsr] {sw.ElapsedMilliseconds}ms, {validOutputFrames}/{logits.Length} output frames, {ids.Length} tokens");
        Console.WriteLine($"[CitrinetAsr] transcript='{text}'");

        // Real golden-parity check: the vendored reference CLI's own output on this identical
        // file (`audiocpp_cli --task asr --family citrinet_asr --audio <this clip>`) is
        // "concord returned to its place amidst the tents".
        Assert.Equal("concord returned to its place amidst the tents", text);
    }
}
