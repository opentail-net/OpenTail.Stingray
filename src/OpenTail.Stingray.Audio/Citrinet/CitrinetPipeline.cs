using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using OpenTail.Stingray.Audio.Rvc;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Audio.Citrinet;

/// <summary>
/// Native NVIDIA NeMo Citrinet ASR (Jasper-family CTC CNN with Squeeze-Excite) pipeline.
/// </summary>
public sealed class CitrinetPipeline : ISpeechToTextPipeline
{
    public string Architecture => "NVIDIA-NeMo-Citrinet-ASR";
    public int SampleRate => CitrinetAsrWeights.SampleRate;

    private readonly GgufModel? _model;
    private readonly CitrinetAsrWeights _weights;
    private readonly string[] _pieces;
    private readonly CitrinetAsrMelExtractor _melExtractor;

    public CitrinetPipeline(GgufModel? model, CitrinetAsrWeights weights, string[] pieces)
    {
        _model = model;
        _weights = weights;
        _pieces = pieces;
        _melExtractor = new CitrinetAsrMelExtractor(weights);
    }

    /// <summary>
    /// Loads a real Citrinet ASR pipeline directly from a GGUF model file.
    /// </summary>
    public static CitrinetPipeline Load(string ggufPath)
    {
        if (string.IsNullOrWhiteSpace(ggufPath) || !File.Exists(ggufPath))
            throw new FileNotFoundException($"Citrinet GGUF model not found: {ggufPath}");

        var model = GgufModel.Open(ggufPath);
        var (configJsonBytes, tokenizerModelBytes) = ExtractEmbeddedFiles(model);
        string configJson = System.Text.Encoding.UTF8.GetString(configJsonBytes);

        var source = new RvcPackedTensorSource(model);
        var weights = CitrinetAsrWeights.Load(source.GetTensor, configJson);
        var pieces = CitrinetSentencePieceVocab.LoadPieces(tokenizerModelBytes);

        return new CitrinetPipeline(model, weights, pieces);
    }

    public static (byte[] ConfigJson, byte[] TokenizerModel) ExtractEmbeddedFiles(GgufModel model)
    {
        var files = AudioCppEmbeddedFiles.ReadAll(model);
        byte[]? configJson = null, tokenizerModel = null;
        if (files.TryGetValue("citrinet_256_config.json", out var cfg))
            configJson = cfg;
        if (files.TryGetValue("citrinet_256_tokenizer.model", out var tok))
            tokenizerModel = tok;

        if (configJson is null || tokenizerModel is null)
        {
            foreach (var kvp in files)
            {
                if (configJson is null && kvp.Key.EndsWith(".json", StringComparison.OrdinalIgnoreCase))
                    configJson = kvp.Value;
                if (tokenizerModel is null && kvp.Key.EndsWith(".model", StringComparison.OrdinalIgnoreCase))
                    tokenizerModel = kvp.Value;
            }
        }

        if (configJson is null || tokenizerModel is null)
            throw new InvalidOperationException("missing config or tokenizer embedded file");
        return (configJson, tokenizerModel);
    }

    /// <summary>
    /// Transcribes audio samples into text.
    /// </summary>
    public SpeechToTextResult Transcribe(SpeechToTextRequest request)
    {
        if (request.AudioSamples == null || request.AudioSamples.Length == 0)
        {
            return new SpeechToTextResult(string.Empty, request.Language ?? "en", TimeSpan.Zero, []);
        }

        float[] waveform = request.AudioSamples;
        if (request.SampleRate != SampleRate)
        {
            waveform = AudioResampler.Resample(request.AudioSamples, request.SampleRate, SampleRate, channels: 1, ResampleQuality.BestQuality);
        }

        TimeSpan totalDuration = TimeSpan.FromSeconds((double)waveform.Length / SampleRate);

        var (melFlat, rawFrames, paddedFrames) = _melExtractor.ExtractMel(waveform, _weights.PadTo);
        if (paddedFrames == 0)
        {
            return new SpeechToTextResult(string.Empty, request.Language ?? "en", totalDuration, []);
        }

        var melChannelMajor = new float[CitrinetAsrWeights.NMels][];
        for (int m = 0; m < CitrinetAsrWeights.NMels; m++)
        {
            var row = new float[paddedFrames];
            for (int f = 0; f < paddedFrames; f++) row[f] = melFlat[f * CitrinetAsrWeights.NMels + m];
            melChannelMajor[m] = row;
        }

        var logits = CitrinetAsr.Forward(_weights, melChannelMajor);

        int validOutputFrames = (rawFrames + _weights.OutputStride - 1) / _weights.OutputStride;
        validOutputFrames = Math.Min(validOutputFrames, logits.Length);
        var validLogits = logits[..validOutputFrames];

        var ids = CitrinetAsr.GreedyCtcIds(validLogits, _weights.BlankId);
        string text = CitrinetSentencePieceVocab.Decode(_pieces, ids);

        var segments = new List<SpeechSegment>
        {
            new SpeechSegment
            {
                Id = 0,
                Start = TimeSpan.Zero,
                End = totalDuration,
                Text = text
            }
        };

        return new SpeechToTextResult(
            text: text,
            language: request.Language ?? "en",
            duration: totalDuration,
            segments: segments);
    }

    /// <summary>
    /// Transcribes real-time streaming audio chunks asynchronously.
    /// </summary>
    public async IAsyncEnumerable<SpeechSegment> TranscribeStreamAsync(
        IAsyncEnumerable<ReadOnlyMemory<float>> audioStream,
        SpeechToTextRequest baseRequest,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        var buffer = new List<float>();
        int chunkFrames = SampleRate * 2;
        TimeSpan timeOffset = TimeSpan.Zero;

        await foreach (var chunk in audioStream.WithCancellation(ct))
        {
            buffer.AddRange(chunk.ToArray());
            if (buffer.Count >= chunkFrames)
            {
                var req = baseRequest with { AudioSamples = buffer.ToArray() };
                var res = Transcribe(req);
                foreach (var seg in res.Segments)
                {
                    yield return seg with { Start = seg.Start + timeOffset, End = seg.End + timeOffset };
                }
                timeOffset += TimeSpan.FromSeconds((double)chunkFrames / SampleRate);
                buffer.RemoveRange(0, chunkFrames);
            }
        }

        if (buffer.Count > 0)
        {
            var req = baseRequest with { AudioSamples = buffer.ToArray() };
            var res = Transcribe(req);
            foreach (var seg in res.Segments)
            {
                yield return seg with { Start = seg.Start + timeOffset, End = seg.End + timeOffset };
            }
        }
    }

    public void Dispose()
    {
        _model?.Dispose();
    }
}
