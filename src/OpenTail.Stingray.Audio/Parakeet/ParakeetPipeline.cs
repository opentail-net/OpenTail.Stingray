using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace OpenTail.Stingray.Audio.Parakeet;

/// <summary>
/// Native NVIDIA NeMo Parakeet ASR (FastConformer CTC / TDT 16kHz speech recognition) pipeline.
/// </summary>
public sealed class ParakeetPipeline : ISpeechToTextPipeline
{
    public string Architecture => "NVIDIA-NeMo-Parakeet-ASR";
    public int SampleRate => 16000;

    private readonly ParakeetMelExtractor _melExtractor;
    private readonly ParakeetTokenizer _tokenizer;
    private readonly ParakeetCtcDecoder _decoder;
    private readonly ParakeetWeights? _weights;

    public ParakeetPipeline(
        ParakeetMelExtractor? melExtractor = null,
        ParakeetTokenizer? tokenizer = null,
        ParakeetCtcDecoder? decoder = null,
        ParakeetWeights? weights = null)
    {
        _weights = weights;
        _melExtractor = melExtractor ?? new ParakeetMelExtractor();
        _tokenizer = tokenizer ?? new ParakeetTokenizer();
        _decoder = decoder ?? new ParakeetCtcDecoder(_tokenizer, weights);
    }

    /// <summary>
    /// Loads a real Parakeet ASR pipeline directly from a GGUF model file.
    /// </summary>
    public static ParakeetPipeline Load(string ggufPath)
    {
        if (string.IsNullOrWhiteSpace(ggufPath) || !File.Exists(ggufPath))
            throw new FileNotFoundException($"Parakeet GGUF model not found: {ggufPath}");

        var weights = new ParakeetWeights(ggufPath);
        var tokenizer = ParakeetTokenizer.FromGguf(weights.Model);
        var melExtractor = ParakeetMelExtractor.FromWeights(weights);
        var decoder = new ParakeetCtcDecoder(tokenizer, weights);

        return new ParakeetPipeline(melExtractor, tokenizer, decoder, weights);
    }

    /// <summary>
    /// Transcribes 16kHz audio samples into text with timestamps and speech segments.
    /// </summary>
    public SpeechToTextResult Transcribe(SpeechToTextRequest request)
    {
        if (request.AudioSamples == null || request.AudioSamples.Length == 0)
        {
            return new SpeechToTextResult(string.Empty, request.Language ?? "en", TimeSpan.Zero, []);
        }

        // Resample to 16kHz if needed
        float[] pcm16k = request.AudioSamples;
        if (request.SampleRate != SampleRate)
        {
            pcm16k = ResampleTo16k(request.AudioSamples, request.SampleRate);
        }

        TimeSpan totalDuration = TimeSpan.FromSeconds((double)pcm16k.Length / SampleRate);

        // 1. Extract 80-channel Log-Mel Spectrogram
        float[] mel = _melExtractor.ExtractMel(pcm16k);
        int inMelFrames = mel.Length / ParakeetMelExtractor.NumMels;

        if (inMelFrames == 0)
        {
            return new SpeechToTextResult(string.Empty, request.Language ?? "en", totalDuration, []);
        }

        if (_weights == null)
            throw new InvalidOperationException("ParakeetPipeline requires real GGUF weights (use ParakeetPipeline.Load) -- no procedural fallback exists for the FastConformer encoder.");

        // 2. FastConformer Acoustic Encoding (8x subsampling + conformer blocks + CTC head)
        var (_, ctcLogits, numConformerFrames) = ParakeetConformerEncoder.Forward(_weights, mel, inMelFrames);

        // 3. CTC Greedy Decoding & Timestamp Alignment
        var (fullText, _, segments) = _decoder.DecodeGreedy(
            ctcLogits: ctcLogits,
            numFrames: numConformerFrames,
            timeOffset: TimeSpan.Zero);

        return new SpeechToTextResult(
            text: fullText,
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
        int chunkFrames = SampleRate * 2; // 2-second streaming chunks
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

    private static float[] ResampleTo16k(float[] input, int srcRate)
    {
        if (srcRate == 16000 || input.Length == 0) return input;
        return AudioResampler.Resample(input, srcRate, 16000);
    }

    public void Dispose()
    {
        _weights?.Dispose();
    }
}
