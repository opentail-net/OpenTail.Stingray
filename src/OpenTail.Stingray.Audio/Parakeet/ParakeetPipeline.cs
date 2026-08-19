using System.Runtime.CompilerServices;
using System.Text;

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
    private readonly ParakeetConformerEncoder _encoder;
    private readonly ParakeetCtcDecoder _decoder;

    public ParakeetPipeline(
        ParakeetMelExtractor? melExtractor = null,
        ParakeetTokenizer? tokenizer = null,
        ParakeetConformerEncoder? encoder = null,
        ParakeetCtcDecoder? decoder = null)
    {
        _melExtractor = melExtractor ?? new ParakeetMelExtractor();
        _tokenizer = tokenizer ?? new ParakeetTokenizer();
        _encoder = encoder ?? new ParakeetConformerEncoder();
        _decoder = decoder ?? new ParakeetCtcDecoder(_tokenizer);
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

        // 2. FastConformer Acoustic Encoding (8x subsampling + conformer blocks)
        var (embeddings, numConformerFrames) = _encoder.Forward(mel, inMelFrames);

        // 3. CTC Greedy Decoding & Timestamp Alignment
        var (fullText, _, segments) = _decoder.DecodeGreedy(
            embeddings: embeddings,
            numFrames: numConformerFrames,
            hiddenDim: _encoder.Config.HiddenDim,
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
        var accumulatedSamples = new List<float>();
        TimeSpan streamTimeOffset = TimeSpan.Zero;
        int chunkSampleThreshold = SampleRate * 2; // 2-second streaming chunk window

        await foreach (var chunk in audioStream.WithCancellation(ct))
        {
            if (ct.IsCancellationRequested) yield break;

            accumulatedSamples.AddRange(chunk.ToArray());

            if (accumulatedSamples.Count >= chunkSampleThreshold)
            {
                float[] chunkArray = accumulatedSamples.ToArray();
                accumulatedSamples.Clear();

                float[] pcm16k = (baseRequest.SampleRate != SampleRate)
                    ? ResampleTo16k(chunkArray, baseRequest.SampleRate)
                    : chunkArray;

                float[] mel = _melExtractor.ExtractMel(pcm16k);
                int inMelFrames = mel.Length / ParakeetMelExtractor.NumMels;

                if (inMelFrames > 0)
                {
                    var (embeddings, numConformerFrames) = _encoder.Forward(mel, inMelFrames);
                    var (_, _, segments) = _decoder.DecodeGreedy(
                        embeddings: embeddings,
                        numFrames: numConformerFrames,
                        hiddenDim: _encoder.Config.HiddenDim,
                        timeOffset: streamTimeOffset);

                    foreach (var seg in segments)
                    {
                        yield return seg;
                    }
                }

                streamTimeOffset += TimeSpan.FromSeconds((double)pcm16k.Length / SampleRate);
                await Task.Yield();
            }
        }

        // Process any remaining tail samples
        if (accumulatedSamples.Count > 0 && !ct.IsCancellationRequested)
        {
            float[] pcm16k = (baseRequest.SampleRate != SampleRate)
                ? ResampleTo16k(accumulatedSamples.ToArray(), baseRequest.SampleRate)
                : accumulatedSamples.ToArray();

            float[] mel = _melExtractor.ExtractMel(pcm16k);
            int inMelFrames = mel.Length / ParakeetMelExtractor.NumMels;

            if (inMelFrames > 0)
            {
                var (embeddings, numConformerFrames) = _encoder.Forward(mel, inMelFrames);
                var (_, _, segments) = _decoder.DecodeGreedy(
                    embeddings: embeddings,
                    numFrames: numConformerFrames,
                    hiddenDim: _encoder.Config.HiddenDim,
                    timeOffset: streamTimeOffset);

                foreach (var seg in segments)
                {
                    yield return seg;
                }
            }
        }
    }

    private static float[] ResampleTo16k(float[] input, int srcRate)
    {
        if (srcRate == 16000) return input;
        double ratio = 16000.0 / srcRate;
        int outLength = (int)(input.Length * ratio);
        var output = new float[outLength];

        for (int i = 0; i < outLength; i++)
        {
            double srcIdx = i / ratio;
            int idx = (int)srcIdx;
            double frac = srcIdx - idx;

            if (idx + 1 < input.Length)
            {
                output[i] = (float)((1.0 - frac) * input[idx] + frac * input[idx + 1]);
            }
            else if (idx < input.Length)
            {
                output[i] = input[idx];
            }
        }

        return output;
    }

    public void Dispose()
    {
        _encoder.Dispose();
        _decoder.Dispose();
    }
}
