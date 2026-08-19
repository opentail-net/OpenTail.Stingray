using System.Runtime.CompilerServices;
using System.Text;

namespace OpenTail.Stingray.Audio.QwenASR;

/// <summary>
/// Native pipeline for Alibaba Qwen3-ASR (0.6B and 1.7B) multilingual speech-to-text and Qwen3-ForcedAligner.
/// </summary>
public sealed class QwenAsrPipeline : ISpeechToTextPipeline
{
    public string Architecture => "Alibaba-Qwen3-ASR";
    public int SampleRate => 16000;

    private readonly QwenAsrMelExtractor _melExtractor;
    private readonly QwenAsrTokenizer _tokenizer;
    private readonly QwenAsrAudioEncoder _encoder;
    private readonly QwenAsrDecoder _decoder;
    private readonly QwenAsrForcedAligner _aligner;

    public QwenAsrPipeline(
        QwenAsrMelExtractor? melExtractor = null,
        QwenAsrTokenizer? tokenizer = null,
        QwenAsrAudioEncoder? encoder = null,
        QwenAsrDecoder? decoder = null,
        QwenAsrForcedAligner? aligner = null)
    {
        _melExtractor = melExtractor ?? new QwenAsrMelExtractor();
        _tokenizer = tokenizer ?? new QwenAsrTokenizer();
        _encoder = encoder ?? new QwenAsrAudioEncoder();
        _decoder = decoder ?? new QwenAsrDecoder();
        _aligner = aligner ?? new QwenAsrForcedAligner(_tokenizer);
    }

    /// <summary>
    /// Transcribes audio speech into text with timestamps and segment breakdowns.
    /// </summary>
    public SpeechToTextResult Transcribe(SpeechToTextRequest request)
    {
        if (request.AudioSamples == null || request.AudioSamples.Length == 0)
        {
            return new SpeechToTextResult(string.Empty, request.Language ?? "en", TimeSpan.Zero, []);
        }

        float[] pcm16k = request.AudioSamples;
        if (request.SampleRate != SampleRate && request.SampleRate > 0)
        {
            pcm16k = AudioResampler.Resample(pcm16k, request.SampleRate, SampleRate);
        }

        TimeSpan totalDuration = TimeSpan.FromSeconds((double)pcm16k.Length / SampleRate);

        // 1. Extract 128-channel Log-Mel Spectrogram
        float[] mel = _melExtractor.ExtractMel(pcm16k);
        int inMelFrames = mel.Length / QwenAsrMelExtractor.NumMels;

        if (inMelFrames == 0)
        {
            return new SpeechToTextResult(string.Empty, request.Language ?? "en", totalDuration, []);
        }

        // 2. Audio Transformer Encoder (AuT) with 8x Conv2D downsampling
        var (audioSoftTokens, numAudioTokens) = _encoder.Forward(mel, inMelFrames);

        // 3. Format ChatML Prompt
        string promptText = _tokenizer.FormatPrompt(request.Language, request.InitialPrompt);
        int[] promptTokens = _tokenizer.Encode(promptText);

        // 4. Qwen3 LLM Decoding
        int[] generatedTokens = _decoder.Generate(
            promptTokens: promptTokens,
            audioSoftTokens: audioSoftTokens,
            numAudioTokens: numAudioTokens,
            temperature: request.Temperature);

        // 5. Decode Tokens with Timestamps
        var (fullText, segments) = _tokenizer.DecodeWithTimestamps(generatedTokens, TimeSpan.Zero);

        // If forced alignment requested or segments empty, run aligner
        if (segments.Count == 0 && !string.IsNullOrWhiteSpace(fullText))
        {
            segments = _aligner.Align(
                referenceText: fullText,
                audioTokens: audioSoftTokens,
                numAudioTokens: numAudioTokens,
                audioDim: _encoder.Config.QwenHiddenDim,
                timeOffset: TimeSpan.Zero);
        }

        return new SpeechToTextResult(
            text: fullText,
            language: request.Language ?? "en",
            duration: totalDuration,
            segments: segments);
    }

    /// <summary>
    /// Performs word-level forced alignment between a reference transcript and audio.
    /// </summary>
    public List<SpeechSegment> Align(string referenceText, float[] audioSamples, int sampleRate = 16000)
    {
        if (string.IsNullOrWhiteSpace(referenceText) || audioSamples.Length == 0)
        {
            return [];
        }

        float[] pcm16k = (sampleRate != SampleRate)
            ? AudioResampler.Resample(audioSamples, sampleRate, SampleRate)
            : audioSamples;

        float[] mel = _melExtractor.ExtractMel(pcm16k);
        int inMelFrames = mel.Length / QwenAsrMelExtractor.NumMels;
        var (audioSoftTokens, numAudioTokens) = _encoder.Forward(mel, inMelFrames);

        return _aligner.Align(
            referenceText: referenceText,
            audioTokens: audioSoftTokens,
            numAudioTokens: numAudioTokens,
            audioDim: _encoder.Config.QwenHiddenDim,
            timeOffset: TimeSpan.Zero);
    }

    /// <summary>
    /// Transcribes streaming audio in real-time.
    /// </summary>
    public async IAsyncEnumerable<SpeechSegment> TranscribeStreamAsync(
        IAsyncEnumerable<ReadOnlyMemory<float>> audioStream,
        SpeechToTextRequest baseRequest,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        var buffer = new List<float>();
        TimeSpan streamOffset = TimeSpan.Zero;
        int chunkSampleThreshold = SampleRate * 2; // 2-second streaming window

        await foreach (var chunk in audioStream.WithCancellation(ct))
        {
            if (ct.IsCancellationRequested) yield break;

            buffer.AddRange(chunk.ToArray());

            if (buffer.Count >= chunkSampleThreshold)
            {
                float[] pcm = buffer.ToArray();
                buffer.Clear();

                var req = baseRequest with { AudioSamples = pcm, SampleRate = SampleRate };
                var result = Transcribe(req);

                foreach (var seg in result.Segments)
                {
                    yield return seg with
                    {
                        Start = seg.Start + streamOffset,
                        End = seg.End + streamOffset
                    };
                }

                streamOffset += TimeSpan.FromSeconds((double)pcm.Length / SampleRate);
                await Task.Yield();
            }
        }

        if (buffer.Count > 0 && !ct.IsCancellationRequested)
        {
            float[] pcm = buffer.ToArray();
            var req = baseRequest with { AudioSamples = pcm, SampleRate = SampleRate };
            var result = Transcribe(req);

            foreach (var seg in result.Segments)
            {
                yield return seg with
                {
                    Start = seg.Start + streamOffset,
                    End = seg.End + streamOffset
                };
            }
        }
    }

    public void Dispose()
    {
        _encoder.Dispose();
        _decoder.Dispose();
        _aligner.Dispose();
    }
}
