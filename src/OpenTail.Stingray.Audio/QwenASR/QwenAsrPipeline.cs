using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

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
    private readonly QwenAsrWeights? _weights;

    public QwenAsrPipeline(
        QwenAsrMelExtractor? melExtractor = null,
        QwenAsrTokenizer? tokenizer = null,
        QwenAsrAudioEncoder? encoder = null,
        QwenAsrDecoder? decoder = null,
        QwenAsrForcedAligner? aligner = null,
        QwenAsrWeights? weights = null)
    {
        _weights = weights;
        _melExtractor = melExtractor ?? new QwenAsrMelExtractor();
        _tokenizer = tokenizer ?? (weights != null ? new QwenAsrTokenizer(weights) : new QwenAsrTokenizer());
        _encoder = encoder ?? new QwenAsrAudioEncoder();
        _decoder = decoder ?? new QwenAsrDecoder();
        _aligner = aligner ?? new QwenAsrForcedAligner(_tokenizer);
    }

    /// <summary>
    /// Loads a real Qwen3-ASR pipeline directly from a GGUF model file. Real BPE tokenizer
    /// (via <see cref="QwenAsrWeights.Tokenizer"/>); the AuT audio encoder and LLM decoder
    /// stages are still procedural pending their own real ports -- see
    /// docs/audio-review-progress.md's QwenASR section.
    /// </summary>
    public static QwenAsrPipeline Load(string ggufPath)
    {
        if (string.IsNullOrWhiteSpace(ggufPath) || !File.Exists(ggufPath))
            throw new FileNotFoundException($"Qwen3-ASR GGUF model not found: {ggufPath}");

        var weights = new QwenAsrWeights(ggufPath);
        var encoderConfig = new QwenAsrEncoderConfig
        {
            EncoderDim = weights.AudioDim,
            NumLayers = weights.AudioLayers,
            NumHeads = weights.AudioHeads,
            QwenHiddenDim = weights.LlmDim
        };
        var decoderConfig = new QwenAsrDecoderConfig
        {
            HiddenDim = weights.LlmDim,
            NumLayers = weights.LlmLayers,
            NumHeads = weights.LlmHeads,
            NumKvHeads = weights.LlmKvHeads,
            VocabSize = weights.LlmVocabSize,
            EosTokenId = weights.EosTokenId
        };

        var melExtractor = new QwenAsrMelExtractor();
        var tokenizer = new QwenAsrTokenizer(weights);
        var encoder = new QwenAsrAudioEncoder(encoderConfig, weights);
        var decoder = new QwenAsrDecoder(decoderConfig);
        var aligner = new QwenAsrForcedAligner(tokenizer);

        return new QwenAsrPipeline(melExtractor, tokenizer, encoder, decoder, aligner, weights);
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

        // 2. Audio Transformer (AuT) Encoder Forward Pass -> Soft Audio Tokens
        var (audioSoftTokens, numAudioTokens) = _encoder.Forward(mel, inMelFrames);

        // 3. Format ChatML Multimodal Prompt
        string promptStr = _tokenizer.FormatPrompt(
            numAudioTokens: numAudioTokens,
            language: request.Language,
            taskInstruction: (request.Task == SpeechTask.Translate) ? "Translate the speech into English." : "Transcribe the audio speech into text.");
        int[] promptTokens = _tokenizer.Encode(promptStr);

        // 4. Qwen3 LLM Transformer Decoder Forward Pass
        int[] generatedTokens = _decoder.Generate(
            promptTokens: promptTokens,
            audioSoftTokens: audioSoftTokens,
            numAudioTokens: numAudioTokens,
            maxNewTokens: 256,
            temperature: (float)request.Temperature);

        // 5. Decode Tokens to Text and Timestamps
        var (text, segments) = _tokenizer.DecodeWithTimestamps(generatedTokens, TimeSpan.Zero, totalDuration);

        // If forced alignment was requested or output has segments
        if (segments.Count == 0 && !string.IsNullOrWhiteSpace(text))
        {
            segments.Add(new SpeechSegment
            {
                Id = 0,
                Start = TimeSpan.Zero,
                End = totalDuration,
                Text = text,
                Tokens = generatedTokens,
                Probability = 0.98f
            });
        }

        return new SpeechToTextResult(
            text: text,
            language: request.Language ?? "en",
            duration: totalDuration,
            segments: segments);
    }

    /// <summary>
    /// Performs word-level forced alignment between reference text and input speech audio.
    /// </summary>
    public IReadOnlyList<SpeechSegment> Align(float[] audioSamples, string referenceText, int sampleRate = 16000)
    {
        if (audioSamples == null || audioSamples.Length == 0 || string.IsNullOrWhiteSpace(referenceText))
        {
            return [];
        }

        float[] pcm16k = (sampleRate == SampleRate) ? audioSamples : AudioResampler.Resample(audioSamples, sampleRate, SampleRate);
        float[] mel = _melExtractor.ExtractMel(pcm16k);
        int inMelFrames = mel.Length / QwenAsrMelExtractor.NumMels;

        var (audioSoftTokens, numAudioTokens) = _encoder.Forward(mel, inMelFrames);
        return _aligner.Align(referenceText, audioSoftTokens, numAudioTokens, _encoder.Config.QwenHiddenDim, TimeSpan.Zero);
    }

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

    public void Dispose()
    {
        _weights?.Dispose();
    }
}
