using OpenTail.Stingray.Audio.Vad;

namespace OpenTail.Stingray.Audio.Whisper;

/// <summary>
/// Native end-to-end Speech-to-Text (ASR) and Speech Translation pipeline for OpenAI Whisper.
/// Supports Silero VAD silence pruning, timestamp decoding, and chunked execution.
/// </summary>
public sealed class WhisperPipeline : ISpeechToTextPipeline
{
    public const int ChunkSamples = 16000 * 30; // 30 seconds = 480,000 samples

    public string Architecture => "OpenAI-Whisper";
    public int SampleRate => 16000;

    private readonly WhisperConfig _config;
    private readonly WhisperMelExtractor _melExtractor;
    private readonly WhisperTokenizer _tokenizer;
    private readonly WhisperEncoder _encoder;
    private readonly WhisperDecoder _decoder;
    private readonly SileroVad _vad;

    public WhisperPipeline(
        WhisperConfig? config = null,
        WhisperMelExtractor? melExtractor = null,
        WhisperTokenizer? tokenizer = null,
        WhisperEncoder? encoder = null,
        WhisperDecoder? decoder = null,
        SileroVad? vad = null)
    {
        _config = config ?? WhisperConfig.Tiny;
        _melExtractor = melExtractor ?? new WhisperMelExtractor(_config.NumMels);
        _tokenizer = tokenizer ?? new WhisperTokenizer();
        _encoder = encoder ?? new WhisperEncoder(_config);
        _decoder = decoder ?? new WhisperDecoder(_config);
        _vad = vad ?? new SileroVad();
    }

    /// <summary>
    /// Transcribes or translates audio samples into text with timestamps and segment breakdown.
    /// </summary>
    public SpeechToTextResult Transcribe(SpeechToTextRequest request)
    {
        if (request.AudioSamples == null || request.AudioSamples.Length == 0)
        {
            return new SpeechToTextResult(string.Empty, request.Language ?? "en", TimeSpan.Zero, []);
        }

        var samples = request.AudioSamples;
        if (request.SampleRate != SampleRate && request.SampleRate > 0)
        {
            samples = AudioResampler.Resample(samples, request.SampleRate, SampleRate);
        }
        int totalSamples = samples.Length;
        var allSegments = new List<SpeechSegment>();
        var fullTextBuilder = new System.Text.StringBuilder();

        // If VAD is enabled, detect speech boundaries first
        if (request.UseVad)
        {
            var speechSegments = _vad.DetectSegments(samples);
            if (speechSegments.Count == 0)
            {
                // No voice detected in the entire recording
                return new SpeechToTextResult(string.Empty, request.Language ?? "en", TimeSpan.FromSeconds((double)totalSamples / SampleRate), []);
            }

            for (int i = 0; i < speechSegments.Count; i++)
            {
                var vadSeg = speechSegments[i];
                int start = Math.Clamp(vadSeg.StartSample, 0, totalSamples);
                int length = Math.Clamp(vadSeg.EndSample - start, 0, totalSamples - start);
                if (length <= 0) continue;

                var segAudio = samples.AsSpan(start, length);
                TimeSpan segTimeOffset = TimeSpan.FromSeconds(vadSeg.StartSeconds);

                ProcessAudioChunk(segAudio, segTimeOffset, request, allSegments, fullTextBuilder);
                request.Progress?.Invoke(i + 1, speechSegments.Count);
            }
        }
        else
        {
            int numChunks = (totalSamples + ChunkSamples - 1) / ChunkSamples;
            if (numChunks == 0) numChunks = 1;

            for (int chunkIdx = 0; chunkIdx < numChunks; chunkIdx++)
            {
                int startOffset = chunkIdx * ChunkSamples;
                int length = Math.Min(ChunkSamples, totalSamples - startOffset);
                TimeSpan chunkTimeOffset = TimeSpan.FromSeconds((double)startOffset / SampleRate);

                var chunkSpan = samples.AsSpan(startOffset, length);

                ProcessAudioChunk(chunkSpan, chunkTimeOffset, request, allSegments, fullTextBuilder);
                request.Progress?.Invoke(chunkIdx + 1, numChunks);
            }
        }

        string detectedLanguage = request.Language ?? "en";
        TimeSpan totalDuration = TimeSpan.FromSeconds((double)totalSamples / SampleRate);
        return new SpeechToTextResult(
            text: fullTextBuilder.ToString(),
            language: detectedLanguage,
            duration: totalDuration,
            segments: allSegments);
    }

    private void ProcessAudioChunk(
        ReadOnlySpan<float> chunkSpan,
        TimeSpan chunkTimeOffset,
        SpeechToTextRequest request,
        List<SpeechSegment> allSegments,
        System.Text.StringBuilder fullTextBuilder)
    {
        // 1. Extract 80/128-channel Log-Mel Spectrogram
        float[] mel = _melExtractor.ExtractMel(chunkSpan, padTo30Seconds: false);
        int numFrames = mel.Length / _config.NumMels;

        // 2. Audio Transformer Encoder Forward Pass
        float[] audioEncoderState = _encoder.Forward(mel, numFrames);
        int encFrames = audioEncoderState.Length / _config.AudioState;

        // 3. Prepare Decoder Prompt Tokens
        int[] prompt = _tokenizer.BuildInitialPrompt(request.Language, request.Task, request.EnableTimestamps);
        var generatedTokens = new List<int>(prompt);

        // 4. Autoregressive KV-Cached Greedy / Temperature Decoding Loop
        var kvCache = new WhisperKvCache(_config.TextLayer, _config.TextCtx, _config.TextState);
        for (int i = 0; i < prompt.Length - 1; i++)
        {
            _decoder.ForwardStep(prompt[i], i, kvCache, audioEncoderState, encFrames);
        }

        int currentToken = prompt[^1];
        int currentPos = prompt.Length - 1;
        int maxTokens = Math.Min(_config.TextCtx - prompt.Length, 64);
        var rng = new Random(42);

        for (int step = 0; step < maxTokens; step++)
        {
            float[] logits = _decoder.ForwardStep(currentToken, currentPos++, kvCache, audioEncoderState, encFrames);
            int nextToken = SampleNextToken(logits, request.Temperature, rng);

            if (nextToken == WhisperTokenizer.EndOfText)
            {
                break;
            }

            generatedTokens.Add(nextToken);
            currentToken = nextToken;
        }

        // 5. Decode Segment Tokens & Timestamps
        var (chunkText, chunkSegments) = _tokenizer.DecodeSegments(generatedTokens.ToArray(), chunkTimeOffset);

        if (chunkSegments.Count > 0)
        {
            allSegments.AddRange(chunkSegments);
        }

        if (!string.IsNullOrWhiteSpace(chunkText))
        {
            if (fullTextBuilder.Length > 0) fullTextBuilder.Append(' ');
            fullTextBuilder.Append(chunkText);
        }
    }

    private int SampleNextToken(ReadOnlySpan<float> logits, float temperature, Random rng)
    {
        if (temperature <= 0.0f)
        {
            // Greedy argmax
            int bestToken = 0;
            float maxLogit = float.MinValue;

            for (int i = 0; i < logits.Length; i++)
            {
                if (logits[i] > maxLogit)
                {
                    maxLogit = logits[i];
                    bestToken = i;
                }
            }

            return bestToken;
        }

        // Temperature Softmax & Categorical Sampling
        Span<float> expProbs = stackalloc float[Math.Min(logits.Length, 1024)];
        float maxL = float.MinValue;
        for (int i = 0; i < expProbs.Length; i++)
        {
            if (logits[i] > maxL) maxL = logits[i];
        }

        float sumExp = 0f;
        for (int i = 0; i < expProbs.Length; i++)
        {
            float e = MathF.Exp((logits[i] - maxL) / temperature);
            expProbs[i] = e;
            sumExp += e;
        }

        float roll = (float)rng.NextDouble() * sumExp;
        float accum = 0f;

        for (int i = 0; i < expProbs.Length; i++)
        {
            accum += expProbs[i];
            if (accum >= roll) return i;
        }

        return 0;
    }

    /// <summary>
    /// Transcribes incoming streaming audio chunks in real-time.
    /// </summary>
    public async IAsyncEnumerable<SpeechSegment> TranscribeStreamAsync(
        IAsyncEnumerable<ReadOnlyMemory<float>> audioStream,
        SpeechToTextRequest baseRequest,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct = default)
    {
        var buffer = new List<float>();
        int segmentCounter = 0;
        TimeSpan streamTimeOffset = TimeSpan.Zero;

        await foreach (var chunk in audioStream.WithCancellation(ct))
        {
            if (chunk.Length == 0) continue;

            ReadOnlySpan<float> span = chunk.Span;
            if (baseRequest.SampleRate != SampleRate && baseRequest.SampleRate > 0)
            {
                float[] resampled = AudioResampler.Resample(span, baseRequest.SampleRate, SampleRate);
                buffer.AddRange(resampled);
            }
            else
            {
                buffer.AddRange(span.ToArray());
            }

            if (buffer.Count >= SampleRate * 3) // 3-second accumulated chunk
            {
                float[] audioToProcess = buffer.ToArray();
                var req = baseRequest with { AudioSamples = audioToProcess, SampleRate = SampleRate };
                var result = Transcribe(req);
                foreach (var seg in result.Segments)
                {
                    yield return seg with
                    {
                        Id = ++segmentCounter,
                        Start = seg.Start + streamTimeOffset,
                        End = seg.End + streamTimeOffset
                    };
                }
                streamTimeOffset += TimeSpan.FromSeconds((double)audioToProcess.Length / SampleRate);
                buffer.Clear();
            }
        }

        if (buffer.Count > 0)
        {
            float[] audioToProcess = buffer.ToArray();
            var req = baseRequest with { AudioSamples = audioToProcess, SampleRate = SampleRate };
            var result = Transcribe(req);
            foreach (var seg in result.Segments)
            {
                yield return seg with
                {
                    Id = ++segmentCounter,
                    Start = seg.Start + streamTimeOffset,
                    End = seg.End + streamTimeOffset
                };
            }
        }
    }

    public void Dispose()
    {
        _vad.Dispose();
    }
}
