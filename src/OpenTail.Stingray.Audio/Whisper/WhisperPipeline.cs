namespace OpenTail.Stingray.Audio.Whisper;

/// <summary>
/// Native end-to-end Speech-to-Text (ASR) and Speech Translation pipeline for OpenAI Whisper.
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

    public WhisperPipeline(
        WhisperConfig? config = null,
        WhisperMelExtractor? melExtractor = null,
        WhisperTokenizer? tokenizer = null,
        WhisperEncoder? encoder = null,
        WhisperDecoder? decoder = null)
    {
        _config = config ?? WhisperConfig.Tiny;
        _melExtractor = melExtractor ?? new WhisperMelExtractor(_config.NumMels);
        _tokenizer = tokenizer ?? new WhisperTokenizer();
        _encoder = encoder ?? new WhisperEncoder(_config);
        _decoder = decoder ?? new WhisperDecoder(_config);
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
        int totalSamples = samples.Length;
        var allSegments = new List<SpeechSegment>();
        var fullTextBuilder = new System.Text.StringBuilder();

        int numChunks = (totalSamples + ChunkSamples - 1) / ChunkSamples;
        if (numChunks == 0) numChunks = 1;

        string detectedLanguage = request.Language ?? "en";

        for (int chunkIdx = 0; chunkIdx < numChunks; chunkIdx++)
        {
            int startOffset = chunkIdx * ChunkSamples;
            int length = Math.Min(ChunkSamples, totalSamples - startOffset);
            TimeSpan chunkTimeOffset = TimeSpan.FromSeconds((double)startOffset / SampleRate);

            var chunkSpan = samples.AsSpan(startOffset, length);

            // 1. Extract 80/128-channel Log-Mel Spectrogram
            float[] mel = _melExtractor.ExtractMel(chunkSpan, padTo30Seconds: false);
            int numFrames = mel.Length / _config.NumMels;

            // 2. Audio Transformer Encoder Forward Pass
            float[] audioEncoderState = _encoder.Forward(mel, numFrames);
            int encFrames = audioEncoderState.Length / _config.AudioState;

            // 3. Prepare Decoder Prompt Tokens
            int[] prompt = _tokenizer.BuildInitialPrompt(request.Language, request.Task, request.EnableTimestamps);
            var generatedTokens = new List<int>(prompt);

            // 4. Autoregressive Greedy / Temperature Decoding Loop
            int maxTokens = Math.Min(_config.TextCtx - prompt.Length, 64);
            var rng = new Random(42);

            for (int step = 0; step < maxTokens; step++)
            {
                float[] logits = _decoder.ForwardNextToken(
                    generatedTokens.ToArray(),
                    audioEncoderState,
                    encFrames);

                int nextToken = SampleNextToken(logits, request.Temperature, rng);

                if (nextToken == WhisperTokenizer.EndOfText)
                {
                    break;
                }

                generatedTokens.Add(nextToken);
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

            request.Progress?.Invoke(chunkIdx + 1, numChunks);
        }

        TimeSpan totalDuration = TimeSpan.FromSeconds((double)totalSamples / SampleRate);
        return new SpeechToTextResult(
            text: fullTextBuilder.ToString(),
            language: detectedLanguage,
            duration: totalDuration,
            segments: allSegments);
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

    public void Dispose()
    {
        // No unmanaged resources
    }
}
