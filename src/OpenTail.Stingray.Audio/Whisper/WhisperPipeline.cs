using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using OpenTail.Stingray.Audio.Vad;

namespace OpenTail.Stingray.Audio.Whisper;

/// <summary>
/// Native end-to-end Speech-to-Text (ASR) and Speech Translation pipeline for OpenAI Whisper (Tiny, Base, Small, Medium, Large-v1/v2/v3, Large-v3-Turbo).
/// Supports Silero VAD silence pruning, timestamp decoding, and chunked execution.
/// </summary>
public sealed class WhisperPipeline : ISpeechToTextPipeline
{
    public const int ChunkSamples = 16000 * 30; // 30 seconds = 480,000 samples

    public string Architecture => _config.IsV3
        ? _config.TextLayer switch
        {
            4 => "OpenAI-Whisper-Large-V3-Turbo",
            2 => "Distil-Whisper-Large-V3",
            _ => "OpenAI-Whisper-Large-V3"
        }
        : "OpenAI-Whisper";
    public int SampleRate => 16000;

    private readonly WhisperConfig _config;
    private readonly WhisperMelExtractor _melExtractor;
    private readonly WhisperTokenizer _tokenizer;
    private readonly WhisperEncoder _encoder;
    private readonly WhisperDecoder _decoder;
    private readonly SileroVad _vad;

    public WhisperConfig Config => _config;

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
        _tokenizer = tokenizer ?? (_config.IsV3 ? WhisperTokenizer.CreateV3() : new WhisperTokenizer());
        _encoder = encoder ?? new WhisperEncoder(_config);
        _decoder = decoder ?? new WhisperDecoder(_config);
        _vad = vad ?? new SileroVad();
    }

    /// <summary>
    /// Loads a real Whisper model from a whisper.cpp GGML binary file (.bin) with full weights, config, and vocabulary.
    /// </summary>
    public static WhisperPipeline Load(string ggmlModelPath, SileroVad? vad = null)
    {
        if (string.IsNullOrWhiteSpace(ggmlModelPath) || !File.Exists(ggmlModelPath))
            throw new FileNotFoundException($"Whisper GGML model not found: {ggmlModelPath}");

        var ggml = WhisperGgmlModel.Load(ggmlModelPath);
        return FromModel(ggml, vad);
    }

    /// <summary>
    /// Loads a real Whisper model from a real GGUF file (produced by
    /// <c>scratch-llamacpp-ref/whisper_ggml_to_gguf.py</c>, a lossless repackaging of the same
    /// whisper.cpp ggml weights into a genuine GGUF container -- see
    /// <see cref="WhisperGgmlModel.LoadFromGguf"/> for why this project self-converts rather than
    /// consuming a third-party "whisper gguf" release).
    /// </summary>
    public static WhisperPipeline LoadFromGguf(string ggufModelPath, SileroVad? vad = null)
    {
        if (string.IsNullOrWhiteSpace(ggufModelPath) || !File.Exists(ggufModelPath))
            throw new FileNotFoundException($"Whisper GGUF model not found: {ggufModelPath}");

        var ggml = WhisperGgmlModel.LoadFromGguf(ggufModelPath);
        return FromModel(ggml, vad);
    }

    /// <summary>
    /// Loads a real Whisper model from the canonical Hugging Face `openai/whisper-*`
    /// Safetensors distribution (`config.json`/`model.safetensors`/`vocab.json` in
    /// <paramref name="checkpointDir"/>) -- the actual format `transformers`' `
    /// WhisperForConditionalGeneration.from_pretrained()` loads, confirmed against real HF repo
    /// file listings before implementing (see <see cref="WhisperGgmlModel.LoadFromSafetensors"/>).
    /// </summary>
    public static WhisperPipeline LoadFromSafetensors(string checkpointDir, SileroVad? vad = null)
    {
        if (string.IsNullOrWhiteSpace(checkpointDir) || !Directory.Exists(checkpointDir))
            throw new DirectoryNotFoundException($"Whisper Safetensors checkpoint directory not found: {checkpointDir}");

        var ggml = WhisperGgmlModel.LoadFromSafetensors(checkpointDir);
        return FromModel(ggml, vad);
    }

    private static WhisperPipeline FromModel(WhisperGgmlModel ggml, SileroVad? vad)
    {
        var config = ggml.ToConfig();
        var melExtractor = new WhisperMelExtractor(config.NumMels);
        var tokenizer = WhisperTokenizer.FromGgml(ggml);
        var encoderWeights = new WhisperEncoderWeights(ggml);
        var decoderWeights = new WhisperDecoderWeights(ggml);
        var encoder = new WhisperEncoder(config, encoderWeights);
        var decoder = new WhisperDecoder(config, decoderWeights);

        return new WhisperPipeline(config, melExtractor, tokenizer, encoder, decoder, vad);
    }

    /// <summary>
    /// Creates a pre-configured Whisper Large-v3 pipeline (128 mels, 1280 dim, 32 enc layers, 32 dec layers, 100 languages).
    /// </summary>
    public static WhisperPipeline CreateLargeV3(
        WhisperMelExtractor? melExtractor = null,
        WhisperTokenizer? tokenizer = null,
        SileroVad? vad = null)
    {
        var config = WhisperConfig.LargeV3;
        return new WhisperPipeline(
            config: config,
            melExtractor: melExtractor,
            tokenizer: tokenizer ?? WhisperTokenizer.CreateV3(),
            vad: vad);
    }

    /// <summary>
    /// Creates a pre-configured Whisper Large-v3-Turbo pipeline (128 mels, 1280 dim, 32 enc layers, 4 dec layers, 100 languages).
    /// </summary>
    public static WhisperPipeline CreateLargeV3Turbo(
        WhisperMelExtractor? melExtractor = null,
        WhisperTokenizer? tokenizer = null,
        SileroVad? vad = null)
    {
        var config = WhisperConfig.LargeV3Turbo;
        return new WhisperPipeline(
            config: config,
            melExtractor: melExtractor,
            tokenizer: tokenizer ?? WhisperTokenizer.CreateV3(),
            vad: vad);
    }

    /// <summary>
    /// Creates a pre-configured Distil-Whisper Large-v3 pipeline (128 mels, 1280 dim, 32 enc layers, 2 dec layers, 100 languages).
    /// </summary>
    public static WhisperPipeline CreateDistilLargeV3(
        WhisperMelExtractor? melExtractor = null,
        WhisperTokenizer? tokenizer = null,
        SileroVad? vad = null)
    {
        var config = WhisperConfig.DistilLargeV3;
        return new WhisperPipeline(
            config: config,
            melExtractor: melExtractor,
            tokenizer: tokenizer ?? WhisperTokenizer.CreateV3(),
            vad: vad);
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
        var fullTextBuilder = new StringBuilder();

        // If VAD is enabled, detect speech boundaries first
        if (request.UseVad)
        {
            var speechSegments = _vad.DetectSegments(samples);
            if (speechSegments.Count == 0)
            {
                return new SpeechToTextResult(string.Empty, request.Language ?? "en", TimeSpan.FromSeconds((double)totalSamples / SampleRate), []);
            }

            foreach (var chunk in speechSegments)
            {
                int startIdx = chunk.StartSample;
                int endIdx = Math.Min(totalSamples, chunk.EndSample);
                if (endIdx <= startIdx) continue;

                var chunkSpan = samples.AsSpan(startIdx, endIdx - startIdx);
                TimeSpan chunkOffset = TimeSpan.FromSeconds(chunk.StartSeconds);
                ProcessAudioChunk(chunkSpan, chunkOffset, request, allSegments, fullTextBuilder);
            }
        }
        else
        {
            // Standard 30s chunking
            int offset = 0;
            while (offset < totalSamples)
            {
                int len = Math.Min(ChunkSamples, totalSamples - offset);
                var chunkSpan = samples.AsSpan(offset, len);
                TimeSpan chunkTime = TimeSpan.FromSeconds((double)offset / SampleRate);

                ProcessAudioChunk(chunkSpan, chunkTime, request, allSegments, fullTextBuilder);
                offset += len;
            }
        }

        return new SpeechToTextResult(
            text: fullTextBuilder.ToString().Trim(),
            language: request.Language ?? "en",
            duration: TimeSpan.FromSeconds((double)totalSamples / SampleRate),
            segments: allSegments);
    }

    private void ProcessAudioChunk(
        ReadOnlySpan<float> chunkSpan,
        TimeSpan chunkTimeOffset,
        SpeechToTextRequest request,
        List<SpeechSegment> allSegments,
        StringBuilder fullTextBuilder)
    {
        // 1. Extract Mel Spectrogram (padded to 30s)
        float[] mel = _melExtractor.ExtractMel(chunkSpan, padTo30Seconds: true);
        int numFrames = mel.Length / _config.NumMels;

        // 2. Audio Encoder Forward Pass
        float[] audioFeatures = _encoder.Forward(mel, numFrames);
        int audioFrames = audioFeatures.Length / _config.AudioState;

        // 3. Construct Initial Decoder Prompt
        int[] initialPrompt = _tokenizer.BuildInitialPrompt(
            request.Language,
            request.Task,
            request.EnableTimestamps);

        // 4. Autoregressive Greedy Decoding with KV Cache
        var generatedTokens = new List<int>(initialPrompt);
        var kvCache = new WhisperKvCache(_config.TextLayer, _config.TextCtx, _config.TextState);

        // Prime cross-attention cache over audio encoder frames
        _decoder.PrimeCrossAttention(kvCache, audioFeatures, audioFrames);

        int maxNewTokens = Math.Min(256, _config.TextCtx - initialPrompt.Length);
        int currentPos = 0;

        // Prefill initial prompt tokens into KV cache
        for (int i = 0; i < initialPrompt.Length - 1; i++)
        {
            _decoder.ForwardStep(initialPrompt[i], currentPos++, kvCache, audioFeatures, audioFrames);
        }

        int lastToken = initialPrompt[^1];
        int? prevGenerated = null;
        int? prevPrevGenerated = null;

        for (int step = 0; step < maxNewTokens; step++)
        {
            float[] logits = _decoder.ForwardStep(lastToken, currentPos++, kvCache, audioFeatures, audioFrames);

            _tokenizer.ApplySamplingFilters(
                logits,
                isInitial: step == 0,
                lastGeneratedToken: prevGenerated,
                secondLastGeneratedToken: prevPrevGenerated);

            // Suppress timestamps entirely if timestamps disabled
            if (!request.EnableTimestamps)
            {
                for (int t = _tokenizer.TimestampBegin; t <= _tokenizer.TimestampEnd && t < logits.Length; t++)
                {
                    logits[t] = float.NegativeInfinity;
                }
            }

            int nextToken = SampleNextToken(logits, request.Temperature);
            generatedTokens.Add(nextToken);
            prevPrevGenerated = prevGenerated;
            prevGenerated = nextToken;
            lastToken = nextToken;

            if (nextToken == WhisperTokenizer.EndOfText || nextToken == 0)
            {
                break;
            }
        }

        // 5. Decode Tokens to Text Segments with Timestamps
        var (chunkText, chunkSegments) = _tokenizer.DecodeWithTimestamps(
            generatedTokens.ToArray(),
            chunkTimeOffset);

        if (!string.IsNullOrWhiteSpace(chunkText))
        {
            if (fullTextBuilder.Length > 0) fullTextBuilder.Append(' ');
            fullTextBuilder.Append(chunkText);
        }

        foreach (var seg in chunkSegments)
        {
            allSegments.Add(seg with { Id = allSegments.Count });
        }
    }

    private static int SampleNextToken(float[] logits, float temperature)
    {
        if (temperature <= 0.0f)
        {
            // Greedy argmax
            int bestIdx = 0;
            float maxVal = logits[0];
            for (int i = 1; i < logits.Length; i++)
            {
                if (logits[i] > maxVal)
                {
                    maxVal = logits[i];
                    bestIdx = i;
                }
            }
            return bestIdx;
        }

        // Scaled softmax sampling
        float maxL = float.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++)
        {
            if (logits[i] > maxL) maxL = logits[i];
        }

        float sumExp = 0f;
        var probs = new float[logits.Length];
        for (int i = 0; i < logits.Length; i++)
        {
            float p = MathF.Exp((logits[i] - maxL) / temperature);
            probs[i] = p;
            sumExp += p;
        }

        float r = (float)Random.Shared.NextDouble() * sumExp;
        float accum = 0f;
        for (int i = 0; i < probs.Length; i++)
        {
            accum += probs[i];
            if (accum >= r) return i;
        }

        return probs.Length - 1;
    }

    public async IAsyncEnumerable<SpeechSegment> TranscribeStreamAsync(
        IAsyncEnumerable<ReadOnlyMemory<float>> audioStream,
        SpeechToTextRequest baseRequest,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        var buffer = new List<float>();
        TimeSpan timeOffset = TimeSpan.Zero;

        await foreach (var chunk in audioStream.WithCancellation(ct))
        {
            buffer.AddRange(chunk.ToArray());

            if (buffer.Count >= ChunkSamples)
            {
                var req = baseRequest with { AudioSamples = buffer.ToArray() };
                var result = Transcribe(req);

                foreach (var seg in result.Segments)
                {
                    yield return seg with { Start = seg.Start + timeOffset, End = seg.End + timeOffset };
                }

                timeOffset += TimeSpan.FromSeconds((double)ChunkSamples / SampleRate);
                buffer.RemoveRange(0, ChunkSamples);
            }
        }

        if (buffer.Count > 0)
        {
            var req = baseRequest with { AudioSamples = buffer.ToArray() };
            var result = Transcribe(req);

            foreach (var seg in result.Segments)
            {
                yield return seg with { Start = seg.Start + timeOffset, End = seg.End + timeOffset };
            }
        }
    }

    public void Dispose() { }
}
