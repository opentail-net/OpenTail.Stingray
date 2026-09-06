
namespace OpenTail.Stingray.Audio.CosyVoice;

/// <summary>
/// Real, weight-driven CosyVoice3 end-to-end pipeline: text -&gt; <see cref="CosyVoice3Llm"/>
/// speech tokens -&gt; <see cref="CosyVoice3FlowEncoder"/> conditioning -&gt;
/// <see cref="CosyVoice3DiTModel"/> CFM ODE solve -&gt; <see cref="CosyVoiceHiftVocoder"/> waveform.
/// Chains three independently real-weights-tested stages built earlier this session (see
/// docs/audio-review-progress.md's CosyVoice3 entries) into one call.
///
/// <para>Speaker conditioning: if a real reference audio file is supplied (`--ref-audio`), a
/// real 192-dim x-vector is extracted via <see cref="CamPlusSpeakerEncoder"/> (the checkpoint's
/// own `campplus.onnx`) -- otherwise falls back to an all-zero vector. The reference audio is
/// also tokenized via <see cref="CosyVoiceSpeechTokenizer"/> (a separate real ONNX speech
/// tokenizer) and its tokens are concatenated with the newly generated tokens before flow
/// encoding, and its real mel is used as `cond`'s prefix -- matching the real reference's
/// `CausalMaskedDiffWithDiT::build_cgraph_encode` zero-shot conditioning mechanism. The
/// synthesized prompt-audio prefix is trimmed from the output before returning. See
/// <see cref="Generate(string, int, int, int?, string?, float)"/>'s own doc comment for the full
/// mechanism. Remaining open gap: the DiT's CFG refinement is still omitted (see
/// <see cref="CosyVoice3DiTModel.SolveFlowMatchingOde"/>'s doc comment).</para>
/// </summary>
public sealed class CosyVoice3Pipeline : ITextToSpeechPipeline
{
    public string Architecture => "CosyVoice3";
    public int SampleRate => 24000;
    public int DefaultSampleRate => 24000;

    public AudioGenerationResult Generate(AudioGenerationRequest request)
    {
        var pcm = Generate(request.Text, referenceAudioPath: request.ReferenceAudioPath, referenceText: request.ReferenceText);
        var result = new AudioGenerationResult(pcm, DefaultSampleRate);
        if (!string.IsNullOrEmpty(request.OutputPath))
        {
            result.SaveWav(request.OutputPath);
        }
        return result;
    }

    public IAsyncEnumerable<float[]> GenerateStreamAsync(AudioGenerationRequest request, System.Threading.CancellationToken ct = default)
        => TtsStreamingHelper.SplitAndGenerateAsync(request, Generate, ct);

    private readonly GgufModel _rawModel;
    private readonly CosyVoice3LlmTensorSource _llmSource;
    private readonly CosyVoice3FlowEncoderWeights _flowWeights;
    private readonly CosyVoice3DiTWeights _ditWeights;
    private readonly CosyVoice3HiftWeights _hiftWeights;
    private readonly string? _campplusOnnxPath;
    private readonly string? _speechTokenizerOnnxPath;
    private readonly Core.IComputeBackend? _backend;

    private CosyVoice3Pipeline(GgufModel rawModel, CosyVoice3LlmTensorSource llmSource, CosyVoice3FlowEncoderWeights flowWeights, CosyVoice3DiTWeights ditWeights, CosyVoice3HiftWeights hiftWeights, string? campplusOnnxPath, string? speechTokenizerOnnxPath, Core.IComputeBackend? backend = null)
    {
        _rawModel = rawModel;
        _llmSource = llmSource;
        _flowWeights = flowWeights;
        _ditWeights = ditWeights;
        _hiftWeights = hiftWeights;
        _campplusOnnxPath = campplusOnnxPath;
        _speechTokenizerOnnxPath = speechTokenizerOnnxPath;
        _backend = backend;
    }

    /// <summary>Loads all real CosyVoice3 weights from the single bundled GGUF file.
    /// <paramref name="backend"/>, when supplied, routes the flow-matching DiT's Sgemm-shaped
    /// projections through it (--backend vulkan option, see
    /// docs/052-vulkan-backend-for-tts-engines-plan.md); the LLM, ConvPositionEmbedding, and
    /// HiFTGenerator vocoder stay CPU-only regardless.</summary>
    public static CosyVoice3Pipeline Load(string ggufPath, Core.IComputeBackend? backend = null)
    {
        if (string.IsNullOrWhiteSpace(ggufPath) || !File.Exists(ggufPath))
            throw new FileNotFoundException($"CosyVoice3 GGUF model not found: {ggufPath}");

        var rawModel = GgufModel.Open(ggufPath);
        var llmSource = new CosyVoice3LlmTensorSource(rawModel);
        llmSource.EnableSpeechGenerationMode();
        var flowWeights = new CosyVoice3FlowEncoderWeights(rawModel);
        var ditWeights = new CosyVoice3DiTWeights(rawModel);
        var hiftWeights = new CosyVoice3HiftWeights(ggufPath); // separate GgufModel.Open under the hood, real GGUF reopen is cheap (mmap)
        string? campplusOnnxPath = ResolveOnnxPath(ggufPath, "campplus.onnx", "models/campplus.onnx");
        string? speechTokenizerOnnxPath = ResolveOnnxPath(ggufPath, "speech_tokenizer_v3.onnx", "models/cosyvoice_speech_tokenizer_v2.onnx");

        return new CosyVoice3Pipeline(rawModel, llmSource, flowWeights, ditWeights, hiftWeights, campplusOnnxPath, speechTokenizerOnnxPath, backend);
    }

    /// <summary>
    /// Looks for the named ONNX file next to the GGUF file first (real per-checkout layout, e.g.
    /// `models/cosyvoice3/frontend-onnx/campplus.onnx`), then falls back to the given shared
    /// default path. Returns null (not a throw) if neither exists.
    /// </summary>
    private static string? ResolveOnnxPath(string ggufPath, string localFileName, string fallbackPath)
    {
        string? dir = Path.GetDirectoryName(Path.GetFullPath(ggufPath));
        foreach (var c in new[]
        {
            dir is null ? null : Path.Combine(dir, "frontend-onnx", localFileName),
            dir is null ? null : Path.Combine(dir, localFileName),
            fallbackPath,
        })
        {
            if (c is not null && File.Exists(c)) return c;
        }
        return null;
    }

    /// <summary>
    /// Synthesizes real 24kHz PCM audio for the given text.
    ///
    /// <para>When a real reference audio file is supplied, this now follows the real reference's
    /// zero-shot conditioning mechanism (`CausalMaskedDiffWithDiT::build_cgraph_encode` in
    /// `examples/cosyvoice.cpp/src/cosyvoice-graph.cpp`): the reference audio's own speech tokens
    /// (via <see cref="CosyVoiceSpeechTokenizer"/>, a real ONNX speech tokenizer -- NOT the same
    /// model as the CamPlus speaker encoder) are concatenated with the newly generated tokens
    /// BEFORE flow encoding, producing one joint `mu`/`cond` sequence; `cond`'s first
    /// `promptFrames` are the reference's own real mel (zero elsewhere). The synthesized
    /// reference-audio-prefix portion of the output waveform is then trimmed off before
    /// returning, since the reference only returns the newly-synthesized continuation.</para>
    /// </summary>
    public float[] Generate(string text, int maxNewSpeechTokens = 200, int odeSteps = 10, int? seed = null, string? referenceAudioPath = null, float cfgRate = 0.7f, string? referenceText = null, string? instruction = null, float temperature = 0.8f, float[]? explicitSpeakerEmbedding = null, float pitchScale = 1.0f, bool useFrameHoldSineGenExperiment = false)
    {
        float[] speakerEmbedding = explicitSpeakerEmbedding ?? ExtractSpeakerEmbedding(referenceAudioPath);
        float[] refMel = ExtractReferenceMel(referenceAudioPath);
        int[] promptTokens = ExtractPromptTokens(referenceAudioPath);

        // refMel (from ExtractReferenceMel's own independent hop/padding) and promptTokens (from
        // the separate ONNX speech tokenizer) are NOT guaranteed to land on exactly matching
        // frame counts under TokenMelRatio -- confirmed via real [DBG] trace on the reference
        // audio: 352 prompt tokens (704 frames under TokenMelRatio=2) vs 703 refMel frames, a real
        // one-frame skew. The real reference (examples/cosyvoice.cpp's
        // cosyvoice_frontend_prompt_speech_finalize) resolves this by truncating BOTH down to
        // their mutually-consistent common length (min(refMelFrames/TokenMelRatio, tokenCount)),
        // not by letting either one dominate -- otherwise `mu` (built from promptTokens) and
        // `cond` (built from refMel) transition from prompt to target content on different
        // frames, which the DiT's causal conv smears and CFG then amplifies over the ODE steps.
        // This showed up as excess variance concentrated in the low mel-frequency (most
        // DC-sensitive) channels of the generated target audio.
        if (promptTokens.Length > 0 && refMel.Length > 0)
        {
            int refMelFrames = refMel.Length / CosyVoice3DiTWeights.MelDim;
            int alignedTokens = Math.Min(refMelFrames / CosyVoice3FlowEncoderWeights.TokenMelRatio, promptTokens.Length);
            if (alignedTokens != promptTokens.Length)
                promptTokens = promptTokens[..alignedTokens];
            int alignedMelFrames = alignedTokens * CosyVoice3FlowEncoderWeights.TokenMelRatio;
            if (alignedMelFrames != refMelFrames)
            {
                var trimmedMel = new float[alignedMelFrames * CosyVoice3DiTWeights.MelDim];
                Array.Copy(refMel, trimmedMel, trimmedMel.Length);
                refMel = trimmedMel;
            }
        }

        // Condition the LLM itself on the reference (promptText/promptTokens) -- see
        // CosyVoice3Llm.GenerateSpeechTokens's own doc comment for why this matters: without it,
        // the newly-generated speech tokens are not a real continuation of promptTokens, even
        // though CosyVoice3Pipeline splices them together below before flow encoding.
        var speechTokens = CosyVoice3Llm.GenerateSpeechTokens(_rawModel, _llmSource, text, maxNewSpeechTokens, promptText: referenceText, promptSpeechTokens: promptTokens, instruction: instruction, temperature: temperature);
        if (speechTokens.Length == 0) return [];

        int[] jointTokens = promptTokens.Length > 0 ? [.. promptTokens, .. speechTokens] : speechTokens;
        var (mu, spks) = CosyVoice3FlowEncoder.ComputeMuAndSpks(_flowWeights, jointTokens, speakerEmbedding);

        int numFrames = mu.Length / CosyVoice3DiTWeights.MelDim;
        var cond = new float[mu.Length];
        // promptTokens and refMel were aligned to a common frame count above, so mu's and cond's
        // prompt/target boundary now land on the same frame -- either source gives the same value.
        int promptFrames = 0;
        if (refMel.Length > 0)
        {
            promptFrames = Math.Min(refMel.Length / CosyVoice3DiTWeights.MelDim, numFrames);
            Array.Copy(refMel, 0, cond, 0, promptFrames * CosyVoice3DiTWeights.MelDim);
        }

        var spksBroadcast = new float[numFrames * CosyVoice3DiTWeights.MelDim];
        for (int f = 0; f < numFrames; f++)
            Array.Copy(spks, 0, spksBroadcast, f * CosyVoice3DiTWeights.MelDim, CosyVoice3DiTWeights.MelDim);

        var rng = new Random(seed ?? 0);
        var mel = CosyVoice3DiTModel.SolveFlowMatchingOde(_ditWeights, cond, mu, spksBroadcast, numFrames, odeSteps, rng, cfgRate: cfgRate, backend: _backend);

        if (Environment.GetEnvironmentVariable("STINGRAY_DEBUG_COSYVOICE3") is { Length: > 0 })
        {
            float melMin = float.MaxValue, melMax = float.MinValue, melSum = 0f, melAbsSum = 0f;
            foreach (var v in mel) { if (v < melMin) melMin = v; if (v > melMax) melMax = v; melSum += v; melAbsSum += MathF.Abs(v); }
            Console.Error.WriteLine($"[DBG] speechTokens={speechTokens.Length} promptTokens={promptTokens.Length} numFrames={numFrames} promptFrames={promptFrames} spk[0..3]={string.Join(",", speakerEmbedding[..Math.Min(4, speakerEmbedding.Length)])} refMel.Length={refMel.Length} mel min={melMin:F4} max={melMax:F4} mean={melSum / mel.Length:F4} meanAbs={melAbsSum / mel.Length:F4}");
        }

        // Slices ONLY target mel frames [targetFrames, MelDim] matching C++ reference
        // (examples/audio.cpp/src/models/cosyvoice3/flow.cpp:810-817 and session.cpp:224-230).
        // HiFT is executed strictly on the target continuation, eliminating boundary clicking
        // and convolutional prompt bleeding.
        int targetFrames = numFrames - promptFrames;
        if (targetFrames <= 0) return [];

        var melChannelFirst = new float[targetFrames * CosyVoice3DiTWeights.MelDim];
        for (int f = 0; f < targetFrames; f++)
        {
            int srcFrame = promptFrames + f;
            for (int c = 0; c < CosyVoice3DiTWeights.MelDim; c++)
            {
                melChannelFirst[c * targetFrames + f] = mel[srcFrame * CosyVoice3DiTWeights.MelDim + c];
            }
        }

        if (Environment.GetEnvironmentVariable("STINGRAY_CFG_TRACE") == "1")
        {
            int worstFrame = -1; float worstAbs = 0f;
            for (int f = 0; f < targetFrames; f++)
            {
                float m = 0f;
                for (int c = 0; c < CosyVoice3DiTWeights.MelDim; c++) m = Math.Max(m, Math.Abs(melChannelFirst[c * targetFrames + f]));
                if (m > worstAbs) { worstAbs = m; worstFrame = f; }
            }
            Console.Error.WriteLine($"[TargetMelTrace] targetFrames={targetFrames} worstFrame={worstFrame} worstAbs={worstAbs:F4}");

            var sb = new System.Text.StringBuilder("[ChannelStats] mean=");
            for (int c = 0; c < CosyVoice3DiTWeights.MelDim; c++)
            {
                double sum = 0;
                for (int f = 0; f < targetFrames; f++) sum += melChannelFirst[c * targetFrames + f];
                sb.Append((sum / targetFrames).ToString("F3")).Append(',');
            }
            Console.Error.WriteLine(sb.ToString());
            var sb2 = new System.Text.StringBuilder("[ChannelStats] std=");
            for (int c = 0; c < CosyVoice3DiTWeights.MelDim; c++)
            {
                double sum = 0, sumsq = 0;
                for (int f = 0; f < targetFrames; f++) { sum += melChannelFirst[c * targetFrames + f]; sumsq += (double)melChannelFirst[c * targetFrames + f] * melChannelFirst[c * targetFrames + f]; }
                double mean = sum / targetFrames;
                double variance = sumsq / targetFrames - mean * mean;
                sb2.Append(Math.Sqrt(Math.Max(0, variance)).ToString("F3")).Append(',');
            }
            Console.Error.WriteLine(sb2.ToString());
        }

        var wav = useFrameHoldSineGenExperiment
            ? CosyVoiceSineGenFrameHoldExperiment.Generate(_hiftWeights, melChannelFirst, targetFrames, rng, pitchScale: pitchScale)
            : CosyVoiceHiftVocoder.Generate(_hiftWeights, melChannelFirst, targetFrames, rng, pitchScale: pitchScale);

        // Peak normalize to 0.85 full scale
        float peak = 0f;
        for (int i = 0; i < wav.Length; i++)
        {
            float a = MathF.Abs(wav[i]);
            if (a > peak) peak = a;
        }
        if (peak > 1e-4f && peak < 0.8f)
        {
            float gain = 0.85f / peak;
            for (int i = 0; i < wav.Length; i++) wav[i] *= gain;
        }

        return wav;
    }

    /// <summary>
    /// Real speech-token extraction from reference audio via <see cref="CosyVoiceSpeechTokenizer"/>
    /// (a separate real ONNX graph from CamPlus), needed to build the joint prompt+target token
    /// sequence the flow encoder expects for real zero-shot conditioning.
    /// </summary>
    private int[] ExtractPromptTokens(string? referenceAudioPath)
    {
        if (string.IsNullOrEmpty(referenceAudioPath) || _speechTokenizerOnnxPath is null || !File.Exists(referenceAudioPath))
            return [];

        try
        {
            var (samples, sr, _) = WavReader.ReadWav(referenceAudioPath);
            if (samples.Length == 0) return [];
            if (sr != CosyVoiceSpeechTokenizer.SampleRate)
                samples = AudioResampler.Resample(samples, sr, CosyVoiceSpeechTokenizer.SampleRate);

            var tokens = CosyVoiceSpeechTokenizer.Extract(_speechTokenizerOnnxPath, samples);
            return tokens ?? [];
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[CosyVoice3Pipeline] Prompt speech-token extraction failed, falling back to no prompt tokens: {ex.Message}");
            return [];
        }
    }

    /// <summary>
    /// Real x-vector extraction (see <see cref="CamPlusSpeakerEncoder"/>) when a reference audio
    /// path and a real `campplus.onnx` are both available; falls back to the pre-existing
    /// all-zero placeholder vector otherwise.
    /// </summary>
    private float[] ExtractSpeakerEmbedding(string? referenceAudioPath)
    {
        var zero = new float[CosyVoice3FlowEncoderWeights.SpeakerEmbedDim];
        if (string.IsNullOrEmpty(referenceAudioPath) || _campplusOnnxPath is null || !File.Exists(referenceAudioPath))
            return zero;

        try
        {
            var (samples, sr, _) = WavReader.ReadWav(referenceAudioPath);
            if (samples.Length == 0) return zero;
            if (sr != CamPlusSpeakerEncoder.SampleRate)
                samples = AudioResampler.Resample(samples, sr, CamPlusSpeakerEncoder.SampleRate);

            var emb = CamPlusSpeakerEncoder.Extract(_campplusOnnxPath, samples);
            return emb is { Length: CosyVoice3FlowEncoderWeights.SpeakerEmbedDim } ? emb : zero;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[CosyVoice3Pipeline] Real speaker-embedding extraction failed, falling back to zero vector: {ex.Message}");
            return zero;
        }
    }

    /// <summary>
    /// Extracts 80-channel 24kHz mel-spectrogram conditioning from reference audio if available.
    /// </summary>
    private static float[] ExtractReferenceMel(string? referenceAudioPath)
    {
        if (string.IsNullOrEmpty(referenceAudioPath) || !File.Exists(referenceAudioPath))
            return [];

        try
        {
            var (samples, sr, _) = WavReader.ReadWav(referenceAudioPath);
            if (samples.Length == 0) return [];
            if (sr != CosyVoiceMelExtractor.SampleRate)
                samples = AudioResampler.Resample(samples, sr, CosyVoiceMelExtractor.SampleRate);

            var mel = CosyVoiceMelExtractor.Shared.ExtractMel(samples);
            if (Environment.GetEnvironmentVariable("STINGRAY_CFG_TRACE") == "1" && mel.Length > 0)
            {
                int melDim = CosyVoice3DiTWeights.MelDim;
                int frames = mel.Length / melDim;
                int worstFrame = -1; float worstAbs = 0f;
                for (int f = 0; f < frames; f++)
                {
                    float m = 0f;
                    for (int c = 0; c < melDim; c++) m = Math.Max(m, Math.Abs(mel[f * melDim + c]));
                    if (m > worstAbs) { worstAbs = m; worstFrame = f; }
                }
                Console.Error.WriteLine($"[RefMelTrace] frames={frames} worstFrame={worstFrame} worstAbs={worstAbs:F4}");
            }
            return mel;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[CosyVoice3Pipeline] Reference mel extraction failed: {ex.Message}");
            return [];
        }
    }

    public void Dispose()
    {
        _llmSource.Dispose();
        _hiftWeights.Dispose();
        _rawModel.Dispose();
    }
}
