
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

    private CosyVoice3Pipeline(GgufModel rawModel, CosyVoice3LlmTensorSource llmSource, CosyVoice3FlowEncoderWeights flowWeights, CosyVoice3DiTWeights ditWeights, CosyVoice3HiftWeights hiftWeights, string? campplusOnnxPath, string? speechTokenizerOnnxPath)
    {
        _rawModel = rawModel;
        _llmSource = llmSource;
        _flowWeights = flowWeights;
        _ditWeights = ditWeights;
        _hiftWeights = hiftWeights;
        _campplusOnnxPath = campplusOnnxPath;
        _speechTokenizerOnnxPath = speechTokenizerOnnxPath;
    }

    /// <summary>Loads all real CosyVoice3 weights from the single bundled GGUF file.</summary>
    public static CosyVoice3Pipeline Load(string ggufPath)
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

        return new CosyVoice3Pipeline(rawModel, llmSource, flowWeights, ditWeights, hiftWeights, campplusOnnxPath, speechTokenizerOnnxPath);
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
    public float[] Generate(string text, int maxNewSpeechTokens = 200, int odeSteps = 10, int? seed = null, string? referenceAudioPath = null, float cfgRate = 0.7f, string? referenceText = null, string? instruction = null, float temperature = 0.8f, float[]? explicitSpeakerEmbedding = null, float pitchScale = 1.0f)
    {
        float[] speakerEmbedding = explicitSpeakerEmbedding ?? ExtractSpeakerEmbedding(referenceAudioPath);
        float[] refMel = ExtractReferenceMel(referenceAudioPath);
        int[] promptTokens = ExtractPromptTokens(referenceAudioPath);

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
        var mel = CosyVoice3DiTModel.SolveFlowMatchingOde(_ditWeights, cond, mu, spksBroadcast, numFrames, odeSteps, rng, cfgRate: cfgRate);

        if (Environment.GetEnvironmentVariable("STINGRAY_DEBUG_COSYVOICE3") is { Length: > 0 })
        {
            float melMin = float.MaxValue, melMax = float.MinValue, melSum = 0f, melAbsSum = 0f;
            foreach (var v in mel) { if (v < melMin) melMin = v; if (v > melMax) melMax = v; melSum += v; melAbsSum += MathF.Abs(v); }
            Console.Error.WriteLine($"[DBG] speechTokens={speechTokens.Length} promptTokens={promptTokens.Length} numFrames={numFrames} promptFrames={promptFrames} spk[0..3]={string.Join(",", speakerEmbedding[..Math.Min(4, speakerEmbedding.Length)])} refMel.Length={refMel.Length} mel min={melMin:F4} max={melMax:F4} mean={melSum / mel.Length:F4} meanAbs={melAbsSum / mel.Length:F4}");
        }

        // mel is channel-last [numFrames, MelDim]; HiFT's real forward expects channel-first [MelDim, T] flat.
        var melChannelFirst = new float[mel.Length];
        for (int f = 0; f < numFrames; f++)
            for (int c = 0; c < CosyVoice3DiTWeights.MelDim; c++)
                melChannelFirst[c * numFrames + f] = mel[f * CosyVoice3DiTWeights.MelDim + c];

        var wav = CosyVoiceHiftVocoder.Generate(_hiftWeights, melChannelFirst, numFrames, rng, pitchScale: pitchScale);

        // Trim off the synthesized reference-audio-prefix portion -- the reference only returns
        // the newly-synthesized continuation, not a regenerated copy of the prompt audio.
        if (promptFrames > 0)
        {
            int trimSamples = Math.Min(promptFrames * CosyVoiceMelExtractor.HopLength, wav.Length);
            wav = wav[trimSamples..];
        }

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

            return CosyVoiceMelExtractor.Shared.ExtractMel(samples);
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
