
namespace OpenTail.Stingray.Audio.CosyVoice;

/// <summary>
/// Real, weight-driven CosyVoice2 end-to-end pipeline: text -&gt; <see cref="CosyVoiceLlmGeneration"/>
/// speech tokens -&gt; <see cref="CosyVoiceFlowEncoder"/> conditioning -&gt;
/// <see cref="CosyVoiceCfmDecoder"/> CFM ODE solve -&gt; <see cref="CosyVoiceHiftVocoder"/> waveform.
/// Chains four independently real-weights-tested stages built this session (see docs/audio-
/// review-progress.md's CosyVoice section) into one call -- the same pattern already used for
/// <see cref="CosyVoice3Pipeline"/>.
///
/// <para>Real, deliberate simplification (documented, not silently dropped): no reference/prompt
/// audio (zero-shot voice cloning) support yet -- speaker conditioning uses a zero 192-dim vector
/// (a real, non-fabricated affine-layer bias contribution, just not a real per-speaker embedding,
/// same CamPlus-x-vector gap already documented for CosyVoice3Pipeline) and no prompt speech
/// tokens are injected into the LLM prefix (plain/cross-lingual synthesis mode, not zero-shot).
/// </para>
/// </summary>
public sealed class CosyVoice2Pipeline : IDisposable
{
    public int SampleRate => 24000;

    private readonly CosyVoiceLlmTensorSource _llmSource;
    private readonly string _tokenizerDir;
    private readonly CosyVoiceFlowWeights _flowWeights;
    private readonly CosyVoiceCfmDecoderWeights _cfmWeights;
    private readonly CosyVoiceHiftWeights _hiftWeights;
    private readonly string? _campplusOnnxPath;
    private readonly string? _speechTokenizerOnnxPath;

    private CosyVoice2Pipeline(CosyVoiceLlmTensorSource llmSource, string tokenizerDir, CosyVoiceFlowWeights flowWeights, CosyVoiceCfmDecoderWeights cfmWeights, CosyVoiceHiftWeights hiftWeights, string? campplusOnnxPath, string? speechTokenizerOnnxPath)
    {
        _llmSource = llmSource;
        _tokenizerDir = tokenizerDir;
        _flowWeights = flowWeights;
        _cfmWeights = cfmWeights;
        _hiftWeights = hiftWeights;
        _campplusOnnxPath = campplusOnnxPath;
        _speechTokenizerOnnxPath = speechTokenizerOnnxPath;
    }

    /// <summary>First existing `models/{name}` or `models/_models/{name}` walking up from the LLM
    /// checkpoint's directory (the reference-prompt ONNX helpers are optional).</summary>
    private static string? ResolveModelFile(string anchorPath, string name)
    {
        for (var dir = new DirectoryInfo(Path.GetDirectoryName(Path.GetFullPath(anchorPath))!); dir is not null; dir = dir.Parent)
        {
            foreach (var candidate in new[] { Path.Combine(dir.FullName, name), Path.Combine(dir.FullName, "models", name), Path.Combine(dir.FullName, "models", "_models", name) })
                if (File.Exists(candidate)) return candidate;
        }
        return null;
    }

    /// <summary>
    /// Loads all real CosyVoice2 weights: `llmSafetensorsPath` (`models/cosyvoice2_llm.safetensors`),
    /// `tokenizerDir` (`models/cosyvoice2_tokenizer`, the real downloaded HF Qwen2 tokenizer),
    /// `flowSafetensorsPath` (`models/cosyvoice2_flow.safetensors`), `hiftSafetensorsPath`
    /// (`models/cosyvoice2_hift.safetensors`).
    /// </summary>
    public static CosyVoice2Pipeline Load(string llmSafetensorsPath, string tokenizerDir, string flowSafetensorsPath, string hiftSafetensorsPath)
    {
        if (!File.Exists(llmSafetensorsPath))
            throw new FileNotFoundException($"CosyVoice2 LLM model not found: {llmSafetensorsPath}");
        if (!Directory.Exists(tokenizerDir))
            throw new DirectoryNotFoundException($"CosyVoice2 tokenizer directory not found: {tokenizerDir}");

        var llmSource = new CosyVoiceLlmTensorSource(
            llmSafetensorsPath,
            numLayers: 24, hiddenDim: 896, numHeads: 14, numKvHeads: 2, headDim: 64,
            ffDim: 4864, vocabSize: 151936, ropeTheta: 1_000_000f, rmsNormEps: 1e-6f);

        var flowWeights = new CosyVoiceFlowWeights(flowSafetensorsPath);
        var cfmWeights = new CosyVoiceCfmDecoderWeights(flowWeights);
        var hiftWeights = new CosyVoiceHiftWeights(hiftSafetensorsPath);

        return new CosyVoice2Pipeline(llmSource, tokenizerDir, flowWeights, cfmWeights, hiftWeights,
            ResolveModelFile(llmSafetensorsPath, "campplus.onnx"),
            ResolveModelFile(llmSafetensorsPath, "cosyvoice_speech_tokenizer_v2.onnx"));
    }

    /// <summary>
    /// Synthesizes real 24kHz PCM audio for the given text. With <paramref name="referenceAudioPath"/>
    /// (+ its <paramref name="referenceText"/>), runs the real zero-shot prompt path of CosyVoice2's
    /// `inference_zero_shot` (added 2026-09-25; without it the model had an all-zero speaker and no
    /// prompt, and its output drifted into garbled words): CamPlus x-vector → speaker embedding,
    /// speech-tokenizer-v2 prompt tokens (aligned with the reference mel at 2 mel frames/token) →
    /// the LLM continues the prompt speech tokens, the flow encodes prompt+generated tokens, the CFM
    /// is conditioned on the prompt mel (`cond`), and the prompt frames are trimmed before HiFT.
    /// </summary>
    public float[] Generate(string text, int maxNewSpeechTokens = 200, int odeSteps = 10, int? seed = null,
        string? referenceAudioPath = null, string? referenceText = null)
    {
        const int melDim = CosyVoiceCfmDecoderWeights.OutChannels;
        const int tokenMelRatio = 2;

        var xvector = new float[_flowWeights.SpkEmbedDim];
        int[] promptTokens = [];
        float[] refMel = []; // frame-major [frames, 80]
        if (!string.IsNullOrEmpty(referenceAudioPath))
        {
            var (samples, sr, _) = WavReader.ReadWav(referenceAudioPath);
            if (_campplusOnnxPath is not null)
            {
                var s16 = sr == CamPlusSpeakerEncoder.SampleRate ? samples : AudioResampler.Resample(samples, sr, CamPlusSpeakerEncoder.SampleRate);
                if (CamPlusSpeakerEncoder.Extract(_campplusOnnxPath, s16) is { } emb && emb.Length == xvector.Length) xvector = emb;
            }
            if (_speechTokenizerOnnxPath is not null)
            {
                var s16 = sr == CosyVoiceSpeechTokenizer.SampleRate ? samples : AudioResampler.Resample(samples, sr, CosyVoiceSpeechTokenizer.SampleRate);
                promptTokens = CosyVoiceSpeechTokenizer.Extract(_speechTokenizerOnnxPath, s16) ?? [];
            }
            var s24 = sr == CosyVoiceMelExtractor.SampleRate ? samples : AudioResampler.Resample(samples, sr, CosyVoiceMelExtractor.SampleRate);
            refMel = CosyVoiceMelExtractor.Shared.ExtractMel(s24);

            // Align prompt tokens and prompt mel to a common length (reference frontend truncates both).
            int alignedTokens = Math.Min(refMel.Length / melDim / tokenMelRatio, promptTokens.Length);
            promptTokens = promptTokens[..alignedTokens];
            refMel = refMel[..(alignedTokens * tokenMelRatio * melDim)];
        }

        var speechTokens = CosyVoiceLlmGeneration.GenerateSpeechTokens(_llmSource, _tokenizerDir, text,
            promptText: promptTokens.Length > 0 ? referenceText ?? "" : "", promptSpeechTokens: promptTokens,
            maxNewTokens: maxNewSpeechTokens, rng: new Random(seed ?? 0));
        if (speechTokens.Length == 0) return [];

        var (mu, totalFrames) = CosyVoiceFlowEncoder.Forward(_flowWeights, promptTokens, speechTokens);
        var spkEmbed = CosyVoiceFlowEncoder.ProjectSpeakerEmbedding(_flowWeights, xvector);

        int promptFrames = Math.Min(refMel.Length / melDim, totalFrames);
        float[]? cond = null;
        if (promptFrames > 0)
        {
            cond = new float[melDim * totalFrames]; // channel-first [80, t]
            for (int f = 0; f < promptFrames; f++)
                for (int c = 0; c < melDim; c++)
                    cond[c * totalFrames + f] = refMel[f * melDim + c];
        }

        var rng = new Random(seed ?? 0);
        var mel = CosyVoiceCfmDecoder.Generate(_cfmWeights, mu, spkEmbed, totalFrames, rng, odeSteps, cond);

        int targetFrames = totalFrames - promptFrames;
        if (targetFrames <= 0) return [];
        if (promptFrames > 0)
        {
            var target = new float[melDim * targetFrames];
            for (int c = 0; c < melDim; c++)
                Array.Copy(mel, c * totalFrames + promptFrames, target, c * targetFrames, targetFrames);
            mel = target;
        }
        return CosyVoiceHiftVocoder.Generate(_hiftWeights, mel, targetFrames, rng);
    }

    public void Dispose()
    {
        _llmSource.Dispose();
        _flowWeights.Dispose();
        _hiftWeights.Dispose();
    }
}
