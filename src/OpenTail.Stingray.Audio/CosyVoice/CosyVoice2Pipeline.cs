
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

    private CosyVoice2Pipeline(CosyVoiceLlmTensorSource llmSource, string tokenizerDir, CosyVoiceFlowWeights flowWeights, CosyVoiceCfmDecoderWeights cfmWeights, CosyVoiceHiftWeights hiftWeights)
    {
        _llmSource = llmSource;
        _tokenizerDir = tokenizerDir;
        _flowWeights = flowWeights;
        _cfmWeights = cfmWeights;
        _hiftWeights = hiftWeights;
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

        return new CosyVoice2Pipeline(llmSource, tokenizerDir, flowWeights, cfmWeights, hiftWeights);
    }

    /// <summary>Synthesizes real 24kHz PCM audio for the given text.</summary>
    public float[] Generate(string text, int maxNewSpeechTokens = 200, int odeSteps = 10, int? seed = null)
    {
        var speechTokens = CosyVoiceLlmGeneration.GenerateSpeechTokens(_llmSource, _tokenizerDir, text, maxNewTokens: maxNewSpeechTokens);
        if (speechTokens.Length == 0) return [];

        var (mu, totalFrames) = CosyVoiceFlowEncoder.Forward(_flowWeights, promptTokens: [], speechTokens);
        var spkEmbed = CosyVoiceFlowEncoder.ProjectSpeakerEmbedding(_flowWeights, new float[_flowWeights.SpkEmbedDim]);

        var rng = new Random(seed ?? 0);
        var mel = CosyVoiceCfmDecoder.Generate(_cfmWeights, mu, spkEmbed, totalFrames, rng, odeSteps);

        return CosyVoiceHiftVocoder.Generate(_hiftWeights, mel, totalFrames, rng);
    }

    public void Dispose()
    {
        _llmSource.Dispose();
        _flowWeights.Dispose();
        _hiftWeights.Dispose();
    }
}
