using System.IO;
using OpenTail.Stingray.Core;

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
/// own `campplus.onnx`, found locally at `models/campplus.onnx`) -- otherwise falls back to an
/// all-zero vector, same as before. Real, deliberate simplification still open (documented, not
/// silently dropped): `cond` (the reference mel prepended to the DiT input) is still all-zero
/// even with a real reference audio -- only the speaker embedding is real so far -- and the
/// DiT's CFG refinement is also still omitted (see
/// <see cref="CosyVoice3DiTModel.SolveFlowMatchingOde"/>'s doc comment).</para>
/// </summary>
public sealed class CosyVoice3Pipeline : ITextToSpeechPipeline
{
    public string Architecture => "CosyVoice3";
    public int SampleRate => 24000;
    public int DefaultSampleRate => 24000;

    public AudioGenerationResult Generate(AudioGenerationRequest request)
    {
        var pcm = Generate(request.Text, referenceAudioPath: request.ReferenceAudioPath);
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

    private CosyVoice3Pipeline(GgufModel rawModel, CosyVoice3LlmTensorSource llmSource, CosyVoice3FlowEncoderWeights flowWeights, CosyVoice3DiTWeights ditWeights, CosyVoice3HiftWeights hiftWeights, string? campplusOnnxPath)
    {
        _rawModel = rawModel;
        _llmSource = llmSource;
        _flowWeights = flowWeights;
        _ditWeights = ditWeights;
        _hiftWeights = hiftWeights;
        _campplusOnnxPath = campplusOnnxPath;
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
        string? campplusOnnxPath = ResolveCampplusOnnxPath(ggufPath);

        return new CosyVoice3Pipeline(rawModel, llmSource, flowWeights, ditWeights, hiftWeights, campplusOnnxPath);
    }

    /// <summary>
    /// Looks for `campplus.onnx` next to the GGUF file first (real per-checkout layout, e.g.
    /// `models/cosyvoice3/frontend-onnx/campplus.onnx`), then falls back to the shared default
    /// `models/campplus.onnx`. Returns null (not a throw) if neither exists -- speaker
    /// conditioning then falls back to the pre-existing all-zero vector, same as before this was
    /// wired up.
    /// </summary>
    private static string? ResolveCampplusOnnxPath(string ggufPath)
    {
        string? dir = Path.GetDirectoryName(Path.GetFullPath(ggufPath));
        foreach (var c in new[]
        {
            dir is null ? null : Path.Combine(dir, "frontend-onnx", "campplus.onnx"),
            dir is null ? null : Path.Combine(dir, "campplus.onnx"),
            "models/campplus.onnx",
        })
        {
            if (c is not null && File.Exists(c)) return c;
        }
        return null;
    }

    /// <summary>Synthesizes real 24kHz PCM audio for the given text.</summary>
    public float[] Generate(string text, int maxNewSpeechTokens = 200, int odeSteps = 10, int? seed = null, string? referenceAudioPath = null)
    {
        var speechTokens = CosyVoice3Llm.GenerateSpeechTokens(_rawModel, _llmSource, text, maxNewSpeechTokens);
        if (speechTokens.Length == 0) return [];

        float[] speakerEmbedding = ExtractSpeakerEmbedding(referenceAudioPath);
        var (mu, spks) = CosyVoice3FlowEncoder.ComputeMuAndSpks(_flowWeights, speechTokens, speakerEmbedding);

        int numFrames = mu.Length / CosyVoice3DiTWeights.MelDim;
        var cond = new float[mu.Length]; // no reference audio -> zero conditioning mel

        var spksBroadcast = new float[numFrames * CosyVoice3DiTWeights.MelDim];
        for (int f = 0; f < numFrames; f++)
            Array.Copy(spks, 0, spksBroadcast, f * CosyVoice3DiTWeights.MelDim, CosyVoice3DiTWeights.MelDim);

        var rng = new Random(seed ?? 0);
        var mel = CosyVoice3DiTModel.SolveFlowMatchingOde(_ditWeights, cond, mu, spksBroadcast, numFrames, odeSteps, rng);

        // mel is channel-last [numFrames, MelDim]; HiFT's real forward expects channel-first [MelDim, T] flat.
        var melChannelFirst = new float[mel.Length];
        for (int f = 0; f < numFrames; f++)
            for (int c = 0; c < CosyVoice3DiTWeights.MelDim; c++)
                melChannelFirst[c * numFrames + f] = mel[f * CosyVoice3DiTWeights.MelDim + c];

        var wav = CosyVoiceHiftVocoder.Generate(_hiftWeights, melChannelFirst, numFrames, rng);

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
    /// Real x-vector extraction (see <see cref="CamPlusSpeakerEncoder"/>) when a reference audio
    /// path and a real `campplus.onnx` are both available; falls back to the pre-existing
    /// all-zero placeholder vector otherwise (missing reference audio, missing ONNX file, or a
    /// load/extraction failure -- never throws, since a degraded speaker embedding is strictly
    /// better than crashing the whole synthesis).
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

    public void Dispose()
    {
        _llmSource.Dispose();
        _hiftWeights.Dispose();
        _rawModel.Dispose();
    }
}
