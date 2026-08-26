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
/// <para>Real, deliberate simplification (documented, not silently dropped): no reference/
/// prompt audio (zero-shot voice cloning) support yet -- speaker conditioning uses a zero
/// 192-dim vector (a real, non-fabricated affine-layer bias contribution, just not a real
/// per-speaker embedding, since this codebase has no CamPlus x-vector extractor ported yet) and
/// `cond` (the reference mel) is all-zero. The DiT's CFG refinement is also omitted (see
/// <see cref="CosyVoice3DiTModel.SolveFlowMatchingOde"/>'s doc comment).</para>
/// </summary>
public sealed class CosyVoice3Pipeline : ITextToSpeechPipeline
{
    public string Architecture => "CosyVoice3";
    public int SampleRate => 24000;
    public int DefaultSampleRate => 24000;

    public AudioGenerationResult Generate(AudioGenerationRequest request)
    {
        var pcm = Generate(request.Text);
        var result = new AudioGenerationResult(pcm, DefaultSampleRate);
        if (!string.IsNullOrEmpty(request.OutputPath))
        {
            result.SaveWav(request.OutputPath);
        }
        return result;
    }

    public async IAsyncEnumerable<float[]> GenerateStreamAsync(AudioGenerationRequest request, [System.Runtime.CompilerServices.EnumeratorCancellation] System.Threading.CancellationToken ct = default)
    {
        if (string.IsNullOrWhiteSpace(request.Text)) yield break;

        var sentences = System.Text.RegularExpressions.Regex.Split(request.Text, @"(?<=[.!?\n])\s+");
        foreach (var s in sentences)
        {
            var trimmed = s.Trim();
            if (string.IsNullOrEmpty(trimmed)) continue;
            ct.ThrowIfCancellationRequested();

            var req = request with { Text = trimmed, OutputPath = null };
            var res = Generate(req);
            if (res.Samples.Length > 0)
            {
                yield return res.Samples;
            }
            await System.Threading.Tasks.Task.Yield();
        }
    }

    private readonly GgufModel _rawModel;
    private readonly CosyVoice3LlmTensorSource _llmSource;
    private readonly CosyVoice3FlowEncoderWeights _flowWeights;
    private readonly CosyVoice3DiTWeights _ditWeights;
    private readonly CosyVoice3HiftWeights _hiftWeights;

    private CosyVoice3Pipeline(GgufModel rawModel, CosyVoice3LlmTensorSource llmSource, CosyVoice3FlowEncoderWeights flowWeights, CosyVoice3DiTWeights ditWeights, CosyVoice3HiftWeights hiftWeights)
    {
        _rawModel = rawModel;
        _llmSource = llmSource;
        _flowWeights = flowWeights;
        _ditWeights = ditWeights;
        _hiftWeights = hiftWeights;
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

        return new CosyVoice3Pipeline(rawModel, llmSource, flowWeights, ditWeights, hiftWeights);
    }

    /// <summary>Synthesizes real 24kHz PCM audio for the given text.</summary>
    public float[] Generate(string text, int maxNewSpeechTokens = 200, int odeSteps = 10, int? seed = null)
    {
        var speechTokens = CosyVoice3Llm.GenerateSpeechTokens(_rawModel, _llmSource, text, maxNewSpeechTokens);
        if (speechTokens.Length == 0) return [];

        var (mu, spks) = CosyVoice3FlowEncoder.ComputeMuAndSpks(_flowWeights, speechTokens, new float[CosyVoice3FlowEncoderWeights.SpeakerEmbedDim]);

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

    public void Dispose()
    {
        _llmSource.Dispose();
        _hiftWeights.Dispose();
        _rawModel.Dispose();
    }
}
