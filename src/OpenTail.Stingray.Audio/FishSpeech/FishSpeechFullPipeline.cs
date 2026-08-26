using System;
using System.Collections.Generic;

namespace OpenTail.Stingray.Audio.FishSpeech;

/// <summary>
/// Full Fish Speech S2 Pro text-to-speech pipeline: text -&gt; <see cref="FishSpeechPipeline"/>'s
/// slow-AR semantic-token generation (which internally also runs the real fast-AR codebook
/// expansion per frame, see <see cref="FishSpeechPipeline.GenerateFrames"/>) -&gt; real
/// <see cref="FishSpeechCodec"/> decode -&gt; mono float32 PCM.
///
/// <para>Analogous to <c>OrpheusPipeline.Synthesize</c>: wires together already golden-verified
/// components (slow-AR, fast-AR, codec -- each independently proven numerically correct against
/// real oracles, see docs/audio-review-progress.md's Fish Speech section) into one callable
/// end-to-end path. No new model math here -- purely plumbing.</para>
/// </summary>
public sealed class FishSpeechFullPipeline : ITextToSpeechPipeline
{
    public string Architecture => "FishSpeech";
    public int SampleRate => 44100;
    public int DefaultSampleRate => 44100;

    private readonly FishSpeechPipeline _talker;
    private readonly FishSpeechCodecWeights _codecWeights;

    public static FishSpeechFullPipeline Load(string modelPath, string? tokDir = null, string? codecGgufPath = null)
    {
        tokDir ??= ResolveTokenizerDir(modelPath);
        codecGgufPath ??= modelPath; // s2-pro checkpoint contains the embedded codec weights
        return new FishSpeechFullPipeline(modelPath, tokDir, codecGgufPath);
    }

    private static string ResolveTokenizerDir(string modelPath)
    {
        string[] candidates = ["examples/s2.cpp", "models/s2.cpp", "models"];
        foreach (var c in candidates)
        {
            if (Directory.Exists(c) && (File.Exists(Path.Combine(c, "tokenizer.json")) || File.Exists(Path.Combine(c, "vocab.json"))))
                return c;
        }
        return "examples/s2.cpp";
    }

    public FishSpeechFullPipeline(string talkerGgufPath, string tokenizerDir, string codecGgufPath, int numLayers = 36, int ctxSize = 2048)
    {
        _talker = new FishSpeechPipeline(talkerGgufPath, tokenizerDir, numLayers, ctxSize);
        _codecWeights = new FishSpeechCodecWeights(codecGgufPath);
    }

    public AudioGenerationResult Generate(AudioGenerationRequest request)
    {
        var pcm = Synthesize(request.Text);
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

    /// <summary>Full pipeline: text -&gt; mono float32 PCM (44.1kHz, matching the real codec's native rate).</summary>
    public float[] Synthesize(string text, int maxTokens = 200)
    {
        var (semanticTokens, codebooksPerFrame) = _talker.GenerateFrames(text, maxTokens);
        if (semanticTokens.Count == 0) return [];

        int t = semanticTokens.Count;
        var semanticCodes = semanticTokens.ToArray();

        // codebooksPerFrame[frame] = [semantic, residual_0, .., residual_8] (NumCodebooks=10 total,
        // index 0 duplicates the already-known semantic code -- see FishSpeechPipeline.GenerateFrames).
        int numResidual = codebooksPerFrame[0].Length - 1;
        var residualCodes = new int[numResidual][];
        for (int cb = 0; cb < numResidual; cb++)
        {
            residualCodes[cb] = new int[t];
            for (int ti = 0; ti < t; ti++)
                residualCodes[cb][ti] = codebooksPerFrame[ti][cb + 1];
        }

        var pcm = FishSpeechCodec.Decode(_codecWeights, semanticCodes, residualCodes);

        // Peak normalize to 0.85 full scale
        float peak = 0f;
        for (int i = 0; i < pcm.Length; i++)
        {
            float a = MathF.Abs(pcm[i]);
            if (a > peak) peak = a;
        }
        if (peak > 1e-4f && peak < 0.8f)
        {
            float gain = 0.85f / peak;
            for (int i = 0; i < pcm.Length; i++) pcm[i] *= gain;
        }

        return pcm;
    }

    public void Dispose()
    {
        _talker.Dispose();
        _codecWeights.Dispose();
    }
}
