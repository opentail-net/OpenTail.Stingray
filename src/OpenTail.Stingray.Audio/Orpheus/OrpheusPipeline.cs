using System;
using System.Collections.Generic;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Audio.Orpheus;

/// <summary>
/// Real Orpheus TTS pipeline: text + voice -> real Llama-3.2-3B-shape talker (via this
/// codebase's EXISTING, unmodified `ForwardPass`/`GgufTokenizer` text-generation infrastructure
/// -- confirmed to load and run this exact checkpoint unchanged, see docs/audio-review-
/// progress.md's Orpheus section) -> real SNAC codec decode (<see cref="SnacDecoder"/>) -> 24kHz
/// mono PCM.
///
/// <para><b>Real prompt/detokenization spec, read directly from the real
/// `canopyai/Orpheus-TTS` Python source (`orpheus_tts_pypi/orpheus_tts/engine_class.py`,
/// `decoder.py`), independently corroborated by `examples/CrispASR/src/orpheus.cpp` and
/// `examples/CrispASR/tools/reference_backends/orpheus_snac.py` -- see docs/audio-review-
/// progress.md's Orpheus section for the full derivation, do not re-derive</b>:</para>
/// <list type="bullet">
/// <item>Prompt = `[128259] + Encode("{voice}: {text}") + [128009, 128260, 128261, 128257]`
/// (raw BPE encode, NOT a chat template -- this checkpoint is not instruct-tuned).</item>
/// <item>Generation stops at token id 49158 (this checkpoint's real default
/// `stop_token_ids`, confirmed via direct tokenizer-vocab inspection -- an ordinary BPE text
/// token this fine-tune learned to emit as an end-of-audio marker, not a reserved special
/// token) or a max-tokens cap.</item>
/// <item>Detokenization: `code = raw_id - 128266 - (index % 7) * 4096`, where `index` is the
/// 0-based count of generated audio tokens (confirmed algebraically equivalent to the real
/// Python `turn_token_into_id`'s `N - 10 - (index%7)*4096` once `N = raw_id - 128256` is
/// substituted -- the "custom_token_0" vocab entry's real id is 128256, confirmed by direct
/// tokenizer inspection, NOT 128266 as `orpheus.cpp`'s own comment mislabels it).</item>
/// <item>Every 7 generated tokens form one SNAC superframe: slot 0 -&gt; codes_0 (coarse),
/// slots 1&amp;4 -&gt; codes_1 (mid), slots 2/3/5/6 -&gt; codes_2 (fine).</item>
/// </list>
/// </summary>
public sealed class OrpheusPipeline : ITextToSpeechPipeline
{
    public string Architecture => "Orpheus-TTS";
    public int SampleRate => 24000;
    public int DefaultSampleRate => 24000;

    private readonly GgufModel _model;
    private readonly CpuBackend _backend;
    private readonly ForwardPass _fwd;
    private readonly GgufTokenizer _tokenizer;
    private readonly SnacWeights _snacWeights;

    private const int PromptStartToken = 128259;
    private static readonly int[] PromptEndTokens = [128009, 128260, 128261, 128257];
    private const int StopToken = 49158;
    private const int CustomTokenBaseOffset = 128266; // = <custom_token_0> id (128256) + the real formula's constant 10

    private static readonly int[] SlotToCodebook = [0, 1, 2, 2, 1, 2, 2];

    public static OrpheusPipeline Load(string modelPath, string? snacGgufPath = null)
    {
        string dir = Path.GetDirectoryName(modelPath) ?? "models";
        snacGgufPath ??= Path.Combine(dir, "snac-24khz.gguf");
        if (!File.Exists(snacGgufPath))
        {
            snacGgufPath = Path.Combine(dir, "snac_24khz.gguf");
        }
        return new OrpheusPipeline(modelPath, snacGgufPath);
    }

    public AudioGenerationResult Generate(AudioGenerationRequest request)
    {
        string voice = string.IsNullOrWhiteSpace(request.Voice) || request.Voice == "af_heart" ? "tara" : request.Voice;
        var pcm = Synthesize(request.Text, voice);
        var result = new AudioGenerationResult(pcm, DefaultSampleRate);
        if (!string.IsNullOrEmpty(request.OutputPath))
        {
            result.SaveWav(request.OutputPath);
        }
        return result;
    }

    public async IAsyncEnumerable<float[]> GenerateStreamAsync(AudioGenerationRequest request, [System.Runtime.CompilerServices.EnumeratorCancellation] System.Threading.CancellationToken ct = default)
    {
        var res = Generate(request);
        yield return res.Samples;
    }

    public OrpheusPipeline(string talkerGgufPath, string snacGgufPath, int ctxSize = 8192)
    {
        _model = GgufModel.Open(talkerGgufPath);
        var hp = ModelHyperparams.FromGgufMetadata(_model.Metadata, _model);
        _backend = new CpuBackend();
        _fwd = new ForwardPass(_model, _backend, hp, maxContextLength: ctxSize);
        _tokenizer = GgufTokenizer.FromGgufModel(_model);
        _snacWeights = new SnacWeights(snacGgufPath);
    }

    /// <summary>Builds the real prompt token sequence for the given voice + text.</summary>
    public List<int> BuildPrompt(string text, string voice)
    {
        var encoded = _tokenizer.Encode($"{voice}: {text}");
        var prompt = new List<int>(encoded.Count + 5) { PromptStartToken };
        prompt.AddRange(encoded);
        prompt.AddRange(PromptEndTokens);
        return prompt;
    }

    /// <summary>
    /// Runs the talker autoregressively using the official Orpheus reference sampling configuration
    /// (temp=0.6, top_p=0.8, repetition_penalty=1.3, stop_token_ids=[49158]).
    /// </summary>
    public int[][] GenerateCodes(string text, string voice = "tara", int maxTokens = 1200, float temperature = 0.6f, float topP = 0.8f, float repetitionPenalty = 1.3f)
    {
        var prompt = BuildPrompt(text, voice);
        _fwd.ResetCache();
        var logits = _fwd.Prefill(prompt);

        int pos = prompt.Count;
        var generated = new List<int>();
        var codes0 = new List<int>();
        var codes1 = new List<int>();
        var codes2 = new List<int>();
        int audioTokenIndex = 0;
        var rng = new Random(42);

        int minTokens = Math.Max(70, prompt.Count * 5);
        int nextToken = SampleToken(logits, generated, temperature, topP, repetitionPenalty, rng);
        for (int step = 0; step < maxTokens; step++)
        {
            if (step > minTokens && nextToken == StopToken) break;

            int slot = audioTokenIndex % 7;
            int code = nextToken - CustomTokenBaseOffset - slot * 4096;
            if (code >= 0 && code < SnacWeights.CodebookSize)
            {
                int cb = SlotToCodebook[slot];
                (cb == 0 ? codes0 : cb == 1 ? codes1 : codes2).Add(code);
                audioTokenIndex++;
            }

            generated.Add(nextToken);
            var stepLogits = _fwd.Forward(nextToken, pos);
            pos++;
            nextToken = SampleToken(stepLogits, generated, temperature, topP, repetitionPenalty, rng);
        }

        // Truncate to a whole number of complete superframes (7 audio tokens each)
        int completeSuperframes = audioTokenIndex / 7;
        return
        [
            [.. codes0.GetRange(0, Math.Min(codes0.Count, completeSuperframes * 1))],
            [.. codes1.GetRange(0, Math.Min(codes1.Count, completeSuperframes * 2))],
            [.. codes2.GetRange(0, Math.Min(codes2.Count, completeSuperframes * 4))],
        ];
    }

    /// <summary>Full pipeline: text + voice -> 24kHz mono PCM.</summary>
    public float[] Synthesize(string text, string voice = "tara", int maxTokens = 1200)
    {
        var codes = GenerateCodes(text, voice, maxTokens);
        if (codes[0].Length == 0) return [];
        var pcm = SnacDecoder.Decode(_snacWeights, codes);

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

    private static int SampleToken(ReadOnlySpan<float> logits, List<int> pastTokens, float temperature, float topP, float repetitionPenalty, Random rng)
    {
        var scaled = new float[logits.Length];
        logits.CopyTo(scaled);

        // Apply repetition penalty to recent tokens
        if (repetitionPenalty > 1.0f && pastTokens.Count > 0)
        {
            int start = Math.Max(0, pastTokens.Count - 64);
            for (int i = start; i < pastTokens.Count; i++)
            {
                int tok = pastTokens[i];
                if (tok >= 0 && tok < scaled.Length)
                {
                    if (scaled[tok] > 0) scaled[tok] /= repetitionPenalty;
                    else scaled[tok] *= repetitionPenalty;
                }
            }
        }

        // Temperature scaling
        if (temperature > 0.01f)
        {
            for (int i = 0; i < scaled.Length; i++) scaled[i] /= temperature;
        }

        // Softmax & Top-P Nucleus Sampling
        float maxLogit = float.NegativeInfinity;
        for (int i = 0; i < scaled.Length; i++)
            if (scaled[i] > maxLogit) maxLogit = scaled[i];

        double sumExp = 0.0;
        for (int i = 0; i < scaled.Length; i++)
        {
            scaled[i] = MathF.Exp(scaled[i] - maxLogit);
            sumExp += scaled[i];
        }

        var indexed = new (int Id, float Prob)[scaled.Length];
        for (int i = 0; i < scaled.Length; i++)
            indexed[i] = (i, (float)(scaled[i] / sumExp));

        Array.Sort(indexed, (a, b) => b.Prob.CompareTo(a.Prob));

        float cumProb = 0f;
        int cutoff = 0;
        for (int i = 0; i < indexed.Length; i++)
        {
            cumProb += indexed[i].Prob;
            cutoff = i;
            if (cumProb >= topP) break;
        }

        float r = (float)rng.NextDouble() * cumProb;
        float running = 0f;
        for (int i = 0; i <= cutoff; i++)
        {
            running += indexed[i].Prob;
            if (running >= r) return indexed[i].Id;
        }

        return indexed[0].Id;
    }

    private static int Argmax(ReadOnlySpan<float> logits)
    {
        int idx = 0;
        float max = logits[0];
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > max) { max = logits[i]; idx = i; }
        return idx;
    }

    public void Dispose()
    {
        _fwd.Dispose();
        _backend.Dispose();
        _model.Dispose();
        _snacWeights.Dispose();
    }
}
