using System;
using System.Collections.Generic;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Cuda;
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
    private readonly IDisposable? _backend;
    private readonly IForwardPass _fwd;
    private readonly GgufTokenizer _tokenizer;
    private readonly SnacWeights _snacWeights;

    private const int PromptStartToken = 128259;
    private static readonly int[] PromptEndTokens = [128009, 128260, 128261, 128257];
    private const int StopToken = 49158;
    private const int CustomTokenBaseOffset = 128266;

    private static readonly int[] SlotToCodebook = [0, 1, 2, 2, 1, 2, 2];

    public static OrpheusPipeline Load(string modelPath, string? snacGgufPath = null, int ctxSize = 2048, bool allowGpu = true)
    {
        string dir = Path.GetDirectoryName(modelPath) ?? "models";
        snacGgufPath ??= Path.Combine(dir, "snac-24khz.gguf");
        if (!File.Exists(snacGgufPath))
        {
            snacGgufPath = Path.Combine(dir, "snac_24khz.gguf");
        }
        return new OrpheusPipeline(modelPath, snacGgufPath, ctxSize: ctxSize, allowGpu: allowGpu);
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

    public IAsyncEnumerable<float[]> GenerateStreamAsync(AudioGenerationRequest request, System.Threading.CancellationToken ct = default)
        => TtsStreamingHelper.SplitAndGenerateAsync(request, Generate, ct);

    public OrpheusPipeline(string talkerGgufPath, string snacGgufPath, int ctxSize = 2048, bool allowGpu = true)
    {
        _model = GgufModel.Open(talkerGgufPath);
        var hp = ModelHyperparams.FromGgufMetadata(_model.Metadata, _model);

        IForwardPass? fwd = null;
        IDisposable? backend = null;

        if (allowGpu)
        {
            // 1. Try CUDA first (highest throughput on NVIDIA hardware)
            try
            {
                if (CudaBackend.IsAvailable())
                {
                    var cuda = CudaBackend.Create();
                    backend = cuda;
                    fwd = new CudaForwardPass(_model, cuda, hp, maxContextLength: ctxSize);
                    Console.WriteLine($"[Orpheus] GPU Acceleration Enabled: CUDA ({cuda.Name}).");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[Orpheus] CUDA initialization failed ({ex.Message}), trying Vulkan.");
                backend?.Dispose();
                backend = null;
                fwd = null;
            }

            // 2. Fall back to Vulkan
            if (fwd == null)
            {
                try
                {
                    var vulkan = new OpenTail.Stingray.Vulkan.VulkanBackend();
                    backend = vulkan;
                    fwd = new OpenTail.Stingray.Engine.GpuForwardPass(_model, vulkan, hp, maxContextLength: ctxSize);
                    Console.WriteLine($"[Orpheus] GPU Acceleration Enabled: Vulkan ({vulkan.Name}).");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[Orpheus] Vulkan GPU unavailable ({ex.Message}), using CPU backend.");
                    backend?.Dispose();
                    backend = null;
                    fwd = null;
                }
            }
        }

        if (fwd == null)
        {
            var cpu = new CpuBackend();
            backend = cpu;
            fwd = new ForwardPass(_model, cpu, hp, maxContextLength: ctxSize);
        }

        _backend = backend;
        _fwd = fwd;
        _tokenizer = GgufTokenizer.FromGgufModel(_model);
        _snacWeights = new SnacWeights(snacGgufPath);
    }

    /// <summary>Builds the real prompt token sequence for the given voice + text.</summary>
    public List<int> BuildPrompt(string text, string voice)
    {
        var encoded = _tokenizer.Encode($"{voice}: {text}");
        var prompt = new List<int>(encoded.Count + 6) { 128259, 128000 };
        prompt.AddRange(encoded);
        prompt.AddRange([128009, 128260, 128261, 128257]);
        return prompt;
    }

    /// <summary>
    /// Runs the talker autoregressively (reference defaults: temp=0.6, top_p=0.8, repetition_penalty=1.1, stop=128258),
    /// then de-interleaves the generated token ids into 3 SNAC codebook streams.
    /// </summary>
    public int[][] GenerateCodes(string text, string voice = "tara", int maxTokens = 1200, float temperature = 0.6f, float topP = 0.8f, float repetitionPenalty = 1.1f)
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

        int minTokens = 42; // at least 6 superframes (~0.25s) before allowing EOS
        int nextToken = SampleToken(logits, generated, temperature, topP, repetitionPenalty, allowStop: false, rng);
        for (int step = 0; step < maxTokens; step++)
        {
            // 128258 is official End-Of-Speech (<EOS>)
            if (step >= minTokens && (nextToken == 128258 || nextToken == 128262 || nextToken == 128009 || nextToken == 128001 || nextToken == 128256))
            {
                break;
            }

            int s = audioTokenIndex % 7;
            int c = nextToken - CustomTokenBaseOffset - s * 4096;
            if (c >= 0 && c < SnacWeights.CodebookSize)
            {
                int cb = SlotToCodebook[s];
                (cb == 0 ? codes0 : cb == 1 ? codes1 : codes2).Add(c);
                generated.Add(nextToken);
                audioTokenIndex++;
            }
            else
            {
                generated.Add(nextToken);
            }

            var stepLogits = _fwd.Forward(nextToken, pos);
            pos++;
            nextToken = SampleToken(stepLogits, generated, temperature, topP, repetitionPenalty, allowStop: step >= minTokens, rng);
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

    private static int SampleToken(ReadOnlySpan<float> logits, List<int> pastTokens, float temperature, float topP, float repetitionPenalty, bool allowStop, Random rng)
    {
        if (temperature < 0.01f)
        {
            float max = float.NegativeInfinity;
            int maxIdx = 0;
            for (int i = 0; i < logits.Length; i++)
            {
                if (!allowStop && (i == 128258 || i == 128256 || i == 49158)) continue;
                float v = logits[i];
                if (repetitionPenalty > 1.0f && pastTokens.Count > 0)
                {
                    int start = Math.Max(0, pastTokens.Count - 64);
                    for (int h = start; h < pastTokens.Count; h++)
                    {
                        if (pastTokens[h] == i)
                        {
                            v = v > 0 ? v / repetitionPenalty : v * repetitionPenalty;
                            break;
                        }
                    }
                }
                if (v > max) { max = v; maxIdx = i; }
            }
            return maxIdx;
        }

        // Bounded candidate fast path: select top (K + penaltyHistory) logits in a single O(N) pass,
        // avoiding full-vocabulary 128k array allocations and 128k Array.Sort on every generated token.
        const int K = 64;
        int penaltyCount = (repetitionPenalty > 1.0f && pastTokens.Count > 0) ? Math.Min(64, pastTokens.Count) : 0;
        int sel = Math.Min(logits.Length, K + penaltyCount);

        Span<int> candIdx = stackalloc int[sel];
        Span<float> candVal = stackalloc float[sel];
        candVal.Fill(float.NegativeInfinity);
        candIdx.Fill(-1);

        for (int i = 0; i < logits.Length; i++)
        {
            if (!allowStop && (i == 128258 || i == 128256 || i == 49158))
                continue;

            float v = logits[i];
            if (v <= candVal[sel - 1]) continue;

            int j = sel - 1;
            while (j > 0 && candVal[j - 1] < v)
            {
                candVal[j] = candVal[j - 1];
                candIdx[j] = candIdx[j - 1];
                j--;
            }
            candVal[j] = v;
            candIdx[j] = i;
        }

        if (candIdx[0] < 0 || float.IsNegativeInfinity(candVal[0]))
            return Argmax(logits);

        // Apply repetition penalty to selected candidates, then insertion re-sort
        if (penaltyCount > 0)
        {
            int start = Math.Max(0, pastTokens.Count - 64);
            for (int t = 0; t < sel; t++)
            {
                int id = candIdx[t];
                if (id < 0) continue;
                for (int h = start; h < pastTokens.Count; h++)
                {
                    if (pastTokens[h] == id)
                    {
                        candVal[t] = candVal[t] > 0 ? candVal[t] / repetitionPenalty : candVal[t] * repetitionPenalty;
                        break;
                    }
                }
            }

            for (int a = 1; a < sel; a++)
            {
                float v = candVal[a]; int id = candIdx[a]; int b = a - 1;
                while (b >= 0 && candVal[b] < v)
                {
                    candVal[b + 1] = candVal[b];
                    candIdx[b + 1] = candIdx[b];
                    b--;
                }
                candVal[b + 1] = v;
                candIdx[b + 1] = id;
            }
        }

        int count = Math.Min(K, sel);
        float invTemp = 1.0f / temperature;
        float maxLogit = candVal[0] * invTemp;
        Span<float> probs = stackalloc float[count];
        float sumExp = 0f;
        for (int t = 0; t < count; t++)
        {
            if (candIdx[t] < 0 || float.IsNegativeInfinity(candVal[t]))
            {
                probs[t] = 0f;
            }
            else
            {
                float e = MathF.Exp(candVal[t] * invTemp - maxLogit);
                probs[t] = e;
                sumExp += e;
            }
        }

        if (sumExp <= 0f) return candIdx[0] >= 0 ? candIdx[0] : 0;

        float invSum = 1.0f / sumExp;
        for (int t = 0; t < count; t++)
            probs[t] *= invSum;

        // Top-P Nucleus cutoff
        if (topP < 1.0f && topP > 0f)
        {
            float cum = 0f;
            for (int t = 0; t < count; t++)
            {
                cum += probs[t];
                if (cum >= topP)
                {
                    count = t + 1;
                    break;
                }
            }
        }

        float cumProb = 0f;
        for (int t = 0; t < count; t++) cumProb += probs[t];

        float r = (float)rng.NextDouble() * cumProb;
        float running = 0f;
        for (int t = 0; t < count; t++)
        {
            running += probs[t];
            if (running >= r && candIdx[t] >= 0)
                return candIdx[t];
        }

        return candIdx[0] >= 0 ? candIdx[0] : 0;
    }

    public float[] Synthesize(string text, string voice = "tara", int maxTokens = 1200, float temperature = 0.6f, float topP = 0.8f, float repetitionPenalty = 1.1f)
    {
        var swTotal = System.Diagnostics.Stopwatch.StartNew();
        var swGen = System.Diagnostics.Stopwatch.StartNew();
        var codes = GenerateCodes(text, voice, maxTokens, temperature, topP, repetitionPenalty);
        swGen.Stop();
        if (codes[0].Length == 0) return [];

        var swVocoder = System.Diagnostics.Stopwatch.StartNew();
        var pcm = SnacDecoder.Decode(_snacWeights, codes);
        swVocoder.Stop();
        swTotal.Stop();

        double audioSec = pcm.Length / (double)SampleRate;
        int totalCodes = codes[0].Length * 7;
        Console.WriteLine($"[Orpheus Benchmark] Transformer: {swGen.Elapsed.TotalMilliseconds:F1}ms ({totalCodes / swGen.Elapsed.TotalSeconds:F1} tok/s) | SNAC Vocoder: {swVocoder.Elapsed.TotalMilliseconds:F1}ms ({audioSec / swVocoder.Elapsed.TotalSeconds:F1}x RTF) | Total: {swTotal.Elapsed.TotalMilliseconds:F1}ms");
        return pcm;
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
        _backend?.Dispose();
        _model.Dispose();
        _snacWeights.Dispose();
    }
}
