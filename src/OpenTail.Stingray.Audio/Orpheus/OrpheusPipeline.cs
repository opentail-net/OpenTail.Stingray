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
public sealed class OrpheusPipeline : IDisposable
{
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
    /// Runs the talker autoregressively (greedy decode -- a deterministic first correctness
    /// pass; the real reference's own defaults are temp=0.6/top_p=0.8/repetition_penalty=1.3,
    /// not yet wired here) until <see cref="StopToken"/> or <paramref name="maxTokens"/>, then
    /// de-interleaves the raw generated token ids into 3 real SNAC codebook streams.
    /// </summary>
    public int[][] GenerateCodes(string text, string voice = "tara", int maxTokens = 1200)
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

        int nextToken = Argmax(logits);
        for (int step = 0; step < maxTokens; step++)
        {
            if (nextToken == StopToken) break;

            int code = nextToken - CustomTokenBaseOffset - (audioTokenIndex % 7) * 4096;
            if (code >= 0 && code < SnacWeights.CodebookSize)
            {
                int cb = SlotToCodebook[audioTokenIndex % 7];
                (cb == 0 ? codes0 : cb == 1 ? codes1 : codes2).Add(code);
                generated.Add(nextToken);
                audioTokenIndex++;
            }
            // Out-of-range tokens (e.g. the model emitting ordinary text/special tokens instead
            // of a valid codec code) are skipped rather than fed to the codec, matching
            // decoder.py's own real `if token > 0` / codebook-range guard.

            var stepLogits = _fwd.Forward(nextToken, pos);
            pos++;
            nextToken = Argmax(stepLogits);
        }

        // Truncate to a whole number of complete superframes (7 audio tokens each) -- a partial
        // trailing superframe has no complete codes_0/1/2 triple and can't be decoded.
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
        return SnacDecoder.Decode(_snacWeights, codes);
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
