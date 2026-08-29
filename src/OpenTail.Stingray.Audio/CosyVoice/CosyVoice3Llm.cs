using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Audio.CosyVoice;

/// <summary>
/// Real CosyVoice3 LLM speech-token generation loop, driving <see cref="CosyVoice3LlmTensorSource"/>
/// through <see cref="ForwardPass"/> autoregressively. Real prompt/token-composition sequence
/// transcribed directly from `examples/cosyvoice.cpp`'s `cosyvoice-llm-job.cpp`
/// (`cosyvoice_model_3::llm_job_ext`) and `cosyvoice-prompt.cpp`
/// (`cosyvoice_prompt_init_from_prompt_speech`/`cosyvoice_model::set_prompt`):
///
/// <code>
/// [sos_token_id]                              (speech-embedded, real GGUF metadata "sos_token_id")
/// + tokenize(instruction_prefix)              (text-embedded, real GGUF metadata "cosyvoice.instruction_prefix")
/// + tokenize("&lt;|endofprompt|&gt;")               (text-embedded, one real special token)
/// + tokenize(promptText)                      (text-embedded, the REFERENCE AUDIO's own transcript --
///                                               `prompt->prompt_text` in `cosyvoice_prompt_init_from_prompt_speech`,
///                                               empty when there is no reference audio)
/// + tokenize(synthesis text)                  (text-embedded, real BPE tokenizer from this GGUF's own
///                                               non-llama.cpp-standard `tokenizer.vocab.*`/`tokenizer.model.merges` keys)
/// + [task_token_id]                           (speech-embedded, real GGUF metadata "task_token_id")
/// + promptSpeechTokens                        (speech-embedded, the reference audio's OWN speech tokens --
///                                               `prompt->llm_prompt_speech_tokens` -- empty when there is no
///                                               reference audio. Real reference feeds all but the last one
///                                               via `prefill_embedding` and treats the last as `cur` (the seed
///                                               for the next decode step) purely as an artifact of its batched
///                                               prefill API; functionally equivalent to just appending ALL of
///                                               them here and reading the resulting logits, since either way
///                                               the KV cache ends up holding the same sequence and the final
///                                               position's logits predict the same next token.)
/// </code>
///
/// Without this (the previous, simplified "cross-lingual"-only version of this method), the flow
/// encoder was being asked to join two token streams that were never generated to be compatible:
/// the reference audio's real prompt speech tokens (spliced in purely at the FLOW stage by
/// <see cref="CosyVoice3Pipeline"/>) followed by speech tokens the LLM generated with zero
/// awareness that ANY prompt/reference existed. Conditioning the LLM itself on
/// `promptText`/`promptSpeechTokens` here makes its own continuation actually match what
/// `CosyVoice3Pipeline` splices in front of it.
///
/// Every id above is fed through <see cref="CosyVoice3LlmTensorSource.EnableSpeechGenerationMode"/>'s
/// combined [text-vocab rows ; speech-vocab rows] embedding table via the ordinary integer
/// <see cref="ForwardPass"/> token-id API -- text ids as-is, speech ids offset by
/// <see cref="CosyVoice3LlmTensorSource.SpeechTokenIdOffset"/> -- so no raw-embedding injection is
/// needed in C# (unlike the C++ reference, which must inject raw embeddings because its two
/// tables are genuinely separate weight tensors).
/// </summary>
public static class CosyVoice3Llm
{
    /// <summary>
    /// Generates real speech token ids for the given synthesis text using the reference sampling
    /// pipeline from examples/cosyvoice.cpp (top_k=25, top_p=0.8, win_size=10, min_len=text_len*2).
    /// <paramref name="promptText"/>/<paramref name="promptSpeechTokens"/> condition the LLM on a
    /// real zero-shot voice-cloning reference (empty/null for plain, unconditioned synthesis).
    /// </summary>
    public static int[] GenerateSpeechTokens(GgufModel rawModel, CosyVoice3LlmTensorSource source, string text, int maxNewTokens = 300, string? promptText = null, int[]? promptSpeechTokens = null)
    {
        var tokenizer = BuildTokenizer(rawModel);
        int sosTokenId = rawModel.GetMetadata("sos_token_id", 0);
        int taskTokenId = rawModel.GetMetadata("task_token_id", 0);

        string instructionPrefix = rawModel.GetMetadata("cosyvoice.instruction_prefix", "You are a helpful assistant.");
        var prefixTokens = tokenizer.Encode(instructionPrefix);
        var endOfPromptTokens = tokenizer.Encode("<|endofprompt|>");
        var promptTextTokens = string.IsNullOrEmpty(promptText) ? [] : tokenizer.Encode(promptText);
        var textTokens = tokenizer.Encode(text);
        promptSpeechTokens ??= [];

        var hp = ModelHyperparams.FromGgufMetadata(source.Metadata, source);
        using var backend = new CpuBackend();
        using var fwd = new ForwardPass(source, backend, hp);

        var prefillIds = new List<int>(prefixTokens.Count + endOfPromptTokens.Count + promptTextTokens.Count + textTokens.Count + promptSpeechTokens.Length + 2)
        {
            source.SpeechTokenIdOffset + sosTokenId
        };
        prefillIds.AddRange(prefixTokens);
        prefillIds.AddRange(endOfPromptTokens);
        prefillIds.AddRange(promptTextTokens);
        prefillIds.AddRange(textTokens);
        prefillIds.Add(source.SpeechTokenIdOffset + taskTokenId);
        foreach (int t in promptSpeechTokens) prefillIds.Add(source.SpeechTokenIdOffset + t);

        var logits = fwd.Prefill(prefillIds).ToArray();

        var generated = new List<int>();
        int pos = prefillIds.Count;
        // Real reference (`llm_job_ext`'s `min_len`/`max_len`, `examples/cosyvoice.cpp`'s own
        // printed run config): min_len = text_len * min_token_text_ratio(2.0), max_len = text_len
        // * max_token_text_ratio(20.0) -- `text_len` is the NEW synthesis text's OWN token count
        // (`textTokens`), NOT including promptText. The previous ad-hoc `max(60, textTokens*6)`
        // formula didn't match this at all and made the model stop far too early (measured: 60
        // tokens generated here vs. the reference's real 160 for the identical text+prompt).
        int minLen = Math.Max(1, (int)(textTokens.Count * 2.0));
        int maxLenFromRatio = (int)(textTokens.Count * 20.0);
        int effectiveMaxNewTokens = maxLenFromRatio > 0 ? Math.Min(maxNewTokens, maxLenFromRatio) : maxNewTokens;
        var rng = new Random(42);

        for (int step = 0; step < effectiveMaxNewTokens; step++)
        {
            int localId = SampleSpeechToken(logits, generated, allowStop: step >= minLen, rng);

            if (localId >= 6561) // Stop token range
            {
                break;
            }

            generated.Add(localId);
            logits = fwd.Forward(source.SpeechTokenIdOffset + localId, pos).ToArray();
            pos++;
        }

        return [.. generated];
    }

    /// <summary>TEST-SUPPORT ONLY: exposes the raw first-step logits (pre-softmax/top-k/top-p,
    /// straight from ForwardPass.Prefill) for the same real prompt composition
    /// GenerateSpeechTokens builds, to cross-check against the real C++ reference's own dumped
    /// `COSY_DUMP_LLM_LOGITS_PATH` tensor for the identical input sequence.</summary>
    internal static float[] GetFirstStepLogitsForTest(GgufModel rawModel, CosyVoice3LlmTensorSource source, string text, string? promptText, int[]? promptSpeechTokens)
    {
        var tokenizer = BuildTokenizer(rawModel);
        int sosTokenId = rawModel.GetMetadata("sos_token_id", 0);
        int taskTokenId = rawModel.GetMetadata("task_token_id", 0);
        string instructionPrefix = rawModel.GetMetadata("cosyvoice.instruction_prefix", "You are a helpful assistant.");
        var prefixTokens = tokenizer.Encode(instructionPrefix);
        var endOfPromptTokens = tokenizer.Encode("<|endofprompt|>");
        var promptTextTokens = string.IsNullOrEmpty(promptText) ? [] : tokenizer.Encode(promptText);
        var textTokens = tokenizer.Encode(text);
        promptSpeechTokens ??= [];

        var hp = ModelHyperparams.FromGgufMetadata(source.Metadata, source);
        using var backend = new CpuBackend();
        using var fwd = new ForwardPass(source, backend, hp);

        var prefillIds = new List<int> { source.SpeechTokenIdOffset + sosTokenId };
        prefillIds.AddRange(prefixTokens);
        prefillIds.AddRange(endOfPromptTokens);
        prefillIds.AddRange(promptTextTokens);
        prefillIds.AddRange(textTokens);
        prefillIds.Add(source.SpeechTokenIdOffset + taskTokenId);
        foreach (int t in promptSpeechTokens) prefillIds.Add(source.SpeechTokenIdOffset + t);

        return fwd.Prefill(prefillIds).ToArray();
    }

    private static int SampleSpeechToken(float[] logits, List<int> pastTokens, bool allowStop, Random rng, int topK = 25, float topP = 0.8f, int winSize = 10)
    {
        int totalVocab = logits.Length;

        // If stop tokens are not allowed, mask them out
        if (!allowStop)
        {
            for (int i = 6561; i < totalVocab; i++)
                logits[i] = float.NegativeInfinity;
        }

        // Repetition penalty over recent window (win_size)
        if (pastTokens.Count > 0)
        {
            int start = Math.Max(0, pastTokens.Count - winSize);
            for (int i = start; i < pastTokens.Count; i++)
            {
                int tok = pastTokens[i];
                if (tok >= 0 && tok < totalVocab)
                {
                    if (logits[tok] > 0) logits[tok] /= 1.2f;
                    else logits[tok] *= 1.2f;
                }
            }
        }

        // Softmax & Nucleus Top-P / Top-K
        float maxLogit = float.NegativeInfinity;
        for (int i = 0; i < totalVocab; i++)
            if (logits[i] > maxLogit) maxLogit = logits[i];

        double sumExp = 0.0;
        var expLogits = new float[totalVocab];
        for (int i = 0; i < totalVocab; i++)
        {
            expLogits[i] = MathF.Exp(logits[i] - maxLogit);
            sumExp += expLogits[i];
        }

        var candidates = new (int Id, float Prob)[totalVocab];
        for (int i = 0; i < totalVocab; i++)
            candidates[i] = (i, (float)(expLogits[i] / sumExp));

        Array.Sort(candidates, (a, b) => b.Prob.CompareTo(a.Prob));

        int k = Math.Min(topK, candidates.Length);
        float cumProb = 0f;
        int cutoff = 0;
        for (int i = 0; i < k; i++)
        {
            cumProb += candidates[i].Prob;
            cutoff = i;
            if (cumProb >= topP) break;
        }

        float r = (float)rng.NextDouble() * cumProb;
        float running = 0f;
        for (int i = 0; i <= cutoff; i++)
        {
            running += candidates[i].Prob;
            if (running >= r) return candidates[i].Id;
        }

        return candidates[0].Id;
    }

    internal static GgufTokenizer BuildTokenizer(GgufModel model)
    {
        var tokensArray = (object[])model.Metadata["tokenizer.vocab.tokens"];
        var mergesArray = model.Metadata.TryGetValue("tokenizer.model.merges", out var mergesObj) ? (object[])mergesObj : [];
        var tokenTypesArray = model.Metadata.TryGetValue("tokenizer.vocab.token_types", out var ttObj) ? (object[])ttObj : null;

        var tokens = Array.ConvertAll(tokensArray, o => (string)o);
        var merges = Array.ConvertAll(mergesArray, o => (string)o);
        int[]? tokenTypes = tokenTypesArray is null ? null : Array.ConvertAll(tokenTypesArray, Convert.ToInt32);

        var source = new TokenizerSource
        {
            Tokens = tokens,
            Merges = merges,
            TokenTypes = tokenTypes,
            TokenizerPre = "qwen2", // real: tokenizer.pre_tokenizer.regex matches the standard GPT-2/Qwen2 pattern exactly, and the LLM backbone is Qwen2
        };
        return GgufTokenizer.FromSource(source);
    }

    private static HashSet<int> ReadIntArray(GgufModel model, string key)
    {
        var set = new HashSet<int>();
        if (model.Metadata.TryGetValue(key, out var raw) && raw is object[] arr)
            foreach (var v in arr) set.Add(Convert.ToInt32(v));
        return set;
    }

    private static int ArgMax(ReadOnlySpan<float> logits)
    {
        int best = 0;
        float bestVal = float.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++)
        {
            if (logits[i] > bestVal) { bestVal = logits[i]; best = i; }
        }
        return best;
    }
}
