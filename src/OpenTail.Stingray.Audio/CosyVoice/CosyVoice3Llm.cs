
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
    public static int[] GenerateSpeechTokens(GgufModel rawModel, CosyVoice3LlmTensorSource source, string text, int maxNewTokens = 300, string? promptText = null, int[]? promptSpeechTokens = null, string? instruction = null, float temperature = 1.0f)
    {
        var tokenizer = BuildTokenizer(rawModel);
        int sosTokenId = rawModel.GetMetadata("sos_token_id", 0);
        int taskTokenId = rawModel.GetMetadata("task_token_id", 0);

        string instructionPrefix = instruction ?? rawModel.GetMetadata("cosyvoice.instruction_prefix", "You are a helpful assistant.");
        var prefixTokens = string.IsNullOrEmpty(instructionPrefix) ? [] : tokenizer.Encode(instructionPrefix);
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

        var logitsSpan = fwd.Prefill(prefillIds);

        var generated = new List<int>();
        int pos = prefillIds.Count;
        int minLen = Math.Max(1, (int)(textTokens.Count * 2.0));
        int maxLenFromRatio = (int)(textTokens.Count * 20.0);
        int effectiveMaxNewTokens = maxLenFromRatio > 0 ? Math.Min(maxNewTokens, maxLenFromRatio) : maxNewTokens;
        var rng = new Random(42);

        for (int step = 0; step < effectiveMaxNewTokens; step++)
        {
            int localId = SampleSpeechToken(logitsSpan, generated, allowStop: step >= minLen, rng, temperature: temperature);

            if (localId >= 6561) // Stop token range
            {
                break;
            }

            generated.Add(localId);
            logitsSpan = fwd.Forward(source.SpeechTokenIdOffset + localId, pos);
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

    private static int SampleSpeechToken(ReadOnlySpan<float> logits, List<int> pastTokens, bool allowStop, Random rng, int topK = 25, float topP = 0.8f, int winSize = 10, float temperature = 1.0f)
    {
        int totalVocab = logits.Length;
        int maxAllowed = allowStop ? totalVocab : Math.Min(totalVocab, 6561);
        float invTemp = temperature > 0f ? 1f / temperature : 1f;

        Span<int> recentWin = stackalloc int[winSize];
        int winCount = 0;
        if (pastTokens.Count > 0)
        {
            int start = Math.Max(0, pastTokens.Count - winSize);
            for (int i = start; i < pastTokens.Count; i++)
            {
                recentWin[winCount++] = pastTokens[i];
            }
        }
        var recentSpan = recentWin.Slice(0, winCount);

        int k = Math.Min(topK, maxAllowed);
        Span<int> topIdx = stackalloc int[k];
        Span<float> topVal = stackalloc float[k];
        int filled = 0;

        for (int i = 0; i < maxAllowed; i++)
        {
            float v = logits[i];
            if (winCount > 0 && recentSpan.Contains(i))
            {
                v = v > 0 ? v / 1.2f : v * 1.2f;
            }
            v *= invTemp;

            if (filled < k)
            {
                int p = filled++;
                while (p > 0 && topVal[p - 1] > v) { topVal[p] = topVal[p - 1]; topIdx[p] = topIdx[p - 1]; p--; }
                topVal[p] = v; topIdx[p] = i;
            }
            else if (v > topVal[0])
            {
                int p = 0;
                while (p < k - 1 && topVal[p + 1] < v) { topVal[p] = topVal[p + 1]; topIdx[p] = topIdx[p + 1]; p++; }
                topVal[p] = v; topIdx[p] = i;
            }
        }

        if (filled == 0) return 0;

        float maxLogit = topVal[filled - 1];
        Span<float> probs = stackalloc float[filled];
        Span<int> orderedIds = stackalloc int[filled];
        float sumExp = 0f;

        for (int i = 0; i < filled; i++)
        {
            int srcIdx = filled - 1 - i; // Descending
            orderedIds[i] = topIdx[srcIdx];
            float e = MathF.Exp(topVal[srcIdx] - maxLogit);
            probs[i] = e;
            sumExp += e;
        }

        float invSum = 1f / sumExp;
        for (int i = 0; i < filled; i++) probs[i] *= invSum;

        float cumProb = 0f;
        int cutoff = 0;
        for (int i = 0; i < filled; i++)
        {
            cumProb += probs[i];
            cutoff = i;
            if (cumProb >= topP) break;
        }

        float r = (float)rng.NextDouble() * cumProb;
        float running = 0f;
        for (int i = 0; i <= cutoff; i++)
        {
            running += probs[i];
            if (running >= r) return orderedIds[i];
        }

        return orderedIds[0];
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
