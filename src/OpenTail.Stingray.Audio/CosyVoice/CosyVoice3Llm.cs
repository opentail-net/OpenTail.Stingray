
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
        var allDecoded = new List<int>();
        int pos = prefillIds.Count;
        int minLen = Math.Max(1, (int)(textTokens.Count * 2.0));
        int maxLenFromRatio = (int)(textTokens.Count * 20.0);
        int effectiveMaxNewTokens = maxLenFromRatio > 0 ? Math.Min(maxNewTokens, maxLenFromRatio) : maxNewTokens;
        var rng = new Random(42);
        int consecutiveSilentTokens = 0;

        for (int step = 0; step < effectiveMaxNewTokens; step++)
        {
            int localId = SampleSpeechToken(logitsSpan, allDecoded, allowStop: step >= minLen, rng, temperature: temperature);

            if (localId >= 6561) // Stop token range
            {
                break;
            }

            allDecoded.Add(localId);

            // Silent token filtering matching examples/audio.cpp/src/models/cosyvoice3/ar.cpp:34-35, 508-520:
            // If consecutive silent tokens exceed kMaxConsecutiveSilentTokens (5), filter them out from the
            // speech token output passed to the flow/HiFT models.
            if (SilentTokens.Contains(localId))
            {
                consecutiveSilentTokens++;
                if (consecutiveSilentTokens <= MaxConsecutiveSilentTokens)
                {
                    generated.Add(localId);
                }
            }
            else
            {
                consecutiveSilentTokens = 0;
                generated.Add(localId);
            }

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

    private static readonly HashSet<int> SilentTokens = [1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323];
    private const int MaxConsecutiveSilentTokens = 5;

    /// <summary>
    /// Repetition-Aware Sampling (RAS) matching examples/audio.cpp/src/models/cosyvoice3/ar.cpp:338-377:
    /// Samples a nucleus candidate without blanket logit penalization. If the candidate appears in the
    /// recent window >= (winSize * tau) times, it is masked out and resampled, preventing phoneme stutter
    /// without distorting natural continuous acoustic code repetitions.
    /// </summary>
    private static int SampleSpeechToken(ReadOnlySpan<float> logits, List<int> pastTokens, bool allowStop, Random rng, int topK = 25, float topP = 0.8f, int winSize = 10, float tauR = 0.1f, float temperature = 1.0f)
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

        // 1. Softmax over valid vocabulary matching cosyvoice-llm.cpp:290 with repetition penalty
        float maxLogit = float.NegativeInfinity;
        for (int i = 0; i < maxAllowed; i++)
        {
            float v = logits[i];
            if (winCount > 0 && recentSpan.Contains(i))
            {
                v = v > 0 ? v / 1.15f : v * 1.15f;
            }
            if (v > maxLogit) maxLogit = v;
        }

        var fullProbs = new float[maxAllowed];
        double sumExp = 0.0;
        for (int i = 0; i < maxAllowed; i++)
        {
            float v = logits[i];
            if (winCount > 0 && recentSpan.Contains(i))
            {
                v = v > 0 ? v / 1.15f : v * 1.15f;
            }
            float e = MathF.Exp((v - maxLogit) * invTemp);
            fullProbs[i] = e;
            sumExp += e;
        }

        float invSum = (float)(1.0 / sumExp);
        for (int i = 0; i < maxAllowed; i++)
        {
            fullProbs[i] *= invSum;
        }

        // 2. Select top-K candidates matching cosyvoice-llm.cpp:294-300
        int k = Math.Min(topK, maxAllowed);
        Span<int> topIdx = stackalloc int[k];
        Span<float> topVal = stackalloc float[k];
        int filled = 0;

        for (int i = 0; i < maxAllowed; i++)
        {
            float v = fullProbs[i];
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

        // 3. Sort top-K descending and truncate at top_p matching cosyvoice_model_3::llm_prepare_probs
        Span<float> nucleusProbs = stackalloc float[filled];
        Span<int> nucleusIds = stackalloc int[filled];
        for (int i = 0; i < filled; i++)
        {
            int src = filled - 1 - i;
            nucleusIds[i] = topIdx[src];
            nucleusProbs[i] = topVal[src];
        }

        float pSum = 0f;
        int nucleusLen = filled;
        for (int i = 0; i < filled; i++)
        {
            pSum += nucleusProbs[i];
            if (pSum >= topP)
            {
                nucleusLen = i + 1;
                break;
            }
        }

        // Renormalize nucleus probs
        float invPSum = 1f / pSum;
        for (int i = 0; i < nucleusLen; i++)
        {
            nucleusProbs[i] *= invPSum;
        }

        // 4. Sample candidate token matching cosyvoice_llm_sampler (cosyvoice-llm.cpp:427-463)
        float fallbackRandom = (float)rng.NextDouble();
        float nucleusRandom = fallbackRandom;
        int sampledToken = nucleusIds[0];

        for (int i = 0; i < nucleusLen; i++)
        {
            nucleusRandom -= nucleusProbs[i];
            if (nucleusRandom <= 0f)
            {
                sampledToken = nucleusIds[i];
                int repeatCount = 0;
                int window = Math.Min(pastTokens.Count, winSize);
                for (int j = pastTokens.Count - window; j < pastTokens.Count; j++)
                {
                    if (pastTokens[j] == sampledToken) repeatCount++;
                }

                // If repetition is within threshold, accept sampled token immediately
                if (repeatCount < winSize * tauR)
                {
                    return sampledToken;
                }

                // Threshold exceeded: fall back to sampling from full vocabulary distribution
                for (int v = 0; v < maxAllowed; v++)
                {
                    fallbackRandom -= fullProbs[v];
                    if (fallbackRandom <= 0f) return v;
                }
                break;
            }
        }

        return sampledToken;
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
