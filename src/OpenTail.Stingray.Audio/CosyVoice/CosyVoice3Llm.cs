using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Audio.CosyVoice;

/// <summary>
/// Real CosyVoice3 LLM speech-token generation loop, driving <see cref="CosyVoice3LlmTensorSource"/>
/// through <see cref="ForwardPass"/> autoregressively. Real prompt/token-composition sequence
/// transcribed directly from `examples/cosyvoice.cpp`'s `cosyvoice-llm-job.cpp`
/// (`cosyvoice_model_3::llm_job_ext`) and `cosyvoice-prompt.cpp`
/// (`cosyvoice_prompt_init_from_prompt_speech`/`cosyvoice_model::set_prompt`), simplified to the
/// no-reference-audio ("cross-lingual"/plain synthesis) case -- no zero-shot voice-cloning prompt
/// speech tokens injected:
///
/// <code>
/// [sos_token_id]                              (speech-embedded, real GGUF metadata "sos_token_id")
/// + tokenize(instruction_prefix)              (text-embedded, real GGUF metadata "cosyvoice.instruction_prefix")
/// + tokenize("&lt;|endofprompt|&gt;")               (text-embedded, one real special token)
/// + tokenize(synthesis text)                  (text-embedded, real BPE tokenizer from this GGUF's own
///                                               non-llama.cpp-standard `tokenizer.vocab.*`/`tokenizer.model.merges` keys)
/// + [task_token_id]                           (speech-embedded, real GGUF metadata "task_token_id" --
///                                               becomes the seed token for the first autoregressive decode step)
/// </code>
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
    /// Generates real speech token ids (0-based within the speech vocabulary, i.e. already
    /// stripped of <see cref="CosyVoice3LlmTensorSource.SpeechTokenIdOffset"/> -- ready to feed
    /// straight into the flow/DiT stage) for the given synthesis text. Greedy decoding (argmax),
    /// stopping at any real stop-token id (GGUF metadata "stop_token_ids") or maxNewTokens.
    /// </summary>
    public static int[] GenerateSpeechTokens(GgufModel rawModel, CosyVoice3LlmTensorSource source, string text, int maxNewTokens = 200)
    {
        var tokenizer = BuildTokenizer(rawModel);

        string instructionPrefix = rawModel.Metadata.TryGetValue("cosyvoice.instruction_prefix", out var ip) ? (string)ip : "You are a helpful assistant.";
        int sosTokenId = rawModel.GetMetadata("sos_token_id", 0);
        int taskTokenId = rawModel.GetMetadata("task_token_id", 0);
        var stopTokenIds = ReadIntArray(rawModel, "stop_token_ids");

        var promptTextTokens = new List<int>(tokenizer.Encode(instructionPrefix));
        promptTextTokens.AddRange(tokenizer.Encode("<|endofprompt|>"));
        promptTextTokens.AddRange(tokenizer.Encode(text));

        var hp = ModelHyperparams.FromGgufMetadata(source.Metadata);
        using var backend = new CpuBackend();
        using var fwd = new ForwardPass(source, backend, hp);

        var prefillIds = new List<int>(promptTextTokens.Count + 2) { source.SpeechTokenIdOffset + sosTokenId };
        prefillIds.AddRange(promptTextTokens);
        prefillIds.Add(source.SpeechTokenIdOffset + taskTokenId);

        var logits = fwd.Prefill(prefillIds).ToArray();

        var generated = new List<int>();
        int pos = prefillIds.Count;
        for (int step = 0; step < maxNewTokens; step++)
        {
            int localId = ArgMax(logits);
            if (stopTokenIds.Contains(localId)) break;

            generated.Add(localId);
            logits = fwd.Forward(source.SpeechTokenIdOffset + localId, pos).ToArray();
            pos++;
        }

        return [.. generated];
    }

    private static GgufTokenizer BuildTokenizer(GgufModel model)
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
