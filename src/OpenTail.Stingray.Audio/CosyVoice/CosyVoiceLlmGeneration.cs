using System.Text.Json;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Audio.CosyVoice;

/// <summary>
/// Real CosyVoice2 LLM speech-token generation loop, driving <see cref="CosyVoiceLlmTensorSource"/>
/// through <see cref="ForwardPass"/> autoregressively. Named separately from the pre-existing
/// `CosyVoiceLlm.cs`/`CosyVoicePipeline.cs` in this file, which are an unrelated, pre-existing
/// fake/procedural stub (random "simulated acoustic transitions", hash-based speaker embeddings)
/// predating this session's real CosyVoice2 work -- left untouched rather than partially patched,
/// same pattern used for QwenTTS's real classes living alongside its own old stub this session.
///
/// Real prompt-composition sequence transcribed directly from the actual upstream
/// `cosyvoice/llm/llm.py`'s `Qwen2LM.inference` (fetched via `gh api`, not guessed) -- CosyVoice2
/// uses the `Qwen2LM` subclass, NOT `TransformerLM`/`CosyVoice3LM` (confirmed by class name checks
/// inside the real source itself):
///
/// <code>
/// lm_input = concat([
///     sos_emb,                    // llm_embedding.weight[0]  (separate 2-row table, NOT speech_embedding -- Qwen2LM-specific, confirmed directly)
///     text_emb(prompt_text + text),  // real Qwen2 text embedding, token_embd.weight rows [0, textVocabSize)
///     task_id_emb,                 // llm_embedding.weight[1]
///     prompt_speech_token_emb,      // speech_embedding.weight rows, real zero-shot prompt tokens (empty for plain synthesis)
/// ])
/// </code>
///
/// Then decode step by step: `logits = llm_decoder(hidden)`, greedy/sampled token, stop if in
/// `stop_token_ids = [speech_token_size + i for i in range(3)]` (real: `speech_token_size=6561`,
/// so stop ids are 6561/6562/6563 -- eos/unused/fill, all three real stop conditions per source),
/// else `lm_input = speech_embedding.weight[token]` for the next step.
///
/// `sos_emb`/`task_id_emb` are addressed via <see cref="CosyVoiceLlmTensorSource.SosTaskTokenIdBase"/>
/// (2 extra synthetic vocab rows appended by `EnableSpeechGenerationMode`), exactly the same
/// composition trick already used for the text/speech vocab halves -- `ForwardPass` only needs
/// ordinary integer token ids, no raw-embedding injection required.
/// </summary>
public static class CosyVoiceLlmGeneration
{
    /// <summary>
    /// Generates real speech token ids (0-based within the speech vocabulary, already stripped
    /// of <see cref="CosyVoiceLlmTensorSource.SpeechTokenIdOffset"/> -- ready to feed straight
    /// into <see cref="CosyVoiceFlowEncoder"/>) for the given synthesis text. Greedy decoding
    /// (argmax over `llm_decoder` logits, bias added back per <see cref="CosyVoiceLlmTensorSource.LlmDecoderBias"/>
    /// since `ForwardPass` has no final-layer-bias support).
    /// </summary>
    public static int[] GenerateSpeechTokens(
        CosyVoiceLlmTensorSource source, string tokenizerDir, string text,
        string promptText = "", int[]? promptSpeechTokens = null, int maxNewTokens = 200)
    {
        source.EnableSpeechGenerationMode();
        if (source.SosTaskTokenIdBase < 0)
            throw new InvalidOperationException("CosyVoiceLlmTensorSource has no real llm_embedding.weight tensor -- cannot address sos/task_id.");

        var tokenizer = BuildTokenizer(tokenizerDir);
        int sosId = source.SosTaskTokenIdBase;
        int taskId = source.SosTaskTokenIdBase + 1;
        // Real: self.stop_token_ids = [speech_token_size + i for i in range(3)], speech_token_size=6561.
        var stopTokenIds = new HashSet<int> { 6561, 6562, 6563 };

        var textTokens = new List<int>();
        if (!string.IsNullOrEmpty(promptText)) textTokens.AddRange(tokenizer.Encode(promptText));
        textTokens.AddRange(tokenizer.Encode(text));

        promptSpeechTokens ??= [];

        var hp = ModelHyperparams.FromGgufMetadata(source.Metadata);
        using var backend = new Cpu.CpuBackend();
        using var fwd = new ForwardPass(source, backend, hp);

        var prefillIds = new List<int>(textTokens.Count + promptSpeechTokens.Length + 2) { sosId };
        prefillIds.AddRange(textTokens);
        prefillIds.Add(taskId);
        foreach (int t in promptSpeechTokens) prefillIds.Add(source.SpeechTokenIdOffset + t);

        var logits = ApplyBias(fwd.Prefill(prefillIds).ToArray(), source.LlmDecoderBias);

        var generated = new List<int>();
        int pos = prefillIds.Count;
        for (int step = 0; step < maxNewTokens; step++)
        {
            int localId = ArgMax(logits);
            if (stopTokenIds.Contains(localId)) break;

            generated.Add(localId);
            logits = ApplyBias(fwd.Forward(source.SpeechTokenIdOffset + localId, pos).ToArray(), source.LlmDecoderBias);
            pos++;
        }

        return [.. generated];
    }

    private static float[] ApplyBias(float[] logits, float[]? bias)
    {
        if (bias is null) return logits;
        for (int i = 0; i < logits.Length; i++) logits[i] += bias[i];
        return logits;
    }

    /// <summary>
    /// Real HF tokenizer construction from `vocab.json`/`merges.txt`/`tokenizer_config.json`
    /// (real byte-level BPE, GPT-2/Qwen2 family -- downloaded from the actual upstream
    /// `FunAudioLLM/CosyVoice2-0.5B/CosyVoice-BlankEN` checkpoint, a plain Qwen2Tokenizer with
    /// no CosyVoice-specific extra tokens). Same real special-token-completion fix already
    /// needed for QwenASR's Safetensors tokenizer (`vocab.json` only holds the base ~151643-
    /// entry vocab; the endoftext/im_start/im_end special tokens live in
    /// `tokenizer_config.json`'s `added_tokens_decoder` and would otherwise resolve to an empty
    /// string, corrupting BPE matching) -- checked directly against this real downloaded file,
    /// not assumed from the QwenASR precedent.
    /// </summary>
    private static GgufTokenizer BuildTokenizer(string tokenizerDir)
    {
        var (tokens, merges, addedByContent) = Primitives.HfBpeTokenizerLoader.Load(tokenizerDir);

        var source = new TokenizerSource
        {
            Tokens = tokens,
            Merges = merges,
            AdditionalSpecialTokens = addedByContent,
            AddBosToken = false,
            ModelFamily = "gpt2",
        };
        return GgufTokenizer.FromSource(source);
    }

    private static int ArgMax(ReadOnlySpan<float> logits)
    {
        int best = 0;
        float bestVal = float.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++)
            if (logits[i] > bestVal) { bestVal = logits[i]; best = i; }
        return best;
    }
}
