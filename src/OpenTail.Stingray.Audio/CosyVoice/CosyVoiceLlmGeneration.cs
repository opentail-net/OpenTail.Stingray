
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
        string promptText = "", int[]? promptSpeechTokens = null, int maxNewTokens = 200, Random? rng = null, float repetitionPenalty = 1.0f)
    {
        rng ??= new Random(0);
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
        int synthTextTokens = tokenizer.Encode(text).Count;
        textTokens.AddRange(tokenizer.Encode(text));

        promptSpeechTokens ??= [];

        var hp = ModelHyperparams.FromGgufMetadata(source.Metadata, source);
        using var backend = new Cpu.CpuBackend();
        using var fwd = new ForwardPass(source, backend, hp);

        var prefillIds = new List<int>(textTokens.Count + promptSpeechTokens.Length + 2) { sosId };
        prefillIds.AddRange(textTokens);
        prefillIds.Add(taskId);
        foreach (int t in promptSpeechTokens) prefillIds.Add(source.SpeechTokenIdOffset + t);

        var logits = ApplyBias(fwd.Prefill(prefillIds).ToArray(), source.LlmDecoderBias);

        // Real CosyVoice2 inference (`llm.inference` + `sampling_ids`): Repetition-Aware Sampling
        // (top_k 25, top_p 0.8, win 10, tau 0.1), stop tokens masked before min_len = 2 x synth-text
        // tokens, capped at max_len = 20 x. This used plain argmax until 2026-09-25, which looped
        // without ever emitting EOS (always ran to the cap) and garbled the speech. Shares
        // CosyVoice3's RAS sampler.
        int minLen = Math.Max(1, synthTextTokens * 2);
        int maxLen = synthTextTokens > 0 ? Math.Min(maxNewTokens, synthTextTokens * 20) : maxNewTokens;
        var generated = new List<int>();
        int pos = prefillIds.Count;
        for (int step = 0; step < maxLen; step++)
        {
            // Upstream CosyVoice2 ras_sampling has no logit repetition penalty (its only repetition handling is the
            // full-distribution resample); the 1.15 default of the shared sampler comes from the CosyVoice3 C++ port.
            int localId = CosyVoice3Llm.SampleSpeechToken(logits, generated, allowStop: step >= minLen, rng, repetitionPenalty: repetitionPenalty);
            if (stopTokenIds.Contains(localId)) break;

            generated.Add(localId);
            logits = ApplyBias(fwd.Forward(source.SpeechTokenIdOffset + localId, pos).ToArray(), source.LlmDecoderBias);
            pos++;
        }

        return [.. generated];
    }

    /// <summary>
    /// Teacher-forced scoring: with the same prefill as <see cref="GenerateSpeechTokens"/> (<c>[sos, text, task_id]</c>),
    /// feeds <paramref name="speechTokens"/> one by one and returns, for each, its rank among the speech logits
    /// (0 = argmax) and its log-probability. A correct LLM ranks real speech tokens of the text's own recording highly
    /// all the way through; decay with position points at position handling, poor ranks from the start at the
    /// embedding/head indexing.
    /// </summary>
    public static (int Rank, float LogProb)[] ScoreSpeechTokens(CosyVoiceLlmTensorSource source, string tokenizerDir, string text, int[] speechTokens)
    {
        source.EnableSpeechGenerationMode();
        var tokenizer = BuildTokenizer(tokenizerDir);
        var prefillIds = new List<int> { source.SosTaskTokenIdBase };
        prefillIds.AddRange(tokenizer.Encode(text));
        prefillIds.Add(source.SosTaskTokenIdBase + 1);

        var hp = ModelHyperparams.FromGgufMetadata(source.Metadata, source);
        using var backend = new Cpu.CpuBackend();
        using var fwd = new ForwardPass(source, backend, hp);
        var logits = ApplyBias(fwd.Prefill(prefillIds).ToArray(), source.LlmDecoderBias);
        int pos = prefillIds.Count;
        var result = new (int, float)[speechTokens.Length];
        for (int i = 0; i < speechTokens.Length; i++)
        {
            int n = source.LlmDecoderBias?.Length ?? logits.Length;
            var l = logits.AsSpan(0, n);
            float max = System.Numerics.Tensors.TensorPrimitives.Max(l);
            double sum = 0;
            foreach (float v in l) sum += Math.Exp(v - max);
            float target = l[speechTokens[i]];
            int rank = 0;
            foreach (float v in l) if (v > target) rank++;
            result[i] = (rank, (float)(target - max - Math.Log(sum)));
            logits = ApplyBias(fwd.Forward(source.SpeechTokenIdOffset + speechTokens[i], pos).ToArray(), source.LlmDecoderBias);
            pos++;
        }
        return result;
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
}
