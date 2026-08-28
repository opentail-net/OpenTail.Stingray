using System;
using System.Collections.Generic;
using System.Linq;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Audio.FishSpeech;

/// <summary>
/// Real Fish Speech S2 Pro slow-AR (semantic-token) pipeline: text -&gt; real ChatML-style prompt
/// -&gt; real per-timestep embedding composition -&gt; existing, UNMODIFIED `ForwardPass` trunk (via
/// <see cref="FishSpeechTensorSource"/>'s tensor-name remapping) -&gt; a sequence of real semantic
/// tokens. Does NOT yet include the fast-AR codebook expander or the codec -- this produces
/// semantic tokens only, not audio. See docs/audio-review-progress.md's Fish Speech section for
/// the full architecture derivation and remaining work.
///
/// <para><b>Real prompt format, from `examples/s2.cpp/src/s2_prompt.cpp` (no-reference-audio
/// case, do not re-derive)</b>: `&lt;|im_start|&gt;system\nconvert the provided text to
/// speech&lt;|im_end|&gt;\n&lt;|im_start|&gt;user\n{text}&lt;|im_end|&gt;\n&lt;|im_start|&gt;
/// assistant\n` + a single `&lt;|voice|&gt;` token.</para>
///
/// <para><b>Real per-timestep embedding composition, from `s2_model.cpp` (confirmed via source,
/// not guessed)</b>: `x = Embeddings[token_id]`; for a SEMANTIC-range token additionally sum
/// `CodebookEmbeddings[value + cb*CodebookSize]` for each of the `NumCodebooks` codebook values
/// at that position (all zero/unused for a plain text-prompt position, matching the real
/// `semantic_mask` which zeroes the codebook contribution entirely for non-semantic positions);
/// then, when `ScaleCodebookEmbeddings` is set, the WHOLE composed embedding (not just the
/// codebook part) is scaled by `1/sqrt(CodebookDim)` for semantic positions and left at 1.0 for
/// text positions. During prompt-only (text) generation this reduces to a plain embedding
/// lookup, since none of the prompt tokens are semantic.</para>
///
/// <para><b>Real semantic-token sampling mask, from `s2_generate.cpp`</b>: logits are biased to
/// `-inf` everywhere except `[SemanticBeginId, SemanticEndId]` and the `im_end` token, so
/// generation can only ever emit a semantic token or terminate.</para>
/// </summary>
public sealed class FishSpeechPipeline : IDisposable
{
    private readonly GgufModel _model;
    private readonly FishSpeechTensorSource _tensorSource;
    private readonly CpuBackend _backend;
    private readonly ForwardPass _fwd;
    private readonly GgufTokenizer _tokenizer;
    private readonly FishSpeechWeights _weights;
    private readonly int _imEndId;
    private readonly int _voiceId;
    private readonly FishSpeechFastArCache _fastArCache;

    public FishSpeechPipeline(string ggufPath, string tokenizerDir, int numLayers = 36, int ctxSize = 2048)
    {
        _model = GgufModel.Open(ggufPath);
        _tensorSource = new FishSpeechTensorSource(_model, numLayers);
        var hp = ModelHyperparams.FromGgufMetadata(_tensorSource.Metadata, _tensorSource);
        _backend = new CpuBackend();
        _fwd = new ForwardPass(_tensorSource, _backend, hp, maxContextLength: ctxSize);
        _fwd.EnableHiddenTaps([numLayers - 1]); // last layer's output = the trunk's post-trunk pre-final-norm hidden
        _weights = new FishSpeechWeights(ggufPath);
        _fastArCache = new FishSpeechFastArCache(_weights.FastBlockCount);

        var tokResult = HuggingFaceTokenizerSource.Load(tokenizerDir);
        if (!tokResult.IsUsable || tokResult.Source is null)
            throw new InvalidOperationException("Failed to load Fish Speech tokenizer: " +
                string.Join("; ", tokResult.Rejections.Select(r => r.Detail)));
        _tokenizer = GgufTokenizer.FromSource(tokResult.Source);

        _imEndId = Single(_tokenizer.Encode("<|im_end|>"));
        _voiceId = Single(_tokenizer.Encode("<|voice|>"));
    }

    private static int Single(IReadOnlyList<int> ids) =>
        ids.Count == 1 ? ids[0] : throw new InvalidOperationException($"expected a single token, got {ids.Count}");

    /// <summary>Applies the slow-AR trunk's final RMSNorm (norm.weight) to the tapped hidden state, matching s2_model.cpp's eval_cached.</summary>
    private float[] GetNormedHidden(int pos)
    {
        var rawHidden = _fwd.HiddenTapsAt(pos).ToArray();
        return FishSpeechFastAr.RmsNorm(rawHidden, _weights.NormWeight, _weights.RmsNormEps);
    }

    /// <summary>Builds the real prompt token id sequence (no reference audio -- the simple zero-shot case).</summary>
    public List<int> BuildPrompt(string text)
    {
        var ids = new List<int>();
        ids.AddRange(_tokenizer.Encode("<|im_start|>system"));
        ids.AddRange(_tokenizer.Encode("\n"));
        ids.AddRange(_tokenizer.Encode("convert the provided text to speech"));
        ids.Add(_imEndId);
        ids.AddRange(_tokenizer.Encode("\n"));
        ids.AddRange(_tokenizer.Encode("<|im_start|>user"));
        ids.AddRange(_tokenizer.Encode("\n"));
        ids.AddRange(_tokenizer.Encode(text));
        ids.Add(_imEndId);
        ids.AddRange(_tokenizer.Encode("\n"));
        ids.AddRange(_tokenizer.Encode("<|im_start|>assistant"));
        ids.AddRange(_tokenizer.Encode("\n"));
        ids.Add(_voiceId);
        return ids;
    }

    /// <summary>Composes the real per-timestep embedding for a plain (non-semantic) text token.</summary>
    private float[] EmbedTextToken(int tokenId)
    {
        var emb = new float[_weights.EmbeddingDim];
        Array.Copy(_weights.Embeddings, (long)tokenId * _weights.EmbeddingDim, emb, 0, _weights.EmbeddingDim);
        // token_scale = 1.0 for non-semantic positions -- no-op, matches real source.
        return emb;
    }

    /// <summary>Composes the real per-timestep embedding for a semantic token plus its (so-far-known) codebook values.</summary>
    private float[] EmbedSemanticToken(int semanticTokenId, int[] codebookValues)
    {
        var emb = new float[_weights.EmbeddingDim];
        Array.Copy(_weights.Embeddings, (long)semanticTokenId * _weights.EmbeddingDim, emb, 0, _weights.EmbeddingDim);

        for (int cb = 0; cb < codebookValues.Length; cb++)
        {
            long row = (long)(codebookValues[cb] + cb * _weights.CodebookSize) * _weights.EmbeddingDim;
            for (int d = 0; d < _weights.EmbeddingDim; d++)
                emb[d] += _weights.CodebookEmbeddings[row + d];
        }

        if (_weights.ScaleCodebookEmbeddings)
        {
            // Real scale factor from s2_model.cpp: 1 / sqrt(num_codebooks + 1) = 1 / sqrt(11)
            float scale = 1f / MathF.Sqrt(_weights.NumCodebooks + 1);
            for (int d = 0; d < _weights.EmbeddingDim; d++) emb[d] *= scale;
        }
        return emb;
    }

    /// <summary>
    /// Generates a sequence of real semantic tokens for the given text (greedy decode, with a
    /// real RAS repetition-avoidance escape hatch -- see <see cref="GenerateFrames"/>'s doc
    /// comment). Does NOT run the fast-AR codebook expansion or the codec -- returns raw semantic
    /// token ids (offset by SemanticBeginId already subtracted), not audio.
    /// </summary>
    public List<int> GenerateSemanticTokens(string text, int maxTokens = 200, int? seed = null) =>
        GenerateFrames(text, maxTokens, seed).SemanticTokens;

    /// <summary>TEMP bisection hook (docs/audio-review-progress.md's Fish Speech NaN investigation): runs just the real prompt prefill and returns the raw logits, for a caller to check for NaN. TODO remove once the bug is found.</summary>
    public float[] PrefillForBisection(List<int> prompt)
    {
        _fwd.ResetCache();
        return _fwd.Prefill(prompt).ToArray();
    }

    /// <summary>TEMP bisection hook: taps a specific layer's hidden state after prefill, to trace activation magnitude growth across layers. TODO remove once the bug is found.</summary>
    public float[] PrefillHiddenTapForBisection(List<int> prompt, int tapLayer)
    {
        _fwd.ResetCache();
        _fwd.EnableHiddenTaps([tapLayer]);
        _ = _fwd.Prefill(prompt);
        return _fwd.HiddenTapsAt(prompt.Count - 1).ToArray();
    }

    /// <summary>
    /// Same generation as <see cref="GenerateSemanticTokens"/>, but also returns the real fast-AR
    /// codebook expansion computed per frame along the way (previously computed then discarded) --
    /// what <see cref="FishSpeechCodec.Decode"/> needs for full text-to-audio synthesis.
    ///
    /// <para><b>Confirmed, listen- and token-dump-verified bug fix (2026-08-28)</b>: plain greedy
    /// `ArgmaxMasked` gets stuck in a hard repetition loop -- direct evidence from dumping the
    /// generated semantic tokens for a real short prompt showed real, varied content for the
    /// first ~45 frames, then the SAME token repeated for the remaining 125+ frames straight
    /// through to `maxTokens`, `im_end` never reached (the residual codebook degenerated the same
    /// way, presumably as a downstream consequence of the now-constant hidden state). Decoded
    /// through the codec, a long run of a near-constant frame produces a sustained near-periodic
    /// tone -- exactly the reported "underwater" symptom -- while the real spoken content ends
    /// after only ~2s, matching "cuts out too soon". Fixed narrowly: greedy `ArgmaxMasked` stays
    /// the default main-token choice (an EARLIER attempt at replacing it entirely with the real
    /// reference's baseline temperature/top_p/top_k sampling made output WORSE, listen-confirmed
    /// -- reverted, see docs/audio-review-progress.md), but a real port of the reference's own
    /// "RAS" (repetition-avoidance sampling, `s2_generate.cpp`) escape hatch is added: if the
    /// greedy choice repeats a semantic token already seen in the last 10 steps, it is discarded
    /// and re-sampled ONCE at a higher temperature/top_p (1.0/0.9, the real `ras_high_temp`/
    /// `ras_high_top_p`) via <see cref="SampleToken"/>, then generation returns to plain greedy
    /// for subsequent steps -- only intervening exactly when the confirmed failure mode is about
    /// to occur, not changing behavior otherwise.</para>
    /// </summary>
    public (List<int> SemanticTokens, List<int[]> CodebooksPerFrame) GenerateFrames(string text, int maxTokens = 200, int? seed = null)
    {
        var rng = new Random(seed ?? 42);
        var prompt = BuildPrompt(text);
        _fwd.ResetCache();

        // Every prompt position is plain text (BuildPrompt never emits a semantic/codebook
        // token), so EmbedTextToken's per-position embedding reduces to the SAME plain embedding-
        // table lookup ForwardPass's own batched Prefill already does internally (see
        // EmbedTextToken's doc comment: "token_scale = 1.0 for non-semantic positions -- no-op").
        // Feeding the whole prompt through one batched Prefill call instead of a sequential
        // per-token ForwardEmbedding loop is numerically identical but lets the engine batch the
        // matmuls across positions instead of redoing full single-token decode overhead per
        // prompt token -- measured this session's performance pass as the dominant remaining cost
        // after the fast-AR KV cache fix (see docs/audio-review-progress.md).
        var logits = _fwd.Prefill(prompt);
        int pos = prompt.Count;

        int semBegin = _weights.SemanticBeginId;
        int semEnd = _weights.SemanticEndId;
        var semanticTokens = new List<int>();
        var codebooksPerFrame = new List<int[]>();

        const int rasWindowSize = 10;
        const float rasHighTemp = 1.0f, rasHighTopP = 0.9f;
        const float defaultTemp = 0.8f, defaultTopP = 0.8f;
        const int defaultTopK = 30;
        var rasWindow = new Queue<int>(rasWindowSize);

        int mainToken = SampleMasked(logits, semBegin, semEnd, _imEndId, defaultTemp, defaultTopP, defaultTopK, rng);
        float[] hidden = GetNormedHidden(pos - 1);

        for (int step = 0; step < maxTokens && mainToken != _imEndId; step++)
        {
            if (rasWindow.Contains(mainToken) && mainToken >= semBegin && mainToken <= semEnd)
                mainToken = SampleMasked(logits, semBegin, semEnd, _imEndId, rasHighTemp, rasHighTopP, defaultTopK, rng);

            rasWindow.Enqueue(mainToken);
            if (rasWindow.Count > rasWindowSize) rasWindow.Dequeue();

            int semCode = Math.Clamp(mainToken - semBegin, 0, _weights.CodebookSize - 1);
            semanticTokens.Add(semCode);

            // Real fast-AR codebook expansion:
            // Position 0 = slow-AR hidden state (projected)
            // Position 1 = fast_embeddings[semCode], which outputs logits for Codebook 1
            // Position 2..9 = fast_embeddings[cbToken], which outputs logits for Codebook 2..9
            var codebookValues = new int[_weights.NumCodebooks];
            codebookValues[0] = semCode;
            _fastArCache.Reset();
            FishSpeechFastAr.ForwardStep(_weights, _fastArCache, hidden);
            var stepLogits = FishSpeechFastAr.ForwardStep(_weights, _fastArCache, FishSpeechFastAr.EmbedFastToken(_weights, semCode));
            for (int cb = 1; cb < _weights.NumCodebooks; cb++)
            {
                int cbToken = SampleToken(stepLogits, defaultTemp, defaultTopP, defaultTopK, rng);
                codebookValues[cb] = cbToken;
                if (cb < _weights.NumCodebooks - 1)
                    stepLogits = FishSpeechFastAr.ForwardStep(_weights, _fastArCache, FishSpeechFastAr.EmbedFastToken(_weights, cbToken));
            }
            codebooksPerFrame.Add(codebookValues);

            var emb = EmbedSemanticToken(mainToken, codebookValues);
            logits = _fwd.ForwardEmbedding(emb, pos);
            pos++;
            hidden = GetNormedHidden(pos - 1);

            mainToken = SampleMasked(logits, semBegin, semEnd, _imEndId, defaultTemp, defaultTopP, defaultTopK, rng);
        }

        return (semanticTokens, codebooksPerFrame);
    }

    /// <summary>
    /// TEST-SUPPORT ONLY: forces semantic tokens from a reference trajectory through slow-AR + fast-AR.
    /// </summary>
    public (List<int> SemanticTokens, List<int[]> CodebooksPerFrame) ForceGenerateFrames(string text, int[] forcedSemanticCodes)
    {
        var prompt = BuildPrompt(text);
        _fwd.ResetCache();
        _fwd.Prefill(prompt);
        int pos = prompt.Count;

        int semBegin = _weights.SemanticBeginId;
        var semanticTokens = new List<int>();
        var codebooksPerFrame = new List<int[]>();
        float[] hidden = GetNormedHidden(pos - 1);

        for (int step = 0; step < forcedSemanticCodes.Length; step++)
        {
            int semCode = forcedSemanticCodes[step];
            int mainToken = semBegin + semCode;
            semanticTokens.Add(semCode);

            var codebookValues = new int[_weights.NumCodebooks];
            codebookValues[0] = semCode;
            _fastArCache.Reset();
            FishSpeechFastAr.ForwardStep(_weights, _fastArCache, hidden);
            var stepLogits = FishSpeechFastAr.ForwardStep(_weights, _fastArCache, FishSpeechFastAr.EmbedFastToken(_weights, semCode));
            for (int cb = 1; cb < _weights.NumCodebooks; cb++)
            {
                int cbToken = ArgmaxLocal(stepLogits);
                codebookValues[cb] = cbToken;
                if (cb < _weights.NumCodebooks - 1)
                    stepLogits = FishSpeechFastAr.ForwardStep(_weights, _fastArCache, FishSpeechFastAr.EmbedFastToken(_weights, cbToken));
            }
            codebooksPerFrame.Add(codebookValues);

            var emb = EmbedSemanticToken(mainToken, codebookValues);
            _fwd.ForwardEmbedding(emb, pos);
            pos++;
            hidden = GetNormedHidden(pos - 1);
        }

        return (semanticTokens, codebooksPerFrame);
    }

    private static int ArgmaxLocal(ReadOnlySpan<float> logits)
    {
        int idx = 0;
        float max = logits[0];
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > max) { max = logits[i]; idx = i; }
        return idx;
    }

    private static int ArgmaxMasked(ReadOnlySpan<float> logits, int semBegin, int semEnd, int imEndId)
    {
        int best = -1;
        float bestVal = float.NegativeInfinity;
        for (int i = semBegin; i <= semEnd && i < logits.Length; i++)
        {
            if (logits[i] > bestVal) { bestVal = logits[i]; best = i; }
        }
        if (imEndId < logits.Length && logits[imEndId] > bestVal) { best = imEndId; }
        return best;
    }

    /// <summary>RAS escape-hatch only: applies the real `-inf`-outside-`[semBegin,semEnd]` (plus `im_end`) mask, then <see cref="SampleToken"/>.</summary>
    private static int SampleMasked(ReadOnlySpan<float> logits, int semBegin, int semEnd, int imEndId, float temperature, float topP, int topK, Random rng)
    {
        var biased = new float[logits.Length];
        Array.Fill(biased, float.NegativeInfinity);
        for (int i = semBegin; i <= semEnd && i < logits.Length; i++) biased[i] = logits[i];
        if (imEndId < logits.Length) biased[imEndId] = logits[imEndId];
        return SampleToken(biased, temperature, topP, topK, rng);
    }

    /// <summary>
    /// Real port of `examples/s2.cpp/src/s2_sampler.cpp`'s `sample_token`: sort logits
    /// descending, compute the UN-tempered softmax over the full sorted list (used only for the
    /// top-p cumulative-mass threshold), keep the intersection of the top-k highest and the
    /// smallest prefix whose cumulative un-tempered probability exceeds top_p (always keeping at
    /// least the top-1), then re-softmax just the kept logits WITH temperature and sample
    /// categorically. Only used by the RAS escape hatch in <see cref="GenerateFrames"/> -- normal
    /// decoding stays plain greedy (see that method's doc comment for why).
    /// </summary>
    private static int SampleToken(ReadOnlySpan<float> logits, float temperature, float topP, int topK, Random rng)
    {
        int n = logits.Length;
        var logitsArr = logits.ToArray();
        var order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;
        Array.Sort(order, (a, b) => logitsArr[b].CompareTo(logitsArr[a]));

        float max = logitsArr[order[0]];
        var sortedProbs = new float[n];
        float sum = 0f;
        for (int i = 0; i < n; i++) { sortedProbs[i] = MathF.Exp(logitsArr[order[i]] - max); sum += sortedProbs[i]; }
        if (sum > 0f) for (int i = 0; i < n; i++) sortedProbs[i] /= sum;

        int k = topK > 0 ? Math.Min(topK, n) : n;
        float p = Math.Clamp(topP, 0f, 1f);

        var kept = new List<int>();
        float cumsum = 0f;
        for (int i = 0; i < n; i++)
        {
            cumsum += sortedProbs[i];
            bool removeForTopK = i >= k;
            bool removeForTopP = i > 0 && cumsum > p;
            if (removeForTopK || removeForTopP) continue;
            kept.Add(order[i]);
        }
        if (kept.Count == 0) kept.Add(order[0]);

        if (temperature <= 0f) return kept[0];

        var probs = new float[kept.Count];
        float keptMax = float.NegativeInfinity;
        for (int i = 0; i < kept.Count; i++) if (logitsArr[kept[i]] > keptMax) keptMax = logitsArr[kept[i]];
        float keptSum = 0f;
        for (int i = 0; i < kept.Count; i++) { probs[i] = MathF.Exp((logitsArr[kept[i]] - keptMax) / temperature); keptSum += probs[i]; }
        if (keptSum <= 0f) return kept[0];
        for (int i = 0; i < probs.Length; i++) probs[i] /= keptSum;

        double r = rng.NextDouble();
        double acc = 0;
        for (int i = 0; i < probs.Length; i++)
        {
            acc += probs[i];
            if (r < acc) return kept[i];
        }
        return kept[^1];
    }

    public void Dispose()
    {
        _fwd.Dispose();
        _backend.Dispose();
        _weights.Dispose();
        _model.Dispose();
    }
}
