
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
    private readonly float[] _normedHidden;
    private readonly float[] _embBuffer;

    // Zero-allocation sampling scratch buffers
    private readonly float[] _candidateLogits;
    private readonly int[] _candidateIds;
    private readonly int[] _candidateOrder;
    private readonly float[] _sortedProbs;
    private readonly int[] _candidateKept;
    private readonly float[] _candidateSampleProbs;
    private readonly int[] _codebookOrder;
    private readonly float[] _codebookSortedProbs;
    private readonly int[] _codebookKept;
    private readonly float[] _codebookSampleProbs;

    public FishSpeechPipeline(string ggufPath, string tokenizerDir, int numLayers = 36, int ctxSize = 2048)
    {
        _model = GgufModel.Open(ggufPath);
        _tensorSource = new FishSpeechTensorSource(_model, numLayers);
        var hp = ModelHyperparams.FromGgufMetadata(_tensorSource.Metadata, _tensorSource);
        _backend = new CpuBackend();
        _fwd = new ForwardPass(_tensorSource, _backend, hp, maxContextLength: ctxSize);
        _fwd.EnableHiddenTaps([numLayers - 1]); // last layer's output = the trunk's post-trunk pre-final-norm hidden
        _weights = new FishSpeechWeights(ggufPath);
        _fastArCache = new FishSpeechFastArCache(_weights);
        _normedHidden = new float[_weights.EmbeddingDim];
        _embBuffer = new float[_weights.EmbeddingDim];

        int maxCandidates = (_weights.SemanticEndId - _weights.SemanticBeginId + 1) + 1;
        _candidateLogits = new float[maxCandidates];
        _candidateIds = new int[maxCandidates];
        _candidateOrder = new int[maxCandidates];
        _sortedProbs = new float[maxCandidates];
        _candidateKept = new int[maxCandidates];
        _candidateSampleProbs = new float[maxCandidates];

        int codebookSize = _weights.CodebookSize;
        _codebookOrder = new int[codebookSize];
        _codebookSortedProbs = new float[codebookSize];
        _codebookKept = new int[codebookSize];
        _codebookSampleProbs = new float[codebookSize];

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

    /// <summary>Reads the slow-AR trunk's final RMSNorm hidden state directly from ForwardPass.</summary>
    private float[] GetNormedHidden(int pos)
    {
        _fwd.LastHidden.CopyTo(_normedHidden);
        return _normedHidden;
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
    private ReadOnlySpan<float> EmbedSemanticToken(int semanticTokenId, int[] codebookValues)
    {
        int dim = _weights.EmbeddingDim;
        Array.Copy(_weights.Embeddings, (long)semanticTokenId * dim, _embBuffer, 0, dim);

        for (int cb = 0; cb < codebookValues.Length; cb++)
        {
            long row = (long)(codebookValues[cb] + cb * _weights.CodebookSize) * dim;
            System.Numerics.Tensors.TensorPrimitives.Add(_embBuffer.AsSpan(0, dim), _weights.CodebookEmbeddings.AsSpan((int)row, dim), _embBuffer.AsSpan(0, dim));
        }

        if (_weights.ScaleCodebookEmbeddings)
        {
            float scale = 1f / MathF.Sqrt(_weights.NumCodebooks + 1);
            System.Numerics.Tensors.TensorPrimitives.Multiply(_embBuffer.AsSpan(0, dim), scale, _embBuffer.AsSpan(0, dim));
        }
        return _embBuffer;
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
        var semanticTokens = new List<int>();
        var codebooksPerFrame = new List<int[]>();
        foreach (var (sem, cb) in GenerateFramesStream(text, maxTokens, seed))
        {
            semanticTokens.Add(sem);
            codebooksPerFrame.Add(cb);
        }
        return (semanticTokens, codebooksPerFrame);
    }

    /// <summary>
    /// Streaming generator: yields each frame's `(semCode, codebookValues)` as soon as it is sampled.
    /// </summary>
    public IEnumerable<(int SemanticToken, int[] Codebooks)> GenerateFramesStream(string text, int maxTokens = 200, int? seed = null)
    {
        var rng = new Random(seed ?? 42);
        var prompt = BuildPrompt(text);
        _fwd.ResetCache();
        _fwd.SkipOutputProjection = false;

        var logits = _fwd.Prefill(prompt);
        int pos = prompt.Count;

        int semBegin = _weights.SemanticBeginId;
        int semEnd = _weights.SemanticEndId;

        const int rasWindowSize = 10;
        const float rasHighTemp = 1.0f, rasHighTopP = 0.9f;
        const float defaultTemp = 0.8f, defaultTopP = 0.8f;
        const int defaultTopK = 30;
        var rasWindow = new Queue<int>(rasWindowSize);

        int mainToken = SampleMasked(logits, semBegin, semEnd, _imEndId, defaultTemp, defaultTopP, defaultTopK, rng);
        float[] hidden = GetNormedHidden(pos - 1);

        // For all subsequent decode steps, skip full 155,776-row LM head projection in ForwardPass
        // and compute candidate dot products directly for the 4,097 valid tokens (38x faster).
        _fwd.SkipOutputProjection = true;

        for (int step = 0; step < maxTokens && mainToken != _imEndId; step++)
        {
            if (rasWindow.Contains(mainToken) && mainToken >= semBegin && mainToken <= semEnd)
                mainToken = SampleCandidateToken(hidden, semBegin, semEnd, _imEndId, rasHighTemp, rasHighTopP, defaultTopK, rng);

            rasWindow.Enqueue(mainToken);
            if (rasWindow.Count > rasWindowSize) rasWindow.Dequeue();

            int semCode = Math.Clamp(mainToken - semBegin, 0, _weights.CodebookSize - 1);

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

            yield return (semCode, codebookValues);

            var emb = EmbedSemanticToken(mainToken, codebookValues);
            _fwd.ForwardEmbedding(emb, pos);
            pos++;
            hidden = GetNormedHidden(pos - 1);

            mainToken = SampleCandidateToken(hidden, semBegin, semEnd, _imEndId, defaultTemp, defaultTopP, defaultTopK, rng);
        }
    }

    private unsafe int SampleCandidateToken(float[] normedHidden, int semBegin, int semEnd, int imEndId, float temperature, float topP, int topK, Random rng)
    {
        int semCount = semEnd - semBegin + 1;
        int totalCandidates = semCount + 1;

        fixed (float* embPtr = _weights.Embeddings, hPtr = normedHidden, logitPtr = _candidateLogits)
        fixed (int* idPtr = _candidateIds)
        {
            int dim = _weights.EmbeddingDim;
            SimdKernels.MatVecF32(logitPtr, embPtr + (long)semBegin * dim, hPtr, semCount, dim);
            for (int i = 0; i < semCount; i++) idPtr[i] = semBegin + i;

            idPtr[semCount] = imEndId;
            logitPtr[semCount] = SimdKernels.DotF32(hPtr, embPtr + (long)imEndId * dim, dim);
        }

        return SampleIndexedTokenZeroAlloc(_candidateLogits.AsSpan(0, totalCandidates), _candidateIds.AsSpan(0, totalCandidates), temperature, topP, topK, rng);
    }

    private int SampleIndexedTokenZeroAlloc(ReadOnlySpan<float> logits, ReadOnlySpan<int> tokenIds, float temperature, float topP, int topK, Random rng)
    {
        int n = logits.Length;
        for (int i = 0; i < n; i++) _candidateOrder[i] = i;

        var logitsArr = _candidateLogits;
        Array.Sort(_candidateOrder, 0, n, Comparer<int>.Create((a, b) => logitsArr[b].CompareTo(logitsArr[a])));

        float max = logitsArr[_candidateOrder[0]];
        float sum = 0f;
        for (int i = 0; i < n; i++)
        {
            float pVal = MathF.Exp(logitsArr[_candidateOrder[i]] - max);
            _sortedProbs[i] = pVal;
            sum += pVal;
        }
        if (sum > 0f)
        {
            float invSum = 1f / sum;
            for (int i = 0; i < n; i++) _sortedProbs[i] *= invSum;
        }

        int k = topK > 0 ? Math.Min(topK, n) : n;
        float p = Math.Clamp(topP, 0f, 1f);

        int keptCount = 0;
        float cumsum = 0f;
        for (int i = 0; i < n; i++)
        {
            cumsum += _sortedProbs[i];
            bool removeForTopK = i >= k;
            bool removeForTopP = i > 0 && cumsum > p;
            if (removeForTopK || removeForTopP) continue;
            _candidateKept[keptCount++] = _candidateOrder[i];
        }
        if (keptCount == 0)
        {
            _candidateKept[0] = _candidateOrder[0];
            keptCount = 1;
        }

        if (temperature <= 0f) return tokenIds[_candidateKept[0]];

        float keptMax = float.NegativeInfinity;
        for (int i = 0; i < keptCount; i++)
        {
            float l = logitsArr[_candidateKept[i]];
            if (l > keptMax) keptMax = l;
        }

        float keptSum = 0f;
        float invTemp = 1f / temperature;
        for (int i = 0; i < keptCount; i++)
        {
            float pExp = MathF.Exp((logitsArr[_candidateKept[i]] - keptMax) * invTemp);
            _candidateSampleProbs[i] = pExp;
            keptSum += pExp;
        }
        if (keptSum <= 0f) return tokenIds[_candidateKept[0]];

        float invKeptSum = 1f / keptSum;
        for (int i = 0; i < keptCount; i++) _candidateSampleProbs[i] *= invKeptSum;

        double r = rng.NextDouble();
        double acc = 0;
        for (int i = 0; i < keptCount; i++)
        {
            acc += _candidateSampleProbs[i];
            if (r < acc || i == keptCount - 1)
                return tokenIds[_candidateKept[i]];
        }
        return tokenIds[_candidateKept[0]];
    }

    /// <summary>RAS escape-hatch only: applies the real `-inf`-outside-`[semBegin,semEnd]` (plus `im_end`) mask, then <see cref="SampleToken"/>.</summary>
    private int SampleMasked(ReadOnlySpan<float> logits, int semBegin, int semEnd, int imEndId, float temperature, float topP, int topK, Random rng)
    {
        int maxCandidates = (semEnd - semBegin + 1) + 1;
        int totalCandidates = maxCandidates;
        int semCount = semEnd - semBegin + 1;

        for (int i = 0; i < semCount; i++)
        {
            int tokId = semBegin + i;
            _candidateIds[i] = tokId;
            _candidateLogits[i] = tokId < logits.Length ? logits[tokId] : float.NegativeInfinity;
        }
        _candidateIds[semCount] = imEndId;
        _candidateLogits[semCount] = imEndId < logits.Length ? logits[imEndId] : float.NegativeInfinity;

        return SampleIndexedTokenZeroAlloc(_candidateLogits.AsSpan(0, totalCandidates), _candidateIds.AsSpan(0, totalCandidates), temperature, topP, topK, rng);
    }

    /// <summary>
    /// Real port of `examples/s2.cpp/src/s2_sampler.cpp`'s `sample_token`: zero-allocation version.
    /// </summary>
    private int SampleToken(float[] logits, float temperature, float topP, int topK, Random rng)
    {
        int n = logits.Length;
        int[] order = n == _codebookOrder.Length ? _codebookOrder : new int[n];
        float[] sortedProbs = n == _codebookSortedProbs.Length ? _codebookSortedProbs : new float[n];
        int[] kept = n == _codebookKept.Length ? _codebookKept : new int[n];
        float[] sampleProbs = n == _codebookSampleProbs.Length ? _codebookSampleProbs : new float[n];

        for (int i = 0; i < n; i++) order[i] = i;
        Array.Sort(order, 0, n, Comparer<int>.Create((a, b) => logits[b].CompareTo(logits[a])));

        float max = logits[order[0]];
        float sum = 0f;
        for (int i = 0; i < n; i++)
        {
            float pVal = MathF.Exp(logits[order[i]] - max);
            sortedProbs[i] = pVal;
            sum += pVal;
        }
        if (sum > 0f)
        {
            float invSum = 1f / sum;
            for (int i = 0; i < n; i++) sortedProbs[i] *= invSum;
        }

        int k = topK > 0 ? Math.Min(topK, n) : n;
        float p = Math.Clamp(topP, 0f, 1f);

        int keptCount = 0;
        float cumsum = 0f;
        for (int i = 0; i < n; i++)
        {
            cumsum += sortedProbs[i];
            bool removeForTopK = i >= k;
            bool removeForTopP = i > 0 && cumsum > p;
            if (removeForTopK || removeForTopP) continue;
            kept[keptCount++] = order[i];
        }
        if (keptCount == 0)
        {
            kept[0] = order[0];
            keptCount = 1;
        }

        if (temperature <= 0f) return kept[0];

        float keptMax = float.NegativeInfinity;
        for (int i = 0; i < keptCount; i++)
        {
            float l = logits[kept[i]];
            if (l > keptMax) keptMax = l;
        }

        float keptSum = 0f;
        float invTemp = 1f / temperature;
        for (int i = 0; i < keptCount; i++)
        {
            float pExp = MathF.Exp((logits[kept[i]] - keptMax) * invTemp);
            sampleProbs[i] = pExp;
            keptSum += pExp;
        }
        if (keptSum <= 0f) return kept[0];

        float invKeptSum = 1f / keptSum;
        for (int i = 0; i < keptCount; i++) sampleProbs[i] *= invKeptSum;

        double r = rng.NextDouble();
        double acc = 0;
        for (int i = 0; i < keptCount; i++)
        {
            acc += sampleProbs[i];
            if (r < acc || i == keptCount - 1)
                return kept[i];
        }
        return kept[0];
    }

    public void Dispose()
    {
        _fwd.Dispose();
        _backend.Dispose();
        _weights.Dispose();
        _model.Dispose();
    }
}
