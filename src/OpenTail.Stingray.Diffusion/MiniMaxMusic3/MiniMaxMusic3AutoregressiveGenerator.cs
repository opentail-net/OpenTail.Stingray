namespace OpenTail.Stingray.Diffusion.MiniMaxMusic3;

/// <summary>
/// Real MiniMax Music 3 autoregressive generation loop (`MiniMaxMusic3AutoregressiveStep`),
/// transcribed directly from `diffusers/modular_pipelines/minimax_music3/encoders.py` -- see
/// docs/066-minimax-music3-future-plan.md, "Real autoregressive generation loop, fully specified
/// from source". Drives the real Global LM (<see cref="MiniMaxMusic3GlobalModel"/>) and Local/depth
/// decoder (<see cref="MiniMaxMusic3RvqDepthDecoder"/>) together, frame by frame, with real
/// classifier-free guidance at every sampling decision (semantic code AND each of the 7 residual
/// codes), producing a <see cref="Music3Representation"/> for the downstream Flow-synthesis stage.
///
/// <para><b>Real CFG mechanism</b>: the conditional and unconditional (CFG-null) prompts run as two
/// PARALLEL sequences (their own KV caches / depth-decoder input sequences, since their hidden
/// states diverge from the very first prompt token), but every sampling DECISION is made once from
/// CFG-combined logits and that single discrete choice (semantic code, then each residual code) is
/// fed to BOTH branches for the next step -- only the branches' own hidden states differ, never
/// which token gets embedded next.</para>
///
/// <para><b>Real prompt-construction and tokenization are NOT this class's job</b> -- callers pass
/// already-tokenized <paramref name="conditionalPromptTokens"/> ending in the real
/// `&lt;|audio_start|&gt;` token id; the real `_clean_caption`/`_normalize_lyrics` text
/// normalization and the real `Qwen2Tokenizer` vocab for this checkpoint are a separate,
/// not-yet-built piece (this checkpoint's own tokenizer files haven't been fetched/verified yet).</para>
/// </summary>
public static class MiniMaxMusic3AutoregressiveGenerator
{
    public static Music3Representation Generate(
        MiniMaxMusic3GlobalModel globalModel,
        MiniMaxMusic3RvqDepthDecoderWeights depthWeights,
        int[] conditionalPromptTokens,
        int maxFrames,
        Random random)
    {
        int hidden = MiniMaxMusic3Config.LanguageModelHiddenSize;
        int numLayers = MiniMaxMusic3Config.LanguageModelNumLayers;
        int numResidualCodebooks = MiniMaxMusic3Config.RvqDepthDecoderNumCodebooks - 1; // 7
        int depthHidden = MiniMaxMusic3Config.RvqDepthDecoderHiddenSize;

        var unconditionalPromptTokens = BuildUnconditionalPrompt(conditionalPromptTokens);

        var condCache = new MiniMaxMusic3GlobalKvCache(numLayers);
        var uncondCache = new MiniMaxMusic3GlobalKvCache(numLayers);
        var condDepthCache = new MiniMaxMusic3RvqDepthKvCache();
        var uncondDepthCache = new MiniMaxMusic3RvqDepthKvCache();

        // Real: frame_index 0 only advances state past <|audio_start|> (the prompt's last token),
        // does not emit a frame -- the prefill IS that step.
        var (condHiddenSeq, condLastLogits) = globalModel.ForwardIncremental(conditionalPromptTokens, condCache);
        var (uncondHiddenSeq, uncondLastLogits) = globalModel.ForwardIncremental(unconditionalPromptTokens, uncondCache);
        float[] condLastHidden = condHiddenSeq[^1];
        float[] uncondLastHidden = uncondHiddenSeq[^1];

        var semanticTokens = new List<int>();
        var acousticTokens = new List<int[]>();
        var globalHiddenStates = new List<float[]>();
        var localHiddenStates = new List<float[]>();

        // Real: `for frame_index in range(max_frames + 1)`, but frame_index == 0 only advances the
        // language model's state past the prefill's <|audio_start|> hidden state -- its sampled
        // semantic/residual codes are used ONLY to compute the next feedback embedding, and are
        // NEVER appended to frame_hiddens. Getting this off-by-one wrong means treating that
        // prompt-boundary "warm-up" sample as real frame 0's content (and correspondingly never
        // generating the real final frame) -- exactly the class of bug that corrupts every frame's
        // conditioning without failing any single-component golden-parity test.
        for (int frameIndex = 0; frameIndex <= maxFrames; frameIndex++)
        {
            int semanticCode = SampleSemanticCode(condLastLogits, uncondLastLogits, random, out bool isEnd);
            if (isEnd) break;

            condDepthCache.Reset();
            uncondDepthCache.Reset();

            var semanticEmbed = globalModel.EmbedToken(MiniMaxMusic3Config.AudioCodeOffset + semanticCode);
            var projectedSemantic = MiniMaxMusic3RvqDepthDecoder.Project(depthWeights, semanticEmbed);

            MiniMaxMusic3RvqDepthDecoder.ForwardStep(depthWeights, MiniMaxMusic3RvqDepthDecoder.Project(depthWeights, condLastHidden), 0, condDepthCache);
            MiniMaxMusic3RvqDepthDecoder.ForwardStep(depthWeights, MiniMaxMusic3RvqDepthDecoder.Project(depthWeights, uncondLastHidden), 0, uncondDepthCache);

            var condDepthLast = MiniMaxMusic3RvqDepthDecoder.ForwardStep(depthWeights, projectedSemantic, 1, condDepthCache);
            var uncondDepthLast = MiniMaxMusic3RvqDepthDecoder.ForwardStep(depthWeights, projectedSemantic, 1, uncondDepthCache);

            var residualCodes = new int[numResidualCodebooks];
            var localHiddenConcat = new float[numResidualCodebooks * depthHidden];

            for (int ci = 0; ci < numResidualCodebooks; ci++)
            {
                Array.Copy(condDepthLast, 0, localHiddenConcat, ci * depthHidden, depthHidden);

                var condLogitsC = MiniMaxMusic3RvqDepthDecoder.CodebookLogits(depthWeights, condDepthLast, ci);
                var uncondLogitsC = MiniMaxMusic3RvqDepthDecoder.CodebookLogits(depthWeights, uncondDepthLast, ci);

                int code = SampleWithCfgTopK(condLogitsC, uncondLogitsC, random);
                residualCodes[ci] = code;

                if (ci + 1 < numResidualCodebooks)
                {
                    var embedded = MiniMaxMusic3RvqDepthDecoder.Project(depthWeights, MiniMaxMusic3RvqDepthDecoder.EmbedResidualCode(depthWeights, ci, code));
                    condDepthLast = MiniMaxMusic3RvqDepthDecoder.ForwardStep(depthWeights, embedded, ci + 2, condDepthCache);
                    uncondDepthLast = MiniMaxMusic3RvqDepthDecoder.ForwardStep(depthWeights, embedded, ci + 2, uncondDepthCache);
                }
            }

            if (frameIndex > 0)
            {
                semanticTokens.Add(semanticCode);
                acousticTokens.Add(residualCodes);
                globalHiddenStates.Add(condLastHidden);
                localHiddenStates.Add(localHiddenConcat);
                if (semanticTokens.Count >= maxFrames) break;
            }

            // Real feedback embedding from THIS iteration's sampled codes drives the next step,
            // regardless of whether this iteration's frame was actually emitted.
            var feedback = ComputeFeedbackEmbedding(globalModel, depthWeights, semanticCode, residualCodes, hidden);
            (condLastHidden, uncondLastHidden, condLastLogits, uncondLastLogits) =
                globalModel.ForwardIncrementalStepPair(feedback, feedback, condCache, uncondCache);
        }

        return new Music3Representation
        {
            SemanticTokens = [.. semanticTokens],
            AcousticTokens = [.. acousticTokens],
            GlobalHiddenStates = [.. globalHiddenStates],
            LocalHiddenStates = [.. localHiddenStates],
        };
    }

    /// <summary>Real `_embed_audio_frame`: sums the semantic code's real language-model token
    /// embedding with the SUM of all 7 residual codes' real `audio_embeddings` rows, scaled by
    /// `numCodebooks**-0.5`.</summary>
    private static float[] ComputeFeedbackEmbedding(MiniMaxMusic3GlobalModel globalModel, MiniMaxMusic3RvqDepthDecoderWeights depthWeights, int semanticCode, int[] residualCodes, int hidden)
    {
        var sum = globalModel.EmbedToken(MiniMaxMusic3Config.AudioCodeOffset + semanticCode);
        for (int ci = 0; ci < residualCodes.Length; ci++)
        {
            var row = MiniMaxMusic3RvqDepthDecoder.EmbedResidualCode(depthWeights, ci, residualCodes[ci]);
            for (int i = 0; i < hidden; i++) sum[i] += row[i];
        }
        float scale = MathF.Pow(MiniMaxMusic3Config.RvqDepthDecoderNumCodebooks, -0.5f);
        for (int i = 0; i < hidden; i++) sum[i] *= scale;
        return sum;
    }

    /// <summary>Real semantic-code sampling: mask `lm_head` logits to just the real audio-code
    /// range (`[AudioCodeOffset, AudioCodeOffset+SemanticVocabSize)`) plus the real END token id,
    /// apply CFG restricted to the conditional branch's real top-50 candidates (avoids NaN from
    /// guiding two `-inf` logits), then real top-50 multinomial sample.</summary>
    private static int SampleSemanticCode(float[] condLogits, float[] uncondLogits, Random random, out bool isEnd)
    {
        int offset = MiniMaxMusic3Config.AudioCodeOffset;
        int vocabSize = MiniMaxMusic3Config.SemanticVocabSize;
        int endId = MiniMaxMusic3Config.AudioEndTokenId;

        // Candidate set: audio-code range indices [0, vocabSize) map to logits[offset+i]; the END
        // token is one extra candidate appended at index vocabSize.
        var candidateIndices = new int[vocabSize + 1];
        for (int i = 0; i < vocabSize; i++) candidateIndices[i] = offset + i;
        candidateIndices[vocabSize] = endId;

        int sampledCandidate = SampleCfgTopKOverIndices(condLogits, uncondLogits, candidateIndices, random);
        if (sampledCandidate == vocabSize) { isEnd = true; return -1; }
        isEnd = false;
        return sampledCandidate;
    }

    /// <summary>Real residual-codebook sampling -- CORRECTED against the real `minimaxmusic.cpp`
    /// reference (`mm3_cfg_sample_guided`, `src/pipeline.cpp`), which is a DIFFERENT ranking order
    /// than the semantic head: CFG is applied to the FULL codebook first
    /// (`guided[id] = uncond[id] + (cond[id]-uncond[id])*ArCfgScale` for every id, unrestricted),
    /// and only THEN is the top-<see cref="MiniMaxMusic3Config.ArSamplingTopK"/> of the GUIDED
    /// logits taken and softmax-sampled -- there is no cond-first restriction at all for this path.
    /// The semantic head restricts to the conditional branch's top-k FIRST (see
    /// <see cref="SampleCfgTopKOverIndices"/>); reusing that same function here (as this port
    /// previously did) silently drops any residual-codebook candidate that CFG would have promoted
    /// into contention but which wasn't already in the conditional branch's own top-50 -- a real,
    /// substantial divergence hitting 7 of every 8 sampling decisions per frame (every residual
    /// codebook, only the semantic token was ever sampled correctly), which plausibly explains
    /// "jitter, not music" surviving the earlier flow-scheduler direction fix.</summary>
    private static int SampleWithCfgTopK(float[] condLogits, float[] uncondLogits, Random random)
    {
        int n = condLogits.Length;
        var guided = new float[n];
        for (int i = 0; i < n; i++) guided[i] = uncondLogits[i] + (condLogits[i] - uncondLogits[i]) * MiniMaxMusic3Config.ArCfgScale;

        var topPositions = TopKIndices(guided, Math.Min(MiniMaxMusic3Config.ArSamplingTopK, n));
        var topVals = new float[topPositions.Length];
        for (int i = 0; i < topPositions.Length; i++) topVals[i] = guided[topPositions[i]];

        int chosenWithinTop = MultinomialSample(topVals, random);
        return topPositions[chosenWithinTop];
    }

    /// <summary>Shared real CFG + top-k sampling core: restrict to the conditional branch's real
    /// top-<see cref="MiniMaxMusic3Config.ArCfgTopK"/> candidates first (avoids the CFG formula
    /// producing NaN by guiding between two `-inf` logits outside that set), apply
    /// `uncond + (cond-uncond)*ArCfgScale` to just those, then real top-
    /// <see cref="MiniMaxMusic3Config.ArSamplingTopK"/> multinomial sample over the CFG-combined
    /// logits (softmax restricted to that final top-k). Returns the ORIGINAL candidate array index
    /// (not the vocab id) of the sampled candidate.</summary>
    private static int SampleCfgTopKOverIndices(float[] condLogits, float[] uncondLogits, int[] candidateIndices, Random random)
    {
        int n = candidateIndices.Length;
        var condVals = new float[n];
        for (int i = 0; i < n; i++) condVals[i] = condLogits[candidateIndices[i]];

        var cfgTopKPositions = TopKIndices(condVals, Math.Min(MiniMaxMusic3Config.ArCfgTopK, n));

        var cfgVals = new float[cfgTopKPositions.Length];
        for (int i = 0; i < cfgTopKPositions.Length; i++)
        {
            int pos = cfgTopKPositions[i];
            float c = condLogits[candidateIndices[pos]];
            float u = uncondLogits[candidateIndices[pos]];
            cfgVals[i] = u + (c - u) * MiniMaxMusic3Config.ArCfgScale;
        }

        var samplingTopKPositions = TopKIndices(cfgVals, Math.Min(MiniMaxMusic3Config.ArSamplingTopK, cfgVals.Length));
        var samplingVals = new float[samplingTopKPositions.Length];
        for (int i = 0; i < samplingTopKPositions.Length; i++) samplingVals[i] = cfgVals[samplingTopKPositions[i]];

        int chosenWithinSampling = MultinomialSample(samplingVals, random);
        int chosenWithinCfg = samplingTopKPositions[chosenWithinSampling];
        return cfgTopKPositions[chosenWithinCfg];
    }

    private static int[] TopKIndices(float[] values, int k)
    {
        var indices = new int[values.Length];
        for (int i = 0; i < indices.Length; i++) indices[i] = i;
        Array.Sort(indices, (a, b) => values[b].CompareTo(values[a]));
        return indices[..k];
    }

    private static int MultinomialSample(float[] logits, Random random)
    {
        float max = float.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++) if (logits[i] > max) max = logits[i];
        var probs = new float[logits.Length];
        float sum = 0f;
        for (int i = 0; i < logits.Length; i++) { probs[i] = MathF.Exp(logits[i] - max); sum += probs[i]; }
        float target = (float)random.NextDouble() * sum;
        float running = 0f;
        for (int i = 0; i < probs.Length; i++)
        {
            running += probs[i];
            if (running >= target) return i;
        }
        return probs.Length - 1;
    }

    /// <summary>Real CFG-null prompt: same length/tokens as the conditional prompt, except every
    /// token except the first and the two trailing structure tokens (`&lt;|im_end|&gt;`,
    /// `&lt;|audio_start|&gt;`) is replaced with the real `_AUDIO_CFG_TOKEN_ID`.</summary>
    private static int[] BuildUnconditionalPrompt(int[] conditionalPromptTokens)
    {
        var result = (int[])conditionalPromptTokens.Clone();
        int cfgToken = MiniMaxMusic3Config.AudioCfgTokenId;
        for (int i = 1; i < result.Length - 2; i++) result[i] = cfgToken;
        return result;
    }
}
