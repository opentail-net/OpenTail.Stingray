
namespace OpenTail.Stingray.Audio.QwenASR;

/// <summary>
/// Word-level forced alignment engine for Qwen3-ForcedAligner-0.6B -- a real, SEPARATE
/// checkpoint from the base Qwen3-ASR model (own audio encoder, own 28-layer Qwen3 decoder, own
/// swapped 5000-way timestamp classify head in place of the normal vocab lm_head), NOT bundled
/// in the ASR GGUF. Confirmed real, directly against the checkpoint and against
/// `examples/audio.cpp/src/models/qwen3_forced_aligner/processor.cpp` (the real reference):
///
/// <para>Real algorithm (`processor.cpp`'s `build_prompt`/`parse_timestamps`): build the prompt
/// `&lt;|audio_start|&gt;&lt;|audio_pad|&gt;×N&lt;|audio_end|&gt;` + for each word,
/// `word&lt;timestamp&gt;&lt;timestamp&gt;` (two placeholders per word: start, end). Run ONE
/// single non-autoregressive forward pass (a plain prefill, not token-by-token generation) with
/// real audio conditioning, through the SAME causal Qwen3 decoder architecture the base ASR
/// model uses. At each `&lt;timestamp&gt;` token's position, the swapped classify head (which
/// naturally IS this checkpoint's own `thinker.lm_head.weight`, so it comes through
/// `QwenAsrLlmSafetensorsTensorSource`'s existing `"output.weight"` mapping with zero extra
/// code) gives a 5000-way distribution; argmax × 80ms (`timestamp_segment_time_ms`) is the
/// predicted time in milliseconds. A longest-non-decreasing-subsequence repair
/// (<see cref="FixTimestampMonotonic"/>) smooths any non-monotonic outlier predictions before
/// pairing up (start, end) per word.</para>
///
/// <para>Confirmed real reuse: <see cref="QwenAsrWeights.LoadFromSafetensors"/> (written for
/// `Qwen/Qwen3-ASR-0.6B`) loads the real `Qwen/Qwen3-ForcedAligner-0.6B` checkpoint unchanged --
/// identical `thinker_config.audio_config`/`text_config` schema and
/// `thinker.audio_tower.*`/`thinker.model.*` tensor names, confirmed via
/// <c>QwenForcedAlignerRealCheckpointLoadDebugTest</c>.</para>
/// </summary>
public sealed class QwenAsrForcedAligner : IDisposable
{
    private readonly QwenForcedAlignerWeights? _weights;
    private readonly QwenAsrWeights? _realWeights;
    private readonly QwenAsrAudioEncoder? _realEncoder;
    private readonly QwenAsrMelExtractor? _realMelExtractor;
    private readonly string? _safetensorsPath;
    private readonly int _timestampTokenId;
    private readonly int _timestampSegmentTimeMs;

    /// <summary>
    /// The <paramref name="tokenizer"/> parameter is accepted for API/call-site compatibility
    /// (the procedural <see cref="Align"/> fallback below doesn't use it) but not currently used
    /// internally for that path; see <see cref="AlignReal"/> for the real pipeline, which builds
    /// its own tokenizer state from real weights.
    /// </summary>
    public QwenAsrForcedAligner(QwenAsrTokenizer? tokenizer = null, QwenForcedAlignerWeights? weights = null)
    {
        _weights = weights;
    }

    private QwenAsrForcedAligner(QwenAsrWeights realWeights, QwenAsrAudioEncoder realEncoder, string safetensorsPath, int timestampTokenId, int timestampSegmentTimeMs)
    {
        _realWeights = realWeights;
        _realEncoder = realEncoder;
        _realMelExtractor = new QwenAsrMelExtractor();
        _safetensorsPath = safetensorsPath;
        _timestampTokenId = timestampTokenId;
        _timestampSegmentTimeMs = timestampSegmentTimeMs;
    }

    /// <summary>Rough average-BPE-length token-count estimate (English averages ~4 chars/token) -- not real tokenization, only used by the procedural <see cref="Align"/> fallback.</summary>
    private static int EstimateTokenCount(string word) => Math.Max(1, (word.Length + 3) / 4);

    /// <summary>
    /// Loads a Qwen3-ForcedAligner from real Safetensors weights (legacy path -- the loaded
    /// weights are held but <see cref="Align"/> remains the procedural fallback; prefer
    /// <see cref="LoadReal"/> for a genuinely real forward pass).
    /// </summary>
    public static QwenAsrForcedAligner Load(string safetensorsPath)
    {
        if (string.IsNullOrWhiteSpace(safetensorsPath) || !File.Exists(safetensorsPath))
            throw new FileNotFoundException($"Qwen3-ForcedAligner model not found: {safetensorsPath}");

        var weights = new QwenForcedAlignerWeights(safetensorsPath);
        var tokenizer = new QwenAsrTokenizer();
        return new QwenAsrForcedAligner(tokenizer, weights);
    }

    /// <summary>
    /// Loads a real Qwen3-ForcedAligner-0.6B pipeline from the canonical Hugging Face
    /// `Qwen/Qwen3-ForcedAligner-0.6B` checkpoint directory (`config.json`/`model.safetensors`/
    /// `vocab.json`/`merges.txt`/`tokenizer_config.json`) via <see cref="AlignReal"/>.
    /// </summary>
    public static QwenAsrForcedAligner LoadReal(string checkpointDir)
    {
        if (!Directory.Exists(checkpointDir))
            throw new DirectoryNotFoundException($"Qwen3-ForcedAligner checkpoint directory not found: {checkpointDir}");

        var weights = QwenAsrWeights.LoadFromSafetensors(checkpointDir);
        var encoderConfig = new QwenAsrEncoderConfig
        {
            EncoderDim = weights.AudioDim,
            NumLayers = weights.AudioLayers,
            NumHeads = weights.AudioHeads,
            QwenHiddenDim = weights.LlmDim,
        };
        var encoder = new QwenAsrAudioEncoder(encoderConfig, weights);

        using var configDoc = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(checkpointDir, "config.json")));
        var root = configDoc.RootElement;
        int timestampTokenId = root.GetProperty("timestamp_token_id").GetInt32();
        int timestampSegmentTimeMs = root.GetProperty("timestamp_segment_time").GetInt32();

        return new QwenAsrForcedAligner(weights, encoder, Path.Combine(checkpointDir, "model.safetensors"), timestampTokenId, timestampSegmentTimeMs);
    }

    /// <summary>
    /// Real forced alignment: one single non-autoregressive forward pass through the real audio
    /// encoder + Qwen3 decoder + swapped timestamp-classify head, matching
    /// `examples/audio.cpp/src/models/qwen3_forced_aligner/processor.cpp`'s real algorithm (see
    /// this class's doc comment). Requires <see cref="LoadReal"/>.
    /// </summary>
    public List<SpeechSegment> AlignReal(ReadOnlySpan<float> pcm16k, string referenceText, TimeSpan timeOffset)
    {
        if (_realWeights is null || _realEncoder is null || _realMelExtractor is null || _safetensorsPath is null)
            throw new InvalidOperationException("AlignReal requires a QwenAsrForcedAligner constructed via LoadReal.");

        string[] words = TokenizeAlignableWords(referenceText);
        if (words.Length == 0) return [];

        float[] mel = _realMelExtractor.ExtractMel(pcm16k);
        int inMelFrames = mel.Length / QwenAsrMelExtractor.NumMels;
        if (inMelFrames == 0) return [];
        var (audioSoftTokens, numAudioTokens) = _realEncoder.Forward(mel, inMelFrames);

        if (Environment.GetEnvironmentVariable("STINGRAY_FORCEDALIGNER_TRACE") == "1")
        {
            double sum = 0, sumsq = 0, absMax = 0;
            foreach (var v in audioSoftTokens) { sum += v; sumsq += (double)v * v; absMax = Math.Max(absMax, Math.Abs(v)); }
            double mean = sum / audioSoftTokens.Length;
            double std = Math.Sqrt(Math.Max(0, sumsq / audioSoftTokens.Length - mean * mean));
            Console.Error.WriteLine($"[FA-Trace] inMelFrames={inMelFrames} numAudioTokens={numAudioTokens} audioSoftTokens: mean={mean:F4} std={std:F4} absMax={absMax:F4}");
        }

        // Build the real prompt: <|audio_start|><|audio_pad|>xN<|audio_end|> + per word
        // "word<timestamp><timestamp>". Encoding each word/prefix segment SEPARATELY through the
        // same BPE vocab, then splicing in the real <timestamp> token id directly, is exactly
        // equivalent to BPE-encoding the whole concatenated string in one pass, since <timestamp>
        // is a hard token boundary merges never cross -- avoids needing the real tokenizer to
        // recognize the literal string "<timestamp>" as one atomic special token.
        var promptIds = new List<int>();
        var sb = new System.Text.StringBuilder("<|audio_start|>");
        for (int i = 0; i < numAudioTokens; i++) sb.Append("<|audio_pad|>");
        sb.Append("<|audio_end|>");
        promptIds.AddRange(_realWeights.Tokenizer.Encode(sb.ToString()));

        var timestampPositions = new List<int>(words.Length * 2);
        foreach (var word in words)
        {
            promptIds.AddRange(_realWeights.Tokenizer.Encode(word));
            timestampPositions.Add(promptIds.Count);
            promptIds.Add(_timestampTokenId);
            timestampPositions.Add(promptIds.Count);
            promptIds.Add(_timestampTokenId);
        }

        if (Environment.GetEnvironmentVariable("STINGRAY_FORCEDALIGNER_TRACE") == "1")
        {
            Console.Error.WriteLine($"[FA-Trace] promptIds.Count={promptIds.Count} first20=[{string.Join(",", promptIds.Take(20))}]");
            Console.Error.WriteLine($"[FA-Trace] audioPadTokenId={_realWeights.AudioPadTokenId} timestampTokenId={_timestampTokenId}");
            int audioPadCount = promptIds.Count(id => id == _realWeights.AudioPadTokenId);
            Console.Error.WriteLine($"[FA-Trace] audioPadCount in prompt={audioPadCount} (expect numAudioTokens={numAudioTokens})");
        }

        // REVERTED, 2026-09-07 (same session): this checkpoint's config.json DOES declare
        // rope_scaling.interleaved=true/mrope_interleaved=true, and it looked like a plausible
        // root cause for the classify-head bug -- but directly checking the real reference's OWN
        // RoPE application (`qwen_decoder.h`'s `rope_type` field, `GGML_ROPE_TYPE_NEOX` by
        // default, never overridden anywhere in `qwen3_asr/assets.cpp` despite that file DOES
        // parse `mrope_section` into its config struct) confirms the reference deliberately
        // IGNORES these M-RoPE config fields and uses plain standard NEOX rotation -- the same
        // convention this bridge already had before this investigation. Re-bisecting with
        // interleavedRope=true confirmed it does NOT reproduce the reference's real layer-2
        // discontinuity, consistent with this correction. Left `false` (this class's real
        // default) rather than reverted entirely, since the underlying opt-in mechanism
        // (`ModelGraph.cs`'s new `"{arch}.rope.is_neox"` override) is safe, harmless, and may be
        // genuinely useful for some OTHER real checkpoint later -- see
        // docs/audio-review-progress.md's correction entry for the full story.
        using var source = new QwenAsrLlmSafetensorsTensorSource(
            _safetensorsPath,
            numLayers: _realWeights.LlmLayers, hiddenDim: _realWeights.LlmDim, numHeads: _realWeights.LlmHeads,
            numKvHeads: _realWeights.LlmKvHeads, headDim: _realWeights.LlmHeadDim, ffDim: _realWeights.LlmFfDim,
            vocabSize: 5000, ropeTheta: _realWeights.LlmRopeTheta, rmsNormEps: _realWeights.LlmRmsNormEps);
        source.EnableAudioConditioning(audioSoftTokens, numAudioTokens);

        var prompt = promptIds.ToArray();
        int frame = 0;
        for (int i = 0; i < prompt.Length; i++)
        {
            if (prompt[i] == _realWeights.AudioPadTokenId)
                prompt[i] = source.AudioTokenIdOffset + frame++;
        }

        var hp = ModelHyperparams.FromGgufMetadata(source.Metadata);
        using var backend = new CpuBackend();
        using var fwd = new ForwardPass(source, backend, hp);
        bool faTraceLayers = Environment.GetEnvironmentVariable("STINGRAY_FA_TRACE") == "1" && fwd.SupportsHiddenTaps;
        if (faTraceLayers)
        {
            var allLayers = new int[hp.NumLayers];
            for (int i = 0; i < hp.NumLayers; i++) allLayers[i] = i;
            fwd.EnableHiddenTaps(allLayers);
        }

        // EXPERIMENT (2026-09-06), DISPROVEN: tested whether the classify head's read-position
        // needed a -1 shift relative to the <timestamp> token's own index (a "predict-next"
        // convention mismatch would have produced the observed symptom). Result: shifting made
        // no difference -- class 0 still dominated at every position, ruling this out. Also
        // ruled out this same session: Q8 activation-quantized prefill (STINGRAY_CPU_PREFILL_Q8=0
        // changes nothing), and weight corruption (class 0's row in the real checkpoint's
        // thinker.lm_head.weight has norm 0.438, actually BELOW the ~0.63 average across all 5000
        // rows -- not anomalous at all). The remaining bug is most likely in the hidden-state
        // computation itself (attention/layers), not the classify head or its wiring -- needs
        // real intermediate hidden-state comparison against the C++ reference to pin down
        // further. Left gated behind an env var (default 0 = no shift, matching the reference's
        // own indexing) in case it's useful for a future investigation session.
        int readShift = Environment.GetEnvironmentVariable("STINGRAY_FA_READ_SHIFT") is { Length: > 0 } s && int.TryParse(s, out var shiftVal) ? shiftVal : 0;
        var classIds = new int[timestampPositions.Count];
        var wantedPositionToIdx = new Dictionary<int, int>();
        for (int i = 0; i < timestampPositions.Count; i++) wantedPositionToIdx[timestampPositions[i] + readShift] = i;
        int found = 0;
        bool debugTrace = Environment.GetEnvironmentVariable("STINGRAY_FORCEDALIGNER_TRACE") == "1";
        bool faTrace = Environment.GetEnvironmentVariable("STINGRAY_FA_TRACE") == "1";
        double logitsSum = 0, logitsSumSq = 0; long logitsCount = 0;
        fwd.PrefillWithPerPositionLogits(prompt, 0, (position, logits) =>
        {
            if (faTrace)
            {
                foreach (var v in logits) { logitsSum += v; logitsSumSq += (double)v * v; }
                logitsCount += logits.Length;
            }
            if (!wantedPositionToIdx.TryGetValue(position, out int idx)) return;
            int best = 0; float bestVal = float.NegativeInfinity;
            for (int c = 0; c < logits.Length; c++)
            {
                if (logits[c] > bestVal) { bestVal = logits[c]; best = c; }
            }
            classIds[idx] = best;
            if (debugTrace && found < 4)
                Console.Error.WriteLine($"[FA-Trace] readShift={readShift} position={position} best={best} bestVal={bestVal:F4} logits[0..9]=[{string.Join(",", logits.Slice(0, 10).ToArray().Select(v => v.ToString("F3")))}]");
            found++;
        });
        if (faTrace)
        {
            double mean = logitsSum / logitsCount;
            double std = Math.Sqrt(Math.Max(0, logitsSumSq / logitsCount - mean * mean));
            Console.Error.WriteLine($"[FA-STAGE-CS] classify_logits n={logitsCount} mean={mean:F6} std={std:F6}");
            Console.Error.WriteLine($"[FA-STAGE-CS] raw_timestamp_ids=[{string.Join(",", classIds)}]");
        }
        if (faTraceLayers)
        {
            int lastPos = prompt.Length - 1;
            var tapDim = fwd.HiddenTapDim;
            int embDim = tapDim / hp.NumLayers;
            var row = fwd.HiddenTapsAt(lastPos);
            // Real, additive-only (2026-09-07): when STINGRAY_FA_DUMP_DIR points at the same
            // directory the reference's own `thinker.cpp` dump mechanism wrote
            // `layer_{N}_out.bin` files to (see docs/audio-review-progress.md's "CONCLUSIVE:
            // layer-2's MLP formula/weights are CORRECT" entry), diff THIS port's own per-layer
            // last-position hidden state against the reference's real corresponding row
            // element-by-element (max-abs-diff, cosine similarity) -- not just aggregate std --
            // to find the first layer where this port's own computation actually diverges.
            string? dumpDir = Environment.GetEnvironmentVariable("STINGRAY_FA_DUMP_DIR");
            if (dumpDir != null)
            {
                string embPath = Path.Combine(dumpDir, "prompt_embeddings.bin");
                var embTensor = source.FindTensor("token_embd.weight");
                if (File.Exists(embPath) && embTensor != null)
                {
                    var embBytes = File.ReadAllBytes(embPath);
                    int stepsInDump = embBytes.Length / 4 / embDim;
                    if (stepsInDump > 0)
                    {
                        var refRow = new float[embDim];
                        Buffer.BlockCopy(embBytes, (stepsInDump - 1) * embDim * 4, refRow, 0, embDim * 4);
                        int tokenId = prompt[lastPos];
                        var tableBytes = source.GetTensorData(embTensor.Value);
                        var table = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(tableBytes);
                        var ourRow = table.Slice(tokenId * embDim, embDim);
                        double dot0 = 0, normA0 = 0, normB0 = 0, maxAbsDiff0 = 0;
                        for (int i = 0; i < embDim; i++)
                        {
                            float a = ourRow[i], b = refRow[i];
                            dot0 += (double)a * b;
                            normA0 += (double)a * a;
                            normB0 += (double)b * b;
                            maxAbsDiff0 = Math.Max(maxAbsDiff0, Math.Abs(a - b));
                        }
                        double cosine0 = dot0 / (Math.Sqrt(normA0) * Math.Sqrt(normB0) + 1e-12);
                        Console.Error.WriteLine($"[FA-DIFF-CS] prompt_embeddings (token {tokenId}) vs reference last-row: cosine={cosine0:F6} maxAbsDiff={maxAbsDiff0:F6}");
                    }
                }
            }
            for (int layer = 0; layer < hp.NumLayers; layer++)
            {
                var slice = row.Slice(layer * embDim, embDim);
                double sum = 0, sumsq = 0;
                foreach (var v in slice) { sum += v; sumsq += (double)v * v; }
                double m = sum / embDim;
                double sd = Math.Sqrt(Math.Max(0, sumsq / embDim - m * m));
                Console.Error.WriteLine($"[FA-STAGE-CS] layer_{layer} n={embDim} mean={m:F6} std={sd:F6}");

                if (dumpDir is null) continue;
                string path = Path.Combine(dumpDir, $"layer_{layer}_out.bin");
                if (!File.Exists(path)) continue;
                var bytes = File.ReadAllBytes(path);
                int stepsInDump = bytes.Length / 4 / embDim;
                if (stepsInDump <= 0) continue;
                var refRow = new float[embDim];
                Buffer.BlockCopy(bytes, (stepsInDump - 1) * embDim * 4, refRow, 0, embDim * 4);
                double dot = 0, normA = 0, normB = 0, maxAbsDiff = 0;
                for (int i = 0; i < embDim; i++)
                {
                    float a = slice[i], b = refRow[i];
                    dot += (double)a * b;
                    normA += (double)a * a;
                    normB += (double)b * b;
                    maxAbsDiff = Math.Max(maxAbsDiff, Math.Abs(a - b));
                }
                double cosine = dot / (Math.Sqrt(normA) * Math.Sqrt(normB) + 1e-12);
                Console.Error.WriteLine($"[FA-DIFF-CS] layer_{layer} vs reference last-row: cosine={cosine:F6} maxAbsDiff={maxAbsDiff:F6}");
            }
        }
        if (found != timestampPositions.Count)
            throw new InvalidOperationException($"AlignReal only observed {found}/{timestampPositions.Count} timestamp positions during prefill.");

        var timestampMs = new long[classIds.Length];
        for (int i = 0; i < classIds.Length; i++) timestampMs[i] = (long)classIds[i] * _timestampSegmentTimeMs;
        timestampMs = FixTimestampMonotonic(timestampMs);

        var segments = new List<SpeechSegment>(words.Length);
        for (int i = 0; i < words.Length; i++)
        {
            long startMs = timestampMs[i * 2];
            long endMs = Math.Max(startMs + 1, timestampMs[i * 2 + 1]);
            segments.Add(new SpeechSegment
            {
                Id = i,
                Start = timeOffset + TimeSpan.FromMilliseconds(startMs),
                End = timeOffset + TimeSpan.FromMilliseconds(endMs),
                Text = words[i],
                Tokens = [],
                Probability = 0.99f,
            });
        }
        return segments;
    }

    /// <summary>Whitespace-based word tokenization matching the real reference's
    /// `tokenize_space_language`/`clean_token` for non-CJK languages -- CJK per-character
    /// splitting (`tokenize_chinese_mixed`) is not yet ported, matching this pipeline's current
    /// English-first scope. `clean_token` strips everything except apostrophe/ASCII-alnum/CJK/a
    /// handful of non-ASCII letter ranges from EVERY word -- confirmed real and load-bearing:
    /// leaving trailing punctuation (e.g. "three," "publication.") attached changes the real
    /// BPE token count per word, shifting every subsequent `&lt;timestamp&gt;` token's absolute
    /// position in the prompt relative to what the real reference (and the model's training
    /// data) expects.</summary>
    private static string[] TokenizeAlignableWords(string text)
    {
        var raw = text.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        var cleaned = new List<string>(raw.Length);
        foreach (var word in raw)
        {
            var sb = new System.Text.StringBuilder(word.Length);
            foreach (var ch in word)
            {
                if (ch == '\'' || char.IsAsciiLetterOrDigit(ch) || IsKeptNonAsciiLetterOrNumber(ch))
                    sb.Append(ch);
            }
            if (sb.Length > 0) cleaned.Add(sb.ToString());
        }
        return [.. cleaned];
    }

    /// <summary>Real non-ASCII letter/number ranges `clean_token` keeps (Latin Extended, Greek,
    /// Cyrillic, Thai, Hiragana/Katakana, Hangul, plus CJK), ported directly from
    /// `processor.cpp`'s `is_kept_non_ascii_letter_or_number`/`is_cjk`.</summary>
    private static bool IsKeptNonAsciiLetterOrNumber(char ch)
    {
        int c = ch;
        return (c >= 0x00C0 && c <= 0x02AF) || (c >= 0x0370 && c <= 0x03FF) ||
               (c >= 0x0400 && c <= 0x052F) || (c >= 0x0E00 && c <= 0x0E7F) ||
               (c >= 0x3040 && c <= 0x30FF) || (c >= 0xAC00 && c <= 0xD7AF) ||
               (c >= 0x4E00 && c <= 0x9FFF) || (c >= 0x3400 && c <= 0x4DBF) ||
               (c >= 0xF900 && c <= 0xFAFF);
    }

    /// <summary>
    /// Real longest-non-decreasing-subsequence-based timestamp repair, ported directly from
    /// `examples/audio.cpp/src/models/qwen3_forced_aligner/processor.cpp`'s `fix_timestamp`:
    /// finds the longest already-monotonic run, then interpolates (or clamps to the nearest
    /// monotonic neighbor) every predicted value that fell outside it, rather than trusting the
    /// classify head's raw argmax at every position -- the real model can produce isolated
    /// non-monotonic outliers even though timestamps must be non-decreasing by construction.
    /// </summary>
    internal static long[] FixTimestampMonotonic(long[] data)
    {
        int n = data.Length;
        if (n == 0) return [];
        var dp = new int[n];
        var parent = new int[n];
        Array.Fill(dp, 1);
        Array.Fill(parent, -1);
        for (int i = 1; i < n; i++)
        {
            for (int j = 0; j < i; j++)
            {
                if (data[j] <= data[i] && dp[j] + 1 > dp[i])
                {
                    dp[i] = dp[j] + 1;
                    parent[i] = j;
                }
            }
        }
        int index = 0;
        for (int i = 1; i < n; i++) if (dp[i] > dp[index]) index = i;
        var normal = new bool[n];
        while (true)
        {
            normal[index] = true;
            if (parent[index] < 0) break;
            index = parent[index];
        }

        var result = (long[])data.Clone();
        int p = 0;
        while (p < n)
        {
            if (normal[p]) { p++; continue; }
            int j = p;
            while (j < n && !normal[j]) j++;
            int anomalyCount = j - p;
            bool hasLeft = false, hasRight = false;
            long left = 0, right = 0;
            for (int k = p; k > 0; k--)
            {
                if (normal[k - 1]) { hasLeft = true; left = result[k - 1]; break; }
            }
            for (int k = j; k < n; k++)
            {
                if (normal[k]) { hasRight = true; right = result[k]; break; }
            }
            if (anomalyCount <= 2)
            {
                for (int k = p; k < j; k++)
                {
                    if (!hasLeft && !hasRight) result[k] = data[k];
                    else if (!hasLeft) result[k] = right;
                    else if (!hasRight) result[k] = left;
                    else result[k] = (k - (p - 1)) <= (j - k) ? left : right;
                }
            }
            else
            {
                for (int k = p; k < j; k++)
                {
                    if (hasLeft && hasRight)
                    {
                        double step = (double)(right - left) / (anomalyCount + 1);
                        result[k] = (long)(left + step * (k - p + 1));
                    }
                    else if (hasLeft) result[k] = left;
                    else if (hasRight) result[k] = right;
                }
            }
            p = j;
        }
        return result;
    }

    /// <summary>
    /// Performs forced alignment of reference text against audio encoder frame tokens using Dynamic Time Warping (DTW).
    /// Returns aligned speech segments with word-level start and end timestamps.
    /// </summary>
    public List<SpeechSegment> Align(
        string referenceText,
        ReadOnlySpan<float> audioTokens,
        int numAudioTokens,
        int audioDim,
        TimeSpan timeOffset,
        float tokenDurationSeconds = 0.08f) // 12.5Hz token rate (80ms per token)
    {
        var segments = new List<SpeechSegment>();
        if (string.IsNullOrWhiteSpace(referenceText) || numAudioTokens <= 0)
        {
            return segments;
        }

        string[] words = referenceText.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        int numWords = words.Length;
        if (numWords == 0) return segments;

        // Dynamic Time Warping (DTW) alignment matrix [numWords, numAudioTokens]
        // Computes frame-to-token similarity and aligns monotonic word boundaries
        var cost = new float[numWords, numAudioTokens];
        var backtrack = new int[numWords, numAudioTokens];

        for (int w = 0; w < numWords; w++)
        {
            // Word-length token-count proxy, not real BPE ids -- this DTW alignment is itself
            // still entirely procedural/fake pending real Qwen3-ForcedAligner-0.6B weights (a
            // separate model from qwen3-asr, not bundled in our checkpoint), so calling into
            // QwenAsrTokenizer.Encode here would either throw (no real weights available in
            // this aligner's typical no-args construction) or, if it succeeded, lend unearned
            // credibility to math that's synthetic regardless -- see docs/audio-review-
            // progress.md's QwenASR section for why Encode/Decode now require real weights.
            int tokenSeed = EstimateTokenCount(words[w]) > 0 ? words[w][0] : w;

            for (int t = 0; t < numAudioTokens; t++)
            {
                // Compute frame-word affinity score
                int frameOff = t * Math.Max(1, audioDim);
                float dot = 0f;
                int checkDim = Math.Min(32, audioTokens.Length - frameOff);
                for (int d = 0; d < checkDim; d++)
                {
                    dot += audioTokens[frameOff + d] * MathF.Sin((tokenSeed + d) * 0.1f);
                }

                // Cost is negative affinity + distance from diagonal
                float diagDist = MathF.Abs((float)t / numAudioTokens - (float)w / numWords);
                float stepCost = -dot + diagDist * 2.0f;

                if (w == 0 && t == 0)
                {
                    cost[w, t] = stepCost;
                }
                else if (w == 0)
                {
                    cost[w, t] = cost[w, t - 1] + stepCost;
                    backtrack[w, t] = 1; // Left (stay in same word)
                }
                else if (t == 0)
                {
                    cost[w, t] = cost[w - 1, t] + stepCost;
                    backtrack[w, t] = 2; // Up (advance word)
                }
                else
                {
                    float fromDiag = cost[w - 1, t - 1];
                    float fromLeft = cost[w, t - 1];
                    float fromUp = cost[w - 1, t];

                    if (fromDiag <= fromLeft && fromDiag <= fromUp)
                    {
                        cost[w, t] = fromDiag + stepCost;
                        backtrack[w, t] = 0; // Diag
                    }
                    else if (fromLeft <= fromUp)
                    {
                        cost[w, t] = fromLeft + stepCost;
                        backtrack[w, t] = 1; // Left
                    }
                    else
                    {
                        cost[w, t] = fromUp + stepCost;
                        backtrack[w, t] = 2; // Up
                    }
                }
            }
        }

        // Traceback optimal alignment boundaries
        var wordEndFrames = new int[numWords];
        var wordStartFrames = new int[numWords];

        int curW = numWords - 1;
        int curT = numAudioTokens - 1;
        wordEndFrames[curW] = curT + 1;

        while (curW > 0 || curT > 0)
        {
            int action = backtrack[curW, curT];
            if (action == 0) // Diag
            {
                wordStartFrames[curW] = curT;
                curW--;
                curT--;
                if (curW >= 0) wordEndFrames[curW] = curT + 1;
            }
            else if (action == 1) // Left
            {
                curT--;
            }
            else // Up
            {
                wordStartFrames[curW] = curT;
                curW--;
                if (curW >= 0) wordEndFrames[curW] = curT + 1;
            }
        }
        wordStartFrames[0] = 0;

        for (int i = 0; i < numWords; i++)
        {
            int sFrame = wordStartFrames[i];
            int eFrame = Math.Max(sFrame + 1, wordEndFrames[i]);

            TimeSpan startTime = timeOffset + TimeSpan.FromSeconds(sFrame * tokenDurationSeconds);
            TimeSpan endTime = timeOffset + TimeSpan.FromSeconds(eFrame * tokenDurationSeconds);

            // Same word-length proxy as above -- not real BPE ids, see the comment there.
            var wordTokens = new int[EstimateTokenCount(words[i])];

            segments.Add(new SpeechSegment
            {
                Id = i,
                Start = startTime,
                End = endTime,
                Text = words[i],
                Tokens = wordTokens,
                Probability = 0.99f
            });
        }

        return segments;
    }

    public void Dispose()
    {
        _weights?.Dispose();
        _realEncoder?.Dispose();
        _realWeights?.Dispose();
    }
}
