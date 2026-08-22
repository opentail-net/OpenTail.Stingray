using System.Threading.Tasks;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Audio.Parler;

/// <summary>
/// Full Parler-TTS text-to-speech pipeline: text -&gt; real <see cref="UnigramTokenizer"/> -&gt;
/// real <see cref="T5Encoder"/> -&gt; real delayed multi-codebook autoregressive generation
/// (<see cref="ParlerDecoder.ForwardStep"/> with <see cref="ParlerDecoderKvCache"/>,
/// <see cref="ParlerDelayPattern"/>, <see cref="ParlerLogitsProcessor"/>) -&gt; real
/// <see cref="DacDecoder"/> -&gt; mono float32 PCM. Assembles this fire's three newly
/// golden-verified generation-loop primitives (KV cache, delay pattern, EOS logits processor)
/// together with the earlier golden-verified single-forward components. See
/// docs/audio-review-progress.md's Parler-TTS section for the full derivation of every piece.
///
/// <para><b>Real config, confirmed from the actual `parler-tts/parler-tts-mini-v1`
/// `generation_config.json`/`config.json` on Hugging Face (fetched this fire, not guessed)</b>:
/// `bos_token_id=1025` (== `decoder_start_token_id`), `eos_token_id=pad_token_id=1024`,
/// `num_codebooks=9`, `min_new_tokens=10`, real default `do_sample=True` (no fixed
/// temperature/top_k/top_p recorded in the checkpoint's own generation_config -- this pipeline
/// uses GREEDY decode instead, as an explicit first-pass simplification consistent with every
/// other pipeline in this codebase, not a silent approximation).</para>
///
/// <para><b>Per-step mask application, not separately verified against a fresh oracle but
/// directly implied by the real `apply_delay_pattern_mask`'s own doc comment ("only preserving
/// predictions where the mask is set to -1, and otherwise setting to the value detailed in the
/// mask")</b>: since every BOS/PAD position in the delay pattern is fully known ahead of
/// generation (computed once via <see cref="ParlerDelayPattern.Build"/>), this pipeline applies
/// the mask PER STEP (forcing the known BOS/PAD value into that step's model input immediately)
/// rather than only as a post-hoc cleanup after the whole sequence is generated -- mathematically
/// identical, since a forced value never depends on anything the model predicts, just applied
/// earlier. Reuses the already golden-tested <see cref="ParlerDelayPattern.Apply"/> exactly.</para>
/// </summary>
public sealed class ParlerFullPipeline : IDisposable
{
    private const int BosTokenId = 1025;
    private const int EosTokenId = 1024;
    private const int PadTokenId = 1024;
    private const int NumCodebooks = ParlerDecoderWeights.NumCodebooks;

    private readonly UnigramTokenizer _tokenizer;
    private readonly T5EncoderWeights _t5Weights;
    private readonly ParlerDecoderWeights _decoderWeights;
    private readonly DacWeights _dacWeights;

    public ParlerFullPipeline(string tokenizerJsonPath, SafetensorsLoader loader)
    {
        _tokenizer = UnigramTokenizer.FromTokenizerJson(tokenizerJsonPath);
        _t5Weights = new T5EncoderWeights(loader);
        _decoderWeights = new ParlerDecoderWeights(loader);
        _dacWeights = new DacWeights(loader);
    }

    /// <summary>
    /// Mixed-source constructor: T5 encoder and DAC codec still come from the real Safetensors
    /// checkpoint (unchanged, already golden-verified), but the decoder loads from a real
    /// community GGUF conversion instead (<see cref="ParlerDecoderWeights(Core.GgufModel)"/>) --
    /// see that constructor's doc comment for the full derivation, including the real, confirmed
    /// limitation that this specific GGUF conversion has NO T5 tensors at all (decoder+DAC only),
    /// which is exactly why this constructor still needs the Safetensors `loader` for T5/DAC.
    /// Golden-verified this session (`ParlerDecoderGgufTests`) that the GGUF-loaded decoder
    /// matches the Safetensors-loaded decoder's real output at cosine &gt; 0.99 -- see
    /// docs/audio-review-progress.md's GGUF-expansion entries.
    /// </summary>
    public ParlerFullPipeline(string tokenizerJsonPath, SafetensorsLoader loader, GgufModel decoderGguf)
    {
        _tokenizer = UnigramTokenizer.FromTokenizerJson(tokenizerJsonPath);
        _t5Weights = new T5EncoderWeights(loader);
        _decoderWeights = new ParlerDecoderWeights(decoderGguf);
        _dacWeights = new DacWeights(loader);
    }

    /// <summary>Full pipeline: text -&gt; mono float32 PCM. Real T5 EOS (id 1) appended to the tokenizer's segmentation-only output, matching the real T5 tokenizer's post-processor.</summary>
    public float[] Synthesize(string text, int maxNewTokens = 300, int minNewTokens = 10)
    {
        var tokenIds = _tokenizer.Encode(text);
        tokenIds.Add(1); // real T5 EOS id, appended by the real tokenizer's post-processor (see UnigramTokenizerTests)
        var encoderHidden = T5Encoder.Forward(_t5Weights, [.. tokenIds]);

        int maxLength = 1 + maxNewTokens;
        var initialPerCodebook = new int[NumCodebooks][];
        for (int cb = 0; cb < NumCodebooks; cb++) initialPerCodebook[cb] = [BosTokenId];

        var (initialInput, patternMask) = ParlerDelayPattern.Build(initialPerCodebook, BosTokenId, PadTokenId, maxLength, NumCodebooks);

        var sequence = new int[NumCodebooks][];
        for (int cb = 0; cb < NumCodebooks; cb++) sequence[cb] = [.. initialInput[cb]];

        var cache = new ParlerDecoderKvCache(ParlerDecoderWeights.NumLayers);
        var logitsProcessor = new ParlerLogitsProcessor(EosTokenId, NumCodebooks);

        // Feed every already-known initial position (real "first_start_id" prefix) through the KV cache.
        int t0 = sequence[0].Length;
        float[] hidden = [];
        for (int pos = 0; pos < t0; pos++)
        {
            var ids = new int[NumCodebooks];
            for (int cb = 0; cb < NumCodebooks; cb++) ids[cb] = sequence[cb][pos];
            var embed = ParlerDecoder.EmbedStep(_decoderWeights, ids, pos);
            hidden = ParlerDecoder.ForwardStep(_decoderWeights, cache, embed, encoderHidden);
        }

        for (int step = 0; step < maxNewTokens; step++)
        {
            int pos = t0 + step;
            if (pos >= maxLength) break;

            // 9 independent lm_head projections off the same shared hidden state -- real
            // architecture has no data dependency between them, so run them in parallel (measured
            // this session's performance pass; each is a full HiddenDim(1024)->OutputVocabSize(1088)
            // matvec, non-trivial work to leave serialized on a 12-core box).
            var logitsPerCodebook = new float[NumCodebooks][];
            Parallel.For(0, NumCodebooks, cb =>
                logitsPerCodebook[cb] = LinearNoBias(hidden, _decoderWeights.LmHeads[cb], ParlerDecoderWeights.HiddenDim, ParlerDecoderWeights.OutputVocabSize));

            if (step >= minNewTokens)
            {
                var history = new int[NumCodebooks][];
                for (int cb = 0; cb < NumCodebooks; cb++) history[cb] = sequence[cb];
                logitsProcessor.Apply(history, logitsPerCodebook);
            }

            var predicted = new int[NumCodebooks];
            for (int cb = 0; cb < NumCodebooks; cb++) predicted[cb] = Argmax(logitsPerCodebook[cb]);

            // Real "only preserve predictions where the mask is -1" -- force the known BOS/PAD value where the pattern already knows it.
            var maskAtPos = new int[NumCodebooks];
            for (int cb = 0; cb < NumCodebooks; cb++) maskAtPos[cb] = pos < patternMask[cb].Length ? patternMask[cb][pos] : PadTokenId;
            var forced = ParlerDelayPattern.Apply(Wrap(predicted), Wrap(maskAtPos));

            for (int cb = 0; cb < NumCodebooks; cb++) sequence[cb] = [.. sequence[cb], forced[cb][0]];

            bool allEos = true;
            for (int cb = 0; cb < NumCodebooks; cb++)
                if (sequence[cb][^1] != EosTokenId) { allEos = false; break; }
            if (step >= minNewTokens && allEos) break;

            var nextIds = new int[NumCodebooks];
            for (int cb = 0; cb < NumCodebooks; cb++) nextIds[cb] = sequence[cb][^1];
            var nextEmbed = ParlerDecoder.EmbedStep(_decoderWeights, nextIds, pos + 1);
            hidden = ParlerDecoder.ForwardStep(_decoderWeights, cache, nextEmbed, encoderHidden);
        }

        // Real un-delay: drop BOS/PAD-only tail/head per codebook offset, keep only genuinely generated audio codes.
        var unDelayed = UnDelay(sequence);
        if (unDelayed[0].Length == 0) return [];
        return DacDecoder.Decode(_dacWeights, unDelayed);
    }

    /// <summary>Strips each codebook's real BOS-offset prefix and any trailing EOS/PAD, then truncates every stream to the shortest resulting length (the real frame count all 9 codebooks agree on).</summary>
    private static int[][] UnDelay(int[][] sequence)
    {
        var stripped = new int[NumCodebooks][];
        int minLen = int.MaxValue;
        for (int cb = 0; cb < NumCodebooks; cb++)
        {
            var row = sequence[cb];
            int start = cb + 1; // skip this codebook's own BOS-offset prefix (cb+1 BOS tokens precede its real content)
            int end = row.Length;
            while (end > start && (row[end - 1] == EosTokenId || row[end - 1] == PadTokenId)) end--;
            int len = Math.Max(0, end - start);
            stripped[cb] = len > 0 ? row[start..end] : [];
            minLen = Math.Min(minLen, len);
        }
        if (minLen <= 0 || minLen == int.MaxValue) return [[], [], [], [], [], [], [], [], []];
        var result = new int[NumCodebooks][];
        for (int cb = 0; cb < NumCodebooks; cb++) result[cb] = stripped[cb][..minLen];
        return result;
    }

    private static int[][] Wrap(int[] row) => [[row[0]], [row[1]], [row[2]], [row[3]], [row[4]], [row[5]], [row[6]], [row[7]], [row[8]]];

    private static int Argmax(float[] logits)
    {
        int idx = 0;
        float max = logits[0];
        for (int i = 1; i < logits.Length; i++) if (logits[i] > max) { max = logits[i]; idx = i; }
        return idx;
    }

    private static unsafe float[] LinearNoBias(float[] input, float[] weight, int inDim, int outDim)
    {
        var output = new float[outDim];
        fixed (float* wp = weight, xp = input, op = output)
        {
            Cpu.SimdKernels.MatVecF32(op, wp, xp, outDim, inDim);
        }
        return output;
    }

    public void Dispose()
    {
        // Weight loaders here own no unmanaged resources of their own beyond the shared SafetensorsLoader,
        // which the caller owns and disposes (see ParlerFullPipelineTests for the real usage pattern).
    }
}
