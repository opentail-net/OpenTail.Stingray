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
    private readonly int _codebookDim;

    public FishSpeechPipeline(string ggufPath, string tokenizerDir, int numLayers = 36, int ctxSize = 2048)
    {
        _model = GgufModel.Open(ggufPath);
        _tensorSource = new FishSpeechTensorSource(_model, numLayers);
        var hp = ModelHyperparams.FromGgufMetadata(_tensorSource.Metadata, _tensorSource);
        _backend = new CpuBackend();
        _fwd = new ForwardPass(_tensorSource, _backend, hp, maxContextLength: ctxSize);
        _fwd.EnableHiddenTaps([numLayers - 1]); // last layer's output = the trunk's post-trunk pre-final-norm hidden, what the real fast_decode conditions on
        _weights = new FishSpeechWeights(ggufPath);

        var tokResult = HuggingFaceTokenizerSource.Load(tokenizerDir);
        if (!tokResult.IsUsable || tokResult.Source is null)
            throw new InvalidOperationException("Failed to load Fish Speech tokenizer: " +
                string.Join("; ", tokResult.Rejections.Select(r => r.Detail)));
        _tokenizer = GgufTokenizer.FromSource(tokResult.Source);

        _imEndId = Single(_tokenizer.Encode("<|im_end|>"));
        _voiceId = Single(_tokenizer.Encode("<|voice|>"));
        _codebookDim = _model.Metadata.TryGetValue("fish_speech.codec.quantizer_codebook_dim", out var cd) ? Convert.ToInt32(cd) : 8;
    }

    private static int Single(IReadOnlyList<int> ids) =>
        ids.Count == 1 ? ids[0] : throw new InvalidOperationException($"expected a single token, got {ids.Count}");

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
            float scale = 1f / MathF.Sqrt(_codebookDim);
            for (int d = 0; d < _weights.EmbeddingDim; d++) emb[d] *= scale;
        }
        return emb;
    }

    /// <summary>
    /// Generates a sequence of real semantic tokens for the given text (greedy decode -- a
    /// deliberate first-pass simplification; the real reference's own sampler uses temperature/
    /// top_p/top_k plus a repetition-avoidance ("RAS") heuristic, not yet wired here). Does NOT
    /// run the fast-AR codebook expansion or the codec -- returns raw semantic token ids
    /// (offset by SemanticBeginId already subtracted), not audio.
    /// </summary>
    public List<int> GenerateSemanticTokens(string text, int maxTokens = 200)
    {
        var prompt = BuildPrompt(text);
        _fwd.ResetCache();

        ReadOnlySpan<float> logits = default;
        int pos = 0;
        foreach (var tok in prompt)
        {
            logits = _fwd.ForwardEmbedding(EmbedTextToken(tok), pos);
            pos++;
        }

        int semBegin = _weights.SemanticBeginId;
        int semEnd = _weights.SemanticEndId;
        var semanticTokens = new List<int>();
        var codebooksPerFrame = new List<int[]>();

        int mainToken = ArgmaxMasked(logits, semBegin, semEnd, _imEndId);
        float[] hidden = _fwd.HiddenTapsAt(pos - 1).ToArray();

        for (int step = 0; step < maxTokens && mainToken != _imEndId; step++)
        {
            int semCode = Math.Clamp(mainToken - semBegin, 0, _weights.CodebookSize - 1);
            semanticTokens.Add(semCode);

            // Real fast-AR codebook expansion: sample codebooks 1..NumCodebooks-1 one at a time,
            // each conditioned on the slow-AR's own hidden state for THIS timestep plus the
            // codebook values already decided so far this timestep -- matches s2_generate.cpp's
            // real per-timestep loop exactly (see FishSpeechFastAr's doc comment).
            var codebookValues = new int[_weights.NumCodebooks];
            codebookValues[0] = semCode;
            var prefix = new List<int>();
            for (int cb = 1; cb < _weights.NumCodebooks; cb++)
            {
                var cbLogits = FishSpeechFastAr.Forward(_weights, hidden, prefix.ToArray());
                int cbToken = Argmax(cbLogits);
                codebookValues[cb] = cbToken;
                prefix.Add(cbToken);
            }
            codebooksPerFrame.Add(codebookValues);

            var emb = EmbedSemanticToken(mainToken, codebookValues);
            logits = _fwd.ForwardEmbedding(emb, pos);
            pos++;
            hidden = _fwd.HiddenTapsAt(pos - 1).ToArray();

            mainToken = ArgmaxMasked(logits, semBegin, semEnd, _imEndId);
        }

        return semanticTokens;
    }

    private static int Argmax(ReadOnlySpan<float> logits)
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

    public void Dispose()
    {
        _fwd.Dispose();
        _backend.Dispose();
        _weights.Dispose();
        _model.Dispose();
    }
}
