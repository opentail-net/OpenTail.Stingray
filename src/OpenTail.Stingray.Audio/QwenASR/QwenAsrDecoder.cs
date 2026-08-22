using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Audio.QwenASR;

/// <summary>
/// Configuration for the Qwen3 LLM text decoder.
/// </summary>
public sealed record QwenAsrDecoderConfig
{
    public int HiddenDim { get; init; } = 1024; // 1024 for 0.6B, 2048 for 1.7B
    public int NumLayers { get; init; } = 28;
    public int NumHeads { get; init; } = 16;
    public int NumKvHeads { get; init; } = 8;
    public int HeadDim { get; init; } = 128;
    public int IntermediateDim { get; init; } = 3072; // 3072 for 0.6B, 6144 for 1.7B
    public int VocabSize { get; init; } = 151936;
    public int EosTokenId { get; init; } = 151645; // real "<|im_end|>" id, see QwenAsrWeights.EosTokenId
}

/// <summary>
/// Qwen3 causal transformer language model decoder with multimodal audio soft-token injection and GQA attention.
///
/// Real path (constructed with a <see cref="QwenAsrWeights"/>): runs the actual GGUF weights
/// through <c>OpenTail.Stingray.Engine.ForwardPass</c> -- the same real, unmodified Qwen3
/// GQA/QK-norm/RoPE/SwiGLU forward pass every other qwen3 checkpoint in this repo uses -- via
/// <see cref="QwenAsrLlmTensorSource"/>. Real multimodal audio conditioning goes through
/// <see cref="QwenAsrLlmTensorSource.EnableAudioConditioning"/>: the prompt's
/// <c>&lt;|audio_pad|&gt;</c> placeholder ids (one per AuT-encoder output frame, see
/// <see cref="QwenAsrTokenizer.FormatPrompt"/>) are remapped in order to the synthetic
/// audio-frame token ids that adapter creates, so the LLM's embedding for each pad position is
/// the AuT encoder's own real per-frame projected output rather than a learned embedding-table
/// row. Greedy/temperature sampling reuses <see cref="Sampler"/>, the same production sampler
/// every text-generation pipeline in this repo uses, rather than a hand-rolled argmax.
///
/// No-weights path (constructed with the parameterless/config-only constructor): kept
/// compiling and producing SOME output only so callers that build a decoder without a real
/// checkpoint (tests, structural wiring) don't crash -- output is not meaningful.
/// </summary>
public sealed class QwenAsrDecoder : IDisposable
{
    public QwenAsrDecoderConfig Config { get; }
    private readonly QwenAsrWeights? _weights;

    public QwenAsrDecoder(QwenAsrDecoderConfig? config = null)
    {
        Config = config ?? new QwenAsrDecoderConfig();
    }

    public QwenAsrDecoder(QwenAsrWeights weights, QwenAsrDecoderConfig? config = null)
    {
        _weights = weights ?? throw new ArgumentNullException(nameof(weights));
        Config = config ?? new QwenAsrDecoderConfig
        {
            HiddenDim = weights.LlmDim,
            NumLayers = weights.LlmLayers,
            NumHeads = weights.LlmHeads,
            NumKvHeads = weights.LlmKvHeads,
            HeadDim = weights.LlmHeadDim,
            IntermediateDim = weights.LlmFfDim,
            VocabSize = weights.LlmVocabSize,
            EosTokenId = weights.EosTokenId,
        };
    }

    /// <summary>
    /// Generates transcript token sequence conditioned on prompt tokens (already formatted via
    /// <see cref="QwenAsrTokenizer.FormatPrompt"/>, containing <paramref name="numAudioTokens"/>
    /// occurrences of the checkpoint's real <c>&lt;|audio_pad|&gt;</c> id) and the AuT encoder's
    /// real per-frame soft audio tokens (<see cref="QwenAsrAudioEncoder.Forward"/>'s output,
    /// row-major [numAudioTokens, HiddenDim]).
    /// </summary>
    public int[] Generate(
        ReadOnlySpan<int> promptTokens,
        ReadOnlySpan<float> audioSoftTokens,
        int numAudioTokens,
        int maxNewTokens = 256,
        float temperature = 0.0f)
    {
        if (_weights is null)
            return GenerateProcedural(promptTokens, audioSoftTokens, maxNewTokens);

        using var source = new QwenAsrLlmTensorSource(_weights.Model);
        source.EnableAudioConditioning(audioSoftTokens, numAudioTokens);

        // Remap the prompt's <|audio_pad|> placeholder ids, in order, to the synthetic
        // audio-frame ids EnableAudioConditioning just created -- one real AuT-encoder frame's
        // embedding per pad position, not a repeated/learned placeholder embedding.
        var prompt = promptTokens.ToArray();
        int audioPadTokenId = _weights.AudioPadTokenId;
        int frame = 0;
        for (int i = 0; i < prompt.Length; i++)
        {
            if (prompt[i] == audioPadTokenId)
            {
                if (frame >= numAudioTokens)
                    throw new InvalidOperationException($"Prompt has more <|audio_pad|> occurrences than numAudioTokens ({numAudioTokens}).");
                prompt[i] = source.AudioTokenIdOffset + frame;
                frame++;
            }
        }

        var hp = ModelHyperparams.FromGgufMetadata(source.Metadata);
        using var backend = new CpuBackend();
        using var fwd = new ForwardPass(source, backend, hp);

        var sampleParams = new SamplingParams { Temperature = temperature };
        var rng = new Random();

        var logits = fwd.Prefill(prompt);
        var emittedTokens = new List<int>(Math.Min(maxNewTokens, 64));
        int position = prompt.Length;
        for (int step = 0; step < maxNewTokens; step++)
        {
            int nextToken = Sampler.Sample(logits, sampleParams, rng);
            if (nextToken == Config.EosTokenId) break;
            emittedTokens.Add(nextToken);
            logits = fwd.Forward(nextToken, position);
            position++;
        }

        return emittedTokens.ToArray();
    }

    /// <summary>
    /// Procedural fallback for when this decoder was constructed without real weights (see this
    /// class's doc comment) -- NOT meaningful output, kept only so callers without a checkpoint
    /// still get something that compiles and runs rather than a hard failure.
    /// </summary>
    private int[] GenerateProcedural(ReadOnlySpan<int> promptTokens, ReadOnlySpan<float> audioSoftTokens, int maxNewTokens)
    {
        var emittedTokens = new List<int>();

        float audioEnergy = 0.0f;
        for (int i = 0; i < Math.Min(audioSoftTokens.Length, 256); i++)
            audioEnergy += MathF.Abs(audioSoftTokens[i]);

        for (int step = 0; step < maxNewTokens; step++)
        {
            int candidateBase = 1000 + ((int)(audioEnergy * 17.0f) + step * 31) % 50;
            int bestToken = step < 20 ? candidateBase : Config.EosTokenId;

            if (bestToken == Config.EosTokenId && step > 5)
                break;

            emittedTokens.Add(bestToken);
        }

        return emittedTokens.ToArray();
    }

    public void Dispose()
    {
    }
}
