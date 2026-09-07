using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Audio.HiggsAudio;

/// <summary>
/// Real, full zero-shot (no reference-audio conditioning) text-to-waveform generation loop for
/// Higgs Audio TTS, ported from `generator.cpp`'s `HiggsGenerator::generate` (not guessed):
/// prefill the real text prompt, sample the FIRST codes directly from the prefill's own last
/// hidden state, then repeatedly embed the previous (delay-masked) codes and step forward until
/// <see cref="HiggsCodebookSampler"/>'s real stop state machine signals done (or `maxTokens` is
/// hit), reverse the real delay pattern, clamp any still-reserved id (`BocId`/`EocId`) to `0`
/// (confirmed real: `generator.cpp` does exactly this, not an error), then decode through the
/// real acoustic codec.
///
/// <para><b>Real, deliberate scope limit</b>: this is the reference-audio-FREE path only (matches
/// this session's established "reference-audio-free first pass" convention) -- `generator.cpp`'s
/// real reference-audio conditioning (fused prompt positions carrying delayed reference codebook
/// ids instead of text tokens, KV-cache reuse across repeated calls with the same reference) is
/// real, separate, unstarted work, precisely scoped in this session's own progress doc entry
/// rather than guessed at here.</para>
/// </summary>
public static class HiggsGenerator
{
    public readonly struct Result(float[] audioSamples, int[][] rawCodes)
    {
        public float[] AudioSamples { get; } = audioSamples;
        public int[][] RawCodes { get; } = rawCodes;
    }

    public static Result Generate(
        IForwardPass fwd, HiggsLlmTensorSource llm, HiggsTtsTextTokenizer tokenizer,
        HiggsCodecDecoderWeights codecWeights,
        string text, int numCodebooks, int audioVocabSize,
        int maxTokens, SamplingParams? options = null, Random? rng = null)
    {
        var prompt = tokenizer.EncodePrompt(text, referenceText: "", delayedReferenceTokens: 0);
        fwd.Prefill(prompt.TokenIds);

        var sampler = new HiggsCodebookSampler(numCodebooks);
        var delayedFrames = new List<int[]>();

        var first = HiggsArStepper.SampleFromHidden(fwd.LastHidden, llm, numCodebooks, audioVocabSize, options, rng);
        var maskedFirst = sampler.Step(first);
        delayedFrames.Add(maskedFirst);

        int position = prompt.TokenIds.Length;
        while (!sampler.GenerationDone && delayedFrames.Count < maxTokens)
        {
            var raw = HiggsArStepper.Step(fwd, llm, sampler.LastCodes, position, numCodebooks, audioVocabSize, options, rng);
            position++;
            var masked = sampler.Step(raw);
            if (masked.Length > 0 && masked[0] != HiggsCodebookSampler.StopCode)
                delayedFrames.Add(masked);
        }
        if (!sampler.GenerationDone)
            throw new InvalidOperationException("Higgs TTS generation reached maxTokens before EOC.");

        int delayedFrameCount = delayedFrames.Count;
        var delayedFlat = new int[delayedFrameCount * numCodebooks];
        for (int t = 0; t < delayedFrameCount; t++)
            Array.Copy(delayedFrames[t], 0, delayedFlat, t * numCodebooks, numCodebooks);

        var rawFlat = HiggsCodebooks.ReverseDelayPattern(delayedFlat, delayedFrameCount, numCodebooks);
        int rawFrameCount = delayedFrameCount - (numCodebooks - 1);

        int codecVocab = audioVocabSize - 2; // real: excludes BocId/EocId, matches HiggsCodecDecoderWeights.CodebookSize
        for (int i = 0; i < rawFlat.Length; i++)
            if (rawFlat[i] >= codecVocab) rawFlat[i] = 0;

        var rawCodes = new int[rawFrameCount][];
        for (int t = 0; t < rawFrameCount; t++)
        {
            var row = new int[numCodebooks];
            Array.Copy(rawFlat, t * numCodebooks, row, 0, numCodebooks);
            rawCodes[t] = row;
        }

        var waveform = HiggsCodecDecoder.Decode(codecWeights, rawCodes);
        return new Result(waveform, rawCodes);
    }
}
