using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Audio.QwenASR;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

public sealed class QwenAsrTests
{
    [Fact]
    public void QwenAsrMelExtractor_ExtractMel_Produces128ChannelLogMelSpectrogram()
    {
        var extractor = new QwenAsrMelExtractor();
        int sampleRate = 16000;
        int numSamples = sampleRate * 1; // 1 second
        var pcm = new float[numSamples];

        for (int i = 0; i < numSamples; i++)
        {
            pcm[i] = 0.5f * MathF.Sin(2.0f * MathF.PI * 500.0f * i / sampleRate);
        }

        float[] mel = extractor.ExtractMel(pcm);

        Assert.NotNull(mel);
        Assert.NotEmpty(mel);
        Assert.Equal(0, mel.Length % QwenAsrMelExtractor.NumMels);

        int numFrames = mel.Length / QwenAsrMelExtractor.NumMels;
        Assert.True(numFrames >= 90);

        for (int i = 0; i < mel.Length; i++)
        {
            Assert.False(float.IsNaN(mel[i]), $"NaN at mel index {i}");
            Assert.False(float.IsInfinity(mel[i]), $"Infinity at mel index {i}");
        }
    }

    [Fact]
    public void QwenAsrTokenizer_FormatPrompt_ProducesChatMlPromptWithRealSpecialAudioTokens()
    {
        // Structural-only: real BPE Encode()/Decode() need the checkpoint's real tokenizer
        // (QwenAsrTokenizer(QwenAsrWeights)) -- see Tests.Audio/QwenAsrTokenizerTests.cs for
        // real-weights encode/decode coverage. This only checks prompt string assembly, which
        // doesn't need real weights.
        var tokenizer = new QwenAsrTokenizer();
        string prompt = tokenizer.FormatPrompt(numAudioTokens: 3, language: "en", taskInstruction: "Transcribe the audio speech.");

        Assert.Contains("<|im_start|>", prompt, StringComparison.Ordinal);
        Assert.Contains("<|audio_start|><|audio_pad|><|audio_pad|><|audio_pad|><|audio_end|>", prompt, StringComparison.Ordinal);
        Assert.Contains("Language: en", prompt, StringComparison.Ordinal);
    }

    // QwenAsrAudioEncoder_Forward_AppliesConv2dStemAndWindowedAttention removed: the AuT
    // encoder is now a real, weight-driven port (no procedural fast-path constructor exists
    // anymore, same policy as Parakeet's encoder -- see docs/audio-review-progress.md's
    // QwenASR section). Real coverage lives in Tests.Audio/QwenAsrAudioEncoderTests.cs
    // (HeavyTestBase, real GGUF weights).

    [Fact]
    public void QwenAsrForcedAligner_Align_ProducesWordLevelTimestamps()
    {
        using var aligner = new QwenAsrForcedAligner();
        string reference = "open source speech recognition with qwen audio";
        int numAudioTokens = 32;
        var audioTokens = new float[numAudioTokens * 512];

        var segments = aligner.Align(
            referenceText: reference,
            audioTokens: audioTokens,
            numAudioTokens: numAudioTokens,
            audioDim: 512,
            timeOffset: TimeSpan.Zero);

        Assert.NotNull(segments);
        Assert.Equal(7, segments.Count); // 7 words

        for (int i = 0; i < segments.Count; i++)
        {
            Assert.True(segments[i].End > segments[i].Start);
            Assert.NotEmpty(segments[i].Text);
        }
    }

    // QwenAsrPipeline_Transcribe end-to-end coverage removed: QwenAsrTokenizer's real
    // Encode/Decode now require the checkpoint's real BPE tokenizer (no fake fallback, same
    // policy as Parakeet's pipeline -- see docs/audio-review-progress.md's QwenASR section),
    // and QwenAsrAudioEncoder is still unported. Real end-to-end coverage will land once both
    // the AuT encoder and the LLM decoder (via QwenAsrLlmTensorSource, see
    // Tests.Audio/QwenAsrLlmTensorSourceTests.cs) are wired together in QwenAsrPipeline.
}
