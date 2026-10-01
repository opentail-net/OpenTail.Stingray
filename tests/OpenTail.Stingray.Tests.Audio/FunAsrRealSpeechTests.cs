using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Audio.FunASR;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real-speech regression for the Paraformer GGUF pipeline (bugstofix item 07). The root causes were
/// (1) FSMN weight read as [C][k] while the GGUF stores [k][C], (2) the encoder input missing funasr's
/// <c>x * sqrt(512)</c> + sinusoidal position encoding, (3) <c>TensorPrimitives.SoftMax</c> overflowing to NaN.
/// The expected text is the ONNX Paraformer control's transcript of the same clip.</summary>
public sealed class FunAsrRealSpeechTests : HeavyTestBase
{
    private const string OnnxControlTranscript = "对我做了介绍啊那么我想说的是呢大家如果对我的研究感兴趣呢嗯";

    private static string? FindWav()
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 10; i++)
        {
            var p = Path.Combine(dir, "docs", "audio-samples", "paraformer-zh-test-0.wav");
            if (File.Exists(p)) return p;
            dir = Directory.GetParent(dir)?.FullName ?? dir;
        }
        return null;
    }

    [Fact]
    public void Paraformer_GgufRealSpeech_ProducesNonEmptyMandarinTranscript()
    {
        string? modelPath = FunAsrGgufTestModelLocator.FindParaformerModelPath();
        Assert.SkipUnless(modelPath is not null, "Paraformer GGUF (architecture=paraformer, pf.vocab present) not found.");
        string? wav = FindWav();
        Assert.SkipUnless(wav is not null, "docs/audio-samples/paraformer-zh-test-0.wav not found.");

        var (pcm, sr, _) = WavReader.ReadWav(wav!);
        using var pipeline = FunAsrPipeline.Load(modelPath!);
        var result = pipeline.Transcribe(new SpeechToTextRequest { AudioSamples = pcm, SampleRate = sr, Language = "zh" });

        Console.WriteLine($"[paraformer] text={result.Text}");
        Assert.False(string.IsNullOrWhiteSpace(result.Text));
        Assert.Single(result.Segments);
        Assert.True(result.Segments[0].Tokens.Length > 10);
        Assert.Equal(OnnxControlTranscript, result.Text);
    }
}
