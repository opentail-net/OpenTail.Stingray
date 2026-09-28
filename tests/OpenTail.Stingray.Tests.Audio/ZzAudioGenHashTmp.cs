using OpenTail.Stingray.Audio.AudioGen;
using OpenTail.Stingray.Core;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

// Scratch (untracked): hash AudioGen greedy output for old/new A/B. Env AG_HASH=1.
public sealed class ZzAudioGenHashTmp
{
    [Fact]
    public void Run()
    {
        if (Environment.GetEnvironmentVariable("AG_HASH") != "1") return;
        string d = @"C:\Git-Public\OpenTail.Stingray\models\audiogen-medium\";
        using var lm = SafetensorsLoader.Open(d + "audiogen-medium-lm.safetensors");
        using var codec = SafetensorsLoader.Open(d + "audiogen-medium-encodec16k.safetensors");
        using var t5 = SafetensorsLoader.Open(d + "t5-large.safetensors");
        var gen = new AudioGenGenerator(AudioGenTextEncoderWeights.Load(t5), T5Tokenizer.FromFile(d + "t5-large-tokenizer.json"),
            new AudioGenTransformerWeights(lm), AudioGenEncodecDecoderWeights.Load(codec));
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var pcm = gen.Generate("dog barking", durationSeconds: 3.0f, seed: 0, guidanceScale: 3.0f, topK: 1);
        var bytes = new byte[pcm.Length * 4]; Buffer.BlockCopy(pcm, 0, bytes, 0, bytes.Length);
        Console.WriteLine($"[ZzAG] {sw.Elapsed.TotalSeconds:F1}s len={pcm.Length} sha={Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(bytes))[..16]}");
    }
}
