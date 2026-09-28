using OpenTail.Stingray.Audio.MusicGen;
using OpenTail.Stingray.Core;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

// Scratch (untracked): hash MusicGen greedy output for old/new A/B. Env AG_HASH=1.
public sealed class ZzMusicGenHashTmp
{
    [Fact]
    public void Run()
    {
        if (Environment.GetEnvironmentVariable("AG_HASH") != "1") return;
        string d = @"C:\Git-Public\OpenTail.Stingray\models\musicgen-small\";
        using var l = SafetensorsLoader.Open(d + "musicgen-small.safetensors");
        var gen = new MusicGenGenerator(MusicGenTextEncoderWeights.Load(l), T5Tokenizer.FromFile(d + "t5-base-tokenizer.json"),
            new MusicGenTransformerWeights(l), MusicGenEncodecDecoderWeights.Load(l));
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var pcm = gen.Generate("acoustic guitar melody", durationSeconds: 3.0f, seed: 0, guidanceScale: 3.0f, topK: 1);
        var bytes = new byte[pcm.Length * 4]; Buffer.BlockCopy(pcm, 0, bytes, 0, bytes.Length);
        Console.WriteLine($"[ZzMG] {sw.Elapsed.TotalSeconds:F1}s len={pcm.Length} sha={Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(bytes))[..16]}");
    }
}
