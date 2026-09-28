using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.MiniMaxMusic3;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

// Scratch (untracked): MiniMax vocoder old-vs-new. Env MMV_OUT (raw f32 output path), MMV_REF (compare to).
public sealed class ZzMiniMaxVocTmp
{
    [Fact]
    public void Run()
    {
        string? outPath = Environment.GetEnvironmentVariable("MMV_OUT");
        if (outPath is null) return;
        using var loader = SafetensorsLoader.Open(@"C:\Git-Public\OpenTail.Stingray\models\minimax-music3\vocoder.safetensors");
        var w = MiniMaxMusic3VocoderWeights.Load(loader);
        int len = 200; // ~8 s at hop 512 / 12.5? latent frames
        var rng = new Random(3);
        var latent = new float[MiniMaxMusic3Config.VocoderLatentChannels * len];
        for (int i = 0; i < latent.Length; i++) latent[i] = (float)(Math.Sqrt(-2 * Math.Log(1 - rng.NextDouble())) * Math.Cos(2 * Math.PI * rng.NextDouble())) * 0.5f;
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var pcm = MiniMaxMusic3Vocoder.Decode(w, latent, len);
        Console.WriteLine($"[ZzMMV] decode {sw.Elapsed.TotalMilliseconds:F0} ms, {pcm.Length} samples");
        var bytes = new byte[pcm.Length * 4]; Buffer.BlockCopy(pcm, 0, bytes, 0, bytes.Length); File.WriteAllBytes(outPath, bytes);
        if (Environment.GetEnvironmentVariable("MMV_REF") is { Length: > 0 } refPath)
        {
            var rb = File.ReadAllBytes(refPath); var r = new float[rb.Length / 4]; Buffer.BlockCopy(rb, 0, r, 0, rb.Length);
            double d = 0, a2 = 0, b2 = 0, mx = 0;
            for (int i = 0; i < r.Length; i++) { d += (double)pcm[i] * r[i]; a2 += (double)pcm[i] * pcm[i]; b2 += (double)r[i] * r[i]; mx = Math.Max(mx, Math.Abs(pcm[i] - r[i])); }
            Console.WriteLine($"[ZzMMV] vs ref cos {d / Math.Sqrt(a2 * b2):F9} maxAbsDiff {mx:E3} rms {Math.Sqrt(b2 / r.Length):F4}");
        }
    }
}
