using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.AceStep;
using OpenTail.Stingray.Diffusion.AceStep.Conditioning;
using OpenTail.Stingray.Diffusion.AceStep.Text;
using OpenTail.Stingray.Diffusion.AceStep.Transformer;
using OpenTail.Stingray.Diffusion.AceStep.Vae;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

// Scratch (untracked): ACE-Step vs audio.cpp. Env: ACE_OUT (wav path), ACE_PROMPT, ACE_SECONDS, ACE_SEED,
// ACE_REF (audio.cpp wav to compare), ACE_TE (text-encoder gguf), plus STINGRAY_ACESTEP_NOISE/_DUMP.
public sealed class ZzAceStepCmpTmp
{
    [Fact]
    public void Run()
    {
        string? outPath = Environment.GetEnvironmentVariable("ACE_OUT");
        if (outPath is null) return;
        string root = @"C:\Git-Public\OpenTail.Stingray\models\";
        if (Environment.GetEnvironmentVariable("ACE_DECODE_LATENT") is { Length: > 0 } latPath)
        {
            using var vl = SafetensorsLoader.Open(Path.Combine(root, "acestep-v15", "vae.safetensors"));
            var vw = AceStepOobleckDecoderWeights.Load(vl);
            var lat = new float[new FileInfo(latPath).Length / 4];
            Buffer.BlockCopy(File.ReadAllBytes(latPath), 0, lat, 0, lat.Length * 4);
            int frames = lat.Length / 64;
            var cm = new float[lat.Length];
            for (int t = 0; t < frames; t++) for (int c = 0; c < 64; c++) cm[c * frames + t] = lat[t * 64 + c];
            var pcm = AceStepOobleckDecoder.Decode(vw, cm, frames);
            int spc = pcm.Length / 2;
            var il = new float[pcm.Length];
            for (int i = 0; i < spc; i++) { il[2 * i] = pcm[i]; il[2 * i + 1] = pcm[spc + i]; }
            WavWriter.WriteWav(outPath, il, 48000, channels: 2, DitherMode.None);
            double mx = 0; foreach (var v in pcm) mx = Math.Max(mx, Math.Abs(v));
            Console.WriteLine($"[ZzAce] decoded {frames} frames, peak {mx:F4}");
            return;
        }
        using var turboLoader = SafetensorsLoader.Open(root + @"acestep-v15\turbo.safetensors");
        using var vaeLoader = SafetensorsLoader.Open(root + @"acestep-v15\vae.safetensors");
        using var te = new AceStepQwen3TextEncoder(Environment.GetEnvironmentVariable("ACE_TE")
            ?? root + @"qwen3-embedding-0.6b\qwen3-embedding-0.6b-f16.gguf");
        var model = new AceStepModel
        {
            Transformer = AceStepDiTWeights.Load(turboLoader),
            Vae = AceStepOobleckDecoderWeights.Load(vaeLoader),
            VaeEncoder = AceStepOobleckEncoderWeights.Load(vaeLoader),
            TextEncoder = te,
            ConditionEncoder = AceStepConditionEncoderWeights.Load(turboLoader),
            TimbreEncoder = AceStepTimbreEncoderWeights.Load(turboLoader),
        };
        using var pipeline = new AceStepPipeline(model);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var r = pipeline.Generate(new AceStepGenerationParams
        {
            Prompt = Environment.GetEnvironmentVariable("ACE_PROMPT") ?? "upbeat electronic dance music with a driving beat",
            Instrumental = true,
            DurationSeconds = float.Parse(Environment.GetEnvironmentVariable("ACE_SECONDS") ?? "10", System.Globalization.CultureInfo.InvariantCulture),
            Seed = int.Parse(Environment.GetEnvironmentVariable("ACE_SEED") ?? "1", System.Globalization.CultureInfo.InvariantCulture),
        });
        Console.WriteLine($"[ZzAce] generate {sw.Elapsed.TotalSeconds:F1}s");
        var inter = new float[r.SampleCount * 2];
        for (int i = 0; i < r.SampleCount; i++) { inter[2 * i] = r.Left[i]; inter[2 * i + 1] = r.Right[i]; }
        WavWriter.WriteWav(outPath, inter, r.SampleRate, channels: 2, DitherMode.None);

        if (Environment.GetEnvironmentVariable("ACE_REF") is { Length: > 0 } refPath)
        {
            var (refS, refRate, refCh) = WavReader.ReadWav(refPath);
            int n = Math.Min(refS.Length, inter.Length);
            double dot = 0, a2 = 0, b2 = 0, maxd = 0;
            for (int i = 0; i < n; i++)
            {
                dot += (double)inter[i] * refS[i]; a2 += (double)inter[i] * inter[i]; b2 += (double)refS[i] * refS[i];
                maxd = Math.Max(maxd, Math.Abs(inter[i] - refS[i]));
            }
            Console.WriteLine($"[ZzAce] ref {refRate}Hz ch={refCh} len={refS.Length} ours len={inter.Length} cos={dot / Math.Sqrt(a2 * b2):F6} rmsOurs={Math.Sqrt(a2 / n):F5} rmsRef={Math.Sqrt(b2 / n):F5} maxAbsDiff={maxd:F5}");
        }
    }
}
