using System.Diagnostics;
using OpenTail.Stingray.Diffusion.StableAudio;

namespace OpenTail.Stingray.Tests.Diffusion;

// Scratch harness (untracked): Stable Audio 3 generation for ours-vs-reference comparison.
// ZZ_SA=1: generate. ZZ_SA_MODEL=small-music|small-sfx|medium, ZZ_PROMPT, ZZ_DUR (6), ZZ_STEPS (50),
// ZZ_CFG (7), ZZ_SEED (1), ZZ_GPU=1 for Vulkan, ZZ_OUT (wav path).
// ZZ_SASTATS=<wav;wav;...>: print duration / rms / peak / HF ratio / ZCR per file.
public sealed class ZzSa3CmpTmp
{
    private readonly ITestOutputHelper _out;
    public ZzSa3CmpTmp(ITestOutputHelper o) => _out = o;
    private void Log(string s) { _out.WriteLine(s); Console.WriteLine(s); }

    [Fact]
    public void Sa3Generate()
    {
        if (Environment.GetEnvironmentVariable("ZZ_SA") != "1") return;
        string m = Environment.GetEnvironmentVariable("ZZ_SA_MODEL") ?? "small-music";
        string root = @"C:\Git-Public\OpenTail.Stingray\models";
        string ditDir = Path.Combine(root, $"stable-audio-3-{m}-base"), t5 = Path.Combine(root, "stable-audio-3-t5gemma");
        string prompt = Environment.GetEnvironmentVariable("ZZ_PROMPT") ?? "upbeat acoustic guitar melody";
        float dur = float.Parse(Environment.GetEnvironmentVariable("ZZ_DUR") ?? "6", System.Globalization.CultureInfo.InvariantCulture);
        int steps = int.Parse(Environment.GetEnvironmentVariable("ZZ_STEPS") ?? "50");
        float cfg = float.Parse(Environment.GetEnvironmentVariable("ZZ_CFG") ?? "7", System.Globalization.CultureInfo.InvariantCulture);
        int seed = int.Parse(Environment.GetEnvironmentVariable("ZZ_SEED") ?? "1");
        string outPath = Environment.GetEnvironmentVariable("ZZ_OUT") ?? Path.Combine(Path.GetTempPath(), $"zz_sa3_{m}.wav");

        using var dit = OpenTail.Stingray.Core.SafetensorsLoader.OpenDirectory(ditDir);
        using var te = OpenTail.Stingray.Core.SafetensorsLoader.OpenDirectory(t5);
        using var vk = Environment.GetEnvironmentVariable("ZZ_GPU") == "1" ? new OpenTail.Stingray.Vulkan.VulkanBackend() : null;
        string sched = Environment.GetEnvironmentVariable("ZZ_SCHED") ?? "flat";
        float rescale = float.TryParse(Environment.GetEnvironmentVariable("ZZ_RESCALE"), System.Globalization.CultureInfo.InvariantCulture, out var r) ? r : 0.0f;
        var req = new StableAudioRequest { Prompt = prompt, DurationSeconds = dur, Steps = steps, CfgScale = cfg, Seed = seed, OutputPath = outPath, GuidanceSchedule = sched, Rescale = rescale };
        var sw = Stopwatch.StartNew();
        float[] pcm;
        if (m == "medium")
        {
            using var p = new StableAudioMediumPipeline(dit, te, t5, backend: vk);
            Log($"[ZZ sa3] load {sw.Elapsed.TotalSeconds:F1}s"); sw.Restart();
            pcm = p.Generate(req);
        }
        else
        {
            using var p = new StableAudioPipeline(dit, te, t5, backend: vk);
            Log($"[ZZ sa3] load {sw.Elapsed.TotalSeconds:F1}s"); sw.Restart();
            pcm = p.Generate(req);
        }
        Log($"[ZZ sa3] {m} gpu={vk is not null} steps={steps} cfg={cfg} seed={seed}: generate {sw.Elapsed.TotalSeconds:F1}s, {pcm.Length} samples -> {outPath}");
    }

    [Fact]
    public void Sa3TokenIds()
    {
        if (Environment.GetEnvironmentVariable("ZZ_SATOK") != "1") return;
        string t5 = @"C:\Git-Public\OpenTail.Stingray\models\stable-audio-3-t5gemma";
        var src = OpenTail.Stingray.Core.HuggingFaceTokenizerSource.Load(t5);
        var tok = OpenTail.Stingray.Core.GgufTokenizer.FromSource(src.Source!);
        var ours = tok.Encode("lofi house loop").ToArray();
        var raw = File.ReadAllBytes(@"C:\Git-Public\OpenTail.Stingray\tests\OpenTail.Stingray.Tests.Diffusion\TestData\StableAudioPipelineGolden\prompt_ids.bin");
        var fixture = new int[raw.Length / 4];
        Buffer.BlockCopy(raw, 0, fixture, 0, raw.Length);
        Log($"[ZZ tok] ours ({ours.Length}): {string.Join(",", ours)}");
        Log($"[ZZ tok] fixture ({fixture.Length}), first 12: {string.Join(",", fixture.Take(12))}; non-zero count {fixture.Count(v => v != 0)}");
    }

    [Fact]
    public void Sa3Stats()
    {
        string? list = Environment.GetEnvironmentVariable("ZZ_SASTATS");
        if (list is null) return;
        foreach (var path in list.Split(';', StringSplitOptions.RemoveEmptyEntries))
        {
            var (s, sr, ch) = ReadWav(path);
            double sumSq = 0, hf = 0; float peak = 0; int zc = 0;
            for (int i = 0; i < s.Length; i++)
            {
                sumSq += (double)s[i] * s[i];
                peak = MathF.Max(peak, MathF.Abs(s[i]));
                if (i >= ch) { double d = s[i] - s[i - ch]; hf += d * d; if ((s[i] >= 0) != (s[i - ch] >= 0)) zc++; }
            }
            double rms = Math.Sqrt(sumSq / s.Length);
            Log($"[ZZ sa3 stats] {Path.GetFileName(path)}: {s.Length / (double)(sr * ch):F2}s sr={sr} ch={ch} rms={rms:F4} peak={peak:F3} hfRatio={hf / (sumSq + 1e-12):F4} zcr={zc / (double)s.Length:F4}");
        }
    }

    private static (float[] Samples, int SampleRate, int Channels) ReadWav(string path)
    {
        var b = File.ReadAllBytes(path);
        int pos = 12, fmt = 1, ch = 1, sr = 0, bits = 16;
        while (pos + 8 <= b.Length)
        {
            string id = System.Text.Encoding.ASCII.GetString(b, pos, 4);
            int len = BitConverter.ToInt32(b, pos + 4);
            if (id == "fmt ")
            {
                fmt = BitConverter.ToInt16(b, pos + 8); ch = BitConverter.ToInt16(b, pos + 10);
                sr = BitConverter.ToInt32(b, pos + 12); bits = BitConverter.ToInt16(b, pos + 22);
                if (fmt == -2) fmt = BitConverter.ToInt16(b, pos + 32); // WAVE_FORMAT_EXTENSIBLE sub-format
            }
            else if (id == "data")
            {
                int n = Math.Min(len, b.Length - pos - 8) / (bits / 8);
                var s = new float[n];
                for (int i = 0; i < n; i++)
                    s[i] = fmt == 3 ? BitConverter.ToSingle(b, pos + 8 + i * 4) : BitConverter.ToInt16(b, pos + 8 + i * 2) / 32768f;
                return (s, sr, ch);
            }
            pos += 8 + len + (len & 1);
        }
        throw new InvalidDataException($"no data chunk in {path}");
    }
}
