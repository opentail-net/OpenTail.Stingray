using System.Diagnostics;
using System.Globalization;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace OpenTail.Stingray.Tests.Cli;

/// <summary>
/// Per-pipeline end-to-end smoke tests for <c>stingray image</c> (backlog §1.2; item 4 of §4 in
/// docs/done/2026-09-24-diffusion-perf-session-handoff.md). Each runs the real binary with real weights at 256x256, few steps,
/// seed 42, on the CPU (bit-deterministic run to run), and compares a fingerprint of the PNG with a stored reference:
/// per-channel mean, luma standard deviation and an 8x8 average hash. Small numeric drift passes; a broken pipeline (black,
/// flat, noise or a different picture) fails. A flat image also fails on its own, so a broken run cannot become the baseline.
///
/// <para>Heavy: skipped unless <c>STINGRAY_RUN_HEAVY_TESTS=1</c>. With it set, a missing checkpoint FAILS (it does not skip):
/// the point is to run before committing shared diffusion code, and a smoke test that silently skips protects nothing.
/// Regenerate references after an intended change with <c>STINGRAY_UPDATE_BASELINE=1</c> and say why in the commit.
/// Build the CLI first: <c>dotnet build src/OpenTail.Stingray.Cli -c Release</c>.</para>
/// </summary>
public sealed class ImageCommandSmokeTests : IDisposable
{
    private const string Prompt = "a red apple on a wooden table";
    private const double MeanTolerance = 4.0;      // 0..255 scale
    private const double StdTolerance = 4.0;
    private const int HashTolerance = 8;           // of 64 bits
    private const double MinLumaStd = 15.0;        // flat or near-flat output is a failure in its own right

    private readonly string _tmp = Path.Combine(Path.GetTempPath(), "ot-image-smoke-" + Guid.NewGuid().ToString("N"));
    public ImageCommandSmokeTests() => Directory.CreateDirectory(_tmp);
    public void Dispose() { try { Directory.Delete(_tmp, true); } catch (IOException) { } }

    // name, steps, model and auxiliary files (relative to the repo root, passed as "--flag path" pairs).
    public static TheoryData<string> Pipelines => new() { "sd15", "sdxl-turbo", "flux1-schnell", "z-image-turbo", "sd35-medium" };

    private static (int Steps, string[] Args) Case(string name) => name switch
    {
        "sd15" => (4, ["-m", "models/sd15/v1-5-pruned-emaonly.safetensors"]),
        "sdxl-turbo" => (1, ["-m", "models/sd_xl_turbo_1.0_fp16.safetensors"]),
        "flux1-schnell" => (2,
        [
            "-m", "models/flux1-schnell/flux1-schnell-Q4_K_S.gguf",
            "--vae", "models/flux1-schnell/ae.safetensors",
            "--clip-l", "models/flux1-schnell/clip_l.safetensors",
            "--clip-tokenizer", "models/flux1-schnell/tokenizer_clip/tokenizer.json",
            "--t5xxl", "models/flux1-schnell/t5xxl_fp8_e4m3fn.safetensors",
            "--t5-tokenizer", "models/flux1-schnell/tokenizer_t5/tokenizer.json",
        ]),
        "z-image-turbo" => (2,
        [
            "-m", "models/_models/z_image_turbo-Q4_0.gguf",
            "--vae", "models/z-image-turbo/vae/diffusion_pytorch_model.safetensors",
            "--qwen-encoder", "models/_models/Z-Image-AbliteratedV1.Q5_K_M.gguf",
            "--qwen-tokenizer", "models/z-image-turbo/tokenizer/tokenizer.json",
        ]),
        // Not distilled: 4 steps gives a flat orange field, 20 gives the apple (measured 2026-10-30).
        "sd35-medium" => (20,
        [
            "-m", "models/_models/sd3.5_medium-Q4_K_M.gguf",
            "--clip-l", "models/sd35-medium-aux/text_encoder/model.fp16.safetensors",
            "--clip-g", "models/sd35-medium-aux/text_encoder_2/model.fp16.safetensors",
            "--vae", "models/sd35-medium-aux/vae/diffusion_pytorch_model.safetensors",
            "--clip-tokenizer", "models/clip_tokenizer.json",
        ]),
        _ => throw new ArgumentOutOfRangeException(nameof(name)),
    };

    [Theory]
    [MemberData(nameof(Pipelines))]
    public void Pipeline_MatchesStoredReference(string name)
    {
        Assert.SkipUnless(Environment.GetEnvironmentVariable("STINGRAY_RUN_HEAVY_TESTS") == "1",
            "heavy: loads real diffusion weights; set STINGRAY_RUN_HEAVY_TESTS=1");

        string root = RepoRoot();
        var (steps, modelArgs) = Case(name);
        var missing = modelArgs.Where((a, i) => i % 2 == 1 && !File.Exists(Path.Combine(root, a))).ToList();
        Assert.True(missing.Count == 0, $"{name}: checkpoint files missing (this smoke test fails rather than skips):\n  " +
            string.Join("\n  ", missing));

        string png = Path.Combine(_tmp, name + ".png");
        string[] args =
        [
            "image", .. modelArgs, "-p", Prompt, "-W", "256", "-H", "256", "--steps", steps.ToString(CultureInfo.InvariantCulture),
            "--seed", "42", "--backend", "cpu", "-o", png,
        ];
        RunCli(args);
        Assert.True(File.Exists(png), $"{name}: no image written to {png}");

        var actual = Fingerprint.Of(png);
        Assert.True(actual.LumaStd >= MinLumaStd,
            $"{name}: output is (near) flat, luma std {actual.LumaStd:F1} < {MinLumaStd}; the pipeline is broken or under-stepped.");

        string refPath = Path.Combine(root, "tests", "OpenTail.Stingray.Tests.Cli", "Baselines", "ImageSmoke", name + ".json");
        if (Environment.GetEnvironmentVariable("STINGRAY_UPDATE_BASELINE") == "1")
        {
            Directory.CreateDirectory(Path.GetDirectoryName(refPath)!);
            File.WriteAllText(refPath, actual.ToJson(string.Join(' ', args.Where(a => a != png))));
            return;
        }
        Assert.True(File.Exists(refPath), $"{name}: missing reference {refPath}. Generate it with STINGRAY_UPDATE_BASELINE=1.");
        var expected = Fingerprint.FromJson(File.ReadAllText(refPath));

        if (actual.Sha256 == expected.Sha256) return;   // bit-identical: nothing more to compare
        var problems = new List<string>();
        for (int c = 0; c < 3; c++)
            if (Math.Abs(actual.Mean[c] - expected.Mean[c]) > MeanTolerance)
                problems.Add($"mean[{"RGB"[c]}] {expected.Mean[c]:F1} -> {actual.Mean[c]:F1}");
        if (Math.Abs(actual.LumaStd - expected.LumaStd) > StdTolerance)
            problems.Add($"luma std {expected.LumaStd:F1} -> {actual.LumaStd:F1}");
        int hamming = System.Numerics.BitOperations.PopCount(actual.AHash ^ expected.AHash);
        if (hamming > HashTolerance)
            problems.Add($"average hash differs in {hamming}/64 bits");
        if (problems.Count == 0) return;
        string kept = Path.Combine(Path.GetTempPath(), "ot-image-smoke-failures", name + ".png");
        Directory.CreateDirectory(Path.GetDirectoryName(kept)!);
        File.Copy(png, kept, overwrite: true);
        Assert.Fail($"{name}: output drifted from the stored reference ({string.Join("; ", problems)}). " +
            $"Image kept at {kept} for inspection; if the change is intended, regenerate with STINGRAY_UPDATE_BASELINE=1.");
    }

    [Fact]
    public void Fingerprint_HashesStructure_NotJustColour()
    {
        // Pure function check (no model): a left-dark/right-light split and its mirror share mean and std but not the hash.
        static byte[] Split(bool darkLeft)
        {
            var px = new byte[16 * 16 * 3];
            for (int y = 0; y < 16; y++)
                for (int x = 0; x < 16; x++)
                {
                    byte v = (x < 8) == darkLeft ? (byte)20 : (byte)220;
                    int i = (y * 16 + x) * 3;
                    px[i] = px[i + 1] = px[i + 2] = v;
                }
            return px;
        }
        var a = Fingerprint.Of(Split(true), 16, 16);
        var b = Fingerprint.Of(Split(false), 16, 16);
        Assert.Equal(a.LumaStd, b.LumaStd, 6);
        Assert.Equal(64, System.Numerics.BitOperations.PopCount(a.AHash ^ b.AHash));
    }

    private sealed record Fingerprint(double[] Mean, double LumaStd, ulong AHash, string Sha256)
    {
        public static Fingerprint Of(string pngPath)
        {
            byte[] rgb = OpenTail.Stingray.Vision.ImageIO.LoadRgb(pngPath, out int w, out int h);
            return Of(rgb, w, h) with { Sha256 = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(pngPath))) };
        }

        public static Fingerprint Of(byte[] rgb, int w, int h)
        {
            var mean = new double[3];
            var luma = new double[w * h];
            for (int i = 0; i < w * h; i++)
            {
                byte r = rgb[3 * i], g = rgb[3 * i + 1], b = rgb[3 * i + 2];
                mean[0] += r; mean[1] += g; mean[2] += b;
                luma[i] = 0.299 * r + 0.587 * g + 0.114 * b;
            }
            for (int c = 0; c < 3; c++) mean[c] /= w * h;
            double lumaMean = luma.Average();
            double std = Math.Sqrt(luma.Sum(v => (v - lumaMean) * (v - lumaMean)) / luma.Length);

            // 8x8 average hash: box-average luma into an 8x8 grid, one bit per cell above the grid mean.
            var cells = new double[64];
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    cells[(y * 8 / h) * 8 + x * 8 / w] += luma[y * w + x];
            double cellMean = cells.Average();
            ulong hash = 0;
            for (int i = 0; i < 64; i++)
                if (cells[i] > cellMean) hash |= 1UL << i;
            return new Fingerprint(mean, std, hash, "");
        }

        public string ToJson(string command) => new JsonObject
        {
            ["command"] = command,
            ["meanR"] = Math.Round(Mean[0], 3), ["meanG"] = Math.Round(Mean[1], 3), ["meanB"] = Math.Round(Mean[2], 3),
            ["lumaStd"] = Math.Round(LumaStd, 3),
            ["aHash"] = AHash.ToString("X16", CultureInfo.InvariantCulture),
            ["sha256"] = Sha256,
        }.ToJsonString(new JsonSerializerOptions { WriteIndented = true }) + "\n";

        public static Fingerprint FromJson(string json)
        {
            var o = JsonNode.Parse(json)!;
            return new Fingerprint(
                [(double)o["meanR"]!, (double)o["meanG"]!, (double)o["meanB"]!],
                (double)o["lumaStd"]!,
                ulong.Parse((string)o["aHash"]!, NumberStyles.HexNumber, CultureInfo.InvariantCulture),
                (string)o["sha256"]!);
        }
    }

    private static string RepoRoot()
    {
        for (var d = new DirectoryInfo(AppContext.BaseDirectory); d is not null; d = d.Parent)
            if (File.Exists(Path.Combine(d.FullName, "OpenTail.Stingray.slnx"))) return d.FullName;
        throw new DirectoryNotFoundException("repository root not found");
    }

    private static string CliExe()
    {
        string name = OperatingSystem.IsWindows() ? "stingray.exe" : "stingray";
        string? overridePath = Environment.GetEnvironmentVariable("STINGRAY_CLI_EXE");
        if (!string.IsNullOrEmpty(overridePath)) return overridePath;
        string release = Path.Combine(RepoRoot(), "src", "OpenTail.Stingray.Cli", "bin", "Release", "net10.0", name);
        return File.Exists(release) ? release : Path.Combine(AppContext.BaseDirectory, name);
    }

    private static void RunCli(string[] args)
    {
        string exe = CliExe();
        Assert.True(File.Exists(exe), $"CLI binary not found: {exe}");
        var psi = new ProcessStartInfo(exe)
        {
            RedirectStandardOutput = true, RedirectStandardError = true, RedirectStandardInput = true, UseShellExecute = false,
            WorkingDirectory = RepoRoot(),
        };
        foreach (var a in args) psi.ArgumentList.Add(a);
        using var p = Process.Start(psi)!;
        p.StandardInput.Close();
        var stdout = p.StandardOutput.ReadToEndAsync();
        var stderr = p.StandardError.ReadToEndAsync();
        if (!p.WaitForExit(TimeSpan.FromMinutes(10)))
        {
            p.Kill(true);
            p.WaitForExit();
            Assert.Fail("image run timed out after 10 minutes; partial stdout:\n" + stdout.Result + "\n--- partial stderr:\n" + stderr.Result);
        }
        Assert.True(p.ExitCode == 0, $"exit {p.ExitCode}\n--- stdout\n{stdout.Result}\n--- stderr\n{stderr.Result}");
    }
}
