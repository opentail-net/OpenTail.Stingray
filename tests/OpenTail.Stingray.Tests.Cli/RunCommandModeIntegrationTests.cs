using System.Diagnostics;
using System.IO.Compression;
using System.Text;
using System.Text.Json;

namespace OpenTail.Stingray.Tests.Cli;

/// <summary>
/// End-to-end behaviour of <c>stingray run</c> per execution mode, against real (small) local models, running the real binary
/// out of process so no static engine state leaks between modes. These exist as the safety net for splitting
/// <c>RunCommand.Execute</c> into per-mode methods: each test pins what a user observes (exit code, backend banner, generated
/// content), not how the code is organised. Heavy: skipped unless <c>STINGRAY_RUN_HEAVY_TESTS=1</c>, and each test skips
/// visibly (never silently passes) when its model is not on this machine.
/// </summary>
public sealed class RunCommandModeIntegrationTests : IDisposable
{
    private readonly string _tmp = Path.Combine(Path.GetTempPath(), "ot-run-modes-" + Guid.NewGuid().ToString("N"));
    public RunCommandModeIntegrationTests() => Directory.CreateDirectory(_tmp);
    public void Dispose() { try { Directory.Delete(_tmp, true); } catch (IOException) { } }

    // ── fixtures ────────────────────────────────────────────────────────────────────────────────────────────────────

    private static string RepoRoot()
    {
        for (var d = new DirectoryInfo(AppContext.BaseDirectory); d is not null; d = d.Parent)
            if (File.Exists(Path.Combine(d.FullName, "OpenTail.Stingray.slnx"))) return d.FullName;
        throw new DirectoryNotFoundException("repository root not found");
    }

    private static string? Find(params string[] relativeCandidates)
    {
        string root = RepoRoot();
        return relativeCandidates.Select(c => Path.Combine(root, c)).FirstOrDefault(p => File.Exists(p) || Directory.Exists(p));
    }

    private static void RequireHeavy() =>
        Assert.SkipUnless(Environment.GetEnvironmentVariable("STINGRAY_RUN_HEAVY_TESTS") == "1",
            "heavy: loads a real model; set STINGRAY_RUN_HEAVY_TESTS=1");

    private static string RequireModel(params string[] candidates)
    {
        RequireHeavy();
        string? p = Find(candidates);
        Assert.SkipUnless(p is not null, $"model not present on this machine: {candidates[0]}");
        return p!;
    }

    private static string SmolGguf() => RequireModel(
        "models/_models/SmolLM2-1.7B-Instruct-Q5_K_M.gguf", "models/SmolLM2-1.7B-Instruct-Q5_K_M.gguf",
        "models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf", "models/_models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf");

    private sealed record Result(int Exit, string Out, string Err)
    {
        public string All => Out + "\n" + Err;
    }

    /// <summary>
    /// The binary under test: <c>STINGRAY_CLI_EXE</c> if set, else the optimized (Release) build when it exists (REBUILD IT before
    /// running, or these test stale code), else the Debug copy beside the test assembly. A Debug CLI is fine for short text generation but
    /// far too slow for long prefills (the vision test), which therefore require an optimized build.
    /// Build it with <c>dotnet build src/OpenTail.Stingray.Cli -c Release</c>.
    /// </summary>
    private static string CliExe()
    {
        string name = OperatingSystem.IsWindows() ? "stingray.exe" : "stingray";
        string? overridePath = Environment.GetEnvironmentVariable("STINGRAY_CLI_EXE");
        if (!string.IsNullOrEmpty(overridePath)) return overridePath;
        string local = Path.Combine(AppContext.BaseDirectory, name);
        string release = Path.Combine(RepoRoot(), "src", "OpenTail.Stingray.Cli", "bin", "Release", "net10.0", name);
        return File.Exists(release) ? release : local;
    }

    private static bool CliIsOptimized() => !CliExe().Contains($"{Path.DirectorySeparatorChar}Debug{Path.DirectorySeparatorChar}", StringComparison.OrdinalIgnoreCase);

    private static Result Run(params string[] args)
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
        p.StandardInput.Close();   // a run must never wait for interactive input
        var stdout = p.StandardOutput.ReadToEndAsync();
        var stderr = p.StandardError.ReadToEndAsync();
        if (!p.WaitForExit(TimeSpan.FromMinutes(5)))
        {
            p.Kill(true);
            p.WaitForExit();
            Assert.Fail("run timed out after 5 minutes; partial stdout:\n" + stdout.Result + "\n--- partial stderr:\n" + stderr.Result);
        }
        return new Result(p.ExitCode, stdout.Result, stderr.Result);
    }

    private static void AssertCleanSuccess(Result r)
    {
        Assert.True(r.Exit == 0, $"exit {r.Exit}\n--- stdout\n{r.Out}\n--- stderr\n{r.Err}");
        Assert.DoesNotContain("Unhandled exception", r.All, StringComparison.Ordinal);
    }

    private const string CountPrompt = "Count to five:";
    private static void AssertCountsToFive(Result r)
    {
        Assert.Contains("1. One", r.Out, StringComparison.Ordinal);
        Assert.Contains("5. Five", r.Out, StringComparison.Ordinal);
    }

    // ── GGUF text generation, per backend ───────────────────────────────────────────────────────────────────────────

    [Fact]
    public void Gguf_Cpu_Greedy()
    {
        var r = Run("-m", SmolGguf(), "-p", CountPrompt, "--temp", "0", "-n", "24", "-g", "0");
        AssertCleanSuccess(r);
        Assert.Contains("Backend: CPU", r.Out, StringComparison.Ordinal);
        AssertCountsToFive(r);
    }

    [Fact]
    public void Gguf_Cpu_PromptFile_And_DeviceNone()
    {
        string m = SmolGguf();
        string prompt = Path.Combine(_tmp, "prompt.txt");
        File.WriteAllText(prompt, CountPrompt);
        var viaFile = Run("-m", m, "-f", prompt, "--temp", "0", "-n", "24", "-g", "0");
        AssertCleanSuccess(viaFile);
        AssertCountsToFive(viaFile);

        var deviceNone = Run("-m", m, "-p", CountPrompt, "--temp", "0", "-n", "24", "--device", "none");
        AssertCleanSuccess(deviceNone);
        Assert.Contains("Backend: CPU", deviceNone.Out, StringComparison.Ordinal);
        AssertCountsToFive(deviceNone);
    }

    [Fact]
    public void Gguf_Vulkan_FullOffload()
    {
        var r = Run("-m", SmolGguf(), "-p", CountPrompt, "--temp", "0", "-n", "24", "-g", "-1");
        AssertCleanSuccess(r);
        Assert.SkipUnless(r.Out.Contains("Backend: GPU", StringComparison.Ordinal), "no usable Vulkan device on this machine");
        AssertCountsToFive(r);
    }

    [Fact]
    public void Gguf_Hybrid_PartialGpuLayers()
    {
        var r = Run("-m", SmolGguf(), "-p", CountPrompt, "--temp", "0", "-n", "24", "-g", "12");
        AssertCleanSuccess(r);
        Assert.SkipUnless(r.Out.Contains("Backend: Hybrid", StringComparison.Ordinal), "no usable Vulkan device on this machine");
        AssertCountsToFive(r);
    }

    /// <summary><c>--auto</c> once crashed with an ArgumentException because the plan's KV dtype ("float32") was written verbatim into STINGRAY_KV_DTYPE.</summary>
    [Fact]
    public void Gguf_Auto_PlansAndRuns()
    {
        var r = Run("-m", SmolGguf(), "-p", CountPrompt, "--temp", "0", "-n", "24", "--auto");
        AssertCleanSuccess(r);
        Assert.Contains("[ExecutionPlan]", r.Out, StringComparison.Ordinal);
        AssertCountsToFive(r);
    }

    // ── GGUF feature modes ──────────────────────────────────────────────────────────────────────────────────────────

    [Fact]
    public void Gguf_Cpu_TurboQuantKvCache()
    {
        var r = Run("-m", SmolGguf(), "-p", CountPrompt, "--temp", "0", "-n", "24", "-g", "0", "--tq");
        AssertCleanSuccess(r);
        Assert.Contains("TurboQuant: enabled", r.Out, StringComparison.Ordinal);
        AssertCountsToFive(r);
    }

    [Fact]
    public void Gguf_Cpu_PromptLookupSpeculation()
    {
        var r = Run("-m", SmolGguf(), "-p", "Count to five: one two three four five. Count to five:", "--temp", "0", "-n", "24", "-g", "0", "--draft-lookup");
        AssertCleanSuccess(r);
        Assert.Contains("Speculative decoding: prompt-lookup", r.Out, StringComparison.Ordinal);
    }

    [Fact]
    public void Gguf_Cpu_JsonSchemaConstrainedOutput_IsValidJsonMatchingTheSchema()
    {
        const string schema = """{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"]}""";
        var r = Run("-m", SmolGguf(), "-p", "Give a person", "--temp", "0", "-n", "60", "-g", "0", "--json-schema", schema);
        AssertCleanSuccess(r);
        Assert.Contains("constrained to the supplied JSON schema", r.Out, StringComparison.Ordinal);
        int start = r.Out.IndexOf('{'), end = r.Out.LastIndexOf('}');
        Assert.True(start >= 0 && end > start, "no JSON object in the output:\n" + r.Out);
        using var doc = JsonDocument.Parse(r.Out[start..(end + 1)]);
        Assert.Equal(JsonValueKind.String, doc.RootElement.GetProperty("name").ValueKind);
        Assert.Equal(JsonValueKind.Number, doc.RootElement.GetProperty("age").ValueKind);
    }

    [Fact]
    public void HybridSsm_Granite_Cpu()
    {
        string m = RequireModel("models/_models/granite-4.0-h-350m-Q8_0.gguf", "models/granite-4.0-h-350m-Q8_0.gguf");
        var r = Run("-m", m, "-p", CountPrompt, "--temp", "0", "-n", "20", "-g", "0");
        AssertCleanSuccess(r);
        Assert.Contains("Backend: CPU", r.Out, StringComparison.Ordinal);
        Assert.Contains(CountPrompt, r.Out, StringComparison.Ordinal);
        Assert.Contains("32L", r.Out, StringComparison.Ordinal);   // the hybrid trunk loaded with its real depth
    }

    // ── SafeTensors package ─────────────────────────────────────────────────────────────────────────────────────────

    [Fact]
    public void SafeTensorsPackage_LoadsAndGenerates()
    {
        string pkg = RequireModel("models/SmolLM2-135M-Instruct");
        Assert.SkipUnless(File.Exists(Path.Combine(pkg, "model.safetensors")), "not a safetensors package");
        var r = Run("-m", pkg, "-p", CountPrompt, "--temp", "0", "-n", "16", "-g", "0");
        AssertCleanSuccess(r);
        Assert.Contains("30L", r.Out, StringComparison.Ordinal);                // SmolLM2-135M's depth
        Assert.Contains(CountPrompt, r.Out, StringComparison.Ordinal);
    }

    [Fact]
    public void SafeTensorsPackage_RefusesGpuOffload_WithAnExplicitError()
    {
        string pkg = RequireModel("models/SmolLM2-135M-Instruct");
        var r = Run("-m", pkg, "-p", CountPrompt, "-n", "4", "-g", "-1");
        Assert.NotEqual(0, r.Exit);
        Assert.DoesNotContain("Unhandled exception", r.All, StringComparison.Ordinal);
        Assert.False(string.IsNullOrWhiteSpace(r.Err), "a refusal must explain itself on stderr");
    }

    // ── vision ──────────────────────────────────────────────────────────────────────────────────────────────────────

    [Fact]
    public void Vision_ImagePrompt_LoadsProjectorAndGenerates()
    {
        Assert.SkipUnless(CliIsOptimized(), "a 729-token vision prefill is too slow in a Debug CLI; build src/OpenTail.Stingray.Cli -c Release (or set STINGRAY_CLI_EXE)");
        string model = RequireModel("models/_models/granite-vision-3.2-2b-Q3_K_S.gguf");
        string mmproj = RequireModel("models/_models/mmproj-granite-vision-3.2-2b-f16.gguf");
        string png = Path.Combine(_tmp, "halves.png");
        WriteTwoColourPng(png, 96, 96);
        var r = Run("-m", model, "--mmproj", mmproj, "--image", png, "-p", "Describe this image in one sentence.", "--temp", "0", "-n", "24", "-g", "0");
        AssertCleanSuccess(r);
        Assert.False(string.IsNullOrWhiteSpace(r.Out.Split('\n').Last(l => l.Length > 0 && !l.StartsWith("Prefill:") && !l.StartsWith("Decode:"))),
            "the vision run produced no text");
    }

    // ── helpers ─────────────────────────────────────────────────────────────────────────────────────────────────────

    /// <summary>A minimal valid PNG (left half red, right half blue), written by hand so the test needs no imaging library.</summary>
    private static void WriteTwoColourPng(string path, int w, int h)
    {
        var raw = new MemoryStream();
        for (int y = 0; y < h; y++)
        {
            raw.WriteByte(0);   // filter: none
            for (int x = 0; x < w; x++) { if (x < w / 2) { raw.WriteByte(220); raw.WriteByte(30); raw.WriteByte(30); } else { raw.WriteByte(30); raw.WriteByte(40); raw.WriteByte(220); } }
        }
        var idat = new MemoryStream();
        using (var z = new ZLibStream(idat, CompressionLevel.Fastest, leaveOpen: true)) z.Write(raw.ToArray());
        using var fs = File.Create(path);
        fs.Write([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
        void Chunk(string type, byte[] data)
        {
            var len = BitConverter.GetBytes((uint)data.Length); Array.Reverse(len); fs.Write(len);
            var body = Encoding.ASCII.GetBytes(type).Concat(data).ToArray(); fs.Write(body);
            var crc = BitConverter.GetBytes(Crc32(body)); Array.Reverse(crc); fs.Write(crc);
        }
        var ihdr = new byte[13];
        BitConverter.GetBytes((uint)w).Reverse().ToArray().CopyTo(ihdr, 0);
        BitConverter.GetBytes((uint)h).Reverse().ToArray().CopyTo(ihdr, 4);
        ihdr[8] = 8; ihdr[9] = 2;   // 8-bit RGB
        Chunk("IHDR", ihdr); Chunk("IDAT", idat.ToArray()); Chunk("IEND", []);
    }

    private static uint Crc32(byte[] data)
    {
        uint crc = 0xFFFFFFFF;
        foreach (byte b in data)
        {
            crc ^= b;
            for (int k = 0; k < 8; k++) crc = (crc & 1) != 0 ? (crc >> 1) ^ 0xEDB88320 : crc >> 1;
        }
        return ~crc;
    }
}
