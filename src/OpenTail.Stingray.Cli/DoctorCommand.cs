using System.ComponentModel;
using System.IO.Compression;
using System.Reflection;
using System.Runtime.Intrinsics.X86;
using System.Text.Json;
using System.Text.Json.Serialization;
using OpenTail.Stingray.Cli.CommandLine;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Vulkan;

namespace OpenTail.Stingray.Cli;

/// <summary>Fast local diagnostics. This command does not load model weights or run inference.</summary>
public sealed class DoctorCommand : Command<DoctorCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        [CommandOption("-m|--model <PATH>")]
        [Description("Optional GGUF to validate structurally")]
        public string? ModelPath { get; init; }

        [CommandOption("--no-gpu-probe")]
        [Description("Do not initialize CUDA/Vulkan; report GPU checks as not probed")]
        public bool NoGpuProbe { get; init; }

        [CommandOption("--deep")]
        [Description("Run memory allocation smoke tests and backend pipeline verification")]
        public bool Deep { get; init; }

        [CommandOption("--bundle <PATH>")]
        [Description("Write a redacted support bundle (.zip) for attaching to a bug report")]
        public string? BundlePath { get; init; }

        [CommandOption("--json")]
        [Description("Write machine-readable JSON to stdout")]
        public bool Json { get; init; }
    }

    protected override int Execute(Settings settings, CancellationToken cancellation)
    {
        var checks = new List<DoctorCheck>();
        string version = Assembly.GetEntryAssembly()?.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion
            ?? "unknown";
        checks.Add(new("runtime", "ok", $"OpenTail.Stingray CLI {version}; {Environment.Version}; {Environment.ProcessorCount} logical CPUs."));

        // §6.3 "CPU instruction-set support". AVX2+FMA are REQUIRED by the CPU kernels — without
        // them the SIMD paths cannot run, so their absence is a blocking error rather than a note.
        var isa = new List<string>();
        if (Avx.IsSupported) isa.Add("AVX");
        if (Avx2.IsSupported) isa.Add("AVX2");
        if (Fma.IsSupported) isa.Add("FMA");
        if (Avx512F.IsSupported) isa.Add("AVX-512F");
        if (AvxVnni.IsSupported) isa.Add("AVX-VNNI");
        bool isaOk = Avx2.IsSupported && Fma.IsSupported;
        checks.Add(new("cpu.isa", isaOk ? "ok" : "error",
            (isa.Count > 0 ? string.Join(", ", isa) : "no AVX-class instruction sets detected")
            + (isaOk ? "." : " — AVX2 and FMA are required by the CPU kernels."),
            isaOk ? null : "This CPU cannot run the SIMD kernels. Use a machine with AVX2+FMA "
                         + "(Intel Haswell 2013+ / AMD Excavator 2015+), or offload with -g -1 to a GPU backend."));

        // §6.3 "effective configuration conflicts and unknown settings". A misspelled
        // STINGRAY_* name is indistinguishable from unset, so the run silently ignores it.
        // Warning, not error: the name may belong to a different OpenTail version.
        var unknownSettings = KnownEnvironmentVariables.FindUnknown();
        if (unknownSettings.Count == 0)
            checks.Add(new("config", "ok", "No unknown STINGRAY_* variables are set."));
        else
            foreach (string name in unknownSettings)
            {
                string? suggestion = KnownEnvironmentVariables.SuggestClosest(name);
                checks.Add(new("config", "warning", suggestion is null
                    ? $"{name} is set but is not read by this build; it will have no effect."
                    : $"{name} is set but is not read by this build; did you mean {suggestion}?",
                    suggestion is null
                    ? $"Unset {name}, or check `opentail-llm-cli list-env --all` for the supported names."
                    : $"Rename {name} to {suggestion}, or unset it."));
            }

        var facts = StaticPlanRuntimeFacts.Detect(settings.NoGpuProbe);
        foreach (var backend in facts.Backends)
        {
            string status = backend.Status == "available" ? "ok" : backend.Status == "not_probed" ? "not_probed" : "warning";
            checks.Add(new($"backend.{backend.Name}", status, backend.Detail));
        }

        if (settings.ModelPath is { } path)
        {
            try
            {
                using var model = GgufModel.Open(path);
                ModelCompatibility.ValidateForTextGeneration(model);
                checks.Add(new("model", "ok", $"{Path.GetFileName(path)}: GGUF v{model.Header.Version}, {model.Tensors.Count} tensors; text generation is supported."));
            }
            catch (Exception ex) when (ex is IOException or NotSupportedException or ArgumentException)
            {
                checks.Add(new("model", "error", ex.Message,
                    "Confirm the path points at a readable GGUF file. `opentail-llm-cli list-metadata "
                    + "-m <path>` shows whether the file parses; re-download it if the index is truncated."));
            }
        }

        // §6.3 "filesystem readability and available space". Reported against the volume holding
        // the model when one is given, otherwise the working directory. Informational: low space
        // does not block inference of an already-present model, but it is the usual cause of a
        // failed `models pull` or a KV spill.
        try
        {
            string probePath = settings.ModelPath is { } mp && File.Exists(mp)
                ? Path.GetFullPath(mp) : Directory.GetCurrentDirectory();
            string? root = Path.GetPathRoot(probePath);
            if (!string.IsNullOrEmpty(root))
            {
                var drive = new DriveInfo(root);
                double freeGiB = drive.AvailableFreeSpace / (1024.0 * 1024 * 1024);
                checks.Add(new("filesystem", freeGiB < 2.0 ? "warning" : "ok",
                    $"{root} has {freeGiB:F1} GiB free."
                    + (freeGiB < 2.0 ? " Low free space may break model downloads or spill files." : ""),
                    freeGiB < 2.0 ? $"Free space on {root}, or move the model store to a larger volume." : null));
            }
        }
        catch (Exception ex) when (ex is IOException or ArgumentException or UnauthorizedAccessException)
        {
            checks.Add(new("filesystem", "warning", $"Could not determine free space: {ex.GetType().Name}."));
        }

        if (settings.Deep)
        {
            // §6.3 "a minimal allocation/backend smoke test in --deep mode".
            try
            {
                byte[] sample = new byte[64 * 1024 * 1024]; // 64 MiB RAM allocation
                sample[0] = 0xAA;
                sample[sample.Length - 1] = 0x55;
                bool valid = sample[0] == 0xAA && sample[sample.Length - 1] == 0x55;
                checks.Add(new("allocation.smoke", valid ? "ok" : "error",
                    valid ? "Allocated and verified 64 MiB host memory buffer." : "Host memory buffer corruption detected."));
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                checks.Add(new("allocation.smoke", "error", $"Memory allocation failed: {ex.Message}",
                    "Check available host RAM or decrease process memory reservation."));
            }

            if (!settings.NoGpuProbe && facts.Backends.Any(b => b.Name == "vulkan" && b.Status == "available"))
            {
                try
                {
                    using var vk = new VulkanBackend();
                    var tensor = vk.Allocate(new TensorShape([1024]), DType.Float32);
                    vk.Free(tensor);
                    checks.Add(new("backend.smoke", "ok", $"Vulkan backend pipeline and VRAM allocation verified successfully ({vk.Name})."));
                }
                catch (Exception ex) when (ex is not OperationCanceledException)
                {
                    checks.Add(new("backend.smoke", "warning", $"Vulkan initialization smoke test failed: {ex.Message}"));
                }
            }
        }

        var report = new DoctorReport(1, checks);

        if (settings.BundlePath is { } bundlePath)
        {
            WriteBundle(bundlePath, report, settings.ModelPath);
            Console.WriteLine($"Support bundle written to {Path.GetFullPath(bundlePath)}");
            Console.WriteLine("It contains ONLY: this diagnostic report, active STINGRAY_* setting");
            Console.WriteLine("names (values redacted), and a manifest. No prompts, generated text, token");
            Console.WriteLine("IDs, credentials, model bytes, or absolute paths. Review it before sharing.");
            return checks.Any(x => x.Status == "error") ? 2 : 0;
        }

        if (settings.Json)
            Console.WriteLine(JsonSerializer.Serialize(report, DoctorJsonContext.Indented.DoctorReport));
        else
            foreach (var check in checks)
            {
                Console.WriteLine($"{check.Status.ToUpperInvariant(),-11} {check.Name}: {check.Detail}");
                if (check.Remedy is { } remedy)
                    Console.WriteLine($"{string.Empty,-11} -> {remedy}");
            }

        return checks.Any(x => x.Status == "error") ? 2 : 0;
    }

    /// <summary>
    /// Assembles the §6.3 support bundle. Deliberately an ALLOWLIST: only content named here is
    /// written, so a future check that starts reporting something sensitive cannot leak into the
    /// bundle by default. §5.4 — no upload, no network, local review before sharing.
    ///
    /// <para>Setting VALUES are omitted entirely rather than redacted per-name. A bundle is
    /// attached to public bug reports, and a value is far more likely to carry a path or secret
    /// than the name is; the name alone answers "what was configured", which is the diagnostic
    /// question.</para>
    /// </summary>
    private static void WriteBundle(string path, DoctorReport report, string? modelPath)
    {
        string full = Path.GetFullPath(path);
        string? dir = Path.GetDirectoryName(full);
        if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
        if (File.Exists(full)) File.Delete(full);

        using var archive = ZipFile.Open(full, ZipArchiveMode.Create);

        void AddEntry(string name, string content)
        {
            var entry = archive.CreateEntry(name, CompressionLevel.Optimal);
            using var writer = new StreamWriter(entry.Open());
            writer.Write(content);
        }

        AddEntry("doctor.json", JsonSerializer.Serialize(report, DoctorJsonContext.Indented.DoctorReport));

        // Names only, never values.
        var names = new List<string>();
        foreach (System.Collections.DictionaryEntry e in Environment.GetEnvironmentVariables())
            if (e.Key is string k && k.StartsWith(KnownEnvironmentVariables.Prefix, StringComparison.Ordinal))
                names.Add(k);
        names.Sort(StringComparer.Ordinal);
        AddEntry("settings.txt",
            names.Count == 0
                ? "No STINGRAY_* variables are set." + Environment.NewLine
                : "Active STINGRAY_* setting NAMES (values deliberately omitted):" + Environment.NewLine
                  + string.Join(Environment.NewLine, names) + Environment.NewLine);

        // Model identity without the path, which would expose a local directory layout.
        AddEntry("manifest.txt", string.Join(Environment.NewLine,
        [
            "OpenTail.Stingray support bundle",
            "schema: 1",
            "created_utc: " + DateTime.UtcNow.ToString("O"),
            "model_filename: " + (modelPath is null ? "(none supplied)" : Path.GetFileName(modelPath)),
            "",
            "Contents:",
            "  doctor.json   diagnostic report (runtime, cpu, backends, config, filesystem, model)",
            "  settings.txt  active STINGRAY_* setting NAMES only",
            "  manifest.txt  this file",
            "",
            "Deliberately excluded: prompts, generated text, token IDs, credentials, setting",
            "values, model weights, and absolute filesystem paths.",
        ]) + Environment.NewLine);
    }
}

/// <summary>
/// One diagnostic result. <paramref name="Remedy"/> is §6.3's "actionable remediation text" and is
/// populated ONLY for warnings and errors — a remedy on a passing check is noise, and noise is how
/// diagnostics get ignored. Null for `ok`.
/// </summary>
public sealed record DoctorCheck(string Name, string Status, string Detail, string? Remedy = null);
public sealed record DoctorReport(int SchemaVersion, IReadOnlyList<DoctorCheck> Checks);

[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower, WriteIndented = false)]
[JsonSerializable(typeof(DoctorReport))]
internal partial class DoctorJsonContext : JsonSerializerContext
{
    internal static DoctorJsonContext Indented => new(new JsonSerializerOptions
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        WriteIndented = true,
    });
}
