namespace OpenTail.Stingray.Cli;

/// <summary>
/// Golden-parity scaffold generator for a new vision (mmproj) architecture — pure C#, no Python
/// (this project ships no Python anywhere; the engine and its tooling are managed .NET code only,
/// per README.md/CLAUDE.md).
///
/// <para>Every existing vision parity test in this repo (Llava/Pixtral/GLM-4.6V/HunyuanVL/
/// Exaone4/MiMoVl/Qwen2.5-VL/Gemma4UV) was hand-written from scratch against the same shape of
/// boilerplate: open the mmproj GGUF, read its tensor inventory and clip.vision.* metadata, and
/// pair that with an xUnit "&lt;Arch&gt;VisionEmbedderParityTests.cs" comparing the C# encoder's output
/// against an independent oracle. This command automates the boilerplate half — real tensor names/
/// shapes/metadata for THIS checkpoint, so nothing is guessed — and leaves the actual
/// per-architecture math (attention masking, RoPE convention, projector shape) as explicit TODOs,
/// same as every hand-written one needed.</para>
///
/// <para>The independent oracle itself is NOT auto-generated here. Earlier vision parity tests in
/// this project used a hand-written numpy reimplementation of llama.cpp's mtmd C++ — that pattern
/// is retired going forward since it put Python source into this repo. Use the real, already-
/// vendored <c>tools/llama.cpp/llama-mtmd-cli.exe</c> (or <c>llama-mtmd-debug.exe</c>) to capture
/// golden embeddings directly from genuine llama.cpp C++ instead — the same "run the real external
/// reference binary" pattern this project already uses for text-generation parity receipts
/// (llama-tokenize/llama-server), just extended to vision.</para>
/// </summary>
public sealed class GenVisionScaffoldCommand : Command<GenVisionScaffoldCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        [CommandOption("-m|--mmproj <PATH>")]
        [Description("mmproj GGUF for the new architecture")]
        public string MmprojPath { get; init; } = "";

        [CommandOption("-a|--arch <NAME>")]
        [Description("Short architecture name, e.g. step3vl (used for the class/file name)")]
        public string Arch { get; init; } = "";
    }

    private static readonly string[] s_interestingMetaKeys =
    [
        "clip.vision.projector_type", "clip.vision.patch_size", "clip.vision.image_size",
        "clip.vision.embedding_length", "clip.vision.feed_forward_length",
        "clip.vision.attention.head_count", "clip.vision.attention.layer_norm_epsilon",
        "clip.vision.block_count", "clip.vision.rope.freq_base", "clip.vision.n_wa_pattern",
        "clip.vision.use_gelu", "clip.vision.use_silu",
    ];

    protected override int Execute(Settings settings, CancellationToken cancellation)
    {
        if (string.IsNullOrEmpty(settings.MmprojPath) || !File.Exists(settings.MmprojPath))
        {
            AnsiConsole.MarkupLine("[red]Error:[/] mmproj file not found. Use [yellow]-m <path>[/]");
            return 1;
        }
        if (string.IsNullOrWhiteSpace(settings.Arch))
        {
            AnsiConsole.MarkupLine("[red]Error:[/] give a short architecture name with [yellow]-a <name>[/] (e.g. step3vl)");
            return 1;
        }
        string arch = settings.Arch.Trim();
        string className = PascalCase(arch);

        using var model = GgufModel.Open(settings.MmprojPath);

        AnsiConsole.MarkupLine($"[bold]Tensor/metadata inventory[/] for {Markup.Escape(Path.GetFileName(settings.MmprojPath))} (arch={Markup.Escape(arch)})");
        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("[bold]Metadata[/]");
        foreach (string key in s_interestingMetaKeys)
        {
            string value = model.Metadata.TryGetValue(key, out var v) ? Convert.ToString(v) ?? "(present, unprintable)" : "(absent)";
            AnsiConsole.MarkupLine($"  {Markup.Escape(key)} = {Markup.Escape(value)}");
        }

        var grouped = model.Tensors
            .GroupBy(t => StripBlockIndex(t.Name))
            .Select(g => new { Suffix = g.Key, Count = g.Count(), Sample = g.First() })
            .OrderBy(g => g.Suffix, StringComparer.Ordinal)
            .ToList();
        int layerCount = model.Tensors
            .Select(t => TryExtractBlockIndex(t.Name))
            .Where(i => i is not null)
            .Select(i => i!.Value)
            .DefaultIfEmpty(-1)
            .Max() + 1;

        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine($"[bold]Tensor groups[/] (block_count detected: {layerCount})");
        foreach (var g in grouped)
            AnsiConsole.MarkupLine($"  {g.Suffix,-42} x{g.Count,-3} {g.Sample.DType,-9} [{string.Join(",", g.Sample.Dimensions.Take(g.Sample.NDimensions))}]");

        string testDir = FindTestDir();
        string csPath = Path.Combine(testDir, $"{className}VisionEmbedderParityTests.cs");
        if (File.Exists(csPath))
        {
            AnsiConsole.WriteLine();
            AnsiConsole.MarkupLine($"[yellow]SKIP:[/] {Markup.Escape(csPath)} already exists, not overwriting.");
        }
        else
        {
            Directory.CreateDirectory(testDir);
            File.WriteAllText(csPath, RenderCSharpTemplate(arch, className, grouped.Select(g => g.Suffix).ToList()));
            AnsiConsole.WriteLine();
            AnsiConsole.MarkupLine($"[green]Wrote[/] {Markup.Escape(csPath)}");
        }

        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("[bold]Next steps[/] (do not skip — this scaffold has no real math or golden values in it yet):");
        AnsiConsole.MarkupLine($"  1. Read the real reference: [yellow]tools/mtmd/models/{Markup.Escape(arch)}.cpp[/] and clip.cpp's shared build_vit().");
        AnsiConsole.MarkupLine($"  2. Implement the real \"{Markup.Escape(className)}VisionModel\"/\"{Markup.Escape(className)}VisionEncoder\" C# classes.");
        AnsiConsole.MarkupLine($"  3. Capture golden embeddings from the real reference: run [yellow]tools/llama.cpp/llama-mtmd-cli.exe[/]");
        AnsiConsole.MarkupLine($"     (or llama-mtmd-debug.exe) against {Markup.Escape(settings.MmprojPath)} + a test image, and save its output");
        AnsiConsole.MarkupLine($"     as tests/fixtures/{Markup.Escape(arch)}/output.f32 (or capture via an embedding-dump flag if it has one).");
        AnsiConsole.MarkupLine($"  4. Fill in the TODOs in {Markup.Escape(csPath)}, then run it and fix real bugs as they surface.");

        return 0;
    }

    private static string RenderCSharpTemplate(string arch, string className, IReadOnlyList<string> tensorSuffixes)
    {
        string inventoryComment = string.Join(Environment.NewLine, tensorSuffixes.Select(s => "///   " + s));
        return $$"""
using System.Runtime.InteropServices;
using System.Text.Json;

namespace OpenTail.Stingray.Tests.Vision;

/// <summary>
/// GENERATED SCAFFOLD (`stingray gen-vision-scaffold`) — parity of the C# "{{arch}}" vision
/// encoder against a golden reference captured from the real llama.cpp mtmd C++ binary
/// (tools/llama.cpp/llama-mtmd-cli.exe or llama-mtmd-debug.exe), NOT a hand-written reimplementation.
/// Follows the same pattern as <see cref="PixtralVisionEmbedderParityTests"/> /
/// <see cref="Glm4VisionEmbedderParityTests"/>, minus their now-retired numpy-reference step.
///
/// Real tensor-suffix inventory for the checkpoint this scaffold was generated from:
{{inventoryComment}}
///
/// TODO before this test is real:
///   - Implement the real "{{className}}VisionModel"/"{{className}}VisionEncoder" C# classes
///     (read tools/mtmd/models/{{arch}}.cpp and clip.cpp's shared build_vit() first — CLAUDE.md
///     rule 8: don't guess math that looks wrong without checking the real reference).
///   - Capture a real golden output for a specific test image from llama-mtmd-cli.exe/
///     llama-mtmd-debug.exe and save it as tests/fixtures/{{arch}}/output.f32 (plus meta.json
///     recording n_tokens/embd and whatever image preprocessing was used, so this test's input
///     image matches exactly what the reference embedded).
///   - Pick a real cosine/meanAbs threshold from a MEASURED first run, not an assumed one — every
///     existing parity test in this project set its threshold from what the first real run showed
///     (0.97-0.9995 range depending on quantization), not a guess.
/// </summary>
public class {{className}}VisionEmbedderParityTests
{
    private static float[] ReadF32(string path)
    {
        var bytes = File.ReadAllBytes(path);
        return MemoryMarshal.Cast<byte, float>(bytes).ToArray();
    }

    [Fact(Skip = "scaffold: capture a real golden reference and wire up the real encoder below first")]
    public void Forward_MatchesRealReference()
    {
        var fx = VisionTestPaths.FindFixtureDir("{{arch}}");
        if (fx is null) return;
        var outPath = Path.Combine(fx, "output.f32");
        var metaPath = Path.Combine(fx, "meta.json");
        if (!File.Exists(outPath) || !File.Exists(metaPath)) return;

        using var doc = JsonDocument.Parse(File.ReadAllText(metaPath));
        int nTokExpected = doc.RootElement.GetProperty("n_tokens").GetInt32();
        float[] golden = ReadF32(outPath);

        // TODO: replace with the real {{className}}VisionModel.Open(mmproj) / encoder.Forward(...) call.
        throw new NotImplementedException("wire up the real {{arch}} vision encoder here");
    }
}

""";
    }

    private static string StripBlockIndex(string name)
    {
        if (!name.Contains(".blk.", StringComparison.Ordinal)) return name;
        int start = name.IndexOf(".blk.", StringComparison.Ordinal) + 5;
        int dot = name.IndexOf('.', start);
        return dot < 0 ? name : name[..(start - 5)] + ".blk.N." + name[(dot + 1)..];
    }

    private static int? TryExtractBlockIndex(string name)
    {
        const string marker = ".blk.";
        int idx = name.IndexOf(marker, StringComparison.Ordinal);
        if (idx < 0) return null;
        int start = idx + marker.Length;
        int end = name.IndexOf('.', start);
        if (end < 0) return null;
        return int.TryParse(name.AsSpan(start, end - start), out int n) ? n : null;
    }

    private static string PascalCase(string arch) =>
        string.Concat(arch.Split(['_', '-'], StringSplitOptions.RemoveEmptyEntries)
            .Select(p => char.ToUpperInvariant(p[0]) + p[1..]));

    private static string FindTestDir()
    {
        string candidate = Path.Combine("tests", "OpenTail.Stingray.Tests.Vision");
        return Directory.Exists(candidate) ? candidate : candidate; // created on write if absent
    }
}
