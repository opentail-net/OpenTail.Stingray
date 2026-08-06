using System.Globalization;
using System.Reflection;
using System.Text;
using System.Text.RegularExpressions;
using OpenTail.Stingray.Vulkan;

// SpirvGen — regenerates src/OpenTail.Stingray.Vulkan/Shaders.Precompiled.g.cs.
//
// Reflects every `internal const string` GLSL shader on OpenTail.Stingray.Vulkan.Shaders,
// compiles each to SPIR-V via ShaderCompiler.Compile, computes the deterministic
// ShaderCompiler.StableHash of the exact const value, and emits a switch keyed by that hash
// returning a static readonly byte[] per shader. The file is rewritten wholesale, so stale
// entries are dropped. Note: Compile consults the linked-in table first, so unchanged
// shaders return their committed bytes (identical) while an edited shader's new hash misses
// and re-runs glslc — producing SPIR-V that matches what the runtime would compile.

// --only Name1,Name2 rewrites ONLY those entries in place, leaving the other ~88 blobs
// byte-identical. This is the safe way to land a shader edit: the local glslc is much newer
// than whatever produced the committed table, so a full regeneration drifts roughly a third
// of the shaders (verified sound on one AMD iGPU, but that is not evidence for other
// drivers). Splicing keeps the diff to the shader that actually changed.
string? only = null;
for (int a = 0; a < args.Length - 1; a++)
    if (args[a] == "--only") only = args[a + 1];

// Always recompile from GLSL source — never reuse the (possibly stale) committed table —
// so regeneration picks up shader edits and glslc-flag changes.
ShaderCompiler.BypassPrecompiled = true;

string repoRoot = FindRepoRoot();
string outPath = Path.Combine(repoRoot, "src", "OpenTail.Stingray.Vulkan", "Shaders.Precompiled.g.cs");

// Reflect the GLSL shader const strings, sorted by name for stable diffs.
//
// PRIVATE consts are deliberately excluded: they are shader FRAGMENTS, not shaders. Some kernels
// exist in two variants that differ only in one helper function (e.g. MatVecBatchedQ4KInt8 and
// MatVecBatchedQ4KInt8Dp4a share a prologue and body, differing only in whether dot4x8u is the
// hand-written loop or the dotPacked4x8AccSatEXT intrinsic). Splitting the common text into
// `private const` pieces and concatenating them keeps the variants from drifting apart while
// staying a compile-time constant — but the pieces are not independently compilable GLSL, so
// feeding them to glslc would abort regeneration. Visibility is the marker: `internal` = a
// complete shader to precompile, `private` = a fragment that only exists to be concatenated.
var shaders = typeof(Shaders)
    .GetFields(BindingFlags.NonPublic | BindingFlags.Public | BindingFlags.Static)
    .Where(f => f.IsLiteral && f.FieldType == typeof(string) && !f.IsPrivate)
    .Select(f => (Name: f.Name, Source: (string)f.GetRawConstantValue()!))
    .OrderBy(s => s.Name, StringComparer.Ordinal)
    .ToList();

if (shaders.Count == 0)
{
    Console.Error.WriteLine("SpirvGen: no shader const strings found on OpenTail.Stingray.Vulkan.Shaders.");
    return 1;
}

if (only is not null)
{
    var wanted = only.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
    var missing = wanted.Where(w => !shaders.Any(s => s.Name == w)).ToList();
    if (missing.Count > 0)
    {
        Console.Error.WriteLine($"SpirvGen: --only named unknown shader(s): {string.Join(", ", missing)}");
        return 1;
    }
    shaders = shaders.Where(s => wanted.Contains(s.Name, StringComparer.Ordinal)).ToList();
}

// Compile each shader and check for stable-hash collisions (would silently lose entries).
// A handful of shaders (SgemmBf16, SgemmFp8) require GLSL extensions that the bundled
// shaderc/glslc does not support; the Vulkan backend already wraps their pipeline creation
// in try/catch and falls back, so we record them as skipped (with their hash) rather than
// abort. ANY OTHER compile failure is fatal — that would be a real regression.
var entries = new List<(string Name, ulong Hash, byte[] Spirv)>(shaders.Count);
var skipped = new List<(string Name, ulong Hash, string Reason)>();
var seenHashes = new Dictionary<ulong, string>();
foreach (var (name, source) in shaders)
{
    ulong hash = ShaderCompiler.StableHash(source);
    if (seenHashes.TryGetValue(hash, out var other))
    {
        Console.Error.WriteLine($"SpirvGen: stable-hash collision between '{name}' and '{other}' (0x{hash:X16}).");
        return 1;
    }
    seenHashes[hash] = name;

    byte[] spirv;
    try
    {
        spirv = ShaderCompiler.Compile(source);
    }
    catch (Exception ex) when (IsUnsupportedExtension(ex.Message, out var ext))
    {
        Console.WriteLine($"SpirvGen: skipping '{name}' — glslc lacks extension {ext} (backend falls back at runtime).");
        skipped.Add((name, hash, ext));
        continue;
    }
    catch (Exception ex)
    {
        Console.Error.WriteLine($"SpirvGen: glslc failed compiling '{name}': {ex.Message}");
        return 1;
    }

    if (spirv.Length == 0 || spirv.Length % 4 != 0 ||
        !(spirv[0] == 0x03 && spirv[1] == 0x02 && spirv[2] == 0x23 && spirv[3] == 0x07))
    {
        Console.Error.WriteLine($"SpirvGen: '{name}' did not produce valid SPIR-V (len={spirv.Length}).");
        return 1;
    }

    entries.Add((name, hash, spirv));
}

if (only is not null)
    return Splice(outPath, entries);

string generated = Emit(entries, skipped);
File.WriteAllText(outPath, generated);

Console.WriteLine($"SpirvGen: wrote {entries.Count} precompiled + {skipped.Count} skipped " +
    $"= {entries.Count + skipped.Count} total ({generated.Length / 1024} KB) to {outPath}");
return 0;

// Rewrite just the named entries in the committed table: swap each shader's hash key and its
// SPIR-V blob, keeping its existing _sN slot so every other line of the file is untouched.
// Only handles shaders ALREADY in the table — adding a brand-new one changes Count and the
// slot numbering, which is what a full Emit is for.
static int Splice(string outPath, List<(string Name, ulong Hash, byte[] Spirv)> entries)
{
    string text = File.ReadAllText(outPath);
    foreach (var (name, hash, spirv) in entries)
    {
        // Plain string matching, not regex: the committed file is CRLF and the emitted
        // format is fixed, so anchoring on the exact trailing comment is both simpler and
        // immune to the line-ending traps that make a multiline regex silently match nothing.
        // The comment also makes the match unambiguous when one shader name prefixes another
        // (Attention vs AttentionBf16), since it is the end of the line.
        string caseTail = $"return true; // {name}";
        int ci = FindLineEndingWith(text, caseTail);
        if (ci < 0)
        {
            // Not in the table yet — append it rather than forcing a full regeneration, which
            // would rewrite all ~88 other blobs with a much newer glslc and drift a third of them.
            if (!Append(ref text, name, hash, spirv)) return 1;
            Console.WriteLine($"SpirvGen --only: APPENDED '{name}', hash 0x{hash:X16}, {spirv.Length} bytes.");
            continue;
        }
        int caseStart = text.LastIndexOf('\n', ci) + 1;
        string caseLine = text[caseStart..(ci + caseTail.Length)];
        int slotAt = caseLine.IndexOf("_s", StringComparison.Ordinal);
        int slotEnd = caseLine.IndexOf(';', slotAt);
        if (slotAt < 0 || slotEnd < 0)
        {
            Console.Error.WriteLine($"SpirvGen --only: could not parse the slot index out of: {caseLine.Trim()}");
            return 1;
        }
        string slot = caseLine[(slotAt + 2)..slotEnd];
        string indent = caseLine[..(caseLine.Length - caseLine.TrimStart().Length)];
        text = text.Remove(caseStart, caseLine.Length).Insert(caseStart,
            $"{indent}case 0x{hash:X16}UL: spirv = _s{slot}; return true; // {name}");

        // Replace the blob's comment line + array line as one unit so the byte count in the
        // comment can never disagree with the array it labels.
        string arrDecl = $"private static readonly byte[] _s{slot} = {{ ";
        int ai = text.IndexOf(arrDecl, StringComparison.Ordinal);
        if (ai < 0)
        {
            Console.Error.WriteLine($"SpirvGen --only: found the case line for '{name}' but not its _s{slot} array.");
            return 1;
        }
        int arrLineStart = text.LastIndexOf('\n', ai) + 1;
        string arrIndent = text[arrLineStart..ai];
        int commentStart = text.LastIndexOf('\n', arrLineStart - 2) + 1; // the "// Name (N bytes)" line
        int arrEnd = text.IndexOf("};", ai, StringComparison.Ordinal) + 2;

        var sb = new StringBuilder();
        sb.Append(arrIndent).Append("// ").Append(name).Append(" (").Append(spirv.Length).Append(" bytes)")
          .Append(text[(arrLineStart - 2)] == '\r' ? "\r\n" : "\n");
        sb.Append(arrIndent).Append("private static readonly byte[] _s").Append(slot).Append(" = { ");
        for (int b = 0; b < spirv.Length; b++)
        {
            if (b > 0) sb.Append(',');
            sb.Append("0x").Append(spirv[b].ToString("x2", CultureInfo.InvariantCulture));
        }
        sb.Append(" };");
        text = text.Remove(commentStart, arrEnd - commentStart).Insert(commentStart, sb.ToString());

        Console.WriteLine($"SpirvGen --only: spliced '{name}' -> slot _s{slot}, hash 0x{hash:X16}, {spirv.Length} bytes.");
    }

    File.WriteAllText(outPath, text);
    return 0;
}

// Add a shader the table has never seen: bump Count, insert one hash-keyed case ahead of the
// default, and append one _sN array. N is the OLD Count, which is exactly the next free slot
// because Emit numbers slots 0..Count-1 densely.
static bool Append(ref string text, string name, ulong hash, byte[] spirv)
{
    const string countMarker = "internal static int Count => ";
    int cAt = text.IndexOf(countMarker, StringComparison.Ordinal);
    int cEnd = cAt < 0 ? -1 : text.IndexOf(';', cAt);
    if (cAt < 0 || cEnd < 0)
    {
        Console.Error.WriteLine("SpirvGen --only: could not find the Count property to bump.");
        return false;
    }
    string countText = text[(cAt + countMarker.Length)..cEnd];
    if (!int.TryParse(countText, out int slot))
    {
        Console.Error.WriteLine($"SpirvGen --only: could not parse Count from '{countText}'.");
        return false;
    }
    text = text.Remove(cAt + countMarker.Length, countText.Length)
               .Insert(cAt + countMarker.Length, (slot + 1).ToString(CultureInfo.InvariantCulture));

    const string defaultMarker = "default: spirv = System.Array.Empty<byte>(); return false;";
    int dAt = text.IndexOf(defaultMarker, StringComparison.Ordinal);
    if (dAt < 0)
    {
        Console.Error.WriteLine("SpirvGen --only: could not find the switch default to insert before.");
        return false;
    }
    int dLineStart = text.LastIndexOf('\n', dAt) + 1;
    string dIndent = text[dLineStart..dAt];
    string nl = dLineStart >= 2 && text[dLineStart - 2] == '\r' ? "\r\n" : "\n";
    text = text.Insert(dLineStart,
        $"{dIndent}case 0x{hash:X16}UL: spirv = _s{slot}; return true; // {name}{nl}");

    // Append the blob just before the class's closing brace (the final "}" in the file).
    int close = text.LastIndexOf('}');
    if (close < 0) { Console.Error.WriteLine("SpirvGen --only: no closing brace found."); return false; }
    var sb = new StringBuilder();
    sb.Append("    // ").Append(name).Append(" (").Append(spirv.Length).Append(" bytes)").Append(nl);
    sb.Append("    private static readonly byte[] _s").Append(slot).Append(" = { ");
    for (int b = 0; b < spirv.Length; b++)
    {
        if (b > 0) sb.Append(',');
        sb.Append("0x").Append(spirv[b].ToString("x2", CultureInfo.InvariantCulture));
    }
    sb.Append(" };").Append(nl);
    text = text.Insert(close, sb.ToString());
    return true;
}

// Index of the start of the LAST occurrence of `tail` that sits at the end of a line.
// Requiring the line ending is what keeps "// Attention" from matching "// AttentionBf16".
static int FindLineEndingWith(string text, string tail)
{
    for (int i = text.IndexOf(tail, StringComparison.Ordinal); i >= 0;
         i = text.IndexOf(tail, i + 1, StringComparison.Ordinal))
    {
        int after = i + tail.Length;
        if (after >= text.Length || text[after] == '\r' || text[after] == '\n') return i;
    }
    return -1;
}

// glslc's "extension not supported" message names the extension; capture it so the
// generated file (and tests) record exactly which shaders are intentionally absent.
static bool IsUnsupportedExtension(string message, out string ext)
{
    ext = "";
    const string marker = "extension not supported: ";
    int idx = message.IndexOf(marker, StringComparison.Ordinal);
    if (idx < 0) return false;
    int start = idx + marker.Length;
    int end = start;
    while (end < message.Length && (char.IsLetterOrDigit(message[end]) || message[end] == '_')) end++;
    ext = message.Substring(start, end - start);
    return ext.Length > 0;
}

static string Emit(List<(string Name, ulong Hash, byte[] Spirv)> entries,
    List<(string Name, ulong Hash, string Reason)> skipped)
{
    // Precompute the 256 byte→hex strings once instead of ToString("x2") per byte
    // (the SPIR-V blobs total ~150 KB → hundreds of thousands of formats otherwise).
    var hex = new string[256];
    for (int i = 0; i < 256; i++) hex[i] = i.ToString("x2", CultureInfo.InvariantCulture);

    var sb = new StringBuilder();
    sb.AppendLine("// <auto-generated/> — regenerated by tools/SpirvGen (scripts/gen-spirv.ps1). DO NOT EDIT.");
    sb.AppendLine("// Maps the FNV-1a 64-bit ShaderCompiler.StableHash of each GLSL shader source in");
    sb.AppendLine("// Shaders.cs to its precompiled SPIR-V, so the published NativeAOT binary needs no");
    sb.AppendLine("// glslc at runtime. This file is committed and compiles like any other source (no");
    sb.AppendLine("// glslc needed to BUILD); glslc is only needed by the dev-run generator.");
    sb.AppendLine("namespace OpenTail.Stingray.Vulkan;");
    sb.AppendLine();
    sb.AppendLine("internal static class ShadersPrecompiled");
    sb.AppendLine("{");
    sb.AppendLine("    /// <summary>");
    sb.AppendLine("    /// Number of distinct precompiled shader entries in the table. Used by tests to");
    sb.AppendLine("    /// catch a partial/stale table.");
    sb.AppendLine($"    /// </summary>");
    sb.AppendLine($"    internal static int Count => {entries.Count};");
    sb.AppendLine();
    sb.AppendLine("    /// <summary>");
    sb.AppendLine("    /// Shaders intentionally NOT precompiled because the generator's glslc lacks the");
    sb.AppendLine("    /// required GLSL extension. The Vulkan backend wraps these in try/catch and falls");
    sb.AppendLine("    /// back at runtime, so they miss this table by design. (Count + SkippedShaders.Length");
    sb.AppendLine("    /// must equal the total number of Shaders consts.)");
    sb.AppendLine("    /// </summary>");
    if (skipped.Count == 0)
    {
        sb.AppendLine("    internal static readonly string[] SkippedShaders = System.Array.Empty<string>();");
    }
    else
    {
        sb.AppendLine("    internal static readonly string[] SkippedShaders =");
        sb.AppendLine("    {");
        foreach (var (name, _, reason) in skipped)
            sb.AppendLine($"        \"{name}\", // requires {reason}");
        sb.AppendLine("    };");
    }
    sb.AppendLine();
    sb.AppendLine("    /// <summary>");
    sb.AppendLine("    /// Look up precompiled SPIR-V by the FNV-1a 64-bit stable hash of its GLSL source.");
    sb.AppendLine("    /// </summary>");
    sb.AppendLine("    internal static bool TryGet(ulong h, out byte[] spirv)");
    sb.AppendLine("    {");
    sb.AppendLine("        switch (h)");
    sb.AppendLine("        {");
    for (int i = 0; i < entries.Count; i++)
        sb.AppendLine($"            case 0x{entries[i].Hash:X16}UL: spirv = _s{i}; return true; // {entries[i].Name}");
    sb.AppendLine("            default: spirv = System.Array.Empty<byte>(); return false;");
    sb.AppendLine("        }");
    sb.AppendLine("    }");
    sb.AppendLine();
    for (int i = 0; i < entries.Count; i++)
    {
        sb.AppendLine($"    // {entries[i].Name} ({entries[i].Spirv.Length} bytes)");
        sb.Append($"    private static readonly byte[] _s{i} = {{ ");
        var spv = entries[i].Spirv;
        for (int b = 0; b < spv.Length; b++)
        {
            if (b > 0) sb.Append(',');
            sb.Append("0x");
            sb.Append(hex[spv[b]]);
        }
        sb.AppendLine(" };");
    }
    sb.AppendLine("}");
    return sb.ToString();
}

static string FindRepoRoot()
{
    var dir = new DirectoryInfo(AppContext.BaseDirectory);
    while (dir is not null)
    {
        if (File.Exists(Path.Combine(dir.FullName, "OpenTail.Stingray.slnx")))
            return dir.FullName;
        dir = dir.Parent;
    }
    throw new InvalidOperationException("SpirvGen: could not locate repo root (OpenTail.Stingray.slnx) from " + AppContext.BaseDirectory);
}
