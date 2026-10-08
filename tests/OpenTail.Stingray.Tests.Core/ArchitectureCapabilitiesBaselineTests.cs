using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Guards for docs/2-coverage/2026-10-08-architecture-capabilities-plan.md (Phase 0).
///
/// <list type="bullet">
/// <item><b>Facts baseline</b>: one line per registered architecture with everything the rest of the engine learns from its
/// descriptor or looks up by its id (admission, family, backends, batching, image/audio input, chat protocol, tool-call
/// adapter). The refactor phases must leave it byte-identical unless a phase says otherwise.</item>
/// <item><b>Literal guard (ratchet)</b>: counts quoted registered architecture ids in <c>src/**/*.cs</c> outside
/// <c>Engine/Architectures/</c>. A file may never gain literals; it must lower its baseline when it loses them, so the
/// number only goes down.</item>
/// </list>
///
/// Regenerate a baseline after an intended change with <c>STINGRAY_UPDATE_BASELINE=1</c>.
/// </summary>
public sealed class ArchitectureCapabilitiesBaselineTests
{
    private static string Root()
    {
        string? root = AppContext.BaseDirectory;
        while (root is not null && !File.Exists(Path.Combine(root, "CLAUDE.md")))
            root = Path.GetDirectoryName(root);
        Assert.NotNull(root);
        return root!;
    }

    private static string BaselineDir() =>
        Path.Combine(Root(), "tests", "OpenTail.Stingray.Tests.Core", "Baselines");

    private static bool Update => Environment.GetEnvironmentVariable("STINGRAY_UPDATE_BASELINE") == "1";

    private static void Compare(string fileName, IReadOnlyList<string> actual, string what)
    {
        string path = Path.Combine(BaselineDir(), fileName);
        if (Update)
        {
            Directory.CreateDirectory(BaselineDir());
            File.WriteAllLines(path, actual);
            return;
        }
        Assert.True(File.Exists(path), $"Missing baseline {path}. Generate it with STINGRAY_UPDATE_BASELINE=1.");
        var expected = File.ReadAllLines(path);
        var missing = expected.Except(actual).ToList();
        var added = actual.Except(expected).ToList();
        Assert.True(missing.Count == 0 && added.Count == 0,
            $"{what} changed.\n  expected-but-gone ({missing.Count}):\n    {string.Join("\n    ", missing.Take(12))}\n" +
            $"  new ({added.Count}):\n    {string.Join("\n    ", added.Take(12))}\n" +
            "If this is intended, regenerate with STINGRAY_UPDATE_BASELINE=1 and say why in the commit.");
    }

    [Fact]
    public void ArchitectureFacts_MatchBaseline()
    {
        var lines = new List<string>();
        foreach (var d in ArchitectureRegistry.All.OrderBy(d => d.Id, StringComparer.Ordinal))
        {
            string chat = ChatProtocolRegistry.For(d.Id).Id;
            string tool = ToolCallAdapterRegistry.Get(d.Id).Architecture;
            lines.Add(string.Join("|",
                d.Id, d.Status, d.ForwardPassFamily, d.SupportedBackends,
                $"batch={d.SupportsContinuousBatching}", $"canBatchPredicate={d.CanBatchPredicate is not null}",
                $"img={d.SupportsImageInput}", $"aud={d.SupportsAudioInput}",
                $"neox={d.UsesNeoxRope}", $"thinkOff={d.ThinkingDefaultOff}",
                $"fallbackChat={d.FallbackChat}", $"chatProtocol={chat}", $"toolAdapter={tool}",
                $"semantics={d.ApplyModelSemantics is not null}", $"aliases={string.Join(",", d.Aliases.OrderBy(a => a, StringComparer.Ordinal))}"));
        }
        Compare("ArchitectureFacts.txt", lines, "Architecture facts");
    }

    [Fact]
    public void ArchitectureLiterals_OutsideTheArchitectureFolder_OnlyShrink()
    {
        string src = Path.Combine(Root(), "src");
        var ids = ArchitectureRegistry.All
            .SelectMany(d => d.Aliases.Prepend(d.Id))
            .Where(s => s.Length >= 3)
            .Distinct(StringComparer.Ordinal)
            .Select(s => "\"" + s + "\"")
            .ToArray();

        var lines = new List<string>();
        foreach (var file in Directory.EnumerateFiles(src, "*.cs", SearchOption.AllDirectories))
        {
            string rel = Path.GetRelativePath(Root(), file).Replace('\\', '/');
            if (rel.Contains("/obj/") || rel.Contains("/bin/")) continue;
            if (rel.StartsWith("src/OpenTail.Stingray.Engine/Architectures/", StringComparison.Ordinal)) continue;
            // Separate registries with their own model families; not LLM architectures.
            if (rel.StartsWith("src/OpenTail.Stingray.Audio/", StringComparison.Ordinal)
                || rel.StartsWith("src/OpenTail.Stingray.Diffusion/", StringComparison.Ordinal)) continue;

            int count = 0;
            foreach (var line in File.ReadLines(file))
            {
                string t = line.TrimStart();
                if (t.StartsWith("//", StringComparison.Ordinal)) continue;
                foreach (var id in ids)
                {
                    int at = 0;
                    while ((at = t.IndexOf(id, at, StringComparison.Ordinal)) >= 0) { count++; at += id.Length; }
                }
            }
            if (count > 0) lines.Add($"{rel}|{count}");
        }
        lines.Sort(StringComparer.Ordinal);

        string path = Path.Combine(BaselineDir(), "ArchitectureLiterals.txt");
        if (Update)
        {
            Directory.CreateDirectory(BaselineDir());
            File.WriteAllLines(path, lines);
            return;
        }
        Assert.True(File.Exists(path), $"Missing baseline {path}. Generate it with STINGRAY_UPDATE_BASELINE=1.");
        var baseline = File.ReadAllLines(path).Select(l => l.Split('|')).ToDictionary(p => p[0], p => int.Parse(p[1]));
        var now = lines.Select(l => l.Split('|')).ToDictionary(p => p[0], p => int.Parse(p[1]));

        var grew = now.Where(kv => kv.Value > baseline.GetValueOrDefault(kv.Key)).Select(kv => $"{kv.Key}: {baseline.GetValueOrDefault(kv.Key)} -> {kv.Value}").ToList();
        var shrank = baseline.Where(kv => now.GetValueOrDefault(kv.Key) < kv.Value).Select(kv => $"{kv.Key}: {kv.Value} -> {now.GetValueOrDefault(kv.Key)}").ToList();
        Assert.True(grew.Count == 0,
            "New architecture-id literals outside Engine/Architectures/ (put the knowledge on the descriptor instead):\n  " + string.Join("\n  ", grew));
        Assert.True(shrank.Count == 0,
            "Literals were removed (good). Lower the baseline so it cannot creep back: STINGRAY_UPDATE_BASELINE=1.\n  " + string.Join("\n  ", shrank));
    }
}
