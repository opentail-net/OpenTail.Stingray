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

    /// <summary>
    /// The model-vs-machine boundary (capabilities plan): the planner and the forward-pass selection decide what to run from a
    /// descriptor's <c>Capabilities</c> only. Descriptive reads (<c>Id</c>, <c>BackendLimitation</c>, used in explanations) are allowed.
    /// A new read of any other descriptor member from these files fails here; add it to <c>ArchitectureCapabilities</c> instead.
    /// </summary>
    [Fact]
    public void PlannerAndSelection_ReadOnlyCapabilities_FromTheDescriptor()
    {
        string engine = Path.Combine(Root(), "src", "OpenTail.Stingray.Engine");
        var files = Directory.EnumerateFiles(Path.Combine(engine, "Planning"), "*.cs")
            .Append(Path.Combine(engine, "Architectures", "ForwardPassSelection.cs"));
        var allowed = new HashSet<string>(StringComparer.Ordinal) { "Capabilities", "Id", "BackendLimitation" };
        var rx = new System.Text.RegularExpressions.Regex(@"\bdescriptor\??\.(\w+)");
        var bad = new List<string>();
        foreach (var file in files)
            foreach (var (line, i) in File.ReadLines(file).Select((l, i) => (l, i + 1)))
            {
                if (line.TrimStart().StartsWith("//", StringComparison.Ordinal)) continue;
                foreach (System.Text.RegularExpressions.Match m in rx.Matches(line))
                    if (!allowed.Contains(m.Groups[1].Value))
                        bad.Add($"{Path.GetFileName(file)}:{i}: descriptor.{m.Groups[1].Value}");
            }
        Assert.True(bad.Count == 0, "Planner/selection read a descriptor member other than Capabilities:\n  " + string.Join("\n  ", bad));
    }

    // Phase 2: the chat protocol now comes from the descriptor. Pin the lookup edge cases the old per-protocol arch lists had.
    [Theory]
    [InlineData("gemma4", "gemma")]
    [InlineData("GEMMA3", "gemma")]            // case-insensitive, as before
    [InlineData("llama", "llama3")]
    [InlineData("llama4", "llama4")]
    [InlineData("granitehybrid", "granite")]
    [InlineData("qwen35", "chatml")]
    [InlineData("gemma3n", "chatml")]          // never listed under the gemma protocol
    [InlineData("not-a-registered-architecture", "chatml")]
    [InlineData(null, "chatml")]
    public void ChatProtocol_ResolvesThroughTheDescriptor(string? architecture, string expectedProtocol) =>
        Assert.Equal(expectedProtocol, ChatProtocolRegistry.For(architecture).Id);

    // Phase 3: descriptors own their structural traits; the by-name table in Core serves direct callers of the baseline parser and
    // qwen3vlmoe (no descriptor). For every architecture the table knows, the descriptor must agree. A NEW architecture is not in the
    // table and must not have to be: declaring Traits on its descriptor is enough (found by the Phase 5 exercise, which this test
    // originally broke by demanding a Core edit).
    [Fact]
    public void DescriptorTraits_MatchTheByNameTable()
    {
        var bad = ArchitectureRegistry.All
            .Where(d => ModelArchitectureTraits.Legacy(d.Id) != ModelArchitectureTraits.None
                        && (d.Traits ?? ModelArchitectureTraits.None) != ModelArchitectureTraits.Legacy(d.Id))
            .Select(d => d.Id).ToList();
        Assert.True(bad.Count == 0, "Descriptor Traits differ from ModelArchitectureTraits.Legacy for: " + string.Join(", ", bad));
    }

    [Fact]
    public void Traits_DriveTheBaselineParser_NotTheArchitectureName()
    {
        // A file whose metadata namespace is an unknown name but whose descriptor says Mamba-2: the traits decide the layout.
        var meta = new Dictionary<string, object>
        {
            ["general.architecture"] = "granitehybrid",
            ["granitehybrid.block_count"] = 4u,
            ["granitehybrid.embedding_length"] = 64u,
            ["granitehybrid.attention.head_count"] = 4u,
            ["granitehybrid.attention.head_count_kv"] = new uint[] { 0, 2, 0, 2 },
            ["granitehybrid.feed_forward_length"] = 128u,
            ["granitehybrid.context_length"] = 256u,
        };
        var viaDescriptor = ArchitectureModelResolver.ResolveHyperparams(meta);
        Assert.NotNull(viaDescriptor.IsMamba2Layer);
        Assert.Contains(viaDescriptor.IsMamba2Layer!, b => b);

        // The same keys under a name nobody registered get no hybrid treatment (no traits, no Legacy entry).
        var renamed = meta.ToDictionary(kv => kv.Key.Replace("granitehybrid", "zz-unknown"), kv => kv.Key == "general.architecture" ? (object)"zz-unknown" : kv.Value);
        Assert.Null(ArchitectureModelResolver.ResolveHyperparams(renamed).IsMamba2Layer);
    }
}
