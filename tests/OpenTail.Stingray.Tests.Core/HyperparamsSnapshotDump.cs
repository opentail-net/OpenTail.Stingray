using System.Globalization;
using System.Reflection;
using System.Text;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using Xunit;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Migration equivalence tool (not an assertion test). With STINGRAY_HP_DUMP=&lt;output file&gt; and
/// STINGRAY_HP_MODELS=&lt;dir&gt;, writes every public ModelHyperparams property for every GGUF header found under the
/// models dir. Run it before and after moving a rule out of ModelGraph.cs and diff the two files.
/// </summary>
public sealed class HyperparamsSnapshotDump
{
    [Fact]
    public void DumpAllRealHeaders()
    {
        var output = Environment.GetEnvironmentVariable("STINGRAY_HP_DUMP");
        var dir = Environment.GetEnvironmentVariable("STINGRAY_HP_MODELS");
        Assert.SkipWhen(string.IsNullOrEmpty(output) || string.IsNullOrEmpty(dir), "snapshot tool: env not set");

        var props = typeof(ModelHyperparams).GetProperties(BindingFlags.Public | BindingFlags.Instance)
            .Where(p => p.GetIndexParameters().Length == 0).OrderBy(p => p.Name, StringComparer.Ordinal).ToArray();
        var sb = new StringBuilder();
        foreach (var file in Directory.EnumerateFiles(dir!, "*.gguf", SearchOption.AllDirectories)
                     .OrderBy(f => f, StringComparer.OrdinalIgnoreCase))
        {
            string name = Path.GetFileName(file);
            try
            {
                using var g = GgufModel.Open(file);
                var hp = ArchitectureModelResolver.ResolveHyperparams(g); // HPCALL
                sb.AppendLine($"## {name} arch={(g.Metadata.TryGetValue("general.architecture", out var ga) ? ga : "?")}");
                foreach (var p in props) sb.AppendLine($"{p.Name}={Fmt(p.GetValue(hp))}");
            }
            catch (Exception e)
            {
                sb.AppendLine($"## {name}\nERROR={e.GetType().Name}");
            }
        }
        File.WriteAllText(output!, sb.ToString());
    }

    private static string Fmt(object? v) => v switch
    {
        null => "null",
        float f => f.ToString("R", CultureInfo.InvariantCulture),
        double d => d.ToString("R", CultureInfo.InvariantCulture),
        string s => s,
        System.Collections.IEnumerable e => "[" + string.Join(",", e.Cast<object?>().Select(Fmt)) + "]",
        _ => Convert.ToString(v, CultureInfo.InvariantCulture) ?? "null",
    };
}
