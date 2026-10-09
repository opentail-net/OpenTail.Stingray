using OpenTail.Stingray.Cli.Scout;
using OpenTail.Stingray.Core.Catalog;
using static OpenTail.Stingray.Tests.Cli.HubFixtures;

namespace OpenTail.Stingray.Tests.Cli;

/// <summary>The shared memory check: same analyzer as scout, blocks only on a KNOWN over-budget estimate, and is wired into chat and setup.</summary>
public sealed class LoadPreflightTests : IDisposable
{
    private readonly List<string> _files = [];
    public void Dispose() { foreach (var f in _files) try { File.Delete(f); } catch { } }

    private static ScoutInput Input(string arch, int blocks = 2)
    {
        byte[] b = Gguf(arch, blocks);
        var i = GgufModel.ParseIndex([(b, b.Length)]);
        return new ScoutInput("m.gguf", b.Length, i.Header.Version, i.Header.TensorCount, i.Header.MetadataKvCount, i.Metadata, i.Tensors);
    }

    private string WriteFile(byte[] bytes, string ext = ".gguf") { string p = Path.Combine(Path.GetTempPath(), "pf-" + Guid.NewGuid().ToString("N") + ext); File.WriteAllBytes(p, bytes); _files.Add(p); return p; }

    private const long Gib = 1L << 30;

    [Fact]
    public void A_small_model_on_a_big_machine_is_allowed_with_an_estimate()
    {
        var r = LoadPreflight.EvaluateIndex(Input("llama"), 256, ramBytes: 64 * Gib);
        Assert.Equal(PreflightVerdict.Allowed, r.Verdict);
        Assert.True(r.EstimatedPeakBytes > 0);
        Assert.Equal(8 * Gib, r.ReserveBytes);           // min(8 GiB, RAM/4)
    }

    [Fact]
    public void A_known_estimate_over_the_machines_memory_is_blocked()
    {
        var r = LoadPreflight.EvaluateIndex(Input("llama"), 256, ramBytes: 100L << 20);   // 100 MiB machine; the fixed overhead alone exceeds it
        Assert.Equal(PreflightVerdict.Blocked, r.Verdict);
        Assert.NotNull(r.EstimatedPeakBytes);
        Assert.Equal(25L << 20, r.ReserveBytes);          // RAM/4 on a small machine
    }

    [Fact]
    public void A_family_the_estimator_does_not_model_is_Unknown_not_blocked()
    {
        var r = LoadPreflight.EvaluateIndex(Input("deepseek2"), 256, ramBytes: 100L << 20);
        Assert.Equal(PreflightVerdict.Unknown, r.Verdict);
        Assert.Null(r.EstimatedPeakBytes);
    }

    [Fact]
    public void A_longer_context_needs_more()
    {
        long small = LoadPreflight.EvaluateIndex(Input("llama"), 128, ramBytes: 64 * Gib).EstimatedPeakBytes!.Value;
        long large = LoadPreflight.EvaluateIndex(Input("llama"), 256, ramBytes: 64 * Gib).EstimatedPeakBytes!.Value;
        Assert.True(large > small);
    }

    [Fact]
    public void It_is_the_same_decision_scout_reports()
    {
        var input = Input("llama", 4);
        var (budget, reserve) = LoadPreflight.MachineBudget(8 * Gib);
        var scout = ScoutAnalyzer.Analyze(input, new ScoutOptions("t", budget, reserve, null, 256));
        var pf = LoadPreflight.EvaluateIndex(input, 256, ramBytes: 8 * Gib);
        Assert.Equal(scout.Resources.HostWorkingSet.Bytes, pf.EstimatedPeakBytes);
        Assert.Equal(scout.Resources.ExecutionDecision == "allowed", pf.Verdict == PreflightVerdict.Allowed);
    }

    [Fact]
    public void Files_that_are_not_readable_ggufs_are_not_applicable_so_the_loader_reports_its_own_error()
    {
        Assert.Equal(PreflightVerdict.NotApplicable, LoadPreflight.EvaluateFile(Path.Combine(Path.GetTempPath(), "nope.gguf"), 256, 64 * Gib).Verdict);
        Assert.Equal(PreflightVerdict.NotApplicable, LoadPreflight.EvaluateFile(WriteFile([1, 2, 3], ".bin"), 256, 64 * Gib).Verdict);
        Assert.Equal(PreflightVerdict.NotApplicable, LoadPreflight.EvaluateFile(WriteFile(new byte[200]), 256, 64 * Gib).Verdict);
        Assert.Equal(PreflightVerdict.Allowed, LoadPreflight.EvaluateFile(WriteFile(Gguf("llama", 2)), 256, 64 * Gib).Verdict);
    }

    [Fact]
    public void Only_a_known_over_budget_result_stops_a_command_and_the_override_is_loud()
    {
        var blocked = new PreflightResult(PreflightVerdict.Blocked, "x", 40L * Gib, 16 * Gib, 4 * Gib);
        Assert.False(LoadPreflight.ShouldProceed(blocked, ignore: false, out var m1));
        Assert.Contains("--ignore-preflight", m1);
        Assert.Contains("40.0 GiB", m1);
        Assert.True(LoadPreflight.ShouldProceed(blocked, ignore: true, out var m2));
        Assert.StartsWith("warning:", m2);

        Assert.True(LoadPreflight.ShouldProceed(new(PreflightVerdict.Unknown, "x", null, 0, 0), false, out var m3));
        Assert.Contains("not checked", m3);
        Assert.True(LoadPreflight.ShouldProceed(new(PreflightVerdict.Allowed, "x", 1, 0, 0), false, out var m4));
        Assert.Null(m4);
        Assert.True(LoadPreflight.ShouldProceed(new(PreflightVerdict.NotApplicable, "x", null, 0, 0), false, out _));
    }
}
