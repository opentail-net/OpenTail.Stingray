using System.Linq;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Engine.Verification;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Qwen 3.8 plan, Phase 7 (telemetry): after real qwen4exp tokens, <see cref="ForwardCounterEvents"/> reports the PLE gather
/// counters the pass keeps. Real trunk; skipped when absent.
/// </summary>
public sealed class Qwen4ExpTelemetryTests
{
    private sealed class CollectingSink : IEngineEventSink
    {
        public List<EngineEvent> Events { get; } = new();

        public void OnEvent(in EngineEvent e) => Events.Add(e);
    }

    private static string? FindTrunk() => ModelLocator.FindDeep("Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf");

    [Fact]
    public void ForwardCounters_ReportTheRealPleGatherAfterTokens()
    {
        string? trunkPath = FindTrunk();
        Assert.SkipUnless(trunkPath is not null, "Qwen3.8-Flash-Next UD-IQ1_S checkpoint is required (paged, 72.5 GB).");

        using var model = GgufModel.Open(trunkPath!);
        int totalLayers = (int)Convert.ToInt64(model.Metadata["qwen4exp.block_count"]);
        var hp = Qwen4ExpHyperparams.FromMetadata(model.Metadata, totalLayers);
        var tensors = Qwen4ExpTensorSet.Load(model, hp, totalLayers);
        using var pass = new Qwen4ExpForwardPass(model, hp, tensors);
        var tok = GgufTokenizer.FromGgufModel(model);
        int[] ids = tok.Encode("The capital of France is").ToArray();

        var sink = new CollectingSink();
        using var subscription = EngineEvents.Subscribe(sink);
        for (int p = 0; p < ids.Length; p++) pass.Forward(ids[p], p);
        ForwardCounterEvents.Emit(pass);

        var e = Assert.Single(sink.Events, ev => ev.Kind == EngineEventKind.ForwardCounters);
        Assert.Equal("qwen4exp", e.Name);
        Assert.True(e.A > 0, "the pass must have gathered PLE rows");
        Assert.Equal(pass.PleGatherRows, e.A);
        Assert.Equal(pass.PleGatherBytes, e.B);
        Assert.Equal(pass.PleTableBytes, e.C);
    }
}
