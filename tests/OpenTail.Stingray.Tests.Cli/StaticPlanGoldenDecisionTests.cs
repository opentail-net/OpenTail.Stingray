using System.Text;

namespace OpenTail.Stingray.Tests.Cli;

/// <summary>
/// Golden capability decisions must not depend on a developer's installed GPU or on a large
/// downloaded model. These index-only GGUF fixtures carry the metadata the planner consumes;
/// they deliberately have no tensors because placement is excluded from this decision suite.
/// </summary>
public sealed class StaticPlanGoldenDecisionTests : IDisposable
{
    private readonly List<string> _files = [];

    [Fact]
    public void CpuDense_AutoSelectsCpuAndKVarN()
    {
        var report = CreateReport(new StaticPlanCommand.Settings { TurboQuant = true }, headDim: 64);

        AssertDecision(report, "backend", true, "Selected cpu");
        AssertDecision(report, "kv_turbo_quant", true, "KVarN is eligible");
        AssertDecision(report, "speculation", false, "No MTP head");
        // Pin what the decision means, not the sentence. "Not exposed by the CLI or server" went
        // stale the moment the server began serving /v1/sessions; what stays true is that the CLI
        // exposes nothing and that durable restart continuation is unimplemented.
        AssertDecision(report, "session_restart_continuation", false, "Not exposed by the CLI");
        AssertDecision(report, "session_restart_continuation", false, "conformance contract");
        Assert.True(report.Compatibility.Selected);
    }

    [Fact]
    public void CpuSnapKv_UsesLloydMaxWhenKVarNIsIneligible()
    {
        var report = CreateReport(new StaticPlanCommand.Settings
        {
            TurboQuant = true,
            TqMode = "auto",
        }, headDim: 128, snapKvBudget: 64);

        AssertDecision(report, "kv_turbo_quant", true, "Lloyd-Max is eligible");
        AssertDecision(report, "snapkv", true, "explicit budget 64");
    }

    [Fact]
    public void CudaDense_AcceptsCudaKvDtypeWhenCudaIsAvailable()
    {
        var report = CreateReport(new StaticPlanCommand.Settings
        {
            Backend = "cuda",
            GpuLayers = -1,
            KvType = "bf16",
        }, headDim: 128, runtimeFacts: CudaRuntimeFacts());

        AssertDecision(report, "backend", true, "Selected cuda");
        Assert.DoesNotContain(report.EffectiveConfiguration.Diagnostics,
            diagnostic => diagnostic.Field == "kv_type" && diagnostic.Kind == "inapplicable");
    }

    [Fact]
    public void CpuServerMtp_ReportsDtypeAndGrammarConstraints()
    {
        var report = CreateReport(new StaticPlanCommand.Settings
        {
            Target = "server",
            MaxBatchSize = 4,
            ToolGrammar = true,
            KvType = "bf16",
            SpecType = "mtp",
        }, headDim: 128, mtpLayers: 1);

        AssertDecision(report, "speculation", true, "MTP head detected");
        AssertDecision(report, "batching", true, "continuous batching requested");
        AssertDecision(report, "tool_grammar", false, "incompatible with continuous batching");
        Assert.Contains(report.EffectiveConfiguration.Diagnostics,
            diagnostic => diagnostic.Field == "kv_type" && diagnostic.Kind == "inapplicable");
        Assert.Contains(report.EffectiveConfiguration.Diagnostics,
            diagnostic => diagnostic.Field == "tool_grammar" && diagnostic.Kind == "inapplicable");
    }

    public void Dispose()
    {
        foreach (string path in _files)
            File.Delete(path);
    }

    private StaticPlanReport CreateReport(StaticPlanCommand.Settings settings, int headDim,
        int snapKvBudget = -1, int mtpLayers = 0, StaticPlanRuntimeFacts? runtimeFacts = null)
    {
        string path = WriteMetadataOnlyLlamaGguf(headDim, mtpLayers);
        var profile = new StaticPlanProfile(SnapKvBudget: snapKvBudget < 0 ? null : snapKvBudget);
        var config = StaticPlanConfiguration.Resolve(settings, profile, _ => null);
        using var model = GgufModel.Open(path);
        return StaticPlanReport.Create(path, model, config,
            runtimeFacts ?? StaticPlanRuntimeFacts.Detect(noGpuProbe: true), includePlacement: false);
    }

    private static StaticPlanRuntimeFacts CudaRuntimeFacts() => new(
    [
        new StaticPlanBackend("cpu", "available", null, null, "fixture"),
        new StaticPlanBackend("cuda", "available", 16L << 30, 12L << 30, "fixture"),
        new StaticPlanBackend("vulkan", "unavailable", null, null, "fixture"),
    ], new StaticPlanHardware(64L << 30, 16, false, "fixture"),
        new HardwareProfile(16L << 30, 64L << 30, 16, 0, false));

    private string WriteMetadataOnlyLlamaGguf(int headDim, int mtpLayers)
    {
        string path = Path.Combine(Path.GetTempPath(), $"opentail-stingray-plan-{Guid.NewGuid():N}.gguf");
        _files.Add(path);
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: false);
        var metadata = new (string Key, GgufValueType Type, object Value)[]
        {
            ("general.architecture", GgufValueType.String, "llama"),
            ("llama.block_count", GgufValueType.Int32, 2 + mtpLayers),
            ("llama.nextn_predict_layers", GgufValueType.Int32, mtpLayers),
            ("llama.context_length", GgufValueType.Int32, 128),
            ("llama.embedding_length", GgufValueType.Int32, headDim * 2),
            ("llama.feed_forward_length", GgufValueType.Int32, headDim * 8),
            ("llama.vocab_size", GgufValueType.Int32, 256),
            ("llama.attention.head_count", GgufValueType.Int32, 2),
            ("llama.attention.head_count_kv", GgufValueType.Int32, 1),
            ("llama.attention.key_length", GgufValueType.Int32, headDim),
        };
        writer.Write(0x46554747u); // GGUF
        writer.Write(3u);
        writer.Write(0ul); // tensors: this is an index-only planning fixture
        writer.Write((ulong)metadata.Length);
        foreach (var (key, type, value) in metadata)
        {
            WriteGgufString(writer, key);
            writer.Write((uint)type);
            switch (type)
            {
                case GgufValueType.Int32: writer.Write((int)value); break;
                case GgufValueType.String: WriteGgufString(writer, (string)value); break;
                default: throw new InvalidOperationException($"Unexpected fixture metadata type {type}.");
            }
        }
        while (stream.Position % 32 != 0)
            writer.Write((byte)0);
        return path;
    }

    private static void WriteGgufString(BinaryWriter writer, string value)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(value);
        writer.Write((ulong)bytes.Length);
        writer.Write(bytes);
    }

    private static void AssertDecision(StaticPlanReport report, string area, bool selected, string why)
    {
        StaticPlanDecision decision = Assert.Single(report.Decisions, item => item.Area == area);
        Assert.Equal(selected, decision.Selected);
        Assert.Contains(why, decision.Why, StringComparison.Ordinal);
    }
}
