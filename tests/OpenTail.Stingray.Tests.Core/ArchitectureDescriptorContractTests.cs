using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using Xunit;

namespace OpenTail.Stingray.Tests.Core;

public sealed class ArchitectureDescriptorContractTests
{
    private static string GetRepoRoot()
    {
        string? root = AppContext.BaseDirectory;
        while (root is not null && !File.Exists(Path.Combine(root, "CLAUDE.md")))
            root = Path.GetDirectoryName(root);
        Assert.NotNull(root);
        return root!;
    }

    [Fact]
    public void EveryAdmittedDescriptor_HasValidCreateForwardPass()
    {
        var admitted = ArchitectureRegistry.All.Where(d => d.Status == AdmissionStatus.Admitted);
        Assert.NotEmpty(admitted);

        foreach (var d in admitted)
        {
            Assert.NotNull(d.CreateForwardPass);
            if (d.ForwardPassFamily != ForwardPassFamily.Dense)
            {
                Assert.NotEqual(CommonForwardPassFactory.CreateDense, d.CreateForwardPass);
            }
        }
    }

    [Fact]
    public void EveryDescriptor_HasValidIdentityAndAliases()
    {
        foreach (var d in ArchitectureRegistry.All)
        {
            Assert.False(string.IsNullOrWhiteSpace(d.Id), "Descriptor ID must not be null or whitespace.");
            foreach (var alias in d.Aliases)
            {
                Assert.False(string.IsNullOrWhiteSpace(alias), $"Descriptor '{d.Id}' has an empty alias.");
            }
        }
    }

    [Fact]
    public void EveryDescriptor_EvidenceDocExistsOnDisk()
    {
        string root = GetRepoRoot();
        foreach (var d in ArchitectureRegistry.All)
        {
            string path = Path.Combine(root, d.EvidenceDoc);
            Assert.True(File.Exists(path), $"Architecture '{d.Id}': evidence doc '{d.EvidenceDoc}' does not exist.");
        }
    }

    [Fact]
    public void NotAdmittedDescriptors_HaveRefusalReason_AndNoAnchors()
    {
        var notAdmitted = ArchitectureRegistry.All.Where(d => d.Status == AdmissionStatus.NotAdmitted);
        foreach (var d in notAdmitted)
        {
            Assert.False(string.IsNullOrWhiteSpace(d.RefusalReason), $"Descriptor '{d.Id}' is NotAdmitted but has no RefusalReason.");
            Assert.Null(d.StatusAnchor);
            Assert.Null(d.StatusExemption);
        }
    }

    [Fact]
    public void ExperimentalDescriptors_HaveKnownEnvVar()
    {
        var experimental = ArchitectureRegistry.All.Where(d => d.Status == AdmissionStatus.Experimental);
        foreach (var d in experimental)
        {
            Assert.False(string.IsNullOrWhiteSpace(d.ExperimentalEnvVar), $"Descriptor '{d.Id}' is Experimental but has no ExperimentalEnvVar.");
            Assert.True(KnownEnvironmentVariables.All.Contains(d.ExperimentalEnvVar!),
                $"Descriptor '{d.Id}' uses unknown env var '{d.ExperimentalEnvVar}'.");
        }
    }

    [Fact]
    public void AdmittedSet_IsExactlyTheSnapshot()
    {
        var expected = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            "qwen", "qwen2", "qwen2moe", "qwen3", "qwen3moe", "qwen2vl", "qwen35", "qwen35moe",
            "mimo", "mimo2", "gemma", "gemma2", "gemma3", "gemma3n", "phi2", "phi3", "phimoe",
            "olmoe", "rwkv7", "rwkv6", "gpt-oss", "deepseek2", "smollm3", "apertus", "gptneox",
            "falcon", "olmo2", "exaone", "orion", "ernie4_5", "paddleocr", "qwen3vl", "deepseek2-ocr",
            "granitehybrid", "nemotron_h", "lfm2", "lfm2moe", "internlm2", "starcoder2", "cohere2", "glm4",
            "glm4moe", "stablelm", "hunyuan-dense", "hunyuan-moe", "afmoe", "gpt2", "granitemoe", "olmo",
            "starcoder", "codeshell", "jais2", "jais", "maincoder", "exaone4", "mistral3", "ministral",
            "xverse", "minicpm", "gemma4", "granite", "llama", "llama4", "muse-glimmer", "muse_glimmer",
        };
        var actual = ArchitectureRegistry.All
            .Where(d => d.Status == AdmissionStatus.Admitted)
            .SelectMany(d => d.Aliases.Prepend(d.Id))
            .ToHashSet(StringComparer.OrdinalIgnoreCase);

        Assert.True(expected.SetEquals(actual),
            $"Missing: {string.Join(", ", expected.Except(actual, StringComparer.OrdinalIgnoreCase))}; " +
            $"unexpected: {string.Join(", ", actual.Except(expected, StringComparer.OrdinalIgnoreCase))}");
    }

    [Fact]
    public void Resolve_ProbeBasedDetection_ResolvesRelabelledAndMetadataFree()
    {
        // 1. Standard Llama resolution
        var llamaSource = new TestProbeTensorSource(new Dictionary<string, object>
        {
            ["general.architecture"] = "llama",
        });
        var llamaProbe = new ArchitectureProbe
        {
            Path = "model.gguf",
            Architecture = "llama",
            TensorSource = llamaSource,
            Hyperparams = ModelHyperparams.FromGgufMetadata(llamaSource.Metadata, llamaSource),
        };
        var resolvedLlama = ArchitectureRegistry.Resolve(llamaProbe);
        Assert.Equal("llama", resolvedLlama.Id);

        // 2. Relabelled file recognition (llama architecture with general.name containing mistral-3)
        var mistralSource = new TestProbeTensorSource(new Dictionary<string, object>
        {
            ["general.architecture"] = "llama",
            ["general.name"] = "mistral-3-instruct-7b",
        });
        var mistralProbe = new ArchitectureProbe
        {
            Path = "model.gguf",
            Architecture = "llama",
            TensorSource = mistralSource,
            Hyperparams = ModelHyperparams.FromGgufMetadata(mistralSource.Metadata, mistralSource),
        };
        var resolvedMistral = ArchitectureRegistry.Resolve(mistralProbe);
        Assert.Equal("mistral3", resolvedMistral.Id);

        // 3. Metadata-free detection (conservative: full Llama structural signature present)
        var metadataFreeSource = new TestProbeTensorSource(
            new Dictionary<string, object>(),
            new GgufTensorInfo("token_embd.weight", 2, [64, 128], DType.Float32, 0),
            new GgufTensorInfo("output.weight", 2, [64, 128], DType.Float32, 0),
            new GgufTensorInfo("output_norm.weight", 1, [128], DType.Float32, 0),
            new GgufTensorInfo("blk.0.attn_q.weight", 2, [64, 64], DType.Float32, 0),
            new GgufTensorInfo("blk.0.attn_k.weight", 2, [64, 64], DType.Float32, 0),
            new GgufTensorInfo("blk.0.attn_v.weight", 2, [64, 64], DType.Float32, 0),
            new GgufTensorInfo("blk.0.attn_output.weight", 2, [64, 64], DType.Float32, 0),
            new GgufTensorInfo("blk.0.attn_norm.weight", 1, [64], DType.Float32, 0),
            new GgufTensorInfo("blk.0.ffn_gate.weight", 2, [64, 128], DType.Float32, 0),
            new GgufTensorInfo("blk.0.ffn_up.weight", 2, [64, 128], DType.Float32, 0),
            new GgufTensorInfo("blk.0.ffn_down.weight", 2, [128, 64], DType.Float32, 0),
            new GgufTensorInfo("blk.0.ffn_norm.weight", 1, [64], DType.Float32, 0));
        var metadataFreeProbe = new ArchitectureProbe
        {
            Path = "model.gguf",
            Architecture = null,
            TensorSource = metadataFreeSource,
            Hyperparams = ModelHyperparams.FromGgufMetadata(metadataFreeSource.Metadata, metadataFreeSource),
        };
        var resolvedMetadataFree = ArchitectureRegistry.Resolve(metadataFreeProbe);
        Assert.Equal("llama", resolvedMetadataFree.Id);

        // 3b. Broad/generic transformer without full structural signature is NOT claimed as Llama
        var genericTransformerSource = new TestProbeTensorSource(
            new Dictionary<string, object>(),
            new GgufTensorInfo("token_embd.weight", 2, [64, 128], DType.Float32, 0),
            new GgufTensorInfo("blk.0.attn_q.weight", 2, [64, 64], DType.Float32, 0));
        var genericTransformerProbe = new ArchitectureProbe
        {
            Path = "model.gguf",
            Architecture = null,
            TensorSource = genericTransformerSource,
            Hyperparams = ModelHyperparams.FromGgufMetadata(genericTransformerSource.Metadata, genericTransformerSource),
        };
        Assert.Throws<NotSupportedException>(() => ArchitectureRegistry.Resolve(genericTransformerProbe));

        // 4. Metadata-free with unknown tensors throws
        var unknownFreeSource = new TestProbeTensorSource(
            new Dictionary<string, object>(),
            new GgufTensorInfo("unknown_tensor.weight", 1, [64], DType.Float32, 0));
        var unknownFreeProbe = new ArchitectureProbe
        {
            Path = "model.gguf",
            Architecture = null,
            TensorSource = unknownFreeSource,
            Hyperparams = ModelHyperparams.FromGgufMetadata(unknownFreeSource.Metadata, unknownFreeSource),
        };
        Assert.Throws<NotSupportedException>(() => ArchitectureRegistry.Resolve(unknownFreeProbe));

        // 5. Unknown architecture name throws NotSupportedException
        var unknownArchSource = new TestProbeTensorSource(new Dictionary<string, object>
        {
            ["general.architecture"] = "completely_unknown_family",
        });
        var unknownArchProbe = new ArchitectureProbe
        {
            Path = "model.gguf",
            Architecture = "completely_unknown_family",
            TensorSource = unknownArchSource,
            Hyperparams = ModelHyperparams.FromGgufMetadata(unknownArchSource.Metadata, unknownArchSource),
        };
        Assert.Throws<NotSupportedException>(() => ArchitectureRegistry.Resolve(unknownArchProbe));
    }

    [Fact]
    public void DescriptorValidation_CatchesViolations()
    {
        // Missing Id
        Assert.Throws<InvalidOperationException>(() => new ArchitectureDescriptor
        {
            Id = "",
            Status = AdmissionStatus.NotAdmitted,
            EvidenceDoc = "docs/STATUS.md",
            RefusalReason = "reason",
        }.Validate());

        // Relabelled file recognizer without description
        Assert.Throws<InvalidOperationException>(() => new ArchitectureDescriptor
        {
            Id = "test",
            Status = AdmissionStatus.Admitted,
            EvidenceDoc = "docs/STATUS.md",
            StatusExemption = "exempt",
            CreateForwardPass = CommonForwardPassFactory.CreateDense,
            RecognizeRelabelledFile = (_, _) => true,
        }.Validate());

        // Admitted descriptor missing CreateForwardPass
        Assert.Throws<InvalidOperationException>(() => new ArchitectureDescriptor
        {
            Id = "test_no_factory",
            Status = AdmissionStatus.Admitted,
            EvidenceDoc = "docs/STATUS.md",
            StatusExemption = "exempt",
            CreateForwardPass = null,
        }.Validate());

        // Non-dense architecture mistakenly using CommonForwardPassFactory.CreateDense
        Assert.Throws<InvalidOperationException>(() => new ArchitectureDescriptor
        {
            Id = "test_wrong_factory",
            Status = AdmissionStatus.Admitted,
            EvidenceDoc = "docs/STATUS.md",
            StatusExemption = "exempt",
            ForwardPassFamily = ForwardPassFamily.GptOss,
            CreateForwardPass = CommonForwardPassFactory.CreateDense,
        }.Validate());
    }

    [Fact]
    public void ConstructForwardPass_InvokesApplyLoadSetup()
    {
        bool setupInvoked = false;
        var desc = new ArchitectureDescriptor
        {
            Id = "test_setup",
            Status = AdmissionStatus.Admitted,
            EvidenceDoc = "docs/STATUS.md",
            StatusExemption = "exempt",
            ApplyLoadSetup = ctx => { setupInvoked = true; },
            CreateForwardPass = ctx => new TestForwardPass(),
        };
        desc.Validate();

        var dummySource = new TestProbeTensorSource(new Dictionary<string, object>());
        var loadCtx = new ArchitectureLoadContext
        {
            Probe = new ArchitectureProbe
            {
                Architecture = "test_setup",
                TensorSource = dummySource,
                Hyperparams = ModelHyperparams.FromGgufMetadata(dummySource.Metadata, dummySource),
            },
            Decision = new ForwardPassDecision(ForwardPassKind.CpuDense, null),
            Backend = ForwardPassBackend.Cpu,
            ContextSize = 128,
            GpuLayers = 0,
        };

        var pass = desc.ConstructForwardPass(loadCtx);
        Assert.True(setupInvoked);
        Assert.NotNull(pass);
    }

    private sealed class TestForwardPass : IForwardPass
    {
        public int VocabSize => 32000;
        public int MaxSeqLen => 2048;
        public ReadOnlySpan<float> Forward(int token, int position) => ReadOnlySpan<float>.Empty;
        public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0) => ReadOnlySpan<float>.Empty;
        public void TruncateTo(int length) { }
        public void ResetCache() { }
        public void Dispose() { }
    }

    private sealed unsafe class TestProbeTensorSource : IModelTensorSource
    {
        private readonly Dictionary<string, GgufTensorInfo> _tensors;

        public TestProbeTensorSource(IReadOnlyDictionary<string, object> metadata, params GgufTensorInfo[] tensors)
        {
            Metadata = metadata;
            _tensors = tensors.ToDictionary(t => t.Name, StringComparer.Ordinal);
            Tensors = tensors;
        }

        public IReadOnlyList<GgufTensorInfo> Tensors { get; }
        public IReadOnlyDictionary<string, object> Metadata { get; }

        public GgufTensorInfo? FindTensor(string name) =>
            _tensors.TryGetValue(name, out var tensor) ? tensor : null;

        public ReadOnlySpan<byte> GetTensorData(GgufTensorInfo tensor) =>
            throw new NotSupportedException();

        public byte* GetTensorDataPtr(GgufTensorInfo tensor) =>
            throw new NotSupportedException();
    }
}
