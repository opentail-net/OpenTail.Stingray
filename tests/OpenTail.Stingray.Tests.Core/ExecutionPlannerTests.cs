#nullable enable

using System.Buffers.Binary;
using System.Text;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Engine.Packaging;
using OpenTail.Stingray.Engine.Planning;

namespace OpenTail.Stingray.Tests.Core;

public sealed class ExecutionPlannerTests : IDisposable
{
    private readonly List<string> _tempFiles = [];
    private readonly List<string> _tempDirs = [];

    public void Dispose()
    {
        foreach (var file in _tempFiles)
        {
            try { if (File.Exists(file)) File.Delete(file); } catch { /* best effort */ }
        }
        foreach (var dir in _tempDirs)
        {
            try { if (Directory.Exists(dir)) Directory.Delete(dir, true); } catch { /* best effort */ }
        }
    }

    private string CreateTempFile(string extension = ".gguf")
    {
        var path = Path.Combine(Path.GetTempPath(), $"stingray_test_{Guid.NewGuid():N}{extension}");
        _tempFiles.Add(path);
        return path;
    }

    [Fact]
    public void Plan_EndToEnd_GeneratesImmutablePlanWithoutReopeningFiles()
    {
        var metadata = new Dictionary<string, (GgufValueType, object)>
        {
            ["general.architecture"] = (GgufValueType.String, "llama"),
            ["llama.block_count"] = (GgufValueType.UInt32, 4u),
            ["llama.context_length"] = (GgufValueType.UInt32, 2048u),
            ["llama.embedding_length"] = (GgufValueType.UInt32, 64u),
            ["llama.feed_forward_length"] = (GgufValueType.UInt32, 128u),
            ["llama.attention.head_count"] = (GgufValueType.UInt32, 2u),
            ["llama.attention.head_count_kv"] = (GgufValueType.UInt32, 2u),
            ["llama.vocab_size"] = (GgufValueType.UInt32, 100u)
        };

        var tensors = new (string name, long[] dims, DType dtype, byte[] data)[]
        {
            ("token_embd.weight", [64, 100], DType.Float32, new byte[64 * 100 * 4]),
            ("output.weight", [64, 100], DType.Float32, new byte[64 * 100 * 4]),
            ("blk.0.attn_q.weight", [64, 64], DType.Float32, new byte[64 * 64 * 4]),
            ("blk.1.attn_q.weight", [64, 64], DType.Float32, new byte[64 * 64 * 4]),
            ("blk.2.attn_q.weight", [64, 64], DType.Float32, new byte[64 * 64 * 4]),
            ("blk.3.attn_q.weight", [64, 64], DType.Float32, new byte[64 * 64 * 4])
        };

        string filePath = CreateGguf(metadata, tensors);
        var pkg = LooseGgufModelPackage.Open(filePath);
        var desc = ModelDescription.FromPackage(pkg);

        var capabilities = new BackendCapabilities(
            CudaAvailable: false,
            VulkanAvailable: true,
            HardwareProfile: new HardwareProfile(
                VramBytes: 8L * 1024 * 1024 * 1024,
                RamBytes: 32L * 1024 * 1024 * 1024,
                CpuCores: 8,
                EstPcieBandwidthGBps: 16.0,
                HasAvx512: false),
            VulkanDeviceName: "Simulated Vulkan GPU",
            RecommendedThreadCount: 8);

        var req = new ExecutionRequest
        {
            Goal = "balanced",
            PinnedContextSize = 1024,
            PinnedKvDtype = "fp32"
        };

        var plan = ExecutionPlanner.Plan(desc, req, capabilities);

        Assert.Equal(2, plan.SchemaVersion);
        Assert.Equal("vulkan", plan.Backend);
        Assert.Equal("vulkan", plan.SelectedBackend);
        Assert.Equal(ForwardPassKind.VulkanDense, plan.ForwardPassKind);
        Assert.Equal(4, plan.GpuLayers);
        Assert.Equal(0, plan.CpuLayers);
        Assert.Equal(1024, plan.ContextSize);
        Assert.Equal("float32", plan.KvDtype);

        Assert.NotNull(plan.BackendPlan);
        Assert.NotNull(plan.Placement);
        Assert.NotNull(plan.State);
        Assert.NotNull(plan.Batching);
        Assert.NotNull(plan.Speculation);
        Assert.NotNull(plan.Modality);
        Assert.NotNull(plan.Memory);
        Assert.NotNull(plan.Provenance);

        // Validation passes
        ExecutionPlanValidator.Validate(plan);
    }

    [Fact]
    public void Plan_CandidateEvaluation_RoutesSafeTensorsToCpu()
    {
        using var testPkg = SafetensorsTextModelPackageTests.TestPackage.Create();
        var pkg = SafeTensorsModelPackage.Open(testPkg.Directory);
        var desc = ModelDescription.FromPackage(pkg);

        // Even when GPU is available, SafeTensors candidate evaluation resolves to CPU
        var capabilities = new BackendCapabilities(
            CudaAvailable: true,
            VulkanAvailable: true,
            HardwareProfile: new HardwareProfile(
                VramBytes: 8L * 1024 * 1024 * 1024,
                RamBytes: 32L * 1024 * 1024 * 1024,
                CpuCores: 8,
                EstPcieBandwidthGBps: 16.0,
                HasAvx512: false));

        var plan = ExecutionPlanner.Plan(desc, new ExecutionRequest(), capabilities);

        Assert.Equal("cpu", plan.Backend);
        Assert.Equal(ForwardPassKind.SafeTensorsCpu, plan.ForwardPassKind);
        Assert.Equal(0, plan.GpuLayers);
        Assert.Equal(1, plan.CpuLayers);
    }

    [Fact]
    public void ExecutionPlanBuilder_Build_DelegatesCorrectly()
    {
        var metadata = new Dictionary<string, (GgufValueType, object)>
        {
            ["general.architecture"] = (GgufValueType.String, "llama"),
            ["llama.block_count"] = (GgufValueType.UInt32, 2u),
            ["llama.context_length"] = (GgufValueType.UInt32, 1024u),
            ["llama.embedding_length"] = (GgufValueType.UInt32, 64u),
            ["llama.feed_forward_length"] = (GgufValueType.UInt32, 128u),
            ["llama.attention.head_count"] = (GgufValueType.UInt32, 2u),
            ["llama.attention.head_count_kv"] = (GgufValueType.UInt32, 2u),
            ["llama.vocab_size"] = (GgufValueType.UInt32, 100u)
        };

        var tensors = new (string name, long[] dims, DType dtype, byte[] data)[]
        {
            ("token_embd.weight", [64, 100], DType.Float32, new byte[64 * 100 * 4]),
            ("output.weight", [64, 100], DType.Float32, new byte[64 * 100 * 4]),
            ("blk.0.attn_q.weight", [64, 64], DType.Float32, new byte[64 * 64 * 4]),
            ("blk.1.attn_q.weight", [64, 64], DType.Float32, new byte[64 * 64 * 4])
        };

        string filePath = CreateGguf(metadata, tensors);
        var plan = ExecutionPlanBuilder.Build(
            modelPath: filePath,
            goal: "quality",
            pinBackend: "cpu",
            pinGpuLayers: 0,
            pinContextSize: 512,
            noGpuProbe: true);

        Assert.Equal(2, plan.SchemaVersion);
        Assert.Equal("cpu", plan.Backend);
        Assert.Equal(0, plan.GpuLayers);
        Assert.Equal(2, plan.CpuLayers);
        Assert.Equal(512, plan.ContextSize);
        Assert.Equal(ForwardPassKind.CpuDense, plan.ForwardPassKind);
    }

    private ExecutionPlan PlanTinyModel(ExecutionRequest request, bool vulkan = true)
    {
        var metadata = new Dictionary<string, (GgufValueType, object)>
        {
            ["general.architecture"] = (GgufValueType.String, "llama"),
            ["llama.block_count"] = (GgufValueType.UInt32, 2u),
            ["llama.context_length"] = (GgufValueType.UInt32, 2048u),
            ["llama.embedding_length"] = (GgufValueType.UInt32, 64u),
            ["llama.feed_forward_length"] = (GgufValueType.UInt32, 128u),
            ["llama.attention.head_count"] = (GgufValueType.UInt32, 2u),
            ["llama.attention.head_count_kv"] = (GgufValueType.UInt32, 2u),
            ["llama.vocab_size"] = (GgufValueType.UInt32, 100u)
        };
        var tensors = new (string name, long[] dims, DType dtype, byte[] data)[]
        {
            ("token_embd.weight", [64, 100], DType.Float32, new byte[64 * 100 * 4]),
            ("output.weight", [64, 100], DType.Float32, new byte[64 * 100 * 4]),
            ("blk.0.attn_q.weight", [64, 64], DType.Float32, new byte[64 * 64 * 4]),
            ("blk.1.attn_q.weight", [64, 64], DType.Float32, new byte[64 * 64 * 4]),
        };
        var desc = ModelDescription.FromPackage(LooseGgufModelPackage.Open(CreateGguf(metadata, tensors)));
        var caps = new BackendCapabilities(
            CudaAvailable: false,
            VulkanAvailable: vulkan,
            HardwareProfile: new HardwareProfile(8L << 30, 32L << 30, 8, 16.0, false),
            VulkanDeviceName: vulkan ? "Simulated Vulkan GPU" : null,
            RecommendedThreadCount: 8);
        return ExecutionPlanner.Plan(desc, request, caps);
    }

    [Fact]
    public void Plan_CarriesRequestedDeviceIndex_ForGpuBackend()
    {
        var plan = PlanTinyModel(new ExecutionRequest { PinnedContextSize = 512, DeviceIndex = 2 });

        Assert.Equal("vulkan", plan.Backend);
        Assert.Equal(2, plan.BackendPlan!.DeviceIndex);
    }

    [Fact]
    public void Plan_DeviceIndexIsZero_ForCpuBackend()
    {
        var plan = PlanTinyModel(new ExecutionRequest { PinnedBackend = "cpu", PinnedGpuLayers = 0, PinnedContextSize = 512, DeviceIndex = 2 });

        Assert.Equal("cpu", plan.Backend);
        Assert.Equal(0, plan.BackendPlan!.DeviceIndex);
    }

    [Fact]
    public void Plan_RecordsMoeExecutionChoices_AndSurvivesAotJsonRoundTrip()
    {
        var plan = PlanTinyModel(new ExecutionRequest
        {
            PinnedContextSize = 512,
            CpuMoe = true,
            GpuMoePrefill = false,
            MoeWarmPin = 4,
            MoeWarmPinAfter = 16,
            MoePredictPrefetch = false,
            ExpertStatsPath = "stats.json",
        });

        Assert.NotNull(plan.Moe);
        Assert.Equal(true, plan.Moe!.CpuMoe);
        Assert.Equal(false, plan.Moe.GpuMoePrefill);
        Assert.Equal(4, plan.Moe.WarmPin);
        Assert.Equal(16, plan.Moe.WarmPinAfter);
        Assert.Equal(false, plan.Moe.PredictPrefetch);
        Assert.Equal("stats.json", plan.Moe.ExpertStatsPath);

        string json = System.Text.Json.JsonSerializer.Serialize(plan, ExecutionPlanJsonContext.Default.ExecutionPlan);
        var back = System.Text.Json.JsonSerializer.Deserialize(json, ExecutionPlanJsonContext.Default.ExecutionPlan)!;
        Assert.Equal(plan.Moe, back.Moe);
        Assert.Equal(plan.BackendPlan!.DeviceIndex, back.BackendPlan!.DeviceIndex);
    }

    [Fact]
    public void Plan_UnspecifiedMoeChoices_StayNull()
    {
        var plan = PlanTinyModel(new ExecutionRequest { PinnedContextSize = 512 });

        Assert.Null(plan.Moe!.CpuMoe);
        Assert.Null(plan.Moe.GpuMoePrefill);
        Assert.Null(plan.Moe.PredictPrefetch);
    }

    [Theory]
    [InlineData("src/OpenTail.Stingray.Engine/Runtime/RuntimeInstance.cs")]
    [InlineData("src/OpenTail.Stingray.Server/InferenceEngineLoader.cs")]
    public void RuntimeLayer_DoesNotMakePolicyDecisions(string relativePath)
    {
        // THE PLANNER DECIDES, THE PLAN RECORDS, THE RUNTIME OBEYS: once an ExecutionPlan exists, neither the
        // runtime nor the server loader may call the placement/selection policy again.
        string? dir = AppContext.BaseDirectory;
        while (dir is not null && !File.Exists(Path.Combine(dir, "OpenTail.Stingray.slnx")))
            dir = Path.GetDirectoryName(dir);
        Assert.NotNull(dir);

        string text = File.ReadAllText(Path.Combine(dir!, relativePath));
        Assert.DoesNotContain("TierPlanner.Plan(", text);
        Assert.DoesNotContain("DSparkPlacementPlanner.Plan(", text);
        Assert.DoesNotContain("ForwardPassSelection.Select", text);
    }

    [Fact]
    public void Plan_ResolvesEngineTuningFromRequest_AndSurvivesJsonRoundTrip()
    {
        var plan = PlanTinyModel(new ExecutionRequest
        {
            PinnedContextSize = 512,
            PinnedKvDtype = "bf16",
            KvStore = "Auto",
            KvBf16MinTokens = 300,
            MtpEnabled = false,
            BatchVerify = false,
            SpecBatchVerify = false,
            PrefillChunkTokens = 128,
            PrefixSlots = 2,
            PrefixScratchTokens = 1024,
            HybridCpuPrefill = "0",
            HybridCpuPrefillMinTokens = 48,
            HybridCpuPrefillKvBudgetMb = 256,
            HybridCpuPrefillWarmExperts = false,
            SnapKvEnabled = true,
            SnapKvBudget = 2048,
            SnapKvWindow = 16,
            SnapKvRecency = 32,
            SnapKvBudgetExplicit = true,
        });

        var t = plan.Tuning!;
        Assert.Equal("bf16", t.Kv.Dtype);
        Assert.Equal(KvStoreMode.Auto, t.Kv.Store);
        Assert.Equal(300, t.Kv.Bf16AutoMinTokens);
        Assert.False(t.Speculation.MtpEnabled);
        Assert.False(t.Speculation.BatchVerify);
        Assert.False(t.Speculation.SpecBatchVerify);
        Assert.Equal(128, t.Prefill.ChunkTokens);
        Assert.Equal(2, t.Prefill.PrefixSlots);
        Assert.Equal(1024, t.Prefill.PrefixScratchTokens);
        Assert.Equal("0", t.Prefill.HybridCpuPrefill);
        Assert.Equal(48, t.Prefill.HybridCpuPrefillMinTokens);
        Assert.Equal(256, t.Prefill.HybridCpuPrefillKvBudgetMb);
        Assert.False(t.Prefill.HybridCpuPrefillWarmExperts);
        Assert.Equal(new SnapKvConfig(2048, 16, 32, true), t.SnapKv);

        string json = System.Text.Json.JsonSerializer.Serialize(plan, ExecutionPlanJsonContext.Default.ExecutionPlan);
        var back = System.Text.Json.JsonSerializer.Deserialize(json, ExecutionPlanJsonContext.Default.ExecutionPlan)!;
        Assert.Equal(plan.Tuning, back.Tuning);
        Assert.Equal(t, EngineSettings.FromPlan(back).Tuning);
    }

    [Fact]
    public void Plan_UnspecifiedTuning_RecordsEngineDefaults_NotEnvironment()
    {
        string?[] saved = [Environment.GetEnvironmentVariable("STINGRAY_KV_STORE"), Environment.GetEnvironmentVariable("STINGRAY_DISABLE_MTP")];
        try
        {
            Environment.SetEnvironmentVariable("STINGRAY_KV_STORE", "bf16");
            Environment.SetEnvironmentVariable("STINGRAY_DISABLE_MTP", "1");

            // The planner never reads the environment: only the frontend bridge does.
            var plan = PlanTinyModel(new ExecutionRequest { PinnedContextSize = 512 });
            Assert.Equal(KvStoreMode.Fp32, plan.Tuning!.Kv.Store);
            Assert.True(plan.Tuning.Speculation.MtpEnabled);
            Assert.Null(plan.Tuning.Kv.Dtype);
        }
        finally
        {
            Environment.SetEnvironmentVariable("STINGRAY_KV_STORE", saved[0]);
            Environment.SetEnvironmentVariable("STINGRAY_DISABLE_MTP", saved[1]);
        }
    }

    [Fact]
    public void RequestEnvironmentBridge_FillsOnlyUnspecifiedFields()
    {
        string?[] saved = [Environment.GetEnvironmentVariable("STINGRAY_KV_STORE"), Environment.GetEnvironmentVariable("STINGRAY_DISABLE_MTP"),
            Environment.GetEnvironmentVariable("STINGRAY_CPU_MOE")];
        try
        {
            Environment.SetEnvironmentVariable("STINGRAY_KV_STORE", "bf16");
            Environment.SetEnvironmentVariable("STINGRAY_DISABLE_MTP", "1");
            Environment.SetEnvironmentVariable("STINGRAY_CPU_MOE", "1");

            var filled = ExecutionRequestEnvironment.ApplyTo(new ExecutionRequest { PinnedContextSize = 512 });
            Assert.Equal("Bf16", filled.KvStore);
            Assert.Equal(false, filled.MtpEnabled);
            Assert.Equal(true, filled.CpuMoe);

            var explicitReq = ExecutionRequestEnvironment.ApplyTo(
                new ExecutionRequest { KvStore = "Fp32", MtpEnabled = true, CpuMoe = false });
            Assert.Equal("Fp32", explicitReq.KvStore);
            Assert.Equal(true, explicitReq.MtpEnabled);
            Assert.Equal(false, explicitReq.CpuMoe);
        }
        finally
        {
            Environment.SetEnvironmentVariable("STINGRAY_KV_STORE", saved[0]);
            Environment.SetEnvironmentVariable("STINGRAY_DISABLE_MTP", saved[1]);
            Environment.SetEnvironmentVariable("STINGRAY_CPU_MOE", saved[2]);
        }
    }

    [Fact]
    public void ExecutionPlanBuilder_FullRequest_EqualsPlannerPlan()
    {
        var metadata = new Dictionary<string, (GgufValueType, object)>
        {
            ["general.architecture"] = (GgufValueType.String, "llama"),
            ["llama.block_count"] = (GgufValueType.UInt32, 2u),
            ["llama.context_length"] = (GgufValueType.UInt32, 2048u),
            ["llama.embedding_length"] = (GgufValueType.UInt32, 64u),
            ["llama.feed_forward_length"] = (GgufValueType.UInt32, 128u),
            ["llama.attention.head_count"] = (GgufValueType.UInt32, 2u),
            ["llama.attention.head_count_kv"] = (GgufValueType.UInt32, 2u),
            ["llama.vocab_size"] = (GgufValueType.UInt32, 100u)
        };
        var tensors = new (string name, long[] dims, DType dtype, byte[] data)[]
        {
            ("token_embd.weight", [64, 100], DType.Float32, new byte[64 * 100 * 4]),
            ("output.weight", [64, 100], DType.Float32, new byte[64 * 100 * 4]),
            ("blk.0.attn_q.weight", [64, 64], DType.Float32, new byte[64 * 64 * 4]),
            ("blk.1.attn_q.weight", [64, 64], DType.Float32, new byte[64 * 64 * 4]),
        };
        string path = CreateGguf(metadata, tensors);
        var request = new ExecutionRequest
        {
            ModelPath = path,
            PinnedContextSize = 512,
            NoGpuProbe = true,
            DeviceIndex = 3,
            CpuMoe = true,
            MoeWarmPin = 4,
            KvStore = "Bf16",
            MtpEnabled = false,
            PrefixSlots = 2,
            DSparkPlace = "cpu",
        };

        var viaBuilder = ExecutionPlanBuilder.Build(request);

        var desc = ModelDescription.FromPackage(LooseGgufModelPackage.Open(path));
        var caps = new BackendCapabilities(false, false, HardwareProfile.Detect(null as OpenTail.Stingray.Vulkan.VulkanBackend),
            RecommendedThreadCount: Environment.ProcessorCount);
        var direct = ExecutionPlanner.Plan(desc, ExecutionRequestEnvironment.ApplyTo(request), caps);

        Assert.Equal(direct.Backend, viaBuilder.Backend);
        Assert.Equal(direct.ContextSize, viaBuilder.ContextSize);
        Assert.Equal(direct.GpuLayers, viaBuilder.GpuLayers);
        Assert.Equal(direct.Moe, viaBuilder.Moe);
        Assert.Equal(direct.Tuning, viaBuilder.Tuning);
        Assert.Equal(direct.BackendPlan!.DeviceIndex, viaBuilder.BackendPlan!.DeviceIndex);
        Assert.Equal(KvStoreMode.Bf16, viaBuilder.Tuning!.Kv.Store);
        Assert.Equal(4, viaBuilder.Moe!.WarmPin);
    }

    // Execution-affecting settings the runtime must receive through EngineSettings, never from the process environment.
    private static readonly string[] PolicyEnvVars =
    [
        "STINGRAY_KV_DTYPE", "STINGRAY_KV_STORE", "STINGRAY_KV_BF16_MIN_TOKENS",
        "STINGRAY_CPU_MOE", "STINGRAY_MOE_GPU_PREFILL", "STINGRAY_MOE_WARMPIN", "STINGRAY_MOE_WARMPIN_AFTER",
        "STINGRAY_MOE_PREDICT_PREFETCH", "STINGRAY_EXPERT_STATS",
        "STINGRAY_DISABLE_MTP", "STINGRAY_DISABLE_BATCH_VERIFY", "STINGRAY_SPEC_BATCH_VERIFY",
        "STINGRAY_PREFIX_SLOTS", "STINGRAY_PREFIX_SCRATCH_TOKENS", "STINGRAY_PREFILL_CHUNK",
        "STINGRAY_HYBRID_CPU_PREFILL", "STINGRAY_CUDA_HYBRID_CPU_PREFILL", "STINGRAY_GPU_CPU_PREFILL",
        "STINGRAY_HYBRID_CPU_PREFILL_MIN_TOKENS", "STINGRAY_HYBRID_CPU_PREFILL_KV_BUDGET_MB", "STINGRAY_HYBRID_CPU_PREFILL_WARM",
        "STINGRAY_SNAPKV_BUDGET", "STINGRAY_SNAPKV_WINDOW", "STINGRAY_SNAPKV_RECENCY",
    ];

    [Fact]
    public void RuntimeCode_DoesNotReadOrWritePolicyEnvironment()
    {
        string? dir = AppContext.BaseDirectory;
        while (dir is not null && !File.Exists(Path.Combine(dir, "OpenTail.Stingray.slnx")))
            dir = Path.GetDirectoryName(dir);
        Assert.NotNull(dir);

        // The only sanctioned readers: the request bridge (EngineEnvironment), SnapKvConfig.FromEnvironment (invoked only
        // by that bridge / EngineSettings.FromEnvironment), and the CUDA pin (GpuDeviceSelection, a process-level constraint).
        string[] allowed = ["EngineEnvironment.cs", "SnapKvSelector.cs", "GpuDeviceSelection.cs"];
        var offenders = new List<string>();
        foreach (string file in Directory.EnumerateFiles(Path.Combine(dir!, "src", "OpenTail.Stingray.Engine"), "*.cs", SearchOption.AllDirectories))
        {
            if (allowed.Contains(Path.GetFileName(file))) continue;
            string text = File.ReadAllText(file);
            foreach (string name in PolicyEnvVars)
                if (text.Contains($"GetEnvironmentVariable(\"{name}\")") || text.Contains($"SetEnvironmentVariable(\"{name}\""))
                    offenders.Add($"{Path.GetFileName(file)}: {name}");
        }
        Assert.True(offenders.Count == 0, "Policy environment variables read/written outside the request bridge:\n" + string.Join("\n", offenders));

        string runtime = File.ReadAllText(Path.Combine(dir!, "src", "OpenTail.Stingray.Engine", "Runtime", "RuntimeInstance.cs"));
        Assert.DoesNotContain("SetEnvironmentVariable", runtime);
    }

    private string CreateDSparkHead()
    {
        string dir = Path.Combine(Path.GetTempPath(), $"stingray_dspark_{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        _tempDirs.Add(dir);
        // Matches the tiny llama fixture: vocab 100, 2 target layers, hidden 64.
        File.WriteAllText(Path.Combine(dir, "config.json"), """
            {"hidden_size":64,"head_dim":32,"num_attention_heads":2,"num_key_value_heads":2,"intermediate_size":128,
             "num_hidden_layers":1,"block_size":4,"mask_token_id":1,"target_layer_ids":[1],"num_target_layers":2,
             "markov_rank":0,"vocab_size":100,"rms_norm_eps":1e-6,"rope_theta":10000.0,"max_position_embeddings":512}
            """);
        File.WriteAllBytes(Path.Combine(dir, "model.safetensors"), []);
        return dir;
    }

    [Fact]
    public void Plan_DSparkPlacementOff_Optional_FallsBackWithWarning()
    {
        var plan = PlanTinyModel(new ExecutionRequest
        {
            PinnedContextSize = 512, DSparkModelPath = CreateDSparkHead(), DSparkPlace = "off", DSparkRequired = false,
        }, vulkan: false);

        Assert.False(plan.Speculation!.DSparkEnabled);
        Assert.Equal(DSparkPlacement.Off, plan.Speculation.DSparkPlacement);
        Assert.NotEqual(SpeculationMode.DSpark, plan.Speculation.Mode);
        Assert.False(plan.Speculation.DSparkRequired);
        Assert.Contains(plan.Warnings, w => w.Contains("normal generation"));
    }

    [Fact]
    public void Plan_DSparkPlacementOff_Required_FailsAtPlanTime_NotAtRuntime()
    {
        var ex = Assert.Throws<NotSupportedException>(() => PlanTinyModel(new ExecutionRequest
        {
            PinnedContextSize = 512, DSparkModelPath = CreateDSparkHead(), DSparkPlace = "off", DSparkRequired = true,
        }, vulkan: false));
        Assert.Contains("placement resolved to Off", ex.Message);
    }

    [Fact]
    public void Plan_DSparkPinnedToCpu_RecordsThePlacementOnce()
    {
        var plan = PlanTinyModel(new ExecutionRequest
        {
            PinnedContextSize = 512, DSparkModelPath = CreateDSparkHead(), DSparkPlace = "cpu", DSparkRequired = true,
        }, vulkan: false);

        Assert.True(plan.Speculation!.DSparkEnabled);
        Assert.Equal(DSparkPlacement.Cpu, plan.Speculation.DSparkPlacement);
        Assert.Equal(SpeculationMode.DSpark, plan.Speculation.Mode);
        Assert.True(plan.Speculation.DSparkRequired);
        Assert.Contains(plan.Decisions, d => d.Code == "DSPARK_PLACEMENT");
    }

    private string CreateGguf(
        Dictionary<string, (GgufValueType type, object value)> metadata,
        (string name, long[] dims, DType dtype, byte[] data)[] tensors)
    {
        var path = CreateTempFile();
        using var fs = File.Create(path);
        using var writer = new GgufWriter(fs);

        writer.WriteHeader(3, (ulong)tensors.Length, (ulong)metadata.Count);
        foreach (var (key, (type, value)) in metadata)
            writer.WriteMetadataKv(key, type, value);

        ulong offset = 0;
        foreach (var (name, dims, dtype, data) in tensors)
        {
            writer.WriteTensorInfo(name, dims, dtype, offset);
            offset += (ulong)data.Length;
        }

        writer.PadToAlignment(32);
        foreach (var (_, _, _, data) in tensors)
            fs.Write(data);

        return path;
    }

    private sealed class GgufWriter(Stream stream) : IDisposable
    {
        private long _bytesWritten;

        public void WriteHeader(uint version, ulong tensorCount, ulong metadataKvCount)
        {
            WriteUInt32(0x46554747);
            WriteUInt32(version);
            WriteUInt64(tensorCount);
            WriteUInt64(metadataKvCount);
        }

        public void WriteTensorInfo(string name, long[] dims, DType dtype, ulong offset)
        {
            WriteGgufString(name);
            WriteUInt32((uint)dims.Length);
            foreach (var dim in dims)
                WriteUInt64((ulong)dim);
            WriteUInt32((uint)dtype);
            WriteUInt64(offset);
        }

        public void WriteMetadataKv(string key, GgufValueType type, object value)
        {
            WriteGgufString(key);
            WriteUInt32((uint)type);
            WriteGgufValue(type, value);
        }

        public void PadToAlignment(int alignment)
        {
            var remainder = _bytesWritten % alignment;
            if (remainder != 0)
            {
                var padding = alignment - (int)remainder;
                Span<byte> zeros = stackalloc byte[padding];
                zeros.Clear();
                stream.Write(zeros);
                _bytesWritten += padding;
            }
        }

        private void WriteGgufString(string s)
        {
            var bytes = Encoding.UTF8.GetBytes(s);
            WriteUInt64((ulong)bytes.Length);
            stream.Write(bytes);
            _bytesWritten += bytes.Length;
        }

        private void WriteGgufValue(GgufValueType type, object value)
        {
            switch (type)
            {
                case GgufValueType.UInt8:   WriteByte((byte)value); break;
                case GgufValueType.Int8:    WriteByte((byte)(sbyte)value); break;
                case GgufValueType.UInt16:  WriteUInt16((ushort)value); break;
                case GgufValueType.Int16:   WriteInt16((short)value); break;
                case GgufValueType.UInt32:  WriteUInt32((uint)value); break;
                case GgufValueType.Int32:   WriteInt32((int)value); break;
                case GgufValueType.Float32: WriteFloat32((float)value); break;
                case GgufValueType.Bool:    WriteByte((bool)value ? (byte)1 : (byte)0); break;
                case GgufValueType.String:  WriteGgufString((string)value); break;
                case GgufValueType.UInt64:  WriteUInt64((ulong)value); break;
                case GgufValueType.Int64:   WriteInt64((long)value); break;
                case GgufValueType.Float64: WriteFloat64((double)value); break;
                default: throw new NotSupportedException($"Unsupported GGUF value type: {type}");
            }
        }

        private void WriteByte(byte v) { stream.WriteByte(v); _bytesWritten += 1; }

        private void WriteUInt16(ushort v)
        {
            Span<byte> buf = stackalloc byte[2];
            BinaryPrimitives.WriteUInt16LittleEndian(buf, v);
            stream.Write(buf);
            _bytesWritten += 2;
        }

        private void WriteInt16(short v)
        {
            Span<byte> buf = stackalloc byte[2];
            BinaryPrimitives.WriteInt16LittleEndian(buf, v);
            stream.Write(buf);
            _bytesWritten += 2;
        }

        private void WriteUInt32(uint v)
        {
            Span<byte> buf = stackalloc byte[4];
            BinaryPrimitives.WriteUInt32LittleEndian(buf, v);
            stream.Write(buf);
            _bytesWritten += 4;
        }

        private void WriteInt32(int v)
        {
            Span<byte> buf = stackalloc byte[4];
            BinaryPrimitives.WriteInt32LittleEndian(buf, v);
            stream.Write(buf);
            _bytesWritten += 4;
        }

        private void WriteFloat32(float v)
        {
            Span<byte> buf = stackalloc byte[4];
            BinaryPrimitives.WriteSingleLittleEndian(buf, v);
            stream.Write(buf);
            _bytesWritten += 4;
        }

        private void WriteUInt64(ulong v)
        {
            Span<byte> buf = stackalloc byte[8];
            BinaryPrimitives.WriteUInt64LittleEndian(buf, v);
            stream.Write(buf);
            _bytesWritten += 8;
        }

        private void WriteInt64(long v)
        {
            Span<byte> buf = stackalloc byte[8];
            BinaryPrimitives.WriteInt64LittleEndian(buf, v);
            stream.Write(buf);
            _bytesWritten += 8;
        }

        private void WriteFloat64(double v)
        {
            Span<byte> buf = stackalloc byte[8];
            BinaryPrimitives.WriteDoubleLittleEndian(buf, v);
            stream.Write(buf);
            _bytesWritten += 8;
        }

        public void Dispose() { }
    }
}
