using System.Text;
using System.Reflection;
using Xunit;

namespace OpenTail.Stingray.Tests.Server.Fast;

public sealed class SafetensorsProductIntegrationTests : IDisposable
{
    private readonly string _tempDir;

    public SafetensorsProductIntegrationTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), "opentail_st_prod_test_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_tempDir);
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
        {
            try { Directory.Delete(_tempDir, recursive: true); } catch { }
        }
    }

    [Fact]
    public void InferenceEngineLoader_LoadsSafetensorsPackageDirectory()
    {
        string packageDir = Path.Combine(_tempDir, "llama_package");
        Directory.CreateDirectory(packageDir);

        BuildTestPackage(packageDir);

        var opts = new OpenTailStingrayServerOptions
        {
            ModelPath = packageDir,
            ContextSize = 128 // Deliberately smaller than config max_position_embeddings (512).
        };

        var loadedEngine = InferenceEngineLoader.Load(opts);
        using var disposableEngine = loadedEngine.Engine as IDisposable;

        Assert.NotNull(loadedEngine.Engine);
        Assert.Equal("llama", loadedEngine.Architecture);
        Assert.Equal(128, GetForwardPassScratchContextLength(loadedEngine));
    }

    [Fact]
    public async Task InferenceEngineLoader_RealSmolLm2Package_CompletesOneRequest()
    {
        string? packagePath = FindRealPackagePath();
        Assert.SkipWhen(packagePath is null, "SmolLM2-135M-Instruct package not present under models/.");

        var loadedEngine = InferenceEngineLoader.Load(new OpenTailStingrayServerOptions
        {
            ModelPath = packagePath,
            ContextSize = 128,
        });
        using var disposableEngine = loadedEngine.Engine as IDisposable;
        using var timeout = new CancellationTokenSource(TimeSpan.FromMinutes(2));

        var chunks = new List<GenerateChunk>();
        await foreach (var chunk in loadedEngine.Engine.GenerateChunksAsync(
            "The capital of France is", new SamplingParams { Temperature = 0f, MaxNewTokens = 1 }, timeout.Token))
        {
            chunks.Add(chunk);
        }

        Assert.Contains(chunks, chunk => chunk.Kind == GenerateChunkKind.Usage && chunk.PromptTokens > 0);
        Assert.Contains(chunks, chunk => chunk.Kind == GenerateChunkKind.Stop);
    }

    private static int GetForwardPassScratchContextLength(LoadedEngine loaded)
    {
        // LoadedEngine deliberately exposes only the serving interface. This integration test
        // verifies loader-to-forward-pass wiring without widening the production API for a
        // test-only diagnostic.
        var owned = loaded.Engine.GetType()
            .GetFields(BindingFlags.Instance | BindingFlags.NonPublic)
            .Select(field => field.GetValue(loaded.Engine))
            .OfType<IList<IDisposable>>()
            .SingleOrDefault()
            ?? throw new InvalidOperationException("Owned server engine lifetime list is unavailable.");
        var forwardPass = owned.OfType<ForwardPass>().Single();
        var contextField = typeof(ForwardPass).GetField("_ctxLen", BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new InvalidOperationException("ForwardPass scratch-context field is unavailable.");
        return (int)(contextField.GetValue(forwardPass)
            ?? throw new InvalidOperationException("ForwardPass scratch-context value is unavailable."));
    }

    private static string? FindRealPackagePath()
    {
        for (var directory = new DirectoryInfo(AppContext.BaseDirectory); directory is not null; directory = directory.Parent)
        {
            string candidate = Path.Combine(directory.FullName, "models", "SmolLM2-135M-Instruct");
            if (File.Exists(Path.Combine(candidate, "config.json"))) return candidate;
        }
        return null;
    }

    private static void BuildTestPackage(string dir)
    {
        string configJson = "{\"model_type\":\"llama\",\"hidden_size\":64,\"num_hidden_layers\":2,\"num_attention_heads\":4,\"num_key_value_heads\":2,\"intermediate_size\":128,\"vocab_size\":128,\"max_position_embeddings\":512,\"rope_theta\":10000.0,\"rms_norm_eps\":1e-5,\"tie_word_embeddings\":true,\"torch_dtype\":\"float32\"}";
        File.WriteAllText(Path.Combine(dir, "config.json"), configJson);

        var vocabSb = new StringBuilder();
        vocabSb.Append("{\"model\":{\"type\":\"BPE\",\"vocab\":{");
        for (int i = 0; i < 128; i++)
        {
            if (i > 0) vocabSb.Append(',');
            vocabSb.Append($"\"tok_{i}\":{i}");
        }
        vocabSb.Append("},\"merges\":[]}}");
        File.WriteAllText(Path.Combine(dir, "tokenizer.json"), vocabSb.ToString());

        var rng = new Random(42);
        long currentOffset = 0;
        var headerSb = new StringBuilder();
        headerSb.Append("{\"__metadata__\":{\"format\":\"pt\"}");

        var tensorData = new List<byte[]>();
        string[] tensorNames = [
            "model.embed_tokens.weight",
            "model.norm.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.1.input_layernorm.weight",
            "model.layers.1.self_attn.q_proj.weight",
            "model.layers.1.self_attn.k_proj.weight",
            "model.layers.1.self_attn.v_proj.weight",
            "model.layers.1.self_attn.o_proj.weight",
            "model.layers.1.post_attention_layernorm.weight",
            "model.layers.1.mlp.gate_proj.weight",
            "model.layers.1.mlp.up_proj.weight",
            "model.layers.1.mlp.down_proj.weight"
        ];

        foreach (string name in tensorNames)
        {
            bool isNorm = name.Contains("norm");
            bool isEmbed = name.Contains("embed_tokens");
            bool isKvProj = name.Contains("k_proj") || name.Contains("v_proj");
            bool isMlpUp = name.Contains("gate_proj") || name.Contains("up_proj");
            bool isMlpDown = name.Contains("down_proj");

            int elements = isNorm ? 64 : (isEmbed ? 128 * 64 : (isKvProj ? 32 * 64 : ((isMlpUp || isMlpDown) ? 128 * 64 : 64 * 64)));
            byte[] bytes = new byte[elements * sizeof(float)];
            var floats = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(bytes.AsSpan());
            for (int i = 0; i < elements; i++) floats[i] = (float)(rng.NextDouble() * 0.1 - 0.05);

            tensorData.Add(bytes);
            long endOffset = currentOffset + bytes.Length;
            string shapeJson = isNorm ? "[64]" : (isEmbed ? "[128,64]" : (isKvProj ? "[32,64]" : (isMlpUp ? "[128,64]" : (isMlpDown ? "[64,128]" : "[64,64]"))));

            headerSb.Append($",\"{name}\":{{\"dtype\":\"F32\",\"shape\":{shapeJson},\"data_offsets\":[{currentOffset},{endOffset}]}}");
            currentOffset = endOffset;
        }

        headerSb.Append('}');
        string headerJson = headerSb.ToString();
        byte[] headerBytes = Encoding.UTF8.GetBytes(headerJson);

        using var fs = new FileStream(Path.Combine(dir, "model.safetensors"), FileMode.Create, FileAccess.Write);
        using var bw = new BinaryWriter(fs, Encoding.UTF8);

        bw.Write((ulong)headerBytes.Length);
        bw.Write(headerBytes);
        foreach (byte[] bytes in tensorData) bw.Write(bytes);
    }
}
