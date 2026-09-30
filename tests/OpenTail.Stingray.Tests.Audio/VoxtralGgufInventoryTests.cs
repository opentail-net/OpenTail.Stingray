namespace OpenTail.Stingray.Tests.Audio;

public sealed class VoxtralGgufInventoryTests
{
    private static string? FindRepoFile(string relPath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relPath);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void InspectGgufMetadataAndTensors()
    {
        string? ggufPath = FindRepoFile("models/Voxtral-Mini-4B-Realtime-2602-GGUF/voxtral-mini-4b-realtime-2602-q8_0.gguf");
        Assert.NotNull(ggufPath);

        using var model = Core.GgufModel.Open(ggufPath!);
        Console.WriteLine($"[GGUF] Architecture: {(model.Metadata.TryGetValue("general.architecture", out var a) ? a : null)}");
        Console.WriteLine($"[GGUF] Tensor Count: {model.Tensors.Count}");

        foreach (var kvp in model.Metadata)
        {
            if (kvp.Key.StartsWith("audiocpp.embedded_files.data")) continue;
            if (kvp.Value is object[] arr)
            {
                Console.WriteLine($"[Metadata] {kvp.Key} (array of {arr.Length}): {string.Join(", ", arr.Take(5))}{(arr.Length > 5 ? "..." : "")}");
            }
            else
            {
                Console.WriteLine($"[Metadata] {kvp.Key} = {kvp.Value}");
            }
        }

        if (model.Metadata.TryGetValue("audiocpp.tensor_names", out var namesObj) && namesObj is object[] names)
        {
            Console.WriteLine($"[audiocpp.tensor_names] length = {names.Length}");
            for (int i = 0; i < Math.Min(15, names.Length); i++)
            {
                var t = model.Tensors[i];
                Console.WriteLine($"  [{i}] RawName={t.Name}, RealName={names[i]}, DType={t.DType}, Dimensions=[{string.Join(", ", t.Dimensions)}]");
            }
        }

        var rvcSource = new OpenTail.Stingray.Audio.Rvc.RvcPackedTensorSource(model);
        var missingTensors = new List<string>();
        var tensorDtypes = new Dictionary<string, (Core.DType dtype, long[] dims)>();

        void CheckTensor(string name)
        {
            if (!rvcSource.HasTensor(name))
            {
                missingTensors.Add(name);
            }
            else
            {
                var info = rvcSource.GetRawInfo(name);
                tensorDtypes[name] = (info.DType, info.Dimensions);
            }
        }

        CheckTensor("audio_tower.embedder.conv1.weight");
        CheckTensor("audio_tower.embedder.conv1.bias");
        CheckTensor("audio_tower.embedder.conv2.weight");
        CheckTensor("audio_tower.embedder.conv2.bias");
        for (int i = 0; i < 32; i++)
        {
            string p = $"audio_tower.layers.{i}";
            CheckTensor($"{p}.self_attn_layer_norm.weight");
            CheckTensor($"{p}.self_attn.q_proj.weight");
            CheckTensor($"{p}.self_attn.q_proj.bias");
            CheckTensor($"{p}.self_attn.k_proj.weight");
            CheckTensor($"{p}.self_attn.v_proj.weight");
            CheckTensor($"{p}.self_attn.v_proj.bias");
            CheckTensor($"{p}.self_attn.o_proj.weight");
            CheckTensor($"{p}.self_attn.o_proj.bias");
            CheckTensor($"{p}.final_layer_norm.weight");
            CheckTensor($"{p}.mlp.gate_proj.weight");
            CheckTensor($"{p}.mlp.up_proj.weight");
            CheckTensor($"{p}.mlp.down_proj.weight");
            CheckTensor($"{p}.mlp.down_proj.bias");
        }
        CheckTensor("audio_tower.norm.weight");
        CheckTensor("multi_modal_projector.linear_1.weight");
        CheckTensor("multi_modal_projector.linear_2.weight");

        CheckTensor("language_model.model.embed_tokens.weight");
        for (int i = 0; i < 26; i++)
        {
            string p = $"language_model.model.layers.{i}";
            CheckTensor($"{p}.input_layernorm.weight");
            CheckTensor($"{p}.self_attn.q_proj.weight");
            CheckTensor($"{p}.self_attn.k_proj.weight");
            CheckTensor($"{p}.self_attn.v_proj.weight");
            CheckTensor($"{p}.self_attn.o_proj.weight");
            CheckTensor($"{p}.post_attention_layernorm.weight");
            CheckTensor($"{p}.mlp.gate_proj.weight");
            CheckTensor($"{p}.mlp.up_proj.weight");
            CheckTensor($"{p}.mlp.down_proj.weight");
            CheckTensor($"{p}.ada_rms_norm.linear1.weight");
            CheckTensor($"{p}.ada_rms_norm.linear2.weight");
        }
        CheckTensor("language_model.model.norm.weight");

        Console.WriteLine($"[Tensors Checked] Total: {tensorDtypes.Count}, Missing: {missingTensors.Count}");
        if (missingTensors.Count > 0)
        {
            Console.WriteLine($"[Missing Tensors]: {string.Join(", ", missingTensors.Take(10))}");
        }
        Assert.Empty(missingTensors);

        // Print representative dtypes
        Console.WriteLine($"embed_tokens: {tensorDtypes["language_model.model.embed_tokens.weight"].dtype} dims=[{string.Join(",", tensorDtypes["language_model.model.embed_tokens.weight"].dims)}]");
        Console.WriteLine($"decoder layer 0 q_proj: {tensorDtypes["language_model.model.layers.0.self_attn.q_proj.weight"].dtype} dims=[{string.Join(",", tensorDtypes["language_model.model.layers.0.self_attn.q_proj.weight"].dims)}]");
        Console.WriteLine($"audio layer 0 q_proj: {tensorDtypes["audio_tower.layers.0.self_attn.q_proj.weight"].dtype} dims=[{string.Join(",", tensorDtypes["audio_tower.layers.0.self_attn.q_proj.weight"].dims)}]");
        Console.WriteLine($"projector linear_1: {tensorDtypes["multi_modal_projector.linear_1.weight"].dtype} dims=[{string.Join(",", tensorDtypes["multi_modal_projector.linear_1.weight"].dims)}]");
        var qInfo = rvcSource.GetRawInfo("language_model.model.layers.0.self_attn.q_proj.weight");
        var qBytes = model.GetTensorData(qInfo);
        int expectedBytes = (int)(qInfo.ElementCount / 32 * 34);
        Console.WriteLine($"[q_proj Q8_0 bytes] actual={qBytes.Length}, expected={expectedBytes}");
        Assert.Equal(expectedBytes, qBytes.Length);

        // Metadata contract assertions
        Assert.Equal("audiocpp", model.Metadata["general.architecture"]);
        Assert.Equal("voxtral_realtime", model.Metadata["audiocpp.model_spec.family"]);
        Assert.True(rvcSource.HasTensor("audio_tower.embedder.conv1.weight"));
        Assert.True(rvcSource.HasTensor("language_model.model.embed_tokens.weight"));
    }

    [Fact]
    public void ValidateAllRawMatrices_AreStrictlyQ8_0_AndHaveExpectedByteSizes()
    {
        string? ggufPath = FindRepoFile("models/Voxtral-Mini-4B-Realtime-2602-GGUF/voxtral-mini-4b-realtime-2602-q8_0.gguf");
        Assert.NotNull(ggufPath);

        using var model = Core.GgufModel.Open(ggufPath!);
        var source = new AudioCppPackedTensorSource(model);

        var rawMatrices = new List<string>();

        // Text decoder layers
        for (int i = 0; i < 26; i++)
        {
            string p = $"language_model.model.layers.{i}";
            rawMatrices.Add($"{p}.self_attn.q_proj.weight");
            rawMatrices.Add($"{p}.self_attn.k_proj.weight");
            rawMatrices.Add($"{p}.self_attn.v_proj.weight");
            rawMatrices.Add($"{p}.self_attn.o_proj.weight");
            rawMatrices.Add($"{p}.mlp.gate_proj.weight");
            rawMatrices.Add($"{p}.mlp.up_proj.weight");
            rawMatrices.Add($"{p}.mlp.down_proj.weight");
            rawMatrices.Add($"{p}.ada_rms_norm.linear1.weight");
            rawMatrices.Add($"{p}.ada_rms_norm.linear2.weight");
        }

        // Audio tower layers
        for (int i = 0; i < 32; i++)
        {
            string p = $"audio_tower.layers.{i}";
            rawMatrices.Add($"{p}.self_attn.q_proj.weight");
            rawMatrices.Add($"{p}.self_attn.k_proj.weight");
            rawMatrices.Add($"{p}.self_attn.v_proj.weight");
            rawMatrices.Add($"{p}.self_attn.o_proj.weight");
            rawMatrices.Add($"{p}.mlp.gate_proj.weight");
            rawMatrices.Add($"{p}.mlp.up_proj.weight");
            rawMatrices.Add($"{p}.mlp.down_proj.weight");
        }

        // Projector
        rawMatrices.Add("multi_modal_projector.linear_1.weight");
        rawMatrices.Add("multi_modal_projector.linear_2.weight");

        Assert.Equal(26 * 9 + 32 * 7 + 2, rawMatrices.Count); // 460 raw matrices

        foreach (string name in rawMatrices)
        {
            Assert.True(source.HasTensor(name), $"Missing raw matrix: {name}");
            var info = source.GetRawInfo(name);
            Assert.Equal(Core.DType.Q8_0, info.DType);

            int expectedBytes = (int)(info.ElementCount / 32 * 34);
            byte[] rawBytes = source.GetRawBytes(name, Core.DType.Q8_0);
            Assert.Equal(expectedBytes, rawBytes.Length);

            // Verify that asking for wrong DType throws InvalidDataException
            Assert.Throws<InvalidDataException>(() => source.GetRawBytes(name, Core.DType.Float32));
        }
    }

    [Fact]
    public void ValidateEmbeddedFiles_IntegrityAndBounds()
    {
        string? ggufPath = FindRepoFile("models/Voxtral-Mini-4B-Realtime-2602-GGUF/voxtral-mini-4b-realtime-2602-q8_0.gguf");
        Assert.NotNull(ggufPath);

        using var model = Core.GgufModel.Open(ggufPath!);
        Assert.True(model.Metadata.TryGetValue("audiocpp.embedded_files.names", out var namesObj));
        Assert.True(model.Metadata.TryGetValue("audiocpp.embedded_files.offsets", out var offsetsObj));
        Assert.True(model.Metadata.TryGetValue("audiocpp.embedded_files.data", out var dataObj));

        var names = Assert.IsType<object[]>(namesObj);
        var offsets = Assert.IsType<object[]>(offsetsObj);
        Assert.True(offsets.Length == names.Length || offsets.Length == names.Length + 1);
        Assert.True(names.Length > 0);

        byte[] data = dataObj switch
        {
            byte[] bArr => bArr,
            object[] oArr => oArr.Select(o => (byte)Convert.ToInt64(o)).ToArray(),
            _ => throw new Exception("Unexpected data format")
        };

        int tekkenIdx = -1;
        for (int i = 0; i < names.Length; i++)
        {
            if (string.Equals((string)names[i], "tekken.json", StringComparison.OrdinalIgnoreCase))
                tekkenIdx = i;
        }

        long prev = 0;
        for (int i = 0; i < offsets.Length; i++)
        {
            long off = Convert.ToInt64(offsets[i]);
            Assert.True(off >= 0 && off <= data.Length, $"Offset {off} out of bounds");
            Assert.True(off >= prev, $"Offset {off} not monotonic (prev={prev})");
            prev = off;
        }

        Assert.True(tekkenIdx >= 0, "tekken.json not found in embedded files");
        long tStart = Convert.ToInt64(offsets[tekkenIdx]);
        long tEnd = tekkenIdx + 1 < offsets.Length ? Convert.ToInt64(offsets[tekkenIdx + 1]) : data.Length;
        Assert.True(tStart < tEnd);
        Assert.True(tEnd <= data.Length);

        // Verify content begins with JSON '{'
        Assert.Equal((byte)'{', data[tStart]);
    }

    [Fact]
    public void GenericTextGeneration_RejectsAudiocppArchitecture()
    {
        string? ggufPath = FindRepoFile("models/Voxtral-Mini-4B-Realtime-2602-GGUF/voxtral-mini-4b-realtime-2602-q8_0.gguf");
        Assert.NotNull(ggufPath);

        using var model = Core.GgufModel.Open(ggufPath!);
        Assert.Throws<NotSupportedException>(() =>
            Engine.ModelCompatibility.ValidateForTextGeneration(model));
    }
}
