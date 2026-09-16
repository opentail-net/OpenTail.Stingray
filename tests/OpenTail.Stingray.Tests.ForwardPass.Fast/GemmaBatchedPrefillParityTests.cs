namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

public sealed class GemmaBatchedPrefillParityTests : IDisposable
{
    private readonly string _tempDir;

    public GemmaBatchedPrefillParityTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), "gemma_prefill_parity_" + Guid.NewGuid().ToString("N"));
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
    public void Gemma4_BatchedPrefill_MatchesSequentialForward_WithSwaAndMixedHeadDims()
    {
        string ggufPath = Path.Combine(_tempDir, "gemma4_synthetic.gguf");

        const int VocabSize = 64;
        const int HiddenSize = 128;
        const int IntermediateSize = 256;
        const int Layers = 2;
        const int Heads = 4;
        const int HeadDimGlobal = 64;
        const int HeadDimSwa = 32;

        var weights = GenerateDeterministicGemma4Weights(VocabSize, HiddenSize, IntermediateSize, Heads, HeadDimGlobal, HeadDimSwa);
        BuildGemma4GgufFile(ggufPath, weights, VocabSize, HiddenSize, IntermediateSize, Layers, Heads, HeadDimGlobal, HeadDimSwa);

        using var model = GgufModel.Open(ggufPath);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        using var backend = new CpuBackend();

        using var fwdBatched = new Engine.ForwardPass(model, backend, hp);
        using var fwdSequential = new Engine.ForwardPass(model, backend, hp);

        var prompt = new int[] { 5, 12, 23, 7, 44, 18, 2, 33 };

        // 1. Sequential execution
        var seqLogits = new List<float[]>();
        for (int i = 0; i < prompt.Length; i++)
        {
            var l = fwdSequential.Forward(prompt[i], i);
            seqLogits.Add(l.ToArray());
        }

        // 2. Batched prefill execution
        var batchedLogits = new List<float[]>();
        fwdBatched.PrefillWithPerPositionLogits(prompt, 0, (pos, logits) =>
        {
            batchedLogits.Add(logits.ToArray());
        });

        Assert.Equal(prompt.Length, batchedLogits.Count);

        // 3. Verify position-by-position parity
        for (int p = 0; p < prompt.Length; p++)
        {
            var s = seqLogits[p];
            var b = batchedLogits[p];
            Assert.Equal(s.Length, b.Length);

            float maxDiff = 0f;
            for (int v = 0; v < s.Length; v++)
            {
                float diff = Math.Abs(s[v] - b[v]);
                if (diff > maxDiff) maxDiff = diff;
            }

            // High precision parity: maxDiff within 5e-3 (FP reassociation noise across 4 layers of GEMM vs vector-dot)
            Assert.True(maxDiff < 5e-3f, $"Position {p} max diff {maxDiff} exceeded tolerance 5e-3");

            // Argmax decision agrees (or near-tie within floating-point reassociation noise)
            AssertArgmaxOrNearTie(s, b, tieEps: 0.01f, $"Position {p}");
        }

        // 4. Verify post-prefill decode step continues identically
        int nextTok = Argmax(batchedLogits[^1]);
        var nextSeq = fwdSequential.Forward(nextTok, prompt.Length);
        var nextBatched = fwdBatched.Forward(nextTok, prompt.Length);

        float decodeMaxDiff = 0f;
        for (int v = 0; v < nextSeq.Length; v++)
        {
            float diff = Math.Abs(nextSeq[v] - nextBatched[v]);
            if (diff > decodeMaxDiff) decodeMaxDiff = diff;
        }
        Assert.True(decodeMaxDiff < 5e-3f, $"Decode continuation max diff {decodeMaxDiff} exceeded tolerance 5e-3");
        AssertArgmaxOrNearTie(nextSeq.ToArray(), nextBatched.ToArray(), tieEps: 0.01f, "Decode continuation");
    }

    [Fact]
    public void Gemma4_Prefill_DirectReturn_MatchesSequentialLastToken()
    {
        string ggufPath = Path.Combine(_tempDir, "gemma4_direct_prefill.gguf");

        const int VocabSize = 64;
        const int HiddenSize = 128;
        const int IntermediateSize = 256;
        const int Layers = 2;
        const int Heads = 4;
        const int HeadDimGlobal = 64;
        const int HeadDimSwa = 32;

        var weights = GenerateDeterministicGemma4Weights(VocabSize, HiddenSize, IntermediateSize, Heads, HeadDimGlobal, HeadDimSwa);
        BuildGemma4GgufFile(ggufPath, weights, VocabSize, HiddenSize, IntermediateSize, Layers, Heads, HeadDimGlobal, HeadDimSwa);

        using var model = GgufModel.Open(ggufPath);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        using var backend = new CpuBackend();

        using var fwdBatched = new Engine.ForwardPass(model, backend, hp);
        using var fwdSequential = new Engine.ForwardPass(model, backend, hp);

        var prompt = new int[] { 3, 15, 29, 41, 10 };

        for (int i = 0; i < prompt.Length - 1; i++)
            fwdSequential.Forward(prompt[i], i);
        var lastSeqLogits = fwdSequential.Forward(prompt[^1], prompt.Length - 1).ToArray();

        var prefillLogits = fwdBatched.Prefill(prompt).ToArray();

        Assert.Equal(lastSeqLogits.Length, prefillLogits.Length);

        float maxDiff = 0f;
        for (int v = 0; v < lastSeqLogits.Length; v++)
        {
            float diff = Math.Abs(lastSeqLogits[v] - prefillLogits[v]);
            if (diff > maxDiff) maxDiff = diff;
        }

        Assert.True(maxDiff < 5e-3f, $"Direct prefill max diff {maxDiff} exceeded tolerance 5e-3");
        AssertArgmaxOrNearTie(lastSeqLogits, prefillLogits, tieEps: 0.01f, "Direct prefill last token");
    }

    private static void AssertArgmaxOrNearTie(float[] reference, float[] candidate, float tieEps, string label)
    {
        int rArg = Argmax(reference), cArg = Argmax(candidate);
        if (rArg == cArg) return;
        float gap = Math.Abs(reference[rArg] - reference[cArg]);
        Assert.True(gap < tieEps,
            $"{label}: batched argmax {cArg} != single-user {rArg}, NOT a near-tie (reference gap {gap:F4} >= {tieEps:F4})");
    }

    private static int Argmax(ReadOnlySpan<float> span)
    {
        int maxIdx = 0;
        float maxVal = span[0];
        for (int i = 1; i < span.Length; i++)
        {
            if (span[i] > maxVal)
            {
                maxVal = span[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    private sealed record SyntheticWeight(string CanonicalName, long[] GgufDims, float[] Data);

    private static List<SyntheticWeight> GenerateDeterministicGemma4Weights(
        int vocabSize, int hiddenSize, int intermediateSize, int heads, int hdGlobal, int hdSwa)
    {
        var rng = new Random(1337);
        var list = new List<SyntheticWeight>();

        float[] Rnd(int count)
        {
            var arr = new float[count];
            for (int i = 0; i < count; i++)
                arr[i] = (float)(rng.NextDouble() * 0.1 - 0.05);
            return arr;
        }

        list.Add(new SyntheticWeight("token_embd.weight", [hiddenSize, vocabSize], Rnd(vocabSize * hiddenSize)));
        list.Add(new SyntheticWeight("output_norm.weight", [hiddenSize], Rnd(hiddenSize)));
        list.Add(new SyntheticWeight("output.weight", [hiddenSize, vocabSize], Rnd(vocabSize * hiddenSize)));

        // Layer 0: SWA (hdSwa=32, heads=4, kvHeads=2)
        int qDim0 = heads * hdSwa;
        int kvDim0 = 2 * hdSwa;
        list.Add(new SyntheticWeight("blk.0.attn_norm.weight", [hiddenSize], Rnd(hiddenSize)));
        list.Add(new SyntheticWeight("blk.0.attn_q.weight", [hiddenSize, qDim0], Rnd(qDim0 * hiddenSize)));
        list.Add(new SyntheticWeight("blk.0.attn_k.weight", [hiddenSize, kvDim0], Rnd(kvDim0 * hiddenSize)));
        list.Add(new SyntheticWeight("blk.0.attn_v.weight", [hiddenSize, kvDim0], Rnd(kvDim0 * hiddenSize)));
        list.Add(new SyntheticWeight("blk.0.attn_output.weight", [qDim0, hiddenSize], Rnd(hiddenSize * qDim0)));
        list.Add(new SyntheticWeight("blk.0.post_attention_norm.weight", [hiddenSize], Rnd(hiddenSize)));
        list.Add(new SyntheticWeight("blk.0.ffn_norm.weight", [hiddenSize], Rnd(hiddenSize)));
        list.Add(new SyntheticWeight("blk.0.ffn_gate.weight", [hiddenSize, intermediateSize], Rnd(intermediateSize * hiddenSize)));
        list.Add(new SyntheticWeight("blk.0.ffn_up.weight", [hiddenSize, intermediateSize], Rnd(intermediateSize * hiddenSize)));
        list.Add(new SyntheticWeight("blk.0.ffn_down.weight", [intermediateSize, hiddenSize], Rnd(hiddenSize * intermediateSize)));
        list.Add(new SyntheticWeight("blk.0.post_ffw_norm.weight", [hiddenSize], Rnd(hiddenSize)));
        list.Add(new SyntheticWeight("blk.0.layer_output_scale.weight", [1], [1.05f]));

        // Layer 1: Global (hdGlobal=64, heads=4, kvHeads=1)
        int qDim1 = heads * hdGlobal;
        int kvDim1 = 1 * hdGlobal;
        list.Add(new SyntheticWeight("blk.1.attn_norm.weight", [hiddenSize], Rnd(hiddenSize)));
        list.Add(new SyntheticWeight("blk.1.attn_q.weight", [hiddenSize, qDim1], Rnd(qDim1 * hiddenSize)));
        list.Add(new SyntheticWeight("blk.1.attn_k.weight", [hiddenSize, kvDim1], Rnd(kvDim1 * hiddenSize)));
        list.Add(new SyntheticWeight("blk.1.attn_v.weight", [hiddenSize, kvDim1], Rnd(kvDim1 * hiddenSize)));
        list.Add(new SyntheticWeight("blk.1.attn_output.weight", [qDim1, hiddenSize], Rnd(hiddenSize * qDim1)));
        list.Add(new SyntheticWeight("blk.1.post_attention_norm.weight", [hiddenSize], Rnd(hiddenSize)));
        list.Add(new SyntheticWeight("blk.1.ffn_norm.weight", [hiddenSize], Rnd(hiddenSize)));
        list.Add(new SyntheticWeight("blk.1.ffn_gate.weight", [hiddenSize, intermediateSize], Rnd(intermediateSize * hiddenSize)));
        list.Add(new SyntheticWeight("blk.1.ffn_up.weight", [hiddenSize, intermediateSize], Rnd(intermediateSize * hiddenSize)));
        list.Add(new SyntheticWeight("blk.1.ffn_down.weight", [intermediateSize, hiddenSize], Rnd(hiddenSize * intermediateSize)));
        list.Add(new SyntheticWeight("blk.1.post_ffw_norm.weight", [hiddenSize], Rnd(hiddenSize)));
        list.Add(new SyntheticWeight("blk.1.layer_output_scale.weight", [1], [0.98f]));

        return list;
    }

    private static void BuildGemma4GgufFile(
        string ggufPath, List<SyntheticWeight> weights,
        int vocabSize, int hiddenSize, int intermediateSize, int layers, int heads, int hdGlobal, int hdSwa)
    {
        using var fs = new FileStream(ggufPath, FileMode.Create, FileAccess.Write);
        using var w = new BinaryWriter(fs, Encoding.UTF8);

        w.Write(0x46554747u); // Magic: GGUF
        w.Write(3u);          // Version
        w.Write((ulong)weights.Count);

        var swaPattern = new object[] { true, false };
        var metadata = new Dictionary<string, (GgufValueType Type, object Value)>
        {
            ["general.architecture"] = (GgufValueType.String, "gemma4"),
            ["general.name"] = (GgufValueType.String, "SyntheticGemma4"),
            ["gemma4.vocab_size"] = (GgufValueType.UInt32, (uint)vocabSize),
            ["gemma4.context_length"] = (GgufValueType.UInt32, 512u),
            ["gemma4.embedding_length"] = (GgufValueType.UInt32, (uint)hiddenSize),
            ["gemma4.block_count"] = (GgufValueType.UInt32, (uint)layers),
            ["gemma4.attention.head_count"] = (GgufValueType.UInt32, (uint)heads),
            ["gemma4.attention.head_count_kv"] = (GgufValueType.UInt32, 2u),
            ["gemma4.attention.key_length"] = (GgufValueType.UInt32, (uint)hdGlobal),
            ["gemma4.attention.value_length"] = (GgufValueType.UInt32, (uint)hdGlobal),
            ["gemma4.attention.key_length_swa"] = (GgufValueType.UInt32, (uint)hdSwa),
            ["gemma4.attention.value_length_swa"] = (GgufValueType.UInt32, (uint)hdSwa),
            ["gemma4.attention.sliding_window"] = (GgufValueType.UInt32, 4u),
            ["gemma4.attention.sliding_window_pattern"] = (GgufValueType.Array, swaPattern),
            ["gemma4.rope.dimension_count"] = (GgufValueType.UInt32, (uint)hdGlobal),
            ["gemma4.rope.dimension_count_swa"] = (GgufValueType.UInt32, (uint)hdSwa),
            ["gemma4.rope.freq_base"] = (GgufValueType.Float32, 1000000f),
            ["gemma4.rope.freq_base_swa"] = (GgufValueType.Float32, 10000f),
            ["gemma4.feed_forward_length"] = (GgufValueType.UInt32, (uint)intermediateSize),
            ["gemma4.attention.layer_norm_rms_epsilon"] = (GgufValueType.Float32, 1e-6f),
            ["gemma4.final_logit_softcapping"] = (GgufValueType.Float32, 30.0f),
            ["_opentailllm.has_post_attn_norm"] = (GgufValueType.Bool, true),
            ["_opentailllm.has_post_ffw_norm"] = (GgufValueType.Bool, true),
            ["_opentailllm.has_layer_output_scale"] = (GgufValueType.Bool, true),
        };

        w.Write((ulong)metadata.Count);

        foreach (var (key, (type, val)) in metadata)
        {
            WriteGgufString(w, key);
            w.Write((uint)type);
            WriteGgufValue(w, type, val);
        }

        long currentOffset = 0;
        var offsets = new List<long>();
        foreach (var tensor in weights)
        {
            long remainder = currentOffset % 32;
            if (remainder != 0) currentOffset += (32 - remainder);
            offsets.Add(currentOffset);
            currentOffset += tensor.Data.Length * sizeof(float);
        }

        for (int i = 0; i < weights.Count; i++)
        {
            var item = weights[i];
            WriteGgufString(w, item.CanonicalName);
            w.Write((uint)item.GgufDims.Length);
            foreach (var dim in item.GgufDims)
                w.Write((ulong)dim);
            w.Write(0u); // DType: Float32 = 0
            w.Write((ulong)offsets[i]);
        }

        long headerBytes = fs.Position;
        long pad = 32 - (headerBytes % 32);
        if (pad != 32)
        {
            for (int p = 0; p < pad; p++) w.Write((byte)0);
        }

        foreach (var tensor in weights)
        {
            long curPos = fs.Position;
            long rem = curPos % 32;
            if (rem != 0)
            {
                for (int p = 0; p < (32 - rem); p++) w.Write((byte)0);
            }
            foreach (var val in tensor.Data)
                w.Write(val);
        }
    }

    private static void WriteGgufString(BinaryWriter w, string s)
    {
        var bytes = Encoding.UTF8.GetBytes(s);
        w.Write((ulong)bytes.Length);
        w.Write(bytes);
    }

    private static void WriteGgufValue(BinaryWriter w, GgufValueType type, object val)
    {
        switch (type)
        {
            case GgufValueType.UInt32:
                w.Write((uint)val);
                break;
            case GgufValueType.Float32:
                w.Write((float)val);
                break;
            case GgufValueType.Bool:
                w.Write((bool)val ? (byte)1 : (byte)0);
                break;
            case GgufValueType.String:
                WriteGgufString(w, (string)val);
                break;
            case GgufValueType.Array:
                var arr = (object[])val;
                w.Write((uint)GgufValueType.Bool);
                w.Write((ulong)arr.Length);
                foreach (var item in arr)
                    w.Write((bool)item ? (byte)1 : (byte)0);
                break;
            default:
                throw new NotSupportedException($"Type {type} not supported in test generator");
        }
    }
}
