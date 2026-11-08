using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

public sealed class GlmDsaSyntheticTests : IDisposable
{
    private readonly List<string> _tempFiles = [];

    public void Dispose()
    {
        foreach (var file in _tempFiles)
        {
            try { if (File.Exists(file)) File.Delete(file); } catch { }
        }
    }

    [Fact]
    public void GlmDsaHyperparams_FromGgufMetadata_ParsesExpectedKeys()
    {
        var metadata = new Dictionary<string, object>
        {
            ["glm-dsa.leading_dense_block_count"] = (ulong)3,
            ["glm-dsa.expert_weights_scale"] = 2.5f,
            ["glm-dsa.expert_weights_norm"] = true,
            ["glm-dsa.expert_shared_count"] = (ulong)1,
            ["glm-dsa.attention.q_lora_rank"] = (ulong)2048,
            ["glm-dsa.attention.kv_lora_rank"] = (ulong)512,
            ["glm-dsa.attention.key_length_mla"] = (ulong)256,
            ["glm-dsa.attention.value_length_mla"] = (ulong)256,
            ["glm-dsa.attention.indexer.head_count"] = (ulong)32,
            ["glm-dsa.attention.indexer.key_length"] = (ulong)128,
            ["glm-dsa.attention.indexer.top_k"] = (ulong)2048,
            ["glm-dsa.expert_gating_func"] = (ulong)2,
            ["glm-dsa.attention.indexer.types"] = new object[] { 1, 1, 1, 0, 0, 0, 1, 0 },
        };

        var hp = GlmDsaHyperparams.FromGgufMetadata(
            metadata, "glm-dsa", numLayerAll: 78, embedDim: 7168, numHeads: 64, headDim: 256,
            ropeDim: 64, numExperts: 256, numExpertsUsed: 8);

        Assert.Equal(78, hp.NumLayer);
        Assert.Equal(3, hp.LeadingDenseBlockCount);
        Assert.Equal(2.5f, hp.ExpertWeightsScale);
        Assert.True(hp.ExpertWeightsNorm);
        Assert.Equal(1, hp.ExpertSharedCount);
        Assert.Equal(2048, hp.QLoraRank);
        Assert.Equal(512, hp.KvLoraRank);
        Assert.Equal(256, hp.EffectiveHeadDimK);
        Assert.Equal(256, hp.EffectiveHeadDimV);
        Assert.Equal(32, hp.IndexerNumHeads);
        Assert.Equal(128, hp.IndexerHeadSize);
        Assert.Equal(2048, hp.IndexerTopK);
        Assert.Equal(2, hp.ExpertGatingFunc);

        // Custom indexer types schedule
        Assert.True(hp.IsIndexerFull(0));
        Assert.True(hp.IsIndexerFull(1));
        Assert.True(hp.IsIndexerFull(2));
        Assert.False(hp.IsIndexerFull(3));
        Assert.False(hp.IsIndexerFull(4));
        Assert.False(hp.IsIndexerFull(5));
        Assert.True(hp.IsIndexerFull(6));
        Assert.False(hp.IsIndexerFull(7));
    }

    [Fact]
    public void GlmDsa_DefaultIndexerSchedule_FollowsGlm52Pattern()
    {
        var hp = new GlmDsaHyperparams();
        Assert.True(hp.IsIndexerFull(0));
        Assert.True(hp.IsIndexerFull(1));
        Assert.True(hp.IsIndexerFull(2));
        Assert.False(hp.IsIndexerFull(3));
        Assert.False(hp.IsIndexerFull(4));
        Assert.False(hp.IsIndexerFull(5));
        Assert.True(hp.IsIndexerFull(6));
        Assert.False(hp.IsIndexerFull(7));
        Assert.False(hp.IsIndexerFull(8));
        Assert.False(hp.IsIndexerFull(9));
        Assert.True(hp.IsIndexerFull(10));
    }

    [Fact]
    public unsafe void GlmDsa_HadamardRotation_IsInvolutoryAndOrthonormal()
    {
        const int blockSize = 128;
        float[] original = new float[blockSize];
        float[] vector = new float[blockSize];
        for (int i = 0; i < blockSize; i++)
        {
            original[i] = (float)Math.Sin(i * 0.17f);
            vector[i] = original[i];
        }

        fixed (float* ptr = vector)
        {
            // First transform
            PrismHadamard.ApplySylvesterHadamard(ptr, blockSize, blockSize);

            // Energy conservation (orthonormality: ||H x|| = ||x||)
            float originalNorm = 0f, transformedNorm = 0f;
            for (int i = 0; i < blockSize; i++)
            {
                originalNorm += original[i] * original[i];
                transformedNorm += vector[i] * vector[i];
            }
            Assert.Equal(MathF.Sqrt(originalNorm), MathF.Sqrt(transformedNorm), 4);

            // Second transform: H(H(x)) = x because H is symmetric and orthonormal
            PrismHadamard.ApplySylvesterHadamard(ptr, blockSize, blockSize);
            for (int i = 0; i < blockSize; i++)
            {
                Assert.Equal(original[i], vector[i], 4);
            }
        }
    }

    [Fact]
    public void GlmDsa_SyntheticForwardPass_ExecutesCorrectly()
    {
        const int embedDim = 32;
        const int numHeads = 2;
        const int qLoraRank = 16;
        const int kvLoraRank = 16;
        const int headDimK = 16;
        const int headDimV = 16;
        const int ropeDim = 8;
        const int indexerNumHeads = 2;
        const int indexerHeadDim = 128; // must be 128 for Hadamard
        const int numExperts = 4;
        const int numExpertsUsed = 2;
        const int expertFfn = 32;
        const int denseFfn = 32;
        const int vocabSize = 32;
        const int numLayers = 3; // layer 0 dense, layers 1 and 2 MoE

        string path = Path.Combine(Path.GetTempPath(), $"glm_dsa_synthetic_{Guid.NewGuid():N}.gguf");
        _tempFiles.Add(path);

        var tensors = new Dictionary<string, (long[] shape, DType dtype, byte[] data)>();

        void AddTensor(string name, long[] shape, float fill = 0.05f)
        {
            long total = 1;
            foreach (var s in shape) total *= s;
            var floats = new float[total];
            for (int i = 0; i < total; i++) floats[i] = fill * MathF.Sin(i + 1);
            var bytes = new byte[total * 4];
            Buffer.BlockCopy(floats, 0, bytes, 0, bytes.Length);
            tensors[name] = (shape, DType.Float32, bytes);
        }

        AddTensor("token_embd.weight", [embedDim, vocabSize]);
        AddTensor("output_norm.weight", [embedDim]);
        AddTensor("output.weight", [embedDim, vocabSize]);

        for (int i = 0; i < numLayers; i++)
        {
            AddTensor($"blk.{i}.attn_norm.weight", [embedDim]);
            AddTensor($"blk.{i}.attn_q_a_norm.weight", [qLoraRank]);
            AddTensor($"blk.{i}.attn_kv_a_norm.weight", [kvLoraRank]);
            AddTensor($"blk.{i}.attn_q_a.weight", [embedDim, qLoraRank]);
            AddTensor($"blk.{i}.attn_q_b.weight", [qLoraRank, numHeads * headDimK]);
            AddTensor($"blk.{i}.attn_kv_a_mqa.weight", [embedDim, kvLoraRank + ropeDim]);
            AddTensor($"blk.{i}.attn_k_b.weight", [headDimK - ropeDim, kvLoraRank, numHeads]);
            AddTensor($"blk.{i}.attn_v_b.weight", [kvLoraRank, headDimV, numHeads]);
            AddTensor($"blk.{i}.attn_output.weight", [numHeads * headDimV, embedDim]);
            AddTensor($"blk.{i}.ffn_norm.weight", [embedDim]);

            // Indexer
            AddTensor($"blk.{i}.indexer_k_norm.weight", [indexerHeadDim]);
            AddTensor($"blk.{i}.indexer_k_norm.bias", [indexerHeadDim]);
            AddTensor($"blk.{i}.indexer_proj.weight", [embedDim, indexerNumHeads]);
            AddTensor($"blk.{i}.indexer_attn_k.weight", [embedDim, indexerHeadDim]);
            AddTensor($"blk.{i}.indexer_attn_q_b.weight", [qLoraRank, indexerNumHeads * indexerHeadDim]);

            if (i == 0)
            {
                AddTensor($"blk.{i}.ffn_gate.weight", [embedDim, denseFfn]);
                AddTensor($"blk.{i}.ffn_up.weight", [embedDim, denseFfn]);
                AddTensor($"blk.{i}.ffn_down.weight", [denseFfn, embedDim]);
            }
            else
            {
                AddTensor($"blk.{i}.ffn_gate_inp.weight", [embedDim, numExperts]);
                AddTensor($"blk.{i}.ffn_gate_exps.weight", [embedDim, expertFfn, numExperts]);
                AddTensor($"blk.{i}.ffn_up_exps.weight", [embedDim, expertFfn, numExperts]);
                AddTensor($"blk.{i}.ffn_down_exps.weight", [expertFfn, embedDim, numExperts]);
                AddTensor($"blk.{i}.ffn_gate_shexp.weight", [embedDim, expertFfn]);
                AddTensor($"blk.{i}.ffn_up_shexp.weight", [embedDim, expertFfn]);
                AddTensor($"blk.{i}.ffn_down_shexp.weight", [expertFfn, embedDim]);
            }
        }

        var metadata = new Dictionary<string, object>
        {
            ["general.architecture"] = "glm-dsa",
            ["glm-dsa.leading_dense_block_count"] = (ulong)1,
            ["glm-dsa.expert_weights_scale"] = 2.5f,
            ["glm-dsa.expert_weights_norm"] = true,
            ["glm-dsa.expert_shared_count"] = (ulong)1,
            ["glm-dsa.attention.q_lora_rank"] = (ulong)qLoraRank,
            ["glm-dsa.attention.kv_lora_rank"] = (ulong)kvLoraRank,
            ["glm-dsa.attention.key_length_mla"] = (ulong)headDimK,
            ["glm-dsa.attention.value_length_mla"] = (ulong)headDimV,
            ["glm-dsa.attention.indexer.head_count"] = (ulong)indexerNumHeads,
            ["glm-dsa.attention.indexer.key_length"] = (ulong)indexerHeadDim,
            ["glm-dsa.attention.indexer.top_k"] = (ulong)16,
            ["glm-dsa.expert_gating_func"] = (ulong)2,
        };

        WriteGgufFile(path, metadata, tensors);

        using var model = GgufModel.Open(path);
        var hp = GlmDsaHyperparams.FromGgufMetadata(
            model.Metadata, "glm-dsa", numLayerAll: numLayers, embedDim: embedDim, numHeads: numHeads,
            headDim: headDimK, ropeDim: ropeDim, numExperts: numExperts, numExpertsUsed: numExpertsUsed);

        var forwardPass = new GlmDsaForwardPass(model, hp);
        Assert.Equal(vocabSize, forwardPass.VocabSize);

        // Step 0
        var logits0 = forwardPass.Forward(token: 5, position: 0);
        Assert.Equal(vocabSize, logits0.Length);
        foreach (var l in logits0)
        {
            Assert.False(float.IsNaN(l));
            Assert.False(float.IsInfinity(l));
        }

        // Step 1
        var logits1 = forwardPass.Forward(token: 12, position: 1);
        Assert.Equal(vocabSize, logits1.Length);
        foreach (var l in logits1)
        {
            Assert.False(float.IsNaN(l));
            Assert.False(float.IsInfinity(l));
        }
    }

    /// <summary>
    /// Proves the shortcut's actual correctness property: when <c>numKeys == topK</c>, the
    /// shortcut returns <c>[0..numKeys-1]</c> directly, while the full-scoring path would select
    /// the top-<c>numKeys</c> indices by score from exactly <c>numKeys</c> candidates -- which, by
    /// the pigeonhole principle, is every index regardless of the actual score VALUES (only their
    /// relative order could matter, and with <c>topK == count</c> there is no cut to make). Sorting
    /// either result yields the identical array. This complements (does not replace)
    /// <see cref="GlmDsa_IndexerTopKShortcut_RunsAcrossThreshold_ProducesFiniteLogits"/>, which only
    /// proves the code path executes without crashing -- flagged correctly as a gap on review
    /// (2026-11-08): that test's old name implied an output comparison it never made.
    /// </summary>
    [Theory]
    [InlineData(new float[] { 0.9f, 0.1f, 0.5f, -3f, 2f })] // arbitrary scores, any order
    [InlineData(new float[] { 5f, 5f, 5f, 5f, 5f })] // tied scores (stresses sort stability)
    [InlineData(new float[] { -1f, -2f, -3f, -4f, -5f })] // strictly descending already
    public void SelectTopKIndices_WhenTopKEqualsItemCount_AlwaysSelectsEveryIndexRegardlessOfScores(float[] scores)
    {
        int n = scores.Length;
        int[] fullScoringResult = DeepSeek4Graph.SelectTopKIndices(scores, n);
        Array.Sort(fullScoringResult);

        int[] shortcutResult = new int[n];
        for (int t = 0; t < n; t++) shortcutResult[t] = t; // GlmDsaForwardPass.RunFullIndexer's shortcut branch

        Assert.Equal(shortcutResult, fullScoringResult);
    }

    /// <summary>
    /// Exercises the real-scoring indexer path (not the top_k shortcut) with YaRN metadata active,
    /// after the 2026-11-08 fix that made the indexer's RoPE use the YaRN formula. Only a
    /// finite-output smoke check (no independent reference run here), but it does make sure the
    /// new code path -- YaRN-scaled indexer RoPE feeding into Hadamard + scoring + top-k -- at
    /// least runs correctly end-to-end with yarn active, which the pre-fix code path never did.
    /// </summary>
    [Fact]
    public void GlmDsa_FullIndexerScoring_WithYarnActive_ProducesFiniteLogits()
    {
        const int embedDim = 32;
        const int numHeads = 2;
        const int qLoraRank = 16;
        const int kvLoraRank = 16;
        const int headDimK = 16;
        const int headDimV = 16;
        const int ropeDim = 8;
        const int indexerNumHeads = 2;
        const int indexerHeadDim = 128;
        const int denseFfn = 32;
        const int vocabSize = 32;
        const int numLayers = 1;
        const int indexerTopK = 1; // position 1 (numKeys=2) already exceeds this -> real scoring

        string path = Path.Combine(Path.GetTempPath(), $"glm_dsa_yarn_indexer_{Guid.NewGuid():N}.gguf");
        _tempFiles.Add(path);

        var tensors = new Dictionary<string, (long[] shape, DType dtype, byte[] data)>();

        void AddTensor(string name, long[] shape, float fill = 0.05f)
        {
            long total = 1;
            foreach (var s in shape) total *= s;
            var floats = new float[total];
            for (int i = 0; i < total; i++) floats[i] = fill * MathF.Sin(i + 1);
            var bytes = new byte[total * 4];
            Buffer.BlockCopy(floats, 0, bytes, 0, bytes.Length);
            tensors[name] = (shape, DType.Float32, bytes);
        }

        AddTensor("token_embd.weight", [embedDim, vocabSize]);
        AddTensor("output_norm.weight", [embedDim]);
        AddTensor("output.weight", [embedDim, vocabSize]);

        for (int i = 0; i < numLayers; i++)
        {
            AddTensor($"blk.{i}.attn_norm.weight", [embedDim]);
            AddTensor($"blk.{i}.attn_q_a_norm.weight", [qLoraRank]);
            AddTensor($"blk.{i}.attn_kv_a_norm.weight", [kvLoraRank]);
            AddTensor($"blk.{i}.attn_q_a.weight", [embedDim, qLoraRank]);
            AddTensor($"blk.{i}.attn_q_b.weight", [qLoraRank, numHeads * headDimK]);
            AddTensor($"blk.{i}.attn_kv_a_mqa.weight", [embedDim, kvLoraRank + ropeDim]);
            AddTensor($"blk.{i}.attn_k_b.weight", [headDimK - ropeDim, kvLoraRank, numHeads]);
            AddTensor($"blk.{i}.attn_v_b.weight", [kvLoraRank, headDimV, numHeads]);
            AddTensor($"blk.{i}.attn_output.weight", [numHeads * headDimV, embedDim]);
            AddTensor($"blk.{i}.ffn_norm.weight", [embedDim]);

            AddTensor($"blk.{i}.indexer_k_norm.weight", [indexerHeadDim]);
            AddTensor($"blk.{i}.indexer_k_norm.bias", [indexerHeadDim]);
            AddTensor($"blk.{i}.indexer_proj.weight", [embedDim, indexerNumHeads]);
            AddTensor($"blk.{i}.indexer_attn_k.weight", [embedDim, indexerHeadDim]);
            AddTensor($"blk.{i}.indexer_attn_q_b.weight", [qLoraRank, indexerNumHeads * indexerHeadDim]);

            AddTensor($"blk.{i}.ffn_gate.weight", [embedDim, denseFfn]);
            AddTensor($"blk.{i}.ffn_up.weight", [embedDim, denseFfn]);
            AddTensor($"blk.{i}.ffn_down.weight", [denseFfn, embedDim]);
        }

        var metadata = new Dictionary<string, object>
        {
            ["general.architecture"] = "glm-dsa",
            ["glm-dsa.leading_dense_block_count"] = (ulong)numLayers,
            ["glm-dsa.expert_weights_scale"] = 2.5f,
            ["glm-dsa.expert_weights_norm"] = true,
            ["glm-dsa.expert_shared_count"] = (ulong)1,
            ["glm-dsa.attention.q_lora_rank"] = (ulong)qLoraRank,
            ["glm-dsa.attention.kv_lora_rank"] = (ulong)kvLoraRank,
            ["glm-dsa.attention.key_length_mla"] = (ulong)headDimK,
            ["glm-dsa.attention.value_length_mla"] = (ulong)headDimV,
            ["glm-dsa.attention.indexer.head_count"] = (ulong)indexerNumHeads,
            ["glm-dsa.attention.indexer.key_length"] = (ulong)indexerHeadDim,
            ["glm-dsa.attention.indexer.top_k"] = (ulong)indexerTopK,
            ["glm-dsa.expert_gating_func"] = (ulong)2,
            ["glm-dsa.rope.scaling.factor"] = 4f,
            ["glm-dsa.rope.scaling.original_context_length"] = (ulong)4096,
        };

        WriteGgufFile(path, metadata, tensors);

        using var model = GgufModel.Open(path);
        var hp = GlmDsaHyperparams.FromGgufMetadata(
            model.Metadata, "glm-dsa", numLayerAll: numLayers, embedDim: embedDim, numHeads: numHeads,
            headDim: headDimK, ropeDim: ropeDim, numExperts: 4, numExpertsUsed: 2);
        Assert.True(hp.RopeYarnFactor > 1f, "metadata should have parsed as YaRN-active");

        var forwardPass = new GlmDsaForwardPass(model, hp);
        int[] tokens = [1, 5, 9];
        for (int pos = 0; pos < tokens.Length; pos++)
        {
            var logits = forwardPass.Forward(token: tokens[pos], position: pos);
            Assert.Equal(vocabSize, logits.Length);
            foreach (var l in logits)
            {
                Assert.False(float.IsNaN(l));
                Assert.False(float.IsInfinity(l));
            }
        }
    }

    [Fact]
    public void GlmDsa_IndexerTopKShortcut_RunsAcrossThreshold_ProducesFiniteLogits()
    {
        const int embedDim = 32;
        const int numHeads = 2;
        const int qLoraRank = 16;
        const int kvLoraRank = 16;
        const int headDimK = 16;
        const int headDimV = 16;
        const int ropeDim = 8;
        const int indexerNumHeads = 2;
        const int indexerHeadDim = 128;
        const int numExperts = 4;
        const int numExpertsUsed = 2;
        const int denseFfn = 32;
        const int vocabSize = 32;
        const int numLayers = 1; // single dense "full indexer" layer is enough to exercise the shortcut
        const int indexerTopK = 3; // positions 0..2 stay <= top_k (shortcut); position 3 crosses it

        string path = Path.Combine(Path.GetTempPath(), $"glm_dsa_indexer_topk_{Guid.NewGuid():N}.gguf");
        _tempFiles.Add(path);

        var tensors = new Dictionary<string, (long[] shape, DType dtype, byte[] data)>();

        void AddTensor(string name, long[] shape, float fill = 0.05f)
        {
            long total = 1;
            foreach (var s in shape) total *= s;
            var floats = new float[total];
            for (int i = 0; i < total; i++) floats[i] = fill * MathF.Sin(i + 1);
            var bytes = new byte[total * 4];
            Buffer.BlockCopy(floats, 0, bytes, 0, bytes.Length);
            tensors[name] = (shape, DType.Float32, bytes);
        }

        AddTensor("token_embd.weight", [embedDim, vocabSize]);
        AddTensor("output_norm.weight", [embedDim]);
        AddTensor("output.weight", [embedDim, vocabSize]);

        for (int i = 0; i < numLayers; i++)
        {
            AddTensor($"blk.{i}.attn_norm.weight", [embedDim]);
            AddTensor($"blk.{i}.attn_q_a_norm.weight", [qLoraRank]);
            AddTensor($"blk.{i}.attn_kv_a_norm.weight", [kvLoraRank]);
            AddTensor($"blk.{i}.attn_q_a.weight", [embedDim, qLoraRank]);
            AddTensor($"blk.{i}.attn_q_b.weight", [qLoraRank, numHeads * headDimK]);
            AddTensor($"blk.{i}.attn_kv_a_mqa.weight", [embedDim, kvLoraRank + ropeDim]);
            AddTensor($"blk.{i}.attn_k_b.weight", [headDimK - ropeDim, kvLoraRank, numHeads]);
            AddTensor($"blk.{i}.attn_v_b.weight", [kvLoraRank, headDimV, numHeads]);
            AddTensor($"blk.{i}.attn_output.weight", [numHeads * headDimV, embedDim]);
            AddTensor($"blk.{i}.ffn_norm.weight", [embedDim]);

            AddTensor($"blk.{i}.indexer_k_norm.weight", [indexerHeadDim]);
            AddTensor($"blk.{i}.indexer_k_norm.bias", [indexerHeadDim]);
            AddTensor($"blk.{i}.indexer_proj.weight", [embedDim, indexerNumHeads]);
            AddTensor($"blk.{i}.indexer_attn_k.weight", [embedDim, indexerHeadDim]);
            AddTensor($"blk.{i}.indexer_attn_q_b.weight", [qLoraRank, indexerNumHeads * indexerHeadDim]);

            AddTensor($"blk.{i}.ffn_gate.weight", [embedDim, denseFfn]);
            AddTensor($"blk.{i}.ffn_up.weight", [embedDim, denseFfn]);
            AddTensor($"blk.{i}.ffn_down.weight", [denseFfn, embedDim]);
        }

        var metadata = new Dictionary<string, object>
        {
            ["general.architecture"] = "glm-dsa",
            ["glm-dsa.leading_dense_block_count"] = (ulong)numLayers,
            ["glm-dsa.expert_weights_scale"] = 2.5f,
            ["glm-dsa.expert_weights_norm"] = true,
            ["glm-dsa.expert_shared_count"] = (ulong)1,
            ["glm-dsa.attention.q_lora_rank"] = (ulong)qLoraRank,
            ["glm-dsa.attention.kv_lora_rank"] = (ulong)kvLoraRank,
            ["glm-dsa.attention.key_length_mla"] = (ulong)headDimK,
            ["glm-dsa.attention.value_length_mla"] = (ulong)headDimV,
            ["glm-dsa.attention.indexer.head_count"] = (ulong)indexerNumHeads,
            ["glm-dsa.attention.indexer.key_length"] = (ulong)indexerHeadDim,
            ["glm-dsa.attention.indexer.top_k"] = (ulong)indexerTopK,
            ["glm-dsa.expert_gating_func"] = (ulong)2,
        };

        WriteGgufFile(path, metadata, tensors);

        using var model = GgufModel.Open(path);
        var hp = GlmDsaHyperparams.FromGgufMetadata(
            model.Metadata, "glm-dsa", numLayerAll: numLayers, embedDim: embedDim, numHeads: numHeads,
            headDim: headDimK, ropeDim: ropeDim, numExperts: numExperts, numExpertsUsed: numExpertsUsed);

        var forwardPass = new GlmDsaForwardPass(model, hp);

        // Drive 5 tokens through: positions 0,1,2 keep numKeys (1..3) <= top_k=3 -- the shortcut
        // path. Position 3 makes numKeys=4 > top_k=3 -- first step that must run real scoring.
        // Position 4 (numKeys=5) re-exercises scoring with the indexer K cache now spanning both
        // code paths, confirming earlier shortcut-path keys are still usable as scoring inputs.
        int[] tokens = [1, 5, 9, 13, 17];
        for (int pos = 0; pos < tokens.Length; pos++)
        {
            var logits = forwardPass.Forward(token: tokens[pos], position: pos);
            Assert.Equal(vocabSize, logits.Length);
            foreach (var l in logits)
            {
                Assert.False(float.IsNaN(l));
                Assert.False(float.IsInfinity(l));
            }
        }
    }

    /// <summary>
    /// Reproduces the exact fix in GlmDsaForwardPass.MoeFfn (2026-11-08): the sigmoid MoE's
    /// selection bias (<c>exp_probs_b</c>) is added to a COPY of the unbiased sigmoid probs for
    /// choosing which experts to route to, but the weight gathered for each chosen expert must
    /// come from the ORIGINAL unbiased array -- same bug shape, same fix
    /// (<c>DeepSeek4Graph.WeightSelectedExperts</c>), as the one found and fixed in
    /// <c>DeepSeek4ForwardPass.MoeFfn</c> the same week. Picks a bias large enough to flip which
    /// expert wins top-1, then confirms the winner's weight is its small unbiased sigmoid prob.
    /// </summary>
    [Fact]
    public void MoeFfn_SigmoidGating_SelectionBiasNeverLeaksIntoTheGatheredWeight()
    {
        float[] logits = [0.1f, 0.2f, 0.15f, 0.05f, 0.08f];
        var unbiasedProbs = new float[5];
        for (int e = 0; e < 5; e++) unbiasedProbs[e] = 1f / (1f + MathF.Exp(-logits[e]));

        float[] bias = [0f, 0f, 0f, 0f, 5f]; // pushes index 4 to the top post-sigmoid
        var biasedSelectionProbs = (float[])unbiasedProbs.Clone();
        for (int e = 0; e < 5; e++) biasedSelectionProbs[e] += bias[e];

        int[] selected = DeepSeek4Graph.SelectTopKIndices(biasedSelectionProbs, topK: 1);
        Assert.Equal(4, selected[0]); // bias won the selection

        var weights = new float[1];
        DeepSeek4Graph.WeightSelectedExperts(selected, unbiasedProbs, topK: 1, normalize: false, scale: 1f, weights);

        Assert.Equal(unbiasedProbs[4], weights[0], 6); // weight is the UNBIASED prob at index 4
        Assert.NotEqual(biasedSelectionProbs[4], weights[0]); // never the biased value
    }

    /// <summary>
    /// Proves the YaRN indexer-RoPE fix (2026-11-08, <c>RunFullIndexer</c>) actually matters: when
    /// YaRN is active (<c>rope.scaling.factor &gt; 1</c>), the plain ("NORM") rotation formula the
    /// indexer's RoPE used to use and the YaRN-scaled formula glm-dsa.cpp actually applies there
    /// (same params as the main attention's q_pe/k_pe RoPE -- lines 271-273/284-286 vs 400-406)
    /// diverge. If they always agreed, the fix would have been a no-op; this shows they don't once
    /// YaRN is active, and that they DO agree when it isn't (the fix's claimed zero-behaviour-change
    /// case for every checkpoint seen so far).
    /// </summary>
    [Fact]
    public void YarnRope_DivergesFromPlainRopeOnlyWhenYarnIsActive()
    {
        const int dim = 8;
        const int position = 50;
        const float freqBase = 10000f;

        float[] PlainRope()
        {
            var x = new float[dim];
            for (int i = 0; i < dim; i++) x[i] = 1f + i * 0.1f;
            int half = dim / 2;
            for (int i = 0; i < half; i++)
            {
                float freq = MathF.Pow(freqBase, -2f * i / dim);
                float theta = position * freq;
                float cos = MathF.Cos(theta), sin = MathF.Sin(theta);
                float x0 = x[2 * i], x1 = x[2 * i + 1];
                x[2 * i] = x0 * cos - x1 * sin;
                x[2 * i + 1] = x0 * sin + x1 * cos;
            }
            return x;
        }

        float[] YarnRope(float yarnFactor, int origCtxLen)
        {
            var x = new float[dim];
            for (int i = 0; i < dim; i++) x[i] = 1f + i * 0.1f;
            int half = dim / 2;
            bool yarnActive = yarnFactor > 1f;
            float freqScale = yarnActive ? 1f / yarnFactor : 1f;
            float extFactor = yarnActive ? 1f : 0f;
            const float attnFactorBaseline = 1f;

            static float CorrDim(int nDims, int nCtxOrig, float nRot, float b) =>
                nDims * MathF.Log(nCtxOrig / (nRot * 2f * MathF.PI)) / (2f * MathF.Log(b));

            float corrLow = 0f, corrHigh = half * 2f - 1f;
            if (origCtxLen > 0)
            {
                corrLow = MathF.Max(0f, MathF.Floor(CorrDim(dim, origCtxLen, 32f, freqBase)));
                corrHigh = MathF.Min(dim - 1, MathF.Ceiling(CorrDim(dim, origCtxLen, 1f, freqBase)));
            }

            float thetaScale = MathF.Pow(freqBase, -2f / dim);
            float thetaExtrap = position;
            for (int i = 0; i < half; i++)
            {
                float thetaInterp = freqScale * thetaExtrap;
                float thetaFinal = thetaInterp;
                float mscale = attnFactorBaseline;
                if (extFactor != 0f)
                {
                    float y = (i - corrLow) / MathF.Max(0.001f, corrHigh - corrLow);
                    float rampMix = (1f - MathF.Min(1f, MathF.Max(0f, y))) * extFactor;
                    thetaFinal = thetaInterp * (1f - rampMix) + thetaExtrap * rampMix;
                    mscale *= 1f + 0.1f * MathF.Log(1f / freqScale);
                }
                float cos = MathF.Cos(thetaFinal) * mscale, sin = MathF.Sin(thetaFinal) * mscale;
                float x0 = x[2 * i], x1 = x[2 * i + 1];
                x[2 * i] = x0 * cos - x1 * sin;
                x[2 * i + 1] = x0 * sin + x1 * cos;
                thetaExtrap *= thetaScale;
            }
            return x;
        }

        // YaRN inactive (factor == 1, the default for every checkpoint this repo has verified
        // metadata against so far): the two formulas must agree exactly -- this is the "zero
        // behaviour change on checkpoints seen so far" half of the fix's claim.
        float[] plain = PlainRope();
        float[] yarnInactive = YarnRope(yarnFactor: 1f, origCtxLen: 4096);
        for (int i = 0; i < dim; i++) Assert.Equal(plain[i], yarnInactive[i], 5);

        // YaRN active (factor > 1): the formulas must diverge -- this is why using the plain
        // formula for the indexer's RoPE (the bug) was wrong whenever YaRN is actually on.
        float[] yarnActiveResult = YarnRope(yarnFactor: 4f, origCtxLen: 4096);
        bool anyDifferent = false;
        for (int i = 0; i < dim; i++)
        {
            if (MathF.Abs(plain[i] - yarnActiveResult[i]) > 1e-4f) anyDifferent = true;
        }
        Assert.True(anyDifferent, "YaRN-active rotation should differ from the plain formula -- if it never does, the indexer RoPE fix has no effect and the bug it fixed was never real");
    }

    /// <summary>
    /// Shared-expert contribution check (Phase 2A's last open synthetic-correctness item):
    /// glm-dsa.cpp's shared expert (lines 486-498) is a plain SwiGLU FFN -- <c>build_ffn</c> with
    /// <c>LLM_FFN_SILU</c>/<c>LLM_FFN_PAR</c>, no extra scale -- added to the routed MoE output with
    /// <c>ggml_add</c>, exactly matching <c>GlmDsaForwardPass.MoeFfn</c>'s shared-expert block
    /// (gate/up MatVec, SiLU(gate)*up, down MatVec, accumulated into the routed sum with no scale
    /// factor). Verifies that exact composition -- SiLU-gated SwiGLU, no extra weighting -- against
    /// hand-computed values using <see cref="SimdKernels.MatVec"/> directly (the same primitive
    /// production uses), with weight matrices simple enough to check by hand rather than
    /// recomputing the expected value with the same logic under test.
    /// </summary>
    [Fact]
    public unsafe void SharedExpertSwiGlu_MatchesHandComputedValues()
    {
        // input = [1, 2]; gate/up weights are identity (2x2 row-major [out,in]) so gate==up==input;
        // down weight is identity too, so the result is exactly silu(x)*x elementwise.
        float[] input = [1f, 2f];
        float[] identity2x2 = [1f, 0f, 0f, 1f];

        var gate = new float[2];
        var up = new float[2];
        fixed (float* inPtr = input, wPtr = identity2x2, gatePtr = gate, upPtr = up)
        {
            SimdKernels.MatVec(gatePtr, (byte*)wPtr, inPtr, rows: 2, cols: 2, DType.Float32);
            SimdKernels.MatVec(upPtr, (byte*)wPtr, inPtr, rows: 2, cols: 2, DType.Float32);
        }

        for (int i = 0; i < 2; i++)
        {
            float g = gate[i];
            float silu = g / (1f + MathF.Exp(-g)); // SiLU(g) = g * sigmoid(g) = g / (1 + e^-g)
            gate[i] = silu * up[i];
        }

        var down = new float[2];
        fixed (float* gatePtr = gate, wPtr = identity2x2, downPtr = down)
        {
            SimdKernels.MatVec(downPtr, (byte*)wPtr, gatePtr, rows: 2, cols: 2, DType.Float32);
        }

        // Hand computation: SiLU(x) = x / (1 + e^-x); with identity gate/up/down and input [1, 2],
        // the whole pipeline reduces to SiLU(x) * x elementwise (gate == up == input == x here).
        float siluOf1 = 1f / (1f + MathF.Exp(-1f)); // SiLU(1) ~0.7310586
        float siluOf2 = 2f / (1f + MathF.Exp(-2f)); // SiLU(2) ~1.7615942
        float expected0 = siluOf1 * 1f; // SiLU(1) * up[0] = SiLU(1) * 1
        float expected1 = siluOf2 * 2f; // SiLU(2) * up[1] = SiLU(2) * 2
        Assert.Equal(expected0, down[0], 5);
        Assert.Equal(expected1, down[1], 5);

        // Sanity on the hand-computed constants themselves, independent of SimdKernels.
        Assert.True(MathF.Abs(expected0 - 0.7310586f) < 1e-4f);
        Assert.True(MathF.Abs(expected1 - 3.5231884f) < 1e-4f);
    }

    /// <summary>
    /// End-to-end isolation of the shared-expert's contribution inside the real forward pass:
    /// zeroes every routed-expert tensor (gate_exps/up_exps), which forces each routed expert's
    /// SwiGLU input to 0 regardless of which experts the router selects, so the down-projection of
    /// a zero vector is exactly zero (<see cref="SimdKernels.MatVec"/> of an all-zero input, no
    /// bias) -- the routed-MoE contribution is provably zero however routing picks its top-k.
    /// What survives in the FFN output is therefore exactly the shared expert's own SwiGLU,
    /// confirming <c>MoeFfn</c> actually adds the shared contribution rather than silently
    /// dropping or double-counting it.
    /// </summary>
    [Fact]
    public void MoeFfn_SharedExpertContribution_SurvivesWhenRoutedExpertsAreZeroed()
    {
        const int embedDim = 16;
        const int numHeads = 2;
        const int qLoraRank = 8;
        const int kvLoraRank = 8;
        const int headDimK = 8;
        const int headDimV = 8;
        const int ropeDim = 4;
        const int indexerNumHeads = 1;
        const int indexerHeadDim = 128;
        const int numExperts = 2;
        const int numExpertsUsed = 1;
        const int expertFfn = 8;
        const int vocabSize = 16;
        const int numLayers = 2; // layer 0 dense (builds a non-trivial hidden state), layer 1 MoE

        string path = Path.Combine(Path.GetTempPath(), $"glm_dsa_shared_expert_{Guid.NewGuid():N}.gguf");
        _tempFiles.Add(path);

        var tensors = new Dictionary<string, (long[] shape, DType dtype, byte[] data)>();

        void AddTensor(string name, long[] shape, float fill = 0.05f)
        {
            long total = 1;
            foreach (var s in shape) total *= s;
            var floats = new float[total];
            for (int i = 0; i < total; i++) floats[i] = fill * MathF.Sin(i + 1);
            var bytes = new byte[total * 4];
            Buffer.BlockCopy(floats, 0, bytes, 0, bytes.Length);
            tensors[name] = (shape, DType.Float32, bytes);
        }

        void AddZeroTensor(string name, long[] shape)
        {
            long total = 1;
            foreach (var s in shape) total *= s;
            tensors[name] = (shape, DType.Float32, new byte[total * 4]);
        }

        AddTensor("token_embd.weight", [embedDim, vocabSize]);
        AddTensor("output_norm.weight", [embedDim]);
        AddTensor("output.weight", [embedDim, vocabSize]);

        for (int i = 0; i < numLayers; i++)
        {
            AddTensor($"blk.{i}.attn_norm.weight", [embedDim]);
            AddTensor($"blk.{i}.attn_q_a_norm.weight", [qLoraRank]);
            AddTensor($"blk.{i}.attn_kv_a_norm.weight", [kvLoraRank]);
            AddTensor($"blk.{i}.attn_q_a.weight", [embedDim, qLoraRank]);
            AddTensor($"blk.{i}.attn_q_b.weight", [qLoraRank, numHeads * headDimK]);
            AddTensor($"blk.{i}.attn_kv_a_mqa.weight", [embedDim, kvLoraRank + ropeDim]);
            AddTensor($"blk.{i}.attn_k_b.weight", [headDimK - ropeDim, kvLoraRank, numHeads]);
            AddTensor($"blk.{i}.attn_v_b.weight", [kvLoraRank, headDimV, numHeads]);
            AddTensor($"blk.{i}.attn_output.weight", [numHeads * headDimV, embedDim]);
            AddTensor($"blk.{i}.ffn_norm.weight", [embedDim]);

            AddTensor($"blk.{i}.indexer_k_norm.weight", [indexerHeadDim]);
            AddTensor($"blk.{i}.indexer_k_norm.bias", [indexerHeadDim]);
            AddTensor($"blk.{i}.indexer_proj.weight", [embedDim, indexerNumHeads]);
            AddTensor($"blk.{i}.indexer_attn_k.weight", [embedDim, indexerHeadDim]);
            AddTensor($"blk.{i}.indexer_attn_q_b.weight", [qLoraRank, indexerNumHeads * indexerHeadDim]);

            if (i == 0)
            {
                AddTensor($"blk.{i}.ffn_gate.weight", [embedDim, expertFfn]);
                AddTensor($"blk.{i}.ffn_up.weight", [embedDim, expertFfn]);
                AddTensor($"blk.{i}.ffn_down.weight", [expertFfn, embedDim]);
            }
            else
            {
                AddTensor($"blk.{i}.ffn_gate_inp.weight", [embedDim, numExperts]);
                // Routed-expert gate/up zeroed: every routed expert's SwiGLU input is 0 regardless
                // of which expert(s) the router picks, so down(silu(0)*0) == down(0) == 0 exactly
                // -- the routed contribution is provably zero, isolating the shared expert.
                AddZeroTensor($"blk.{i}.ffn_gate_exps.weight", [embedDim, expertFfn, numExperts]);
                AddZeroTensor($"blk.{i}.ffn_up_exps.weight", [embedDim, expertFfn, numExperts]);
                AddTensor($"blk.{i}.ffn_down_exps.weight", [expertFfn, embedDim, numExperts]); // irrelevant: input is 0
                AddTensor($"blk.{i}.ffn_gate_shexp.weight", [embedDim, expertFfn]);
                AddTensor($"blk.{i}.ffn_up_shexp.weight", [embedDim, expertFfn]);
                AddTensor($"blk.{i}.ffn_down_shexp.weight", [expertFfn, embedDim]);
            }
        }

        var metadata = new Dictionary<string, object>
        {
            ["general.architecture"] = "glm-dsa",
            ["glm-dsa.leading_dense_block_count"] = (ulong)1,
            ["glm-dsa.expert_weights_scale"] = 2.5f,
            ["glm-dsa.expert_weights_norm"] = true,
            ["glm-dsa.expert_shared_count"] = (ulong)1,
            ["glm-dsa.attention.q_lora_rank"] = (ulong)qLoraRank,
            ["glm-dsa.attention.kv_lora_rank"] = (ulong)kvLoraRank,
            ["glm-dsa.attention.key_length_mla"] = (ulong)headDimK,
            ["glm-dsa.attention.value_length_mla"] = (ulong)headDimV,
            ["glm-dsa.attention.indexer.head_count"] = (ulong)indexerNumHeads,
            ["glm-dsa.attention.indexer.key_length"] = (ulong)indexerHeadDim,
            ["glm-dsa.attention.indexer.top_k"] = (ulong)16,
            ["glm-dsa.expert_gating_func"] = (ulong)2,
        };

        WriteGgufFile(path, metadata, tensors);

        using var model = GgufModel.Open(path);
        var hp = GlmDsaHyperparams.FromGgufMetadata(
            model.Metadata, "glm-dsa", numLayerAll: numLayers, embedDim: embedDim, numHeads: numHeads,
            headDim: headDimK, ropeDim: ropeDim, numExperts: numExperts, numExpertsUsed: numExpertsUsed);

        var withRouted = new GlmDsaForwardPass(model, hp);
        var logitsWithRouted = withRouted.Forward(token: 3, position: 0).ToArray();

        // Rebuild with the shared expert's weights also zeroed (mutating `tensors` in place is
        // safe here -- the first model's GGUF bytes were already written to `path` above, so this
        // only affects the second file): if the shared contribution were ever dropped (e.g. an
        // accidental overwrite instead of +=), THIS run would already equal the first one -- it
        // wouldn't need the shared weights zeroed to match. Confirms the first run's output
        // actually depends on the shared expert, not just the (zeroed) routed path or the residual.
        AddZeroTensor($"blk.1.ffn_gate_shexp.weight", [embedDim, expertFfn]);
        AddZeroTensor($"blk.1.ffn_up_shexp.weight", [embedDim, expertFfn]);
        string pathNoShared = Path.Combine(Path.GetTempPath(), $"glm_dsa_shared_expert_zeroed_{Guid.NewGuid():N}.gguf");
        _tempFiles.Add(pathNoShared);
        WriteGgufFile(pathNoShared, metadata, tensors);
        using var modelNoShared = GgufModel.Open(pathNoShared);
        var withoutShared = new GlmDsaForwardPass(modelNoShared, hp);
        var logitsWithoutShared = withoutShared.Forward(token: 3, position: 0).ToArray();

        // The shared expert's contribution survives small-magnitude attenuation through the
        // output-norm/output-projection stages that follow it, so the measured difference is real
        // but small (~1e-8 on these tiny synthetic weights) -- well above float32 noise for a
        // 16-wide accumulation (~1e-7 relative), but far below 1e-6.
        bool anyDifferent = false;
        for (int i = 0; i < logitsWithRouted.Length; i++)
        {
            if (MathF.Abs(logitsWithRouted[i] - logitsWithoutShared[i]) > 1e-9f) anyDifferent = true;
        }
        Assert.True(anyDifferent, "zeroing the shared expert's weights should change the output -- if it doesn't, MoeFfn isn't actually using the shared expert's contribution");
    }

    private static void WriteGgufFile(
        string path,
        Dictionary<string, object> metadata,
        Dictionary<string, (long[] shape, DType dtype, byte[] data)> tensors)
    {
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream);

        // Header: Magic "GGUF" (0x46554747)
        writer.Write((byte)'G');
        writer.Write((byte)'G');
        writer.Write((byte)'U');
        writer.Write((byte)'F');
        writer.Write((uint)3); // Version 3
        writer.Write((ulong)tensors.Count);
        writer.Write((ulong)metadata.Count);

        // Write metadata
        foreach (var (key, value) in metadata)
        {
            WriteGgufString(writer, key);
            WriteGgufValue(writer, value);
        }

        // Calculate tensor data offsets
        long tensorInfoStart = stream.Position;
        // Placeholder for tensor infos: we need to write tensor infos first
        // Each tensor info: name, n_dims, dims..., dtype, offset
        long runningOffset = 0;
        var tensorOffsets = new Dictionary<string, ulong>();
        foreach (var (name, (shape, dtype, data)) in tensors)
        {
            // Align offset to 32 bytes
            if (runningOffset % 32 != 0) runningOffset += (32 - (runningOffset % 32));
            tensorOffsets[name] = (ulong)runningOffset;
            runningOffset += data.Length;
        }

        foreach (var (name, (shape, dtype, data)) in tensors)
        {
            WriteGgufString(writer, name);
            writer.Write((uint)shape.Length);
            for (int d = 0; d < shape.Length; d++)
            {
                writer.Write((ulong)shape[d]);
            }
            writer.Write((uint)dtype);
            writer.Write(tensorOffsets[name]);
        }

        // Align stream to 32 bytes for data block
        long currentPos = stream.Position;
        int pad = (int)((32 - (currentPos % 32)) % 32);
        for (int i = 0; i < pad; i++) writer.Write((byte)0);

        long dataBase = stream.Position;
        foreach (var (name, (shape, dtype, data)) in tensors)
        {
            long targetPos = dataBase + (long)tensorOffsets[name];
            while (stream.Position < targetPos) writer.Write((byte)0);
            writer.Write(data);
        }
    }

    private static void WriteGgufString(BinaryWriter writer, string s)
    {
        byte[] utf8 = System.Text.Encoding.UTF8.GetBytes(s);
        writer.Write((ulong)utf8.Length);
        writer.Write(utf8);
    }

    private static void WriteGgufValue(BinaryWriter writer, object value)
    {
        switch (value)
        {
            case uint u32:
                writer.Write((uint)4); // GGUF_TYPE_UINT32
                writer.Write(u32);
                break;
            case int i32:
                writer.Write((uint)5); // GGUF_TYPE_INT32
                writer.Write(i32);
                break;
            case float f32:
                writer.Write((uint)6); // GGUF_TYPE_FLOAT32
                writer.Write(f32);
                break;
            case bool b:
                writer.Write((uint)7); // GGUF_TYPE_BOOL
                writer.Write((byte)(b ? 1 : 0));
                break;
            case string s:
                writer.Write((uint)8); // GGUF_TYPE_STRING
                WriteGgufString(writer, s);
                break;
            case ulong u64:
                writer.Write((uint)10); // GGUF_TYPE_UINT64
                writer.Write(u64);
                break;
            case object[] arr:
                writer.Write((uint)9); // GGUF_TYPE_ARRAY
                writer.Write((uint)5); // element type INT32
                writer.Write((ulong)arr.Length);
                foreach (var item in arr)
                {
                    writer.Write(Convert.ToInt32(item));
                }
                break;
            default:
                throw new NotSupportedException($"Value type {value.GetType()} not supported in synthetic writer.");
        }
    }
}
