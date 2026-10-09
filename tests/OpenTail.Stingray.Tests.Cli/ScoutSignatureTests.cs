using System.Text.Json;
using OpenTail.Stingray.Cli.Scout;

namespace OpenTail.Stingray.Tests.Cli;

/// <summary>Structural-signature ranking: variants of an admitted family must land on it, regardless of size or quantization.</summary>
public sealed class ScoutSignatureTests
{
    private static GgufTensorInfo T(string name, DType dt, params long[] dims) => new(name, dims.Length, dims, dt, 0);

    /// <summary>A llama-shaped tensor set. <paramref name="width"/> and <paramref name="layers"/> change size, <paramref name="dt"/> the quantization.</summary>
    private static List<GgufTensorInfo> LlamaTensors(int layers, long width, DType dt, bool qkNorm = false)
    {
        var l = new List<GgufTensorInfo> { T("token_embd.weight", dt, width, 1000), T("output_norm.weight", DType.Float32, width), T("output.weight", dt, width, 1000) };
        for (int i = 0; i < layers; i++)
        {
            l.Add(T($"blk.{i}.attn_norm.weight", DType.Float32, width));
            l.Add(T($"blk.{i}.attn_q.weight", dt, width, width));
            l.Add(T($"blk.{i}.attn_k.weight", dt, width, width / 4));
            l.Add(T($"blk.{i}.attn_v.weight", dt, width, width / 4));
            l.Add(T($"blk.{i}.attn_output.weight", dt, width, width));
            if (qkNorm) { l.Add(T($"blk.{i}.attn_q_norm.weight", DType.Float32, 64)); l.Add(T($"blk.{i}.attn_k_norm.weight", DType.Float32, 64)); }
            l.Add(T($"blk.{i}.ffn_norm.weight", DType.Float32, width));
            l.Add(T($"blk.{i}.ffn_gate.weight", dt, width, width * 3));
            l.Add(T($"blk.{i}.ffn_up.weight", dt, width, width * 3));
            l.Add(T($"blk.{i}.ffn_down.weight", dt, width * 3, width));
        }
        return l;
    }

    private static ScoutInput Input(string arch, IEnumerable<GgufTensorInfo> tensors, string file = "m.gguf", Action<Dictionary<string, object>>? tweak = null)
    {
        var md = new Dictionary<string, object>
        {
            ["general.architecture"] = arch,
            [$"{arch}.attention.head_count"] = 32u,
            [$"{arch}.attention.head_count_kv"] = 8u,
            ["tokenizer.ggml.model"] = "gpt2",
            ["tokenizer.ggml.merges"] = new string[] { "a b" },
        };
        tweak?.Invoke(md);
        var list = tensors.ToArray();
        return new ScoutInput(file, 1, 3, (ulong)list.Length, (ulong)md.Count, md, list);
    }

    private static readonly ScoutOptions Plain = new("t");

    private static SigOrigin OriginOf(ScoutInput input, string? sha = null) => new(input.FileName, input.FileBytes, sha, null, null, "t");

    private static ArchSignature SnapshotOf(ScoutInput input, string? sha = null)
    {
        var sig = SignatureBuilder.TryBuild(input, ScoutAnalyzer.Analyze(input, Plain), OriginOf(input, sha), out string why);
        Assert.True(sig is not null, why);
        return sig!;
    }

    private static ScoutReport Run(ScoutInput input, params ArchSignature[] sigs) =>
        ScoutAnalyzer.Analyze(input, new ScoutOptions("t", Signatures: sigs));

    // ── what the signature is, and is not, sensitive to ──

    [Fact]
    public void Size_and_quantization_do_not_change_the_signature()
    {
        var small = SnapshotOf(Input("llama", LlamaTensors(2, 64, DType.Q4_K), "tiny-q4.gguf"));
        var big = Input("llama", LlamaTensors(40, 4096, DType.Float16), "big-f16.gguf");
        var r = Run(big, small);
        var c = Assert.Single(r.Architecture!.Candidates);
        Assert.Equal("identical_structure", c.Kind);
        Assert.Equal(0, c.DifferenceCount);
        Assert.Equal(["tiny-q4.gguf"], c.ReferenceFiles);
    }

    [Fact]
    public void Head_count_differences_between_sizes_do_not_matter()
    {
        var mha = SnapshotOf(Input("llama", LlamaTensors(2, 64, DType.Q4_K), tweak: md => md["llama.attention.head_count_kv"] = 32u));
        var gqa = Input("llama", LlamaTensors(2, 64, DType.Q4_K), tweak: md => md["llama.attention.head_count_kv"] = 4u);
        Assert.Equal(0, Run(gqa, mha).Architecture!.Candidates[0].DifferenceCount);
    }

    [Fact]
    public void A_finetune_with_a_new_tensor_family_is_reported_with_exact_differences()
    {
        var parent = SnapshotOf(Input("llama", LlamaTensors(2, 64, DType.Q4_K)));
        var variant = Input("llama", LlamaTensors(2, 64, DType.Q4_K, qkNorm: true));
        var c = Run(variant, parent).Architecture!.Candidates[0];
        Assert.Equal("differs", c.Kind);
        Assert.Equal(3, c.DifferenceCount);
        Assert.Contains("extra tensor pattern blk.*.attn_q_norm.weight", c.Differing);
        Assert.Contains("extra tensor pattern blk.*.attn_k_norm.weight", c.Differing);
        Assert.Contains("feature not in reference: attn.qk_norm", c.Differing);
    }

    [Fact]
    public void A_layer_that_loses_a_tensor_changes_coverage_not_just_presence()
    {
        var parent = SnapshotOf(Input("llama", LlamaTensors(4, 64, DType.Q4_K)));
        var hybridish = LlamaTensors(4, 64, DType.Q4_K).Where(t => t.Name != "blk.2.attn_q.weight").ToList();
        var c = Run(Input("llama", hybridish), parent).Architecture!.Candidates[0];
        Assert.Contains("blk.*.attn_q.weight: layer coverage subset, reference all", c.Differing);
    }

    [Fact]
    public void Tensor_rank_difference_is_reported()
    {
        var parent = SnapshotOf(Input("llama", LlamaTensors(2, 64, DType.Q4_K)));
        var changed = LlamaTensors(2, 64, DType.Q4_K).Select(t => t.Name == "token_embd.weight" ? T(t.Name, t.DType, 64, 10, 10) : t);
        Assert.Contains("token_embd.weight: rank 3, reference 2", Run(Input("llama", changed), parent).Architecture!.Candidates[0].Differing);
    }

    // ── the community-variant cases ──

    [Fact]
    public void Same_structure_under_an_unregistered_name_is_a_relabel_candidate_not_an_admission()
    {
        var parent = SnapshotOf(Input("llama", LlamaTensors(2, 64, DType.Q4_K), "base.gguf"));
        var r = Run(Input("mycoolllm", LlamaTensors(24, 2048, DType.Q8_0), "mycool-q8.gguf"), parent);

        Assert.False(r.Architecture!.Resolved);
        var f = Assert.Single(r.Findings, x => x.Id == "arch.relabel_candidate");
        Assert.Equal(Certainty.Hypothesis, f.Certainty);
        Assert.Contains("not a proof", f.Caveat.Replace("never proof", "not a proof"));
        Assert.Contains(r.Blockers, b => b.Id == "scout.arch.not_registered" && b.Kind == BlockerKind.Confirmed); // still refused
        var next = r.NextActions[0];
        Assert.StartsWith("capture-golden", next.Command, StringComparison.Ordinal);
        Assert.Contains("llama", next.Why);
    }

    [Fact]
    public void Unregistered_with_real_differences_points_at_the_parent_and_its_differences()
    {
        var parent = SnapshotOf(Input("llama", LlamaTensors(2, 64, DType.Q4_K)));
        var r = Run(Input("newarch", LlamaTensors(2, 64, DType.Q4_K, qkNorm: true)), parent);
        Assert.Contains(r.Findings, f => f.Id == "arch.nearest_parent");
        Assert.DoesNotContain(r.Findings, f => f.Id == "arch.relabel_candidate");
        Assert.Contains("3 difference(s)", r.NextActions[0].Why);
    }

    [Fact]
    public void A_registered_admitted_file_matching_its_own_family_gets_the_Known_structural_parent_finding()
    {
        var parent = SnapshotOf(Input("llama", LlamaTensors(2, 64, DType.Q4_K)));
        var r = Run(Input("llama", LlamaTensors(8, 256, DType.Q6_K)), parent);
        Assert.Equal(Certainty.Known, r.Findings.Single(f => f.Id == "arch.structural_parent").Certainty);
    }

    // ── ranking mechanics ──

    [Fact]
    public void One_candidate_per_architecture_closest_snapshot_wins_and_order_is_deterministic()
    {
        var plain = SnapshotOf(Input("llama", LlamaTensors(2, 64, DType.Q4_K), "plain.gguf"));
        var normed = SnapshotOf(Input("llama", LlamaTensors(2, 64, DType.Q4_K, qkNorm: true), "normed.gguf"));
        var file = Input("llama", LlamaTensors(3, 64, DType.Q4_K, qkNorm: true));

        var a = Run(file, plain, normed).Architecture!.Candidates;
        var b = Run(file, normed, plain).Architecture!.Candidates; // input order must not matter
        var only = Assert.Single(a);
        Assert.Equal(["normed.gguf"], only.ReferenceFiles);
        Assert.Equal(0, only.DifferenceCount);
        Assert.Equal(only.ReferenceFiles, b[0].ReferenceFiles);
    }

    [Fact]
    public void Candidates_are_ordered_by_fewest_differences_and_limited_to_three()
    {
        var llama = SnapshotOf(Input("llama", LlamaTensors(2, 64, DType.Q4_K)));
        var qwen3 = SnapshotOf(Input("qwen3", LlamaTensors(2, 64, DType.Q4_K, qkNorm: true)));
        var jais = SnapshotOf(Input("jais", [T("token_embd.weight", DType.Q4_K, 64, 10), T("blk.0.attn_qkv.weight", DType.Q4_K, 64, 192)]));
        var phi = SnapshotOf(Input("phimoe", [T("token_embd.weight", DType.Q4_K, 64, 10), T("blk.0.ffn_gate_inp.weight", DType.Q4_K, 64, 8)]));
        var r = Run(Input("zzz", LlamaTensors(2, 64, DType.Q4_K)), jais, phi, qwen3, llama);
        var c = r.Architecture!.Candidates;
        Assert.Equal(3, c.Count);
        Assert.Equal("llama", c[0].Id);
        Assert.True(c[0].DifferenceCount <= c[1].DifferenceCount && c[1].DifferenceCount <= c[2].DifferenceCount);
    }

    [Fact]
    public void A_snapshot_whose_architecture_left_the_registry_is_dropped()
    {
        var ghost = SnapshotOf(Input("llama", LlamaTensors(2, 64, DType.Q4_K))) with { ArchitectureId = "no_such_arch_anywhere" };
        var r = Run(Input("llama", LlamaTensors(2, 64, DType.Q4_K)), ghost);
        Assert.Equal("computed", r.Architecture!.CandidatesState);
        Assert.Empty(r.Architecture.Candidates);
    }

    [Fact]
    public void Without_signatures_the_ranking_is_not_computed_rather_than_empty()
    {
        var r = ScoutAnalyzer.Analyze(Input("llama", LlamaTensors(2, 64, DType.Q4_K)), Plain);
        Assert.Equal("not_computed", r.Architecture!.CandidatesState);
        Assert.Empty(r.Architecture.Candidates);
    }

    // ── what may become a reference ──

    [Fact]
    public void Only_admitted_architectures_can_be_emitted_as_references()
    {
        var unregistered = Input("mycoolllm", LlamaTensors(2, 64, DType.Q4_K));
        Assert.Null(SignatureBuilder.TryBuild(unregistered, ScoutAnalyzer.Analyze(unregistered, Plain), OriginOf(unregistered), out string why));
        Assert.Contains("registered", why);

        var notAdmitted = ArchitectureRegistry.All.FirstOrDefault(d => d.Status != AdmissionStatus.Admitted);
        if (notAdmitted is null) return; // nothing is currently held back
        var input = Input(notAdmitted.Id, LlamaTensors(2, 64, DType.Q4_K));
        Assert.Null(SignatureBuilder.TryBuild(input, ScoutAnalyzer.Analyze(input, Plain), OriginOf(input), out string why2));
        Assert.Contains("not Admitted", why2);
    }

    [Fact]
    public void Signature_json_roundtrips_and_carries_no_paths_dtypes_or_sizes()
    {
        var sig = SnapshotOf(Input("llama", LlamaTensors(2, 64, DType.Q4_K), "my-model.gguf"));
        string json = JsonSerializer.Serialize(sig, ScoutJsonContext.Default.ArchSignature);
        Assert.Equal(sig.Structure.Patterns.Count, JsonSerializer.Deserialize(json, ScoutJsonContext.Default.ArchSignature)!.Structure.Patterns.Count);
        Assert.DoesNotContain(":\\", json);
        Assert.DoesNotContain("Q4_K", json);
        Assert.DoesNotContain("1000", json); // no absolute dimensions
    }

    [Fact]
    public void Structural_feature_filter_excludes_size_dependent_ids()
    {
        foreach (var id in new[] { "attn.gqa", "attn.mha", "attn.mqa", "rope.partial", "rope.scaling", "quant.low_bit_or_block_fp", "tokenizer.spm_scores_only" })
            Assert.False(SignatureBuilder.IsStructuralFeature(id), id);
        foreach (var id in new[] { "ffn.expert_routed", "attn.qk_norm", "recurrent.hybrid_ssm", "mtp.nextn_head", "rope.multi_axis", "attn.mla_low_rank" })
            Assert.True(SignatureBuilder.IsStructuralFeature(id), id);
    }

    // ── the shipped data ──

    [Fact]
    public void Embedded_signatures_are_valid_and_belong_to_admitted_architectures()
    {
        var sigs = SignatureStore.LoadEmbedded();
        Assert.NotEmpty(sigs);
        foreach (var s in sigs)
        {
            string who = s.Origins.FirstOrDefault()?.FileName ?? s.StructureId;
            Assert.Equal(ArchSignature.CurrentSchemaVersion, s.SchemaVersion);
            var d = ArchitectureRegistry.Find(s.ArchitectureId);
            Assert.True(d is not null, $"{who}: architecture '{s.ArchitectureId}' is no longer registered; regenerate or remove the signature");
            Assert.True(d!.Status == AdmissionStatus.Admitted, $"{who}: '{s.ArchitectureId}' is {d.Status}; a reference signature must be of an Admitted architecture");
            Assert.NotEmpty(s.Structure.Patterns);
            Assert.Equal(SignatureBuilder.StructureId(s.Structure), s.StructureId); // edited by hand or built by an older scout
            Assert.NotEmpty(s.Origins);
            foreach (var o in s.Origins)
            {
                Assert.True(o.Sha256 is { Length: 64 }, $"{who}: shipped evidence must be pinned to a SHA-256 ({o.FileName})");
                Assert.DoesNotContain("\\", o.FileName);
                Assert.DoesNotContain("/", o.FileName);
                Assert.False(string.IsNullOrWhiteSpace(o.ScoutBuild));
            }
        }
    }

    [Fact]
    public void Directory_loading_reports_unusable_files_instead_of_skipping_them_silently()
    {
        string dir = Path.Combine(Path.GetTempPath(), "scout-sig-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var good = SnapshotOf(Input("llama", LlamaTensors(2, 64, DType.Q4_K)));
            File.WriteAllText(Path.Combine(dir, "good.signature.json"), JsonSerializer.Serialize(good, ScoutJsonContext.Default.ArchSignature));
            File.WriteAllText(Path.Combine(dir, "bad.signature.json"), "{ not json");
            File.WriteAllText(Path.Combine(dir, "future.signature.json"), JsonSerializer.Serialize(good with { SchemaVersion = 99 }, ScoutJsonContext.Default.ArchSignature));
            File.WriteAllText(Path.Combine(dir, "ignored.txt"), "x");

            var problems = new List<string>();
            var loaded = SignatureStore.LoadDirectory(dir, problems);
            Assert.Single(loaded);
            Assert.Equal(2, problems.Count);
            Assert.Contains(problems, p => p.StartsWith("bad.signature.json", StringComparison.Ordinal));
            Assert.Contains(problems, p => p.StartsWith("future.signature.json", StringComparison.Ordinal));
        }
        finally { Directory.Delete(dir, recursive: true); }
    }
    // ── structure vs origin ──

    [Fact]
    public void Structure_id_ignores_size_quantization_and_origin_but_changes_with_structure()
    {
        var a = SnapshotOf(Input("llama", LlamaTensors(2, 64, DType.Q4_K), "a.gguf"), sha: new string('a', 64));
        var b = SnapshotOf(Input("llama", LlamaTensors(40, 4096, DType.Float16), "b.gguf"), sha: new string('b', 64));
        var c = SnapshotOf(Input("llama", LlamaTensors(2, 64, DType.Q4_K, qkNorm: true), "c.gguf"));
        Assert.Equal(a.StructureId, b.StructureId);
        Assert.NotEqual(a.StructureId, c.StructureId);
        Assert.Equal(64, a.StructureId.Length);
        Assert.NotEqual(a.Origins[0], b.Origins[0]);
    }

    [Fact]
    public void Origin_records_what_is_known_and_leaves_the_rest_null()
    {
        var input = Input("llama", LlamaTensors(2, 64, DType.Q4_K), "x.gguf");
        var s = SignatureBuilder.TryBuild(input, ScoutAnalyzer.Analyze(input, Plain),
            new SigOrigin("x.gguf", 1, null, "owner/repo", null, "build-1"), out _)!;
        var o = Assert.Single(s.Origins);
        Assert.Equal("owner/repo", o.SourceRepo);
        Assert.Null(o.Sha256);
        Assert.Null(o.SourceRevision);
        string json = JsonSerializer.Serialize(s, ScoutJsonContext.Default.ArchSignature);
        using var doc = JsonDocument.Parse(json);
        Assert.Equal(JsonValueKind.Null, doc.RootElement.GetProperty("origins")[0].GetProperty("sha256").ValueKind);
        Assert.False(doc.RootElement.GetProperty("structure").TryGetProperty("origins", out _)); // origin never leaks into the structure
    }

    [Fact]
    public void Merging_adds_origins_for_the_same_structure_and_refuses_a_different_one()
    {
        var a = SnapshotOf(Input("llama", LlamaTensors(2, 64, DType.Q4_K), "small.gguf"), sha: new string('1', 64));
        var b = SnapshotOf(Input("llama", LlamaTensors(30, 2048, DType.Q8_0), "large.gguf"), sha: new string('2', 64));
        var merged = SignatureBuilder.TryMerge(a, b, out _)!;
        Assert.Equal(["large.gguf", "small.gguf"], merged.Origins.Select(o => o.FileName));
        Assert.Equal(a.StructureId, merged.StructureId);

        var again = SignatureBuilder.TryMerge(merged, b, out _)!;
        Assert.Equal(2, again.Origins.Count); // same file and hash is not added twice

        var different = SnapshotOf(Input("llama", LlamaTensors(2, 64, DType.Q4_K, qkNorm: true), "normed.gguf"));
        Assert.Null(SignatureBuilder.TryMerge(a, different, out string why));
        Assert.Contains("structure_id", why);
    }

    [Fact]
    public void Origins_of_equal_structures_are_pooled_into_one_candidate_and_never_change_the_ranking()
    {
        var a = SnapshotOf(Input("llama", LlamaTensors(2, 64, DType.Q4_K), "small.gguf"), sha: new string('1', 64));
        var b = SnapshotOf(Input("llama", LlamaTensors(30, 2048, DType.Q8_0), "large.gguf"), sha: new string('2', 64));
        var file = Input("llama", LlamaTensors(8, 256, DType.Q6_K));

        var one = Run(file, a).Architecture!.Candidates[0];
        var two = Run(file, a, b).Architecture!.Candidates;
        Assert.Single(two);
        Assert.Equal(one.DifferenceCount, two[0].DifferenceCount);
        Assert.Equal(one.StructureId, two[0].StructureId);
        Assert.Equal(2, two[0].ReferenceFiles.Count);
        Assert.Contains("large.gguf (sha256 222222222222...)", two[0].ReferenceFiles);
    }

    [Fact]
    public void A_hand_edited_structure_is_rejected_on_load()
    {
        string dir = Path.Combine(Path.GetTempPath(), "scout-sig-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var good = SnapshotOf(Input("llama", LlamaTensors(2, 64, DType.Q4_K)));
            var edited = good with { Structure = good.Structure with { Features = [.. good.Structure.Features, "ffn.expert_routed"] } };
            File.WriteAllText(Path.Combine(dir, "edited.signature.json"), JsonSerializer.Serialize(edited, ScoutJsonContext.Default.ArchSignature));
            var problems = new List<string>();
            Assert.Empty(SignatureStore.LoadDirectory(dir, problems));
            Assert.Contains("structure_id does not match", Assert.Single(problems));
        }
        finally { Directory.Delete(dir, recursive: true); }
    }
}