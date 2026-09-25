using System.Diagnostics;

namespace OpenTail.Stingray.Tests.Embeddings;

/// <summary>
/// google/electra-base-discriminator has no ONNX export and no safetensors on its main branch, so:
/// (1) every tensor of the safetensors-convert bot's <c>model.safetensors</c> (refs/pr/6, SHA-256 matching the hub's
/// linked ETag) is compared value-for-value with the repo's own <c>flax_model.msgpack</c> (Flax kernels are [in, out],
/// so Linear weights are compared transposed); the encoder math itself is the ONNX-verified BERT path;
/// (2) the model card's example "The quick brown fox fake over the lazy dog" must flag exactly the substituted
/// word "fake" as replaced (logit &gt; 0).
/// </summary>
public sealed class ElectraDiscriminatorTests
{
    private static string? Dir() =>
        BertEncoderOnnxParityTests.FindRepoDir("models/_models/hf/google__electra-base-discriminator") is { } d
        && File.Exists(Path.Combine(d, "model.safetensors")) && File.Exists(Path.Combine(d, "flax_model.msgpack")) ? d : null;

    [Fact]
    public void ConvertedSafetensors_MatchRepoFlaxWeights()
    {
        string? dir = Dir();
        Assert.SkipUnless(dir != null, "electra-base-discriminator model.safetensors/flax_model.msgpack not found");
        var sw = Stopwatch.StartNew();
        var flax = FlaxMsgpackReader.ReadParams(Path.Combine(dir!, "flax_model.msgpack"));
        using var st = SafetensorsLoader.OpenDirectory(dir!);
        int compared = 0;
        var missing = new List<string>();
        foreach (var name in st.TensorNames.OrderBy(n => n, StringComparer.Ordinal))
        {
            if (name.Contains("embeddings_project", StringComparison.Ordinal)) continue; // unused when embedding_size == hidden_size
            var parts = name.Split('.');
            string leaf = parts[^1];
            string owner = parts[^2];
            bool isEmbedding = owner.EndsWith("embeddings", StringComparison.Ordinal);
            string flaxLeaf = leaf == "bias" ? "bias" : isEmbedding ? "embedding" : owner == "LayerNorm" ? "scale" : "kernel";
            string path = string.Join("/", parts[..^1]) + "/" + flaxLeaf;
            if (!flax.TryGetValue(path, out var f)) { missing.Add(name + " -> " + path); continue; }
            var ours = st.ReadF32(name);
            var shape = st.GetShape(name);
            Assert.Equal(ours.Length, f.Data.Length);
            if (flaxLeaf == "kernel" && shape.Length == 2)
            {
                int rows = shape[0], cols = shape[1]; // torch [out, in]; flax [in, out]
                for (int r = 0; r < rows; r++)
                    for (int c = 0; c < cols; c++)
                        Assert.Equal(f.Data[c * rows + r], ours[r * cols + c]);
            }
            else
            {
                Assert.Equal(f.Data, ours);
            }
            compared++;
        }
        Console.WriteLine($"[Electra] {compared} tensors identical to flax_model.msgpack in {sw.ElapsedMilliseconds} ms; unmatched: {string.Join(", ", missing)}");
        Assert.Empty(missing);
        Assert.True(compared >= 199, $"only {compared} tensors compared");
    }

    [Fact]
    public void ModelCardExample_FlagsTheReplacedWord()
    {
        string? dir = Dir();
        Assert.SkipUnless(dir != null, "electra-base-discriminator model.safetensors not found");
        var sw = Stopwatch.StartNew();
        using var disc = ElectraDiscriminator.Load(dir!);
        var input = disc.Tokenizer.Encode("The quick brown fox fake over the lazy dog");
        var logits = disc.ReplacedTokenLogits(input);
        Console.WriteLine($"[Electra] ids {string.Join(" ", input.Ids)}");
        Console.WriteLine($"[Electra] logits {string.Join(" ", logits.Select(l => l.ToString("F2", System.Globalization.CultureInfo.InvariantCulture)))} ({sw.ElapsedMilliseconds} ms)");
        int fakeId = 8275; // "fake" in the uncased BERT vocab
        for (int t = 1; t < input.Ids.Length - 1; t++)
            Assert.True(input.Ids[t] == fakeId ? logits[t] > 0 : logits[t] < 0, $"token {input.Ids[t]} at {t}: logit {logits[t]}");
    }
}
