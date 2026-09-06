
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real load + blend validation for RvcRetrievalIndex against the real bundled
/// voice_v2_default_index_vectors in rvc-f16.gguf.</summary>
public sealed class RvcRetrievalIndexTests : HeavyTestBase
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
    public void Load_RealCheckpoint_AndBlend_ProducesFiniteBlendedFeatures()
    {
        string? path = FindRepoFile("examples/audio.cpp/models/RVC-GGUF/rvc-f16.gguf");
        Assert.SkipUnless(path != null, "rvc-f16.gguf not found");

        using var model = OpenTail.Stingray.Core.GgufModel.Open(path!);
        var source = new OpenTail.Stingray.Audio.Rvc.RvcPackedTensorSource(model);

        const int dim = 768; // v2 HuBERT content width
        var index = new OpenTail.Stingray.Audio.Rvc.RvcRetrievalIndex(source, "voice_v2_default_index_vectors", dim);

        Assert.True(index.NList > 0);
        Assert.True(index.Vectors.Length > 0);
        Assert.Equal(index.NList, index.ListOffsets.Length);
        Assert.Equal(index.NList, index.ListLengths.Length);
        foreach (var v in index.Centroids) Assert.True(float.IsFinite(v));
        foreach (var v in index.Vectors) Assert.True(float.IsFinite(v));

        // Real synthetic content features to blend (structural check only -- not golden-verified
        // against the reference's own retrieval_blend trace, a follow-on step).
        var rng = new Random(7);
        const int frames = 20;
        var features = new float[frames * dim];
        for (int i = 0; i < features.Length; i++) features[i] = (float)(rng.NextDouble() * 0.4 - 0.2);
        var original = (float[])features.Clone();

        index.ApplyBlend(features, original, frames, dim, retrievalBlend: 0.75f);

        foreach (var v in features) Assert.True(float.IsFinite(v));
        bool changed = false;
        for (int i = 0; i < features.Length; i++)
            if (MathF.Abs(features[i] - original[i]) > 1e-6f) { changed = true; break; }
        Assert.True(changed, "retrieval blend did not change any feature values");
    }
}
