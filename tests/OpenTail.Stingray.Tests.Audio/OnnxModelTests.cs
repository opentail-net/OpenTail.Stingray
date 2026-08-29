
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Validates the hand-rolled ONNX protobuf reader (OnnxModel.cs) against the real Piper
/// en_US-lessac-medium.onnx file, cross-checked against `python -c "import onnx; ..."`'s output
/// captured while scoping the Piper rebuild: 401 initializers, inputs
/// [input, input_lengths, scales], output [output], and specific known tensor names/shapes
/// (e.g. enc_p.encoder.attn_layers.0.emb_rel_k).
/// </summary>
public sealed class OnnxModelTests : HeavyTestBase
{
    private static string? FindRepoFile(string relativePath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativePath);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void OnnxModel_RealPiperFile_ParsesExpectedGraphShape()
    {
        string? path = FindRepoFile("models/en_US-lessac-medium.onnx");
        if (path is null) return;

        var model = OnnxModel.Open(path);

        Assert.Equal(401, model.Initializers.Count);
        Assert.Equal(new[] { "input", "input_lengths", "scales" }, model.GraphInputs);
        Assert.Equal(new[] { "output" }, model.GraphOutputs);

        // Known tensor: enc_p.encoder.attn_layers.0.emb_rel_k -- VITS relative-position
        // attention embedding, per the ONNX initializer name sweep done while scoping this.
        var relK = model.GetTensor("enc_p.encoder.attn_layers.0.emb_rel_k");
        Assert.Equal(1, relK.DataType); // FLOAT
        Assert.True(relK.Dims.Length >= 2, "emb_rel_k should be at least 2D");
        Assert.Equal(relK.RawData.Length, (int)(relK.ElementCount * 4));

        var conv1x1 = model.TryGetTensor("dec.conv_pre.weight");
        Assert.NotNull(conv1x1);

        var floats = OnnxModel.ToFloat32(relK);
        Assert.Equal(relK.ElementCount, floats.Length);
        foreach (float f in floats)
        {
            Assert.False(float.IsNaN(f));
            Assert.False(float.IsInfinity(f));
        }
    }
}
