
namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class FluxVaeRealWeightsTests
{
    private const string ModelFileName = "ae.safetensors";

    private static string? FindModelPath(string fileName)
    {
        string[] absoluteCandidates =
        {
            $@"C:\Git-Public\OpenTail.Stingray\models\{fileName}",
            $@"C:\p\opentail-llm\models\{fileName}",
            $@"E:\models\{fileName}",
        };
        foreach (var p in absoluteCandidates)
        {
            if (File.Exists(p)) return p;
        }

        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "models", fileName);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void FluxVae_RealModelFile_SafetensorsValidAndLoads()
    {
        string? modelPath = FindModelPath(ModelFileName);
        if (modelPath is null) return;

        using var st = SafetensorsLoader.Open(modelPath);
        Assert.NotNull(st);
        Assert.True(st.TensorCount > 0, "Flux VAE safetensors must contain tensors");
        Assert.True(st.Contains("decoder.conv_in.weight") || st.Contains("decoder.conv_out.weight") || st.Contains("first_stage_model.decoder.conv_in.weight"),
            "Flux VAE must contain decoder convolution weights");
    }
}
