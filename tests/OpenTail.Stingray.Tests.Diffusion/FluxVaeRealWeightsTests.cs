
namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class FluxVaeRealWeightsTests : HeavyTestBase
{
    private const string ModelFileName = "ae.safetensors";

    private static string? FindModelPath(string fileName) => OpenTail.Stingray.Tests.Audio.AudioTestPaths.FindModelFile(fileName);

    [Fact]
    public void FluxVae_RealModelFile_SafetensorsValidAndLoads()
    {
        string? modelPath = FindModelPath(ModelFileName);
        if (modelPath is null) Assert.Skip("modelPath not found (checkpoint or fixture missing).");

        using var st = SafetensorsLoader.Open(modelPath);
        Assert.NotNull(st);
        Assert.True(st.TensorCount > 0, "Flux VAE safetensors must contain tensors");
        Assert.True(st.Contains("decoder.conv_in.weight") || st.Contains("decoder.conv_out.weight") || st.Contains("first_stage_model.decoder.conv_in.weight"),
            "Flux VAE must contain decoder convolution weights");
    }
}
