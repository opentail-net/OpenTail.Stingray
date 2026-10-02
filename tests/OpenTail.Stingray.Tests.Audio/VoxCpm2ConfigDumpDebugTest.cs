namespace OpenTail.Stingray.Tests.Audio;

public sealed class VoxCpm2ConfigDumpDebugTest : HeavyTestBase
{
    [Fact]
    public void DumpConfigJson()
    {
        const string path = "F:/_models/voxcpm2/VoxCPM2-GGUF/voxcpm2-q8_0.gguf";
        Assert.SkipUnless(File.Exists(path), "not found");
        using var model = GgufModel.Open(path);
        if (OpenTail.Stingray.Audio.AudioCppEmbeddedFiles.TryGet(model, "config.json", out var bytes))
        {
            var json = System.Text.Encoding.UTF8.GetString(bytes);
            Console.Error.WriteLine(json);
        }
    }
}
