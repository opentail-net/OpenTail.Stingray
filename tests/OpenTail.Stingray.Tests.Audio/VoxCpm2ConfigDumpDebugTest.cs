namespace OpenTail.Stingray.Tests.Audio;

public sealed class VoxCpm2ConfigDumpDebugTest : HeavyTestBase
{
    [Fact]
    public void DumpConfigJson()
    {
        const string path = "F:/_models/voxcpm2/VoxCPM2-GGUF/voxcpm2-q8_0.gguf";
        Assert.SkipUnless(File.Exists(path), "not found");
        using var model = GgufModel.Open(path);
        var names = (object[])model.Metadata["audiocpp.embedded_files.names"];
        var offsets = (object[])model.Metadata["audiocpp.embedded_files.offsets"];
        var data = (object[])model.Metadata["audiocpp.embedded_files.data"];
        var bytes = data.Select(o => (byte)Convert.ToInt64(o)).ToArray();
        for (int i = 0; i < names.Length; i++)
        {
            if ((string)names[i] != "config.json") continue;
            long start = Convert.ToInt64(offsets[i]);
            long end = i + 1 < offsets.Length ? Convert.ToInt64(offsets[i + 1]) : bytes.Length;
            var json = System.Text.Encoding.UTF8.GetString(bytes[(int)start..(int)end]);
            Console.Error.WriteLine(json);
        }
    }
}
