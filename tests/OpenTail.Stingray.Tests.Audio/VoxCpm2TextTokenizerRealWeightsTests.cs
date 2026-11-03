using OpenTail.Stingray.Audio.VoxCpm2;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real-weight smoke test for <see cref="VoxCpm2TextTokenizer"/>.</summary>
public sealed class VoxCpm2TextTokenizerRealWeightsTests : HeavyTestBase
{
    private static string? FindRepoFile(string relPath) => OpenTail.Stingray.Tests.Audio.AudioTestPaths.FindRepoOrModel(relPath);


    [Fact]
    public void LoadFromPackedGguf_EncodesRealText_AndResolvesAudioTokenIds()
    {
        string? path = FindRepoFile("models/_models/voxcpm2/VoxCPM2-GGUF/voxcpm2-q8_0.gguf");
        Assert.SkipUnless(path != null, "voxcpm2-q8_0.gguf not found");

        using var model = OpenTail.Stingray.Core.GgufModel.Open(path!);
        var tokenizer = VoxCpm2TextTokenizer.LoadFromPackedGguf(model);

        Assert.True(tokenizer.AudioStartTokenId >= 0);
        Assert.True(tokenizer.AudioEndTokenId >= 0);
        Assert.True(tokenizer.ReferenceAudioStartTokenId >= 0);
        Assert.True(tokenizer.ReferenceAudioEndTokenId >= 0);
        Assert.NotEqual(tokenizer.AudioStartTokenId, tokenizer.AudioEndTokenId);

        var ids = tokenizer.Encode("Hello, this is a real VoxCPM2 tokenizer test.");
        Assert.NotEmpty(ids);

        var cjkIds = tokenizer.Encode("你好世界");
        Assert.NotEmpty(cjkIds);
    }
}
