using System.Security.Cryptography;
using OpenTail.Stingray.Engine.Verification;

namespace OpenTail.Stingray.Tests.Cli;

/// <summary>A golden recorded on a split model pins every shard, not just the first.</summary>
public sealed class GoldenSplitPinTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "splitpin-" + Guid.NewGuid().ToString("N"));
    public GoldenSplitPinTests() => Directory.CreateDirectory(_dir);
    public void Dispose() { try { Directory.Delete(_dir, true); } catch { } }

    private static string Sha(byte[] b) => Convert.ToHexString(SHA256.HashData(b)).ToLowerInvariant();

    private (GoldenFile Golden, string First) Setup(byte[] s1, byte[] s2)
    {
        string p1 = Path.Combine(_dir, "m-00001-of-00002.gguf"), p2 = Path.Combine(_dir, "m-00002-of-00002.gguf");
        File.WriteAllBytes(p1, s1); File.WriteAllBytes(p2, s2);
        var g = new GoldenFile
        {
            Model = new GoldenModel
            {
                FileName = "m-00001-of-00002.gguf", SizeBytes = s1.Length, Sha256 = Sha(s1),
                Files = [new("m-00001-of-00002.gguf", s1.Length, Sha(s1)), new("m-00002-of-00002.gguf", s2.Length, Sha(s2))],
            },
        };
        return (g, p1);
    }

    [Fact]
    public void All_shards_matching_is_verified()
    {
        var (g, first) = Setup([1, 2, 3], [4, 5, 6]);
        Assert.Equal(PinStatus.Verified, ModelFingerprinter.CheckPin(g, first).Status);
    }

    [Fact]
    public void A_changed_second_shard_is_a_mismatch_even_though_the_first_matches()
    {
        var (g, first) = Setup([1, 2, 3], [4, 5, 6]);
        File.WriteAllBytes(Path.Combine(_dir, "m-00002-of-00002.gguf"), [9, 9, 9]);
        Assert.Equal(PinStatus.Mismatch, ModelFingerprinter.CheckPin(g, first).Status);
    }

    [Fact]
    public void A_missing_second_shard_is_a_mismatch()
    {
        var (g, first) = Setup([1, 2, 3], [4, 5, 6]);
        File.Delete(Path.Combine(_dir, "m-00002-of-00002.gguf"));
        Assert.Equal(PinStatus.Mismatch, ModelFingerprinter.CheckPin(g, first).Status);
    }
}
