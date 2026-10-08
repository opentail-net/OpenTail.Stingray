using System.Security.Cryptography;
using OpenTail.Stingray.Engine.Verification;

namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

public sealed class ModelFingerprintTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "stingray-fp-" + Guid.NewGuid().ToString("N"));

    public ModelFingerprintTests() => Directory.CreateDirectory(_dir);

    public void Dispose() { try { Directory.Delete(_dir, recursive: true); } catch { /* best effort */ } }

    private string Write(string name, byte[] bytes) { string p = Path.Combine(_dir, name); File.WriteAllBytes(p, bytes); return p; }

    private static string Hex(byte[] b) => Convert.ToHexString(SHA256.HashData(b)).ToLowerInvariant();

    [Fact]
    public void Compute_HashesTheFile_ThenReusesTheSidecar_UntilSizeOrTimeChange()
    {
        byte[] bytes = new byte[5000]; new Random(3).NextBytes(bytes);
        string p = Write("m.gguf", bytes);

        var first = ModelFingerprinter.Compute(p);
        Assert.Equal(Hex(bytes), first.Sha256);
        Assert.Equal(5000, first.SizeBytes);
        Assert.False(first.FromCache);
        Assert.True(File.Exists(ModelFingerprinter.SidecarPath(p)));

        var second = ModelFingerprinter.Compute(p);
        Assert.True(second.FromCache);
        Assert.Equal(first.Sha256, second.Sha256);

        // Same size, different content and a new modification time: the cache must not be trusted.
        byte[] other = new byte[5000]; new Random(4).NextBytes(other);
        File.WriteAllBytes(p, other);
        File.SetLastWriteTimeUtc(p, DateTime.UtcNow.AddMinutes(5));
        var third = ModelFingerprinter.Compute(p);
        Assert.False(third.FromCache);
        Assert.Equal(Hex(other), third.Sha256);

        // A different size invalidates it too.
        File.WriteAllBytes(p, [.. other, 1]);
        Assert.Equal(Hex([.. other, 1]), ModelFingerprinter.Compute(p).Sha256);
    }

    [Fact]
    public void Compute_StillWorks_WhenTheSidecarCannotBeWritten()
    {
        byte[] bytes = [1, 2, 3, 4, 5];
        string p = Write("ro.gguf", bytes);
        Directory.CreateDirectory(ModelFingerprinter.SidecarPath(p));       // a directory where the sidecar file would go
        var fp = ModelFingerprinter.Compute(p);
        Assert.Equal(Hex(bytes), fp.Sha256);
        Assert.False(fp.FromCache);
    }

    [Fact]
    public void CorruptSidecar_IsIgnored_NotTrusted()
    {
        byte[] bytes = [9, 9, 9];
        string p = Write("c.gguf", bytes);
        File.WriteAllText(ModelFingerprinter.SidecarPath(p), "not a hash");
        Assert.Equal(Hex(bytes), ModelFingerprinter.Compute(p).Sha256);
    }

    [Fact]
    public void CheckPin_ReportsVerified_Mismatch_AndNotRecorded()
    {
        byte[] bytes = [7, 7, 7, 7];
        string p = Write("pin.gguf", bytes);
        GoldenFile G(string? sha) => new() { Architecture = "t", Model = new GoldenModel { FileName = "pin.gguf", Sha256 = sha } };

        var ok = ModelFingerprinter.CheckPin(G(Hex(bytes).ToUpperInvariant()), p);          // case-insensitive
        Assert.Equal(PinStatus.Verified, ok.Status);
        Assert.Contains("matches the verified file", ok.Describe("pin.gguf"));

        var bad = ModelFingerprinter.CheckPin(G(new string('a', 64)), p);
        Assert.Equal(PinStatus.Mismatch, bad.Status);
        Assert.Contains("NOT the file", bad.Describe("pin.gguf"));

        var none = ModelFingerprinter.CheckPin(G(null), p);
        Assert.Equal(PinStatus.NotRecorded, none.Status);
        Assert.Contains("recorded no hash", none.Describe("pin.gguf"));
    }
}
