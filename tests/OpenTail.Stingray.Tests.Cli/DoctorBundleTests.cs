using System.IO.Compression;
using OpenTail.Stingray.Cli;
using OpenTail.Stingray.Cli.CommandLine;

namespace OpenTail.Stingray.Tests.Cli;

/// <summary>
/// §10 privacy tests for the support bundle. A bundle is attached to public bug reports, so the
/// question these answer is not "does it work" but "can it leak".
/// </summary>
public sealed class DoctorBundleTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ot-bundle-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        try { if (Directory.Exists(_dir)) Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private string RunBundle()
    {
        string path = Path.Combine(_dir, "support.zip");
        var command = (ICommand)new DoctorCommand();
        var original = Console.Out;
        Console.SetOut(new StringWriter());
        try { command.Run(["--no-gpu-probe", "--bundle", path], CancellationToken.None); }
        finally { Console.SetOut(original); }
        return path;
    }

    private static Dictionary<string, string> ReadAll(string zipPath)
    {
        using var archive = ZipFile.OpenRead(zipPath);
        var result = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var entry in archive.Entries)
        {
            using var reader = new StreamReader(entry.Open());
            result[entry.FullName] = reader.ReadToEnd();
        }
        return result;
    }

    [Fact]
    public void BundleContainsExactlyTheAllowlistedEntries()
    {
        var entries = ReadAll(RunBundle());
        Assert.Equal(
            ["doctor.json", "manifest.txt", "settings.txt"],
            entries.Keys.OrderBy(k => k, StringComparer.Ordinal));
    }

    /// <summary>
    /// The load-bearing privacy guarantee: setting VALUES never appear. Names alone answer the
    /// diagnostic question ("what was configured"), while a value is what carries a secret or a
    /// local path.
    /// </summary>
    [Fact]
    public void SettingValuesAreNeverWritten()
    {
        const string name = "STINGRAY_HF_TOKEN";
        const string secret = "hf_thismustnotappear_9137";
        string? previous = Environment.GetEnvironmentVariable(name);
        Environment.SetEnvironmentVariable(name, secret);
        try
        {
            var entries = ReadAll(RunBundle());
            foreach ((string entryName, string content) in entries)
                Assert.False(content.Contains(secret, StringComparison.Ordinal),
                    $"{entryName} leaked a setting value into the support bundle.");

            // The NAME is expected — that is the diagnostic signal being preserved.
            Assert.Contains(name, entries["settings.txt"], StringComparison.Ordinal);
        }
        finally { Environment.SetEnvironmentVariable(name, previous); }
    }

    [Fact]
    public void ManifestDeclaresWhatIsExcluded()
    {
        var entries = ReadAll(RunBundle());
        Assert.Contains("Deliberately excluded", entries["manifest.txt"], StringComparison.Ordinal);
        Assert.Contains("credentials", entries["manifest.txt"], StringComparison.Ordinal);
    }
}
