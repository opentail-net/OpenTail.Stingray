namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Plan acceptance (docs/3-product-and-runtime/2026-10-12-opentelemetry-observability-plan.md): the libraries emit telemetry through
/// <c>System.Diagnostics.Metrics</c> and <c>ActivitySource</c> only, so a consumer who never wires a listener pays for no OpenTelemetry package.
/// Export lives in the server host. A package reference to OpenTelemetry anywhere else under <c>src/</c> breaks that.
/// </summary>
public sealed class LibraryTelemetryDependencyTests
{
    private static string FindRepoRoot()
    {
        for (var d = new DirectoryInfo(AppContext.BaseDirectory); d is not null; d = d.Parent)
            if (File.Exists(Path.Combine(d.FullName, "OpenTail.Stingray.slnx"))) return d.FullName;
        throw new InvalidOperationException("repository root (OpenTail.Stingray.slnx) not found above " + AppContext.BaseDirectory);
    }

    // A real reference, not a comment that mentions the word.
    private static readonly System.Text.RegularExpressions.Regex s_otelReference =
        new(@"<(PackageReference|ProjectReference)\b[^>]*OpenTelemetry", System.Text.RegularExpressions.RegexOptions.IgnoreCase | System.Text.RegularExpressions.RegexOptions.CultureInvariant);

    [Fact]
    public void OnlyTheServerHost_ReferencesAnOpenTelemetryPackage()
    {
        string src = Path.Combine(FindRepoRoot(), "src");
        var projects = Directory.GetFiles(src, "*.csproj", SearchOption.AllDirectories)
            .Where(p => !p.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}", StringComparison.Ordinal)).ToArray();
        Assert.True(projects.Length >= 10, $"expected the solution's source projects, found {projects.Length}");

        var offenders = projects
            .Where(p => !Path.GetFileName(p).Equals("OpenTail.Stingray.Server.Host.csproj", StringComparison.Ordinal))
            .Where(p => s_otelReference.IsMatch(File.ReadAllText(p)))
            .Select(p => Path.GetRelativePath(src, p))
            .ToArray();

        Assert.True(offenders.Length == 0, "These library projects reference OpenTelemetry: " + string.Join(", ", offenders));
        // Positive control: the host does, so the scan can see such a reference when there is one.
        string host = projects.Single(p => Path.GetFileName(p).Equals("OpenTail.Stingray.Server.Host.csproj", StringComparison.Ordinal));
        Assert.Matches(s_otelReference, File.ReadAllText(host));
    }
}
