using OpenTail.Stingray.Engine.Verification;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// One lookup for the audio tests' repo-relative paths. Each test used to carry its own copy of "walk up from the working directory looking for rel", which
/// only ever finds files under the repo's own <c>models</c> folder; 74 checkpoints that the audio and diffusion tests name sit in the <c>_models</c> folders of
/// other drives and were skipped as "not found" (CLAUDE.md rule 12). The repo walk-up still goes first, so nothing that used to be found changes; only when it
/// finds nothing, and only for a path under <c>models/</c>, the conventional model folders are searched through <see cref="ModelLocator"/>.
/// </summary>
internal static class AudioTestPaths
{
    public static string? FindRepoOrModel(string rel)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, rel);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return FindUnderModelFolders(rel);
    }

    /// <summary>
    /// A model file by name, or by a path relative to a model folder: the shared locator first (<c>STINGRAY_MODEL_DIRS</c>, the repo's models folders, the conventional folders
    /// on every drive, one folder down), then the repo-relative forms with and without the <c>models/</c> and <c>models/_models/</c> prefixes.
    /// </summary>
    public static string? FindModelFile(string name) =>
        ModelLocator.FindDeep(name)
        ?? FindRepoOrModel(name)
        ?? FindRepoOrModel("models/" + name.Replace('\\', '/'))
        ?? FindRepoOrModel("models/_models/" + name.Replace('\\', '/'));

    /// <summary>"models/x/y.bin" or "models/_models/y.bin": try <c>x/y.bin</c> (or <c>y.bin</c>) under every model root, then a flat file name one folder down.</summary>
    internal static string? FindUnderModelFolders(string rel)
    {
        string norm = rel.Replace('\\', '/');
        if (!norm.StartsWith("models/", StringComparison.OrdinalIgnoreCase)) return null;
        string tail = norm["models/".Length..];
        if (tail.StartsWith("_models/", StringComparison.OrdinalIgnoreCase)) tail = tail["_models/".Length..];
        if (tail.Length == 0) return null;
        foreach (var root in ModelLocator.Roots())
        {
            var candidate = Path.Combine(root, tail);
            if (File.Exists(candidate)) return candidate;
        }
        return tail.Contains('/') ? null : ModelLocator.FindDeep(tail);
    }
}
