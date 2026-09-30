using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Tests.Audio;

internal static class FunAsrGgufTestModelLocator
{
    private const string FileName = "paraformer-q8.gguf";

    public static string? FindNanoModelPath() => FindModelPath(
        "audiocpp",
        model => GetMetadataString(model, "general.name")?.StartsWith("Fun-ASR-Nano", StringComparison.OrdinalIgnoreCase) == true,
        "Fun-ASR-Nano* (audiocpp)");

    public static string? FindParaformerModelPath() => FindModelPath(
        "paraformer",
        model => model.Metadata.TryGetValue("pf.vocab", out var vocab)
            && vocab is object[] entries
            && entries.Length == 8404
            && entries.All(entry => entry is string),
        "Paraformer (paraformer with pf.vocab)");

    private static string? FindModelPath(string expectedArchitecture, Func<GgufModel, bool> hasExpectedMetadata, string identity)
    {
        var existingPaths = GetCandidatePaths()
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .Where(File.Exists)
            .ToArray();
        if (existingPaths.Length == 0)
            return null;

        var rejected = new List<string>();
        foreach (var path in existingPaths)
        {
            using var model = GgufModel.Open(path);
            string? architecture = GetMetadataString(model, "general.architecture");
            if (string.Equals(architecture, expectedArchitecture, StringComparison.OrdinalIgnoreCase)
                && hasExpectedMetadata(model))
            {
                string? modelName = GetMetadataString(model, "general.name");
                int vocabCount = model.Metadata.TryGetValue("pf.vocab", out var vocab) && vocab is object[] entries
                    ? entries.Length : 0;
                Console.WriteLine($"[FunAsrFixture] path={path} bytes={new FileInfo(path).Length} architecture={architecture} name={modelName ?? "<missing>"} pf.vocab={vocabCount} tensors={model.Tensors.Count}");
                return path;
            }

            string? name = GetMetadataString(model, "general.name");
            rejected.Add($"{path} (architecture={architecture ?? "<missing>"}, name={name ?? "<missing>"}, pf.vocab={model.Metadata.ContainsKey("pf.vocab")})");
        }

        throw new InvalidDataException($"Found {FileName} candidates, but none matched expected {identity}: {string.Join("; ", rejected)}");
    }

    private static IEnumerable<string> GetCandidatePaths()
    {
        var modelDirectories = new List<string>();
        var directory = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            modelDirectories.Add(Path.Combine(directory, "models"));
            var parent = Directory.GetParent(directory);
            if (parent is null)
                break;
            directory = parent.FullName;
        }

        modelDirectories.AddRange([@"C:\p\opentail-llm\models", @"E:\models"]);
        foreach (var modelDirectory in modelDirectories)
        {
            yield return Path.Combine(modelDirectory, FileName);
            yield return Path.Combine(modelDirectory, "_models", FileName);
        }
    }

    private static string? GetMetadataString(GgufModel model, string key) =>
        model.Metadata.TryGetValue(key, out var value) ? Convert.ToString(value) : null;
}
