namespace OpenTail.Stingray.Tests.Vision;

/// <summary>Locates the Gemma 4 12B mmproj + text model + golden fixtures across dev machines.</summary>
internal static class VisionTestPaths
{
    public const string MmprojFile = "mmproj-gemma-4-12b-it-qat-q4_0.gguf";
    public const string TextModelFile = "gemma-4-12b-it-qat-q4_0.gguf";
    public const string E4BMmprojFile = "gemma-4-E4B-it-mmproj.gguf";
    public const string Gemma3MmprojFile = "mmproj-gemma-3-4b-it-f16.gguf";
    public const string Gemma3TextModelFile = "gemma-3-4b-it-Q4_K_M.gguf";
    public const string Llama4MmprojFile = "mmproj-llama-4-scout-17b-16e-instruct-f16.gguf";
    public const string LlavaMmprojFile = "mmproj-llava-v1.5-7b-f16.gguf";
    public const string PixtralMmprojFile = "mmproj-pixtral-12b-f16.gguf";

    private static string? FindModel(string file)
    {
        string[] candidates = { $@"E:\models\{file}", $@"C:\p\opentail-llm\models\{file}" };
        foreach (var p in candidates)
            if (File.Exists(p)) return p;

        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8 && dir is not null; i++)
        {
            var p = Path.Combine(dir, "models", file);
            if (File.Exists(p)) return p;
            dir = Directory.GetParent(dir)?.FullName;
        }
        return null;
    }

    public static string? FindMmproj() => FindModel(MmprojFile);
    public static string? FindE4BMmproj() => FindModel(E4BMmprojFile);
    public static string? FindTextModel() => FindModel(TextModelFile);
    public static string? FindGemma3Mmproj() => FindModel(Gemma3MmprojFile);
    public static string? FindGemma3TextModel() => FindModel(Gemma3TextModelFile);
    public static string? FindLlama4Mmproj() => FindModel(Llama4MmprojFile);
    public static string? FindLlavaMmproj() => FindModel(LlavaMmprojFile);
    public static string? FindPixtralMmproj() => FindModel(PixtralMmprojFile);

    /// <summary>Repo-root-relative golden fixtures produced by scripts/gemma4uv_ref.py.</summary>
    public static string? FindFixtureDir() => FindFixtureDir("gemma4uv");

    /// <summary>Repo-root-relative golden fixtures produced by scripts/&lt;name&gt;_ref.py.</summary>
    public static string? FindFixtureDir(string name)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8 && dir is not null; i++)
        {
            var p = Path.Combine(dir, "tests", "fixtures", name);
            if (Directory.Exists(p)) return p;
            dir = Directory.GetParent(dir)?.FullName;
        }
        return null;
    }
}
