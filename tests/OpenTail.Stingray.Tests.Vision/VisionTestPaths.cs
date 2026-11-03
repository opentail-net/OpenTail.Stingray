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
    public const string Glm4MmprojFile = "mmproj-glm-4.6v-q4.gguf";
    public const string Qwen25VlMmprojFile = "mmproj-qwen2.5-vl-7b-f16.gguf";
    public const string Exaone4MmprojFile = "mmproj-exaone-4.5-q8_0.gguf";
    public const string MimoVlMmprojFile = "mmproj-mimovl-7b-q8_0.gguf";
    public const string HunyuanVlMmprojFile = "mmproj-hunyuanocr-q8_0.gguf";

    private static string? FindModel(string file) => FindAnywhere(file);

    /// <summary>
    /// One lookup for every vision test, so a checkpoint that exists on this machine is never skipped as "not present" just because one test searched fewer places than
    /// another (seven copies of this search existed, most of them with hard-coded paths of old machines, and 15 mmproj files on F:\_models were skipped by tests that only
    /// looked in the repo's models folder). Order: STINGRAY_TEST_MODEL_DIRS (';'-separated), the repo's models and models\_models walking up from the current directory,
    /// then every ready drive's _models and _other_models, one sub-folder level deep.
    /// </summary>
    public static string? FindAnywhere(string file)
    {
        var dirs = new List<string>();
        if (Environment.GetEnvironmentVariable("STINGRAY_TEST_MODEL_DIRS") is { Length: > 0 } extra)
            dirs.AddRange(extra.Split(';', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries));
        foreach (var start in new[] { Directory.GetCurrentDirectory(), AppContext.BaseDirectory })
        {
            var d = start;
            for (int i = 0; i < 10 && d is not null; i++)
            {
                dirs.Add(Path.Combine(d, "models"));
                dirs.Add(Path.Combine(d, "models", "_models"));
                d = Directory.GetParent(d)?.FullName;
            }
        }
        try
        {
            foreach (var drive in DriveInfo.GetDrives())
            {
                if (!drive.IsReady) continue;
                dirs.Add(Path.Combine(drive.Name, "_models"));
                dirs.Add(Path.Combine(drive.Name, "_other_models"));
                dirs.Add(Path.Combine(drive.Name, "models"));
            }
        }
        catch (IOException) { }
        foreach (var d in dirs.Distinct(StringComparer.OrdinalIgnoreCase))
        {
            try
            {
                if (!Directory.Exists(d)) continue;
                var direct = Path.Combine(d, file);
                if (File.Exists(direct)) return direct;
                foreach (var sub in Directory.EnumerateDirectories(d))
                {
                    var nested = Path.Combine(sub, file);
                    if (File.Exists(nested)) return nested;
                }
            }
            catch (Exception e) when (e is IOException or UnauthorizedAccessException) { }
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
    public static string? FindGlm4Mmproj() => FindModel(Glm4MmprojFile);
    public static string? FindQwen25VlMmproj() => FindModel(Qwen25VlMmprojFile);
    public static string? FindExaone4Mmproj() => FindModel(Exaone4MmprojFile);
    public static string? FindMimoVlMmproj() => FindModel(MimoVlMmprojFile);
    public static string? FindHunyuanVlMmproj() => FindModel(HunyuanVlMmprojFile);

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
