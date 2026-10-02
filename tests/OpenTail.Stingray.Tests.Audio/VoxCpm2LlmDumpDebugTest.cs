
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>ONE-OFF DEBUG: dumps config.json + base_lm/residual_lm tensor names for VoxCPM2's
/// packed GGUF. Real ground truth for the MiniCPM backbone's real config numbers.</summary>
public sealed class VoxCpm2LlmDumpDebugTest : HeavyTestBase
{
    private static string? FindRepoFile(string relPath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relPath);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void DumpConfigAndLlmTensorNames()
    {
        string? path = FindRepoFile("models/_models/voxcpm2/VoxCPM2-GGUF/voxcpm2-q8_0.gguf");
        Assert.SkipUnless(path != null, "voxcpm2-q8_0.gguf not found");

        using var model = OpenTail.Stingray.Core.GgufModel.Open(path!);
        string? outDir = FindRepoFile("docs/00-current-work.md");
        string root = Path.GetDirectoryName(outDir!)!;

        Console.Error.WriteLine($"[VoxCpm2LlmDump] embedded files: {string.Join(" | ", OpenTail.Stingray.Audio.AudioCppEmbeddedFiles.Names(model))}");
        var files = OpenTail.Stingray.Audio.AudioCppEmbeddedFiles.ReadAll(model);
        foreach (string name in new[] { "config.json", "tokenizer.json", "tokenizer_config.json" })
        {
            if (files.TryGetValue(name, out var bytes))
            {
                string content = System.Text.Encoding.UTF8.GetString(bytes);
                File.WriteAllText(Path.Combine(root, "..", $"voxcpm2-{name}"), content);
            }
        }

        if (model.Metadata.TryGetValue("audiocpp.tensor_names", out var tnObj) && tnObj is object[] tensorNames)
        {
            var sb = new System.Text.StringBuilder();
            for (int i = 0; i < tensorNames.Length; i++)
            {
                string name = (string)tensorNames[i];
                if (!name.StartsWith("weights/base_lm.") && !name.StartsWith("weights/residual_lm.") &&
                    !name.StartsWith("weights/lm_head") && !name.StartsWith("weights/fsq_layer") &&
                    !name.StartsWith("weights/enc_to_lm") && !name.StartsWith("weights/lm_to_dit") &&
                    !name.StartsWith("weights/res_to_dit") && !name.StartsWith("weights/fusion_concat") &&
                    !name.StartsWith("weights/stop_") && !name.StartsWith("weights/feat_encoder") &&
                    !name.StartsWith("weights/feat_decoder"))
                    continue;
                var t = model.Tensors[i];
                sb.AppendLine($"{name}\t{string.Join(",", t.Dimensions)}\t{t.DType}");
            }
            File.WriteAllText(Path.Combine(root, "..", "voxcpm2-llm-tensor-names.txt"), sb.ToString());
            Console.Error.WriteLine($"[VoxCpm2LlmDump] wrote llm-relevant tensor names");
        }
    }
}
