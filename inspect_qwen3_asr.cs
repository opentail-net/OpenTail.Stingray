using System;
using System.IO;
using System.Text;
using OpenTail.Stingray.Core;

string path = @"C:\Git-Public\OpenTail.Stingray\models\qwen3-asr-0.6b-q4_k.gguf";
if (File.Exists(path))
{
    using var model = GgufModel.Open(path);
    var sb = new StringBuilder();
    sb.AppendLine($"Qwen3 ASR Metadata ({model.Metadata.Count} entries):");
    foreach (var kvp in model.Metadata)
    {
        sb.AppendLine($"  {kvp.Key} = {kvp.Value}");
    }
    sb.AppendLine($"\nQwen3 ASR Tensors ({model.Tensors.Count}):");
    foreach (var t in model.Tensors)
    {
        sb.AppendLine($"  {t.Name} | {t.DType} | dims=[{string.Join(",", t.Dimensions)}]");
    }
    File.WriteAllText(@"C:\Git-Public\OpenTail.Stingray\qwen3_asr_tensors.txt", sb.ToString());
}
