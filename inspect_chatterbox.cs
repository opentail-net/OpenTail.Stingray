using System;
using System.IO;
using System.Text;
using OpenTail.Stingray.Core;

string path = @"C:\Git-Public\OpenTail.Stingray\models\chatterbox-turbo-t3-q4_k.gguf";
if (File.Exists(path))
{
    using var model = GgufModel.Open(path);
    var sb = new StringBuilder();
    sb.AppendLine($"Chatterbox T3 Metadata ({model.Metadata.Count} entries):");
    foreach (var kvp in model.Metadata)
    {
        sb.AppendLine($"  {kvp.Key} = {kvp.Value}");
    }
    sb.AppendLine($"\nChatterbox T3 Tensors ({model.Tensors.Count}):");
    foreach (var t in model.Tensors)
    {
        sb.AppendLine($"  {t.Name} | {t.DType} | dims=[{string.Join(",", t.Dimensions)}]");
    }
    File.WriteAllText(@"C:\Git-Public\OpenTail.Stingray\chatterbox_t3_inspect.txt", sb.ToString());
}
