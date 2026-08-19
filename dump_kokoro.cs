using System;
using System.IO;
using System.Text;
using OpenTail.Stingray.Core;

string modelPath = @"C:\Git-Public\OpenTail.Stingray\models\kokoro-82m-q8_0.gguf";
if (File.Exists(modelPath))
{
    using var model = GgufModel.Open(modelPath);
    var sb = new StringBuilder();
    sb.AppendLine($"Kokoro Total Tensors: {model.Tensors.Count}");
    foreach (var t in model.Tensors)
    {
        sb.AppendLine($"{t.Name} | {t.Type} | dims=[{string.Join(",", t.Dimensions)}]");
    }
    File.WriteAllText(@"C:\Git-Public\OpenTail.Stingray\kokoro_tensors.txt", sb.ToString());
}
