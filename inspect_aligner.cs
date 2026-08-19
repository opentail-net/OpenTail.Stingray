using System;
using System.IO;
using System.Text;
using OpenTail.Stingray.Core;

string path = @"C:\Git-Public\OpenTail.Stingray\models\qwen3-forcedaligner-0.6b.safetensors";
if (File.Exists(path))
{
    using var st = SafetensorsLoader.Open(path);
    var sb = new StringBuilder();
    sb.AppendLine($"Qwen3 ForcedAligner Tensors ({st.TensorCount}):");
    foreach (var name in st.TensorNames)
    {
        var desc = st.GetDescriptor(name);
        sb.AppendLine($"  {name} | {desc.DType} | shape=[{string.Join(",", desc.Shape)}]");
    }
    File.WriteAllText(@"C:\Git-Public\OpenTail.Stingray\qwen3_aligner_tensors.txt", sb.ToString());
}
