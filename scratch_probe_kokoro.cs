using System;
using System.Linq;
using OpenTail.Stingray.Core;

string modelPath = @"C:\Git-Public\OpenTail.Stingray\models\kokoro-82m-q8_0.gguf";
using var model = GgufModel.Open(modelPath);
Console.WriteLine($"Kokoro Tensors: {model.Tensors.Count}");
foreach (var t in model.Tensors.Take(25))
{
    Console.WriteLine($"  {t.Name} : {t.Type} shape=[{string.Join(",", t.Dimensions)}]");
}
