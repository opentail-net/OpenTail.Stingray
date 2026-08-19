using System;
using System.IO;
using System.Linq;
using System.Text;
using OpenTail.Stingray.Core;

string path = @"C:\Git-Public\OpenTail.Stingray\models\hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors";
if (File.Exists(path))
{
    using var st = SafetensorsLoader.Open(path);
    foreach (var n in st.TensorNames.Where(n => n.Contains("time") || n.Contains("txt_in") || n.Contains("vector") || n.Contains("guidance")))
    {
        Console.WriteLine(n);
    }
}
