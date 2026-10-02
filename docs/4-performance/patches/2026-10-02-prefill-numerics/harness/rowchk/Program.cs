using System.Globalization;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine;
// At one prompt row, batched-projection GEMM and per-token int8 matvec vs a double dot of the dequantized weights.
CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
var model = GgufModel.Open(args[0]);
var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
var tok = GgufTokenizer.FromGgufModel(model);
var ids = tok.Encode(File.ReadAllText(args[1])).Take(int.Parse(args[2])).ToArray();
using var backend = new CpuBackend();
using var fwd = new HybridGdnForwardPass(model, backend, hp, maxContextLength: ids.Length + 16);
foreach (var row in args.Skip(3).Select(int.Parse))
{
    HybridGdnForwardPass.XSeqRec = true; HybridGdnForwardPass.XSeqProj = false; HybridGdnForwardPass.XSeqFfn = true;
    HybridGdnForwardPass.GdnChunkedPrefillEnabled = true;
    var list = new List<(string, double, double, double, double)>();
    HybridGdnForwardPass.XRowCheck = list; HybridGdnForwardPass.XCheckRow = row;
    fwd.ResetCache(); fwd.Prefill(ids);
    HybridGdnForwardPass.XRowCheck = null;
    Console.WriteLine($"\n=== row {row} of n={ids.Length}: per projection call, relRMS vs double-dot reference (gemm / int8-matvec), maxAbs (gemm / matvec)");
    foreach (var (name, g, m, gm, mm) in list)
        Console.WriteLine($"  {name,-28} gemm {g,9:E2}  matvec {m,9:E2}   max {gm,9:E2} / {mm,9:E2}{(g > 10 * m ? "   <-- GEMM WORSE" : "")}");
    Console.WriteLine($"  mean relRMS: gemm {list.Average(t => t.Item2):E2}  matvec {list.Average(t => t.Item3):E2};  worst gemm {list.Max(t => t.Item2):E2}  worst matvec {list.Max(t => t.Item3):E2}");
}
