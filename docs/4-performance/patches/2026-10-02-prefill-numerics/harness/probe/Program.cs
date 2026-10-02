using System.Globalization;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine;

// GDN-internal probe.
// (1) Local recurrence error per GDN layer on IDENTICAL inputs/state (taken from the double run): FP32 sequential vs
//     double, FP32 chunked vs double, chunked vs sequential (output and final state).
// (2) End-of-layer hidden-state divergence per layer, end-to-end: S (=A) vs D, R vs D, S vs R.
// (3) Prefill -> decode continuity: prefill N tokens with each path, then K teacher-forced single-token decode steps
//     (the normal decode kernels) on the same tokens; per-step drift between paths.
// args: model.gguf prompts.tsv
CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
var model = GgufModel.Open(args[0]);
var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
var tok = GgufTokenizer.FromGgufModel(model);
const int K = 16;
var prompts = new List<(string Name, int[] Ids)>();
foreach (var line in File.ReadAllLines(args[1]))
{
    if (string.IsNullOrWhiteSpace(line) || line.StartsWith('#')) continue;
    var f = line.Split('\t');
    string text = f[3].StartsWith('@') ? File.ReadAllText(f[3][1..]) : f[3].Replace("\\n", "\n");
    if (f[2] == "chat") text = "<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n";
    prompts.Add((f[0], tok.Encode(text).Take(int.Parse(f[1])).ToArray()));
}
using var backend = new CpuBackend();
using var fwd = new HybridGdnForwardPass(model, backend, hp, maxContextLength: prompts.Max(p => p.Ids.Length) + K + 16);
int L = hp.NumLayers, e = hp.EmbeddingDim, V = hp.VocabSize;
HybridGdnForwardPass.XAllowMtp = true;

void Setup(bool dbl, bool seqRec, bool seqProj, bool seqFfn)
{
    HybridGdnForwardPass.XDblRec = dbl; HybridGdnForwardPass.XDblState.Clear();
    HybridGdnForwardPass.XSeqRec = seqRec; HybridGdnForwardPass.XSeqProj = seqProj; HybridGdnForwardPass.XSeqFfn = seqFfn;
    HybridGdnForwardPass.GdnChunkedPrefillEnabled = true;
    fwd.ResetCache();
}

static double Rel(float[] x, float[] r, int from, int len)
{ double e2 = 0, r2 = 0; for (int i = from; i < from + len; i++) { double d = x[i] - r[i]; e2 += d * d; r2 += (double)r[i] * r[i]; } return Math.Sqrt(e2 / Math.Max(r2, 1e-300)); }

foreach (var (name, ids0) in prompts)
{
    // Continuation tokens: greedy from the exact path, then reused (teacher-forced) for every path.
    Setup(false, true, true, true);
    var ids = ids0.ToList();
    var lg = fwd.Prefill(ids0).ToArray();
    for (int k = 0; k < K; k++) { int nx = Argmax(lg); ids.Add(nx); lg = fwd.Forward(nx, ids.Count - 1).ToArray(); }
    var cont = ids.Skip(ids0.Length).ToArray();
    int n = ids0.Length;

    // (1)+(2)
    var probe = new List<(int, double, double, double, double, double, double, double, double, double, double)>();
    float[][] Layers(bool dbl, bool seqRec)
    {
        Setup(dbl, seqRec, true, true);
        var lh = new float[L][];
        HybridGdnForwardPass.XLayerHidden = lh;
        HybridGdnForwardPass.XRecProbe = dbl ? probe : null;
        fwd.Prefill(ids0);
        HybridGdnForwardPass.XLayerHidden = null; HybridGdnForwardPass.XRecProbe = null;
        return lh;
    }
    var hS = Layers(false, true); var hR = Layers(false, false); var hD = Layers(true, true);
    var byLayer = probe.ToDictionary(p => p.Item1);

    Console.WriteLine($"\n=== {name} n={n}  continuation: {Esc(tok.Decode(cont))}");
    Console.WriteLine("layer type | LOCAL (same inputs) out relRMS: seq-dbl  chk-dbl  chk-seq | state relRMS: seq-dbl  chk-dbl  chk-seq | state max: seq-dbl chk-dbl | HIDDEN relRMS: S-D      R-D      S-R    | last: S-D      R-D");
    for (int l = 0; l < L; l++)
    {
        bool attn = hp.LayerTypes![l] == LayerType.Attention;
        string loc = byLayer.TryGetValue(l, out var p)
            ? $"{p.Item2,8:E2} {p.Item3,8:E2} {p.Item8,8:E2} | {p.Item6,8:E2} {p.Item7,8:E2} {p.Item9,8:E2} | {p.Item10,8:E2} {p.Item11,8:E2}"
            : $"{"-",8} {"",8} {"",8} | {"",8} {"",8} {"",8} | {"",8} {"",8}";
        Console.WriteLine($"{l,5} {(attn ? "attn" : "gdn "),-4} |                     {loc} |               {Rel(hS[l], hD[l], 0, n * e),8:E2} {Rel(hR[l], hD[l], 0, n * e),8:E2} {Rel(hS[l], hR[l], 0, n * e),8:E2} | {Rel(hS[l], hD[l], (n - 1) * e, e),8:E2} {Rel(hR[l], hD[l], (n - 1) * e, e),8:E2}");
    }
    var all = probe.ToArray();
    Console.WriteLine($"mean LOCAL out relRMS: seq-dbl {all.Average(p => p.Item2):E2}  chk-dbl {all.Average(p => p.Item3):E2}  chk-seq {all.Average(p => p.Item8):E2}   state: seq-dbl {all.Average(p => p.Item6):E2}  chk-dbl {all.Average(p => p.Item7):E2}");

    // (3) Continuity: prefill n, then K teacher-forced decode steps. Logits at step -1 (last prefill position) .. K-1.
    float[][] Cont(bool dbl, bool seqRec, bool seqProj, bool seqFfn)
    {
        Setup(dbl, seqRec, seqProj, seqFfn);
        var outp = new float[K + 1][];
        outp[0] = fwd.Prefill(ids0).ToArray();
        HybridGdnForwardPass.XDblRec = false; // decode steps always use the normal FP32 decode kernels
        for (int k = 0; k < K; k++) outp[k + 1] = fwd.Forward(cont[k], n + k).ToArray();
        return outp;
    }
    var cA = Cont(false, true, true, true); var cR = Cont(false, false, true, true); var cE = Cont(false, false, false, false); var cD = Cont(true, true, true, true);
    Console.WriteLine("continuity  step | R vs A: delta  KL       flip | E vs A: delta  KL       flip | A vs D: delta  KL       flip | R vs D: delta  KL       flip");
    for (int s = 0; s <= K; s++)
    {
        string C(float[] x, float[] r) { var (d, kl, f) = Diff(r, x); return $"{d,7:F3} {kl,8:E1} {(f ? "FLIP" : "    ")}"; }
        Console.WriteLine($"  {(s == 0 ? "prefill" : $"dec {s - 1,2}"),-14} | {C(cR[s], cA[s])} | {C(cE[s], cA[s])} | {C(cA[s], cD[s])} | {C(cR[s], cD[s])}");
    }
}

static int Argmax(float[] a) { int b = 0; for (int i = 1; i < a.Length; i++) if (a[i] > a[b]) b = i; return b; }
static string Esc(string s) => "'" + s.Replace("\n", "\\n") + "'";
static (double Delta, double Kl, bool Flip) Diff(float[] r, float[] x)
{
    double d = 0; int r1 = 0, x1 = 0;
    for (int i = 0; i < r.Length; i++) { d = Math.Max(d, Math.Abs(r[i] - x[i])); if (r[i] > r[r1]) r1 = i; if (x[i] > x[x1]) x1 = i; }
    double sr = 0, sx = 0; for (int i = 0; i < r.Length; i++) { sr += Math.Exp(r[i] - r[r1]); sx += Math.Exp(x[i] - x[x1]); }
    double lzr = r[r1] + Math.Log(sr), lzx = x[x1] + Math.Log(sx), kl = 0;
    for (int i = 0; i < r.Length; i++) { double lp = r[i] - lzr; kl += Math.Exp(lp) * (lp - (x[i] - lzx)); }
    return (d, kl, r1 != x1);
}
