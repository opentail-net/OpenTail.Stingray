using System.Globalization;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine;

// Attribution matrix: per-token reference (A) vs chunked prefill with each optimisation toggled, plus a
// double-precision-recurrence reference (D) to tell "different" from "less accurate".
// args: model.gguf prompts.tsv out.csv [allowMtp]
CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
var model = GgufModel.Open(args[0]);
var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
var tok = GgufTokenizer.FromGgufModel(model);
int V = hp.VocabSize;
HybridGdnForwardPass.XAllowMtp = args.Length > 3 && args[3] == "1";

var prompts = new List<(string Cls, string Name, int[] Ids)>();
int idx = 0;
foreach (var line in File.ReadAllLines(args[1]))
{
    if (string.IsNullOrWhiteSpace(line) || line.StartsWith('#')) continue;
    var f = line.Split('\t');
    string text = f[3].StartsWith('@') ? File.ReadAllText(f[3][1..]) : f[3].Replace("\\n", "\n");
    if (f[2] == "chat") text = "<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n";
    var ids = tok.Encode(text).Take(int.Parse(f[1])).ToArray();
    prompts.Add((f[0], $"{f[0]}{idx++}", ids));
}
int maxN = prompts.Max(p => p.Ids.Length);
Console.WriteLine($"model {Path.GetFileName(args[0])} vocab {V} prompts {prompts.Count} maxN {maxN} mtpHead {hp.NumMtpLayers}");

using var backend = new CpuBackend();
using var fwd = new HybridGdnForwardPass(model, backend, hp, maxContextLength: maxN + 16);
long buf = (long)maxN * V;
float[] aL = new float[buf], xL = new float[buf], dL = new float[buf], rL = new float[buf];

var configs = new (string Name, bool Dbl, bool Rec, bool Proj, bool Ffn)[]
{
    ("D_dblRec", true, true, true, true),      // double recurrence, everything else per-token
    ("S_allseq", false, true, true, true),     // must equal A bit-for-bit
    ("R_chunkRec", false, false, true, true),  // B in the ChatGPT table
    ("P_batchProj", false, true, false, true), // C
    ("F_batchFfn", false, true, true, false),  // D (ChatGPT's) - batched FFN
    ("E_allopt", false, false, false, false),  // production
};
// comparisons: label, reference buffer, candidate buffer (filled when the candidate config has run)
var compSpecs = new (string Label, string RefCfg, string XCfg)[]
{
    ("S|A", "A", "S_allseq"), ("R|A", "A", "R_chunkRec"), ("P|A", "A", "P_batchProj"), ("F|A", "A", "F_batchFfn"),
    ("E|A", "A", "E_allopt"), ("E|R", "R_chunkRec", "E_allopt"), ("A|D", "D_dblRec", "A"), ("R|D", "D_dblRec", "R_chunkRec"),
    ("E|D", "D_dblRec", "E_allopt"),
};

using var csv = new StreamWriter(args[2]);
csv.WriteLine("comp,prompt,cls,n,pos,refTop1,xTop1,margin,delta,dTop,dRest,cos,kl,top5");
// per comparison, per prompt: (cls, n, vulnerable, vulnerableTight, flips)
var perPrompt = new Dictionary<string, List<(string Cls, int N, int Vul, int VulT, int Flips, double MaxD, double MinCos, double MeanKl)>>();

float[] Buf(string cfg) => cfg switch { "A" => aL, "D_dblRec" => dL, "R_chunkRec" => rL, _ => xL };

foreach (var (cls, name, ids) in prompts)
{
    int n = ids.Length;
    fwd.ResetCache();
    HybridGdnForwardPass.XDblRec = false;
    var sw = System.Diagnostics.Stopwatch.StartNew();
    for (int t = 0; t < n; t++)
        fwd.Forward(ids[t], t).CopyTo(aL.AsSpan((int)((long)t * V), V));
    Console.WriteLine($"[{name}] n={n} A per-token {sw.Elapsed.TotalSeconds:F1}s ({n / sw.Elapsed.TotalSeconds:F1} t/s)");

    foreach (var c in configs)
    {
        HybridGdnForwardPass.XDblRec = c.Dbl; HybridGdnForwardPass.XDblState.Clear();
        HybridGdnForwardPass.XSeqRec = c.Rec; HybridGdnForwardPass.XSeqProj = c.Proj; HybridGdnForwardPass.XSeqFfn = c.Ffn;
        HybridGdnForwardPass.GdnChunkedPrefillEnabled = true;
        var target = c.Name switch { "D_dblRec" => dL, "R_chunkRec" => rL, _ => xL };
        HybridGdnForwardPass.XAllLogits = target;
        fwd.ResetCache();
        sw.Restart();
        fwd.Prefill(ids);
        double secs = sw.Elapsed.TotalSeconds;
        HybridGdnForwardPass.XAllLogits = null;
        HybridGdnForwardPass.XDblRec = false;
        Console.WriteLine($"  {c.Name,-12} {secs,6:F1}s ({n / secs,5:F1} t/s)");

        foreach (var spec in compSpecs.Where(s => s.XCfg == c.Name || (s.XCfg == "A" && c.Name == "D_dblRec")))
            Compare(spec.Label, Buf(spec.RefCfg), Buf(spec.XCfg), name, cls, n);
    }
}

void Compare(string label, float[] refB, float[] xB, string name, string cls, int n)
{
    var rows = new double[n][];
    Parallel.For(0, n, t =>
    {
        var r = refB.AsSpan((int)((long)t * V), V);
        var x = xB.AsSpan((int)((long)t * V), V);
        int r1 = 0, x1 = 0;
        double dot = 0, nr = 0, nx = 0;
        for (int i = 0; i < V; i++)
        {
            if (r[i] > r[r1]) r1 = i;
            if (x[i] > x[x1]) x1 = i;
            dot += (double)r[i] * x[i]; nr += (double)r[i] * r[i]; nx += (double)x[i] * x[i];
        }
        int r2 = r1 == 0 ? 1 : 0;
        double dRest = 0;
        for (int i = 0; i < V; i++)
        {
            if (i != r1 && r[i] > r[r2]) r2 = i;
            if (i != r1) dRest = Math.Max(dRest, Math.Abs(r[i] - x[i]));
        }
        double dTop = Math.Abs(r[r1] - x[r1]);
        double delta = Math.Max(dTop, dRest);
        double mr = r[r1], mx = x[x1], sr = 0, sx = 0;
        for (int i = 0; i < V; i++) { sr += Math.Exp(r[i] - mr); sx += Math.Exp(x[i] - mx); }
        double lzr = mr + Math.Log(sr), lzx = mx + Math.Log(sx), kl = 0;
        for (int i = 0; i < V; i++) { double lp = r[i] - lzr; kl += Math.Exp(lp) * (lp - (x[i] - lzx)); }
        int ov = Top(r, 5).Intersect(Top(x, 5)).Count();
        rows[t] = [r1, x1, r[r1] - r[r2], delta, dTop, dRest, nr * nx > 0 ? dot / Math.Sqrt(nr * nx) : 1, kl, ov];
    });
    for (int t = 0; t < n; t++)
    {
        var q = rows[t];
        csv.WriteLine($"{label},{name},{cls},{n},{t},{q[0]},{q[1]},{q[2]:G6},{q[3]:G6},{q[4]:G6},{q[5]:G6},{q[6]:F8},{q[7]:G6},{q[8]}");
    }
    csv.Flush();
    int vul = rows.Count(q => q[2] <= 2 * q[3]);
    int vulT = rows.Count(q => q[2] <= q[4] + q[5]);
    int flips = rows.Count(q => q[0] != q[1]);
    if (!perPrompt.TryGetValue(label, out var list)) perPrompt[label] = list = new();
    list.Add((cls, n, vul, vulT, flips, rows.Max(q => q[3]), rows.Min(q => q[6]), rows.Average(q => q[7])));
    Console.WriteLine($"    {label,-5} flips {flips,3} vul {vul,4}/{n} vulTight {vulT,4} maxDelta {rows.Max(q => q[3]),8:G4} minCos {rows.Min(q => q[6]):F6} meanKL {rows.Average(q => q[7]):G3}");

    int ff = Array.FindIndex(rows, q => q[0] != q[1]);
    if (ff >= 0)
    {
        var r = refB.AsSpan((int)((long)ff * V), V);
        var x = xB.AsSpan((int)((long)ff * V), V);
        string Fmt(Span<float> a) { var arr = a.ToArray(); return string.Join(" ", Top(a, 5).Select(i => $"{Esc(tok.Decode([i]))}:{arr[i]:F3}")); }
        Console.WriteLine($"      first flip pos {ff}/{n} margin {rows[ff][2]:G4} delta {rows[ff][3]:G4} dTop {rows[ff][4]:G4} dRest {rows[ff][5]:G4}");
        Console.WriteLine($"        ref: {Fmt(r)}");
        Console.WriteLine($"        x  : {Fmt(x)}");
    }
}

static string Esc(string s) => "'" + s.Replace("\n", "\\n").Replace("\t", "\\t") + "'";

static int[] Top(Span<float> a, int k)
{
    var top = new int[k]; var val = new float[k]; Array.Fill(val, float.NegativeInfinity);
    for (int i = 0; i < a.Length; i++)
    {
        if (a[i] <= val[k - 1]) continue;
        int j = k - 1;
        while (j > 0 && a[i] > val[j - 1]) { val[j] = val[j - 1]; top[j] = top[j - 1]; j--; }
        val[j] = a[i]; top[j] = i;
    }
    return top;
}

// ── Prompt-level summary (prompt = statistical unit), bootstrap 95% CI of the mean over prompts ──
Console.WriteLine();
Console.WriteLine("comp  cls    prompts posns  V%(m<=2d)            Vt%(tight)           F%(flip)             C%=F/V  maxDelta  minCos     meanKL");
var rnd = new Random(12345);
foreach (var label in compSpecs.Select(s => s.Label))
{
    if (!perPrompt.TryGetValue(label, out var all)) continue;
    foreach (var cls in all.Select(p => p.Cls).Distinct().Append("ALL"))
    {
        var ps = cls == "ALL" ? all : all.Where(p => p.Cls == cls).ToList();
        string Ci(Func<(string Cls, int N, int Vul, int VulT, int Flips, double MaxD, double MinCos, double MeanKl), double> f)
        {
            var v = ps.Select(f).ToArray();
            var means = new double[2000];
            for (int b = 0; b < means.Length; b++) { double s = 0; for (int i = 0; i < v.Length; i++) s += v[rnd.Next(v.Length)]; means[b] = s / v.Length; }
            Array.Sort(means);
            return $"{v.Average() * 100,6:F2} [{means[50] * 100:F2},{means[1949] * 100:F2}]";
        }
        int totV = ps.Sum(p => p.Vul), totF = ps.Sum(p => p.Flips);
        Console.WriteLine($"{label,-5} {cls,-6} {ps.Count,7} {ps.Sum(p => p.N),5}  {Ci(p => (double)p.Vul / p.N),-20} {Ci(p => (double)p.VulT / p.N),-20} {Ci(p => (double)p.Flips / p.N),-20} {(totV > 0 ? 100.0 * totF / totV : 0),6:F2} {ps.Max(p => p.MaxD),8:G4} {ps.Min(p => p.MinCos),10:F6} {ps.Average(p => p.MeanKl),10:G3}");
    }
}
