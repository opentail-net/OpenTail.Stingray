// dotnet run analyze.cs -- ornith9b.csv
// Post-hoc analysis of the attribution CSV, prompt as the statistical unit.
using System.Globalization;
CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;

var rows = File.ReadLines(args[0]).Skip(1).Select(l => l.Split(',')).Select(f => new Row(
    f[0], f[1], f[2], int.Parse(f[4]), int.Parse(f[5]), int.Parse(f[6]),
    double.Parse(f[7]), double.Parse(f[8]), double.Parse(f[11]), double.Parse(f[12]))).ToList();
var by = rows.GroupBy(r => (r.Comp, r.Prompt)).ToDictionary(g => g.Key, g => g.OrderBy(r => r.Pos).ToArray());
var promptList = rows.Select(r => (r.Prompt, r.Cls)).Distinct().ToList();
var rnd = new Random(7);

(double Mean, double Lo, double Hi) Boot(double[] v, Func<double[], double> stat)
{
    var s = new double[2000];
    for (int b = 0; b < s.Length; b++) { var x = new double[v.Length]; for (int i = 0; i < v.Length; i++) x[i] = v[rnd.Next(v.Length)]; s[b] = stat(x); }
    Array.Sort(s);
    return (stat(v), s[50], s[1949]);
}
double Median(double[] v) { var c = v.OrderBy(x => x).ToArray(); return c.Length % 2 == 1 ? c[c.Length / 2] : (c[c.Length / 2 - 1] + c[c.Length / 2]) / 2; }

// 1. Arbitration: where A and R disagree on top-1, which agrees with D?
Console.WriteLine("== Arbitration: positions where argmax(A) != argmax(R); who agrees with D (higher-precision recurrence)?");
foreach (var (x, xa) in new[] { ("R_chunkRec", "R|A"), ("E_allopt", "E|A") })
{
    string xd = xa[0] + "|D";
    foreach (var cls in promptList.Select(p => p.Cls).Distinct().Append("ALL"))
    {
        int dis = 0, xWins = 0, aWins = 0, neither = 0, prompts = 0;
        foreach (var (p, c) in promptList.Where(p => cls == "ALL" || p.Cls == cls))
        {
            if (!by.TryGetValue((xa, p), out var ra) || !by.TryGetValue(("A|D", p), out var ad)) continue;
            prompts++;
            for (int i = 0; i < ra.Length; i++)
            {
                int a = ra[i].RefTop1, r = ra[i].XTop1, d = ad[i].RefTop1;
                if (a == r) continue;
                dis++;
                if (r == d) xWins++; else if (a == d) aWins++; else neither++;
            }
        }
        Console.WriteLine($"  {xa[0]} vs A  {cls,-5} prompts {prompts,3}  disagreements {dis,4}  {xa[0]}==D {xWins,4}  A==D {aWins,4}  neither {neither,4}");
    }
}

// 2. Paired distance to D, per prompt: log(KL(D||X) / KL(D||A)) and log((1-cos)_X / (1-cos)_A). Median over prompts + CI.
Console.WriteLine();
Console.WriteLine("== Paired distance to D (per-prompt median log-ratio; <0 means X closer to D than A)");
foreach (var x in new[] { "R", "E" })
    foreach (var cls in promptList.Select(p => p.Cls).Distinct().Append("ALL"))
    {
        var lrKl = new List<double>(); var lrCos = new List<double>(); var flipA = new List<double>(); var flipX = new List<double>();
        foreach (var (p, c) in promptList.Where(p => cls == "ALL" || p.Cls == cls))
        {
            if (!by.TryGetValue(("A|D", p), out var ad) || !by.TryGetValue((x + "|D", p), out var xd)) continue;
            double kA = ad.Average(r => r.Kl), kX = xd.Average(r => r.Kl);
            double cA = ad.Average(r => 1 - r.Cos), cX = xd.Average(r => 1 - r.Cos);
            if (kA > 0 && kX > 0) lrKl.Add(Math.Log(kX / kA));
            if (cA > 0 && cX > 0) lrCos.Add(Math.Log(cX / cA));
            flipA.Add(ad.Count(r => r.RefTop1 != r.XTop1) / (double)ad.Length);
            flipX.Add(xd.Count(r => r.RefTop1 != r.XTop1) / (double)xd.Length);
        }
        if (lrKl.Count == 0) continue;
        var k = Boot(lrKl.ToArray(), Median); var cc = Boot(lrCos.ToArray(), Median);
        var fa = Boot(flipA.ToArray(), v => v.Average()); var fx = Boot(flipX.ToArray(), v => v.Average());
        int xCloser = lrKl.Count(v => v < 0);
        Console.WriteLine($"  {x} {cls,-5} n={lrKl.Count,3}  medLog KL {k.Mean,6:F2} [{k.Lo:F2},{k.Hi:F2}] (exp {Math.Exp(k.Mean):F2}x; {x} closer on {xCloser}/{lrKl.Count})  " +
                          $"medLog(1-cos) {cc.Mean,6:F2} [{cc.Lo:F2},{cc.Hi:F2}]  flip% vs D: A {fa.Mean * 100:F2} [{fa.Lo * 100:F2},{fa.Hi * 100:F2}]  {x} {fx.Mean * 100:F2} [{fx.Lo * 100:F2},{fx.Hi * 100:F2}]");
    }

// 3. Certificate profile per comparison, prompt as the unit.
Console.WriteLine();
Console.WriteLine("== Certificate (m <= 2*delta = vulnerable), prompt-level means with 95% CI");
foreach (var comp in rows.Select(r => r.Comp).Distinct())
    foreach (var cls in promptList.Select(p => p.Cls).Distinct().Append("ALL"))
    {
        var V = new List<double>(); var F = new List<double>(); int totV = 0, totF = 0, totFv = 0, totFc = 0;
        foreach (var (p, c) in promptList.Where(p => cls == "ALL" || p.Cls == cls))
        {
            if (!by.TryGetValue((comp, p), out var rs)) continue;
            int v = rs.Count(r => r.Margin <= 2 * r.Delta), f = rs.Count(r => r.RefTop1 != r.XTop1);
            V.Add(v / (double)rs.Length); F.Add(f / (double)rs.Length);
            totV += v; totF += f; totFv += rs.Count(r => r.RefTop1 != r.XTop1 && r.Margin <= 2 * r.Delta); totFc += rs.Count(r => r.RefTop1 != r.XTop1 && r.Margin > 2 * r.Delta);
        }
        if (V.Count == 0) continue;
        var bv = Boot(V.ToArray(), v => v.Average()); var bf = Boot(F.ToArray(), v => v.Average());
        Console.WriteLine($"  {comp,-5} {cls,-5} n={V.Count,3}  V% {bv.Mean * 100,6:F2} [{bv.Lo * 100:F2},{bv.Hi * 100:F2}]  F% {bf.Mean * 100,5:F2} [{bf.Lo * 100:F2},{bf.Hi * 100:F2}]  C%=F/V {(totV > 0 ? 100.0 * totF / totV : 0),5:F2}  flipsCertified(should be 0) {totFc}");
    }

record Row(string Comp, string Prompt, string Cls, int Pos, int RefTop1, int XTop1, double Margin, double Delta, double Cos, double Kl);
