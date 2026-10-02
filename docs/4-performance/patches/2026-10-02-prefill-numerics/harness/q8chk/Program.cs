using System.Globalization;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine;
// Repeated-token int8 collapse check on the hybrid-GDN chunked prefill: per-token (exact) vs chunked Q8 off/on.
CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
var model = GgufModel.Open(args[0]);
var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
var tok = GgufTokenizer.FromGgufModel(model);
using var backend = new CpuBackend();
using var fwd = new HybridGdnForwardPass(model, backend, hp, maxContextLength: 256);
int the = tok.Encode(" the")[^1], sp = tok.Encode(" ")[^1], nl = tok.Encode("\n")[^1];
var cases = new (string, int[])[] {
    ("' the' x8", Enumerable.Repeat(the, 8).ToArray()),
    ("' the' x16", Enumerable.Repeat(the, 16).ToArray()),
    ("' the' x48", Enumerable.Repeat(the, 48).ToArray()),
    ("'\n' x16", Enumerable.Repeat(nl, 16).ToArray()),
    ("' ' x16", Enumerable.Repeat(sp, 16).ToArray()),
    ("prose 16", tok.Encode("The quick brown fox jumps over the lazy dog while the cat sleeps on the warm mat").Take(16).ToArray()),
    ("prose 48", tok.Encode("The quick brown fox jumps over the lazy dog while the cat sleeps on the warm mat. Pack my box with five dozen liquor jugs. How razorback-jumping frogs can level six piqued gymnasts! Sphinx of black quartz, judge my vow.").Take(48).ToArray()),
};
float[] Run(bool chunked, bool q8, int[] ids) { HybridGdnForwardPass.GdnChunkedPrefillEnabled = chunked; SimdKernels.Q8PrefillEnabled = q8; fwd.ResetCache(); return fwd.Prefill(ids).ToArray(); }
static (double cos, double d, bool same) Cmp(float[] a, float[] b) { double dot=0,na=0,nb=0,d=0; int ia=0,ib=0; for (int i=0;i<a.Length;i++){dot+=(double)a[i]*b[i];na+=(double)a[i]*a[i];nb+=(double)b[i]*b[i];d=Math.Max(d,Math.Abs(a[i]-b[i])); if(a[i]>a[ia])ia=i; if(b[i]>b[ib])ib=i;} return (dot/Math.Sqrt(na*nb), d, ia==ib); }
Console.WriteLine("case          n | chunked F32 vs exact: cos      maxDiff argmax | chunked int8 vs exact: cos      maxDiff argmax");
foreach (var (name, ids) in cases)
{
    var ex = Run(false, false, ids); var f = Run(true, false, ids); var q = Run(true, true, ids);
    var a = Cmp(ex, f); var b = Cmp(ex, q);
    Console.WriteLine($"{name,-12} {ids.Length,3} | {a.cos,10:F6} {a.d,8:F3} {(a.same ? "same" : "DIFF"),-6} | {b.cos,10:F6} {b.d,8:F3} {(b.same ? "same" : "DIFF")}");
}
