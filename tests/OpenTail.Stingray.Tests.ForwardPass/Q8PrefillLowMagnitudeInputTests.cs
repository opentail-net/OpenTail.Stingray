using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// REGRESSION GUARD for a fixed defect: int8 activation prefill collapsed on prompts made of a
/// single repeated token.
///
/// <para>int8 activation prefill (default ON) collapses on low-magnitude inputs. A whitespace-only
/// prompt of 8 tokens produces a final-logit cosine of <b>-0.124</b> against the exact F32 path,
/// with a different argmax — the logits point in roughly the opposite direction, not merely a
/// degraded one. The result is deterministic and reproduces exactly.</para>
///
/// <para><b>The cause was exact token repetition, not whitespace.</b> Every single-token prompt
/// collapsed regardless of which token: space -0.124, tab 0.030, comma 0.031, "9" -0.013, and
/// ordinary words too — "the" 0.470, "scheduler" 0.324. Adding ONE differing token restored 0.995+,
/// at every length from 2 to 64. `ForwardPass.IsSingleDistinctTokenPrompt` now routes such prompts
/// to the exact F32 path, so this case returns cosine 1.0 exactly.</para>
///
/// <para>Two hypotheses were measured and disproved first, which is why the fix keys on token
/// identity rather than anything numerical: embedding outlier ratio does not separate the failing
/// class (healthy code scores worse than collapsing whitespace), and neither does the activation
/// outlier ratio taken at the point of quantisation (healthy prose contains rows with the same
/// maximum as the collapsing case).</para>
///
/// <para>Measured across input classes at n=8/32 (cosine int8 vs F32), so the scope is clear rather
/// than alarmist:</para>
/// <code>
///   prose   0.9973 / 0.9697      hex     0.9980 / 0.9981
///   code    0.9776 / 0.9953      base64  0.9968 / 0.9984
///   repeat  0.9965 / 0.9972      cjk     0.9986 / 0.9971
///   punct   0.9972 / 0.9602      ws     -0.1241 / 0.9776   ** collapse **
/// </code>
///
/// <para>Ordinary text is fine. The collapse is specific to near-empty, low-magnitude input — but
/// that IS reachable: a user sending a blank-ish prompt, a document with a long whitespace run, or
/// an empty template slot. Several other classes also sit below 0.99, which is worth knowing when
/// calibrating any numerics tolerance against this path.</para>
///
/// <para>This test keeps the whitespace case because it was the reproduction that surfaced the
/// defect. The general contract is covered alongside it.</para>
/// </summary>
public sealed class Q8PrefillLowMagnitudeInputTests : HeavyTestBase
{
    [Fact]
    public void Q8Prefill_OnSingleRepeatedToken_StaysCloseToExactF32()
    {
        string? path = FindModelPath();
        Assert.SkipWhen(path is null, "SmolLM2 GGUF not present");
        using var modelHandle = SharedModelCacheFixture.Instance.Acquire(path!);
        var model = modelHandle.Model;
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        var tk = GgufTokenizer.FromGgufModel(model);

        var acc = new List<int>();
        while (acc.Count < 8) acc.AddRange(tk.Encode("        \n\n\t\t    \n   \t \n\n        "));
        int[] toks = acc.Take(8).ToArray();

        float[] q8 = Prefill(model, hp, toks, useQ8: true);
        float[] f32 = Prefill(model, hp, toks, useQ8: false);

        double cos = Cosine(q8, f32);
        Assert.True(cos > 0.9,
            $"int8 prefill diverged from exact F32 on a whitespace-only prompt: cosine {cos:F6}. " +
            "Ordinary text measures 0.96-0.999 on this model.");

        // The general contract, not just the whitespace reproduction: ORDINARY words repeated are
        // equally degenerate (bare "the" x8 measured 0.470 before the fix), so a guard that only
        // special-cased whitespace would have left the real defect in place.
        int the = tk.Encode("the")[0];
        int[] repeated = Enumerable.Repeat(the, 8).ToArray();
        double cosRepeat = Cosine(
            Prefill(model, hp, repeated, useQ8: true),
            Prefill(model, hp, repeated, useQ8: false));
        Assert.True(cosRepeat > 0.9,
            $"int8 prefill diverged on a repeated ordinary token: cosine {cosRepeat:F6}.");
    }

    private static float[] Prefill(GgufModel model, ModelHyperparams hp, int[] toks, bool useQ8)
    {
        bool prev = SimdKernels.Q8PrefillEnabled;
        SimdKernels.Q8PrefillEnabled = useQ8;
        try
        {
            using var backend = new CpuBackend();
            using var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 512);
            return fwd.Prefill(toks).ToArray();
        }
        finally { SimdKernels.Q8PrefillEnabled = prev; }
    }

    private static double Cosine(float[] a, float[] b)
    {
        double d = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++) { d += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i]; }
        return d / (Math.Sqrt(na) * Math.Sqrt(nb));
    }

    private static string? FindModelPath()
    {
        string? dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8 && dir is not null; i++)
        {
            var p = Path.Combine(dir, "models", "SmolLM2-1.7B-Instruct-Q4_K_M.gguf");
            if (File.Exists(p)) return p;
            dir = Directory.GetParent(dir)?.FullName;
        }
        return null;
    }
}
