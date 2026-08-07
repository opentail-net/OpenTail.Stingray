using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// KNOWN DEFECT, pinned as a skipped test rather than asserted green.
///
/// <para>int8 activation prefill (default ON) collapses on low-magnitude inputs. A whitespace-only
/// prompt of 8 tokens produces a final-logit cosine of <b>-0.124</b> against the exact F32 path,
/// with a different argmax — the logits point in roughly the opposite direction, not merely a
/// degraded one. The result is deterministic and reproduces exactly.</para>
///
/// <para><b>Why the shipped mitigation does not cover it.</b> `ForwardPass` skips int8 when a prompt
/// is composed ENTIRELY of control/user-defined tokens. Whitespace tokens are ordinary vocabulary
/// entries, so this prompt takes the int8 path. The mitigation keys on token TYPE; the failure is
/// driven by activation MAGNITUDE, which is a different property. Per-row int8 scaling degrades
/// badly when a row's dynamic range collapses toward zero, and near-empty input is exactly that.</para>
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
/// <para><b>Un-skip when fixed.</b> The likely fix is to gate int8 on activation dynamic range
/// rather than (or in addition to) token type, so the guard tracks the property that actually
/// causes the failure.</para>
/// </summary>
public sealed class Q8PrefillLowMagnitudeInputTests
{
    [Fact(Skip = "KNOWN DEFECT: int8 prefill collapses to cosine -0.124 on whitespace-only input. " +
                 "The shipped mitigation keys on token type (all-control prompts) but the failure is " +
                 "driven by activation magnitude. Un-skip when int8 is gated on dynamic range.")]
    public void Q8Prefill_OnWhitespaceOnlyPrompt_StaysCloseToExactF32()
    {
        string? path = FindModelPath();
        Assert.SkipWhen(path is null, "SmolLM2 GGUF not present");
        using var model = GgufModel.Open(path!);
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
