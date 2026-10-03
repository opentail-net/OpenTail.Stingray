namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// Model-level check for <c>STINGRAY_CPU_DECODE_VIA_GEMM=1</c> (<see cref="SimdKernels.DecodeViaGemm"/>):
/// teacher-forced token-by-token decode must produce the same logits at each position as batched
/// prefill of the same tokens. The kernel-level property (each GEMM row is independent of the batch
/// size) is pinned exactly by <c>BatchInvariantGemmTests</c>; whether the whole model inherits it also
/// depends on decode and prefill agreeing in every other op (attention, norms, RoPE), which only a real
/// model shows. The switch-off arm is the baseline (decode on its own Q8_KS matvec).
/// </summary>
public sealed class DecodeViaGemmInvarianceTests : HeavyTestBase
{
    private const string ModelFile = "SmolLM2-360M-Instruct-Q4_K_M.gguf";

    private const string Prompt =
        "The history of the printing press begins in the fifteenth century, when Johannes Gutenberg " +
        "combined movable metal type, oil-based ink and a wooden screw press. Within fifty years";

    private static (double maxAbs, int identical, int argmaxAgree, int n) Compare(bool viaGemm)
    {
        var path = TeacherForcedParity.FindModel(ModelFile);
        Assert.SkipWhen(path is null, $"{ModelFile} is required.");
        using var modelHandle = SharedModelCacheFixture.Instance.Acquire(path!);
        var model = modelHandle.Model;
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var tokens = TeacherForcedParity.Tokenize(ModelFile, Prompt);

        bool saved = SimdKernels.DecodeViaGemm;
        SimdKernels.DecodeViaGemm = viaGemm;
        try
        {
            using var backend = new CpuBackend();
            var prefill = new float[tokens.Count][];
            using (var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 512))
                fwd.PrefillWithPerPositionLogits(tokens, 0, (pos, logits) => prefill[pos] = logits.ToArray());

            double maxAbs = 0;
            int identical = 0, agree = 0;
            using (var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 512))
                for (int i = 0; i < tokens.Count; i++)
                {
                    var dec = fwd.Forward(tokens[i], i);
                    bool same = true;
                    for (int v = 0; v < dec.Length; v++)
                    {
                        double d = Math.Abs(dec[v] - prefill[i][v]);
                        if (d > maxAbs) maxAbs = d;
                        if (BitConverter.SingleToInt32Bits(dec[v]) != BitConverter.SingleToInt32Bits(prefill[i][v])) same = false;
                    }
                    if (same) identical++;
                    if (Sampler.Greedy(dec) == Sampler.Greedy(prefill[i])) agree++;
                }
            return (maxAbs, identical, agree, tokens.Count);
        }
        finally { SimdKernels.DecodeViaGemm = saved; }
    }

    [Fact]
    public void DecodeViaGemm_DecodeMatchesPrefill_TeacherForced()
    {
        Assert.SkipUnless(SimdKernels.Q8PrefillEnabled, "needs the int8 prefill GEMM (STINGRAY_CPU_PREFILL_Q8 not 0)");
        var off = Compare(viaGemm: false);
        var on = Compare(viaGemm: true);
        Console.Error.WriteLine($"[DecodeViaGemm] {ModelFile}, {on.n} positions");
        Console.Error.WriteLine($"  matvec decode : maxAbs {off.maxAbs:E3}, bit-identical positions {off.identical}/{off.n}, argmax agree {off.argmaxAgree}/{off.n}");
        Console.Error.WriteLine($"  GEMM decode   : maxAbs {on.maxAbs:E3}, bit-identical positions {on.identical}/{on.n}, argmax agree {on.argmaxAgree}/{on.n}");

        Assert.Equal(on.n, on.argmaxAgree);
        Assert.True(on.maxAbs <= off.maxAbs, $"GEMM decode is further from prefill ({on.maxAbs:E3}) than the matvec ({off.maxAbs:E3})");
    }
}
