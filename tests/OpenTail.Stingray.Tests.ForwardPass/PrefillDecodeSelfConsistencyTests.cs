using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// Pins the <b>oracle-free</b> invariant that a whole-prompt prefill agrees with processing the
/// same prompt token by token: <c>Prefill(t0..tN)</c> must produce the logits that
/// <c>Prefill(t0)</c> followed by <c>Forward(t1..tN)</c> produces.
///
/// <para>
/// Why this shape of test earns its place next to the existing kernel suites: every other
/// cross-check here compares one backend against another, which can only tell you that two
/// implementations disagree — never which one is wrong. Self-consistency needs no reference
/// implementation, so it localises a defect to a single backend on its own. Prefill and decode
/// take genuinely different code paths (<see cref="SimdKernels.MatMulBatched"/> with
/// <c>allowQ8</c> versus per-token <c>MatVec</c>), and only prefill can take the int8 activation
/// path, so the two drifting apart is a real and previously untested failure mode.
/// </para>
///
/// <para>
/// <see cref="SimdKernels.Q8PrefillEnabled"/> is a process-wide static; it is saved and restored
/// around each test. This project's <c>xunit.runner.json</c> sets
/// <c>parallelizeTestCollections=false</c>, so that is sufficient — the same discipline
/// <see cref="MatMulBatchedQ8EquivalenceTests"/> documents.
/// </para>
/// </summary>
public sealed class PrefillDecodeSelfConsistencyTests : IDisposable
{
    private const string ModelFile = "SmolLM2-1.7B-Instruct-Q4_K_M.gguf";

    private readonly bool _savedQ8Gate = SimdKernels.Q8PrefillEnabled;

    public void Dispose() => SimdKernels.Q8PrefillEnabled = _savedQ8Gate;

    private static string? FindModelPath(string filename)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var candidate = Path.Combine(dir, "models", filename);
            if (File.Exists(candidate)) return candidate;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    /// <summary>Ordinary mid-vocabulary ids — a stand-in for real prose, not special tokens.</summary>
    private static int[] OrdinaryTokens(int count, int seed)
    {
        var rng = new Random(seed);
        var tokens = new int[count];
        for (int i = 0; i < tokens.Length; i++) tokens[i] = 100 + rng.Next(3000);
        return tokens;
    }

    private static int[] ControlTokens(GgufModel model, int count)
    {
        Assert.True(model.Metadata.TryGetValue("tokenizer.ggml.token_type", out object? raw)
            && raw is object[], "reference model must expose GGUF token types");
        var tokenTypes = (object[])raw!;
        var tokens = new List<int>(count);
        for (int i = 0; i < tokenTypes.Length && tokens.Count < count; i++)
        {
            int type = Convert.ToInt32(tokenTypes[i], System.Globalization.CultureInfo.InvariantCulture);
            if (type is TokenizerSource.ControlTokenType or TokenizerSource.UserDefinedTokenType)
                tokens.Add(i);
        }
        Assert.Equal(count, tokens.Count);
        return tokens.ToArray();
    }

    private static double Cosine(float[] a, float[] b)
    {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += (double)a[i] * b[i];
            na += (double)a[i] * a[i];
            nb += (double)b[i] * b[i];
        }
        return dot / (Math.Sqrt(na) * Math.Sqrt(nb));
    }

    /// <summary>Logits for the last prompt position, reached in one batched prefill.</summary>
    private static float[] ViaPrefill(GgufModel model, ModelHyperparams hp, int[] tokens)
    {
        using var backend = new CpuBackend();
        using var pass = new Engine.ForwardPass(model, backend, hp);
        return pass.Prefill(tokens).ToArray();
    }

    /// <summary>
    /// The per-sequence cache path used by <see cref="Engine.ContinuousBatchingEngine"/>. It is a
    /// separate public route from <see cref="Engine.ForwardPass.Prefill"/> and must retain the
    /// same special-token accuracy guard.
    /// </summary>
    private static float[] ViaPrefillWithCache(GgufModel model, ModelHyperparams hp, int[] tokens)
    {
        using var backend = new CpuBackend();
        using var pass = new Engine.ForwardPass(model, backend, hp);
        using var cache = pass.CreateCache();
        return pass.PrefillWithCache(tokens, cache).ToArray();
    }

    private static float[] ViaPrefillPacked(GgufModel model, ModelHyperparams hp, int[] tokens)
    {
        using var backend = new CpuBackend();
        using var pass = new Engine.ForwardPass(model, backend, hp);
        using var cache = pass.CreateCache();
        float[]?[] results = pass.PrefillPackedMulti([tokens], [0], [cache], [true]);
        return Assert.IsType<float[]>(results[0]);
    }

    /// <summary>The same logits, reached one token at a time. Never takes the int8 path.</summary>
    private static float[] ViaDecode(GgufModel model, ModelHyperparams hp, int[] tokens)
    {
        using var backend = new CpuBackend();
        using var pass = new Engine.ForwardPass(model, backend, hp);
        pass.Prefill(tokens[..1]);
        float[] logits = null!;
        for (int i = 1; i < tokens.Length; i++)
            logits = pass.Forward(tokens[i], i).ToArray();
        return logits;
    }

    /// <summary>
    /// With the int8 prefill gate off, the two routes run the same F32 arithmetic and must agree
    /// to floating-point noise. This is the invariant proper: any drift here is a genuine defect
    /// in batched prefill, not a quantisation trade-off.
    /// </summary>
    [Theory]
    [InlineData(2)]
    [InlineData(8)]
    [InlineData(33)]   // not a multiple of the 4/8-token dispatch groups
    public void F32Prefill_MatchesTokenByTokenDecode(int promptLength)
    {
        string? path = FindModelPath(ModelFile);
        if (path is null)
            Assert.Skip($"{ModelFile} not present under models/.");

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        int[] tokens = OrdinaryTokens(promptLength, seed: 3);

        SimdKernels.Q8PrefillEnabled = false;
        double cosine = Cosine(ViaDecode(model, hp, tokens), ViaPrefill(model, hp, tokens));

        Assert.True(cosine > 0.9999,
            $"F32 batched prefill diverged from token-by-token decode at {promptLength} tokens "
            + $"(cosine {cosine:F6}). Both routes run the same arithmetic with the int8 gate off, "
            + "so this is a batched-prefill defect, not a quantisation trade-off.");
    }

    /// <summary>
    /// With the gate on (the shipped default) prefill quantizes activations to int8 while decode
    /// stays in F32, so the two are NOT bit-identical by construction — see
    /// <see cref="SimdKernels.Q8PrefillEnabled"/>. They must still describe the same distribution.
    /// The bound is deliberately loose: this guards against the path collapsing, and is not a
    /// quality gate (perplexity is). Measured ≈0.988-0.999 on this model across 2-64 tokens.
    /// </summary>
    [Theory]
    [InlineData(2)]
    [InlineData(8)]
    [InlineData(33)]
    public void Q8Prefill_StaysCloseToTokenByTokenDecode(int promptLength)
    {
        string? path = FindModelPath(ModelFile);
        if (path is null)
            Assert.Skip($"{ModelFile} not present under models/.");

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        int[] tokens = OrdinaryTokens(promptLength, seed: 3);

        SimdKernels.Q8PrefillEnabled = true;
        double cosine = Cosine(ViaDecode(model, hp, tokens), ViaPrefill(model, hp, tokens));

        Assert.True(cosine > 0.98,
            $"int8 prefill diverged from token-by-token decode at {promptLength} tokens "
            + $"(cosine {cosine:F6}). Expected ≳0.99 — a collapse here means the int8 activation "
            + "path is broken for this shape, not merely lossy. STINGRAY_CPU_PREFILL_Q8=0 "
            + "disables it.");
    }

    /// <summary>
    /// An all-control prompt is a structural input rather than ordinary text. It previously drove
    /// the Q8 path to a negative final-logit cosine against decode; it must now use the exact F32
    /// sequential fallback even though the global Q8 gate remains enabled for normal prompts.
    /// </summary>
    [Fact]
    public void Q8Prefill_AllControlTokenPrompt_FallsBackToF32DecodeParity()
    {
        string? path = FindModelPath(ModelFile);
        if (path is null)
            Assert.Skip($"{ModelFile} not present under models/.");

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        int[] tokens = ControlTokens(model, count: 2);

        SimdKernels.Q8PrefillEnabled = true;
        double cosine = Cosine(ViaDecode(model, hp, tokens), ViaPrefill(model, hp, tokens));

        Assert.True(cosine > 0.9999,
            $"all-control prompt must use the F32 fallback when Q8 is enabled (cosine {cosine:F6}).");
    }

    [Fact]
    public void Q8PrefillWithCache_AllControlTokenPrompt_FallsBackToF32DecodeParity()
    {
        string? path = FindModelPath(ModelFile);
        if (path is null)
            Assert.Skip($"{ModelFile} not present under models/.");

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        int[] tokens = ControlTokens(model, count: 2);

        SimdKernels.Q8PrefillEnabled = true;
        double cosine = Cosine(ViaDecode(model, hp, tokens), ViaPrefillWithCache(model, hp, tokens));

        Assert.True(cosine > 0.9999,
            $"all-control prompt must use the F32 fallback through PrefillWithCache when Q8 is enabled " +
            $"(cosine {cosine:F6}).");
    }

    [Fact]
    public void Q8PrefillPacked_AllControlTokenPrompt_FallsBackToF32DecodeParity()
    {
        string? path = FindModelPath(ModelFile);
        if (path is null)
            Assert.Skip($"{ModelFile} not present under models/.");

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        int[] tokens = ControlTokens(model, count: 2);

        SimdKernels.Q8PrefillEnabled = true;
        double cosine = Cosine(ViaDecode(model, hp, tokens), ViaPrefillPacked(model, hp, tokens));

        Assert.True(cosine > 0.9999,
            $"all-control prompt must use the F32 fallback through PrefillPackedMulti when Q8 is enabled " +
            $"(cosine {cosine:F6}).");
    }
}
