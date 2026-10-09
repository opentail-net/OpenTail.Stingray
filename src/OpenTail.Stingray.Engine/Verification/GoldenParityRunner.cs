using System.Text;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Engine.Verification;

/// <summary>How a case compares with its reference.</summary>
public enum CaseVerdict
{
    /// <summary>Every compared token matches.</summary>
    Exact,
    /// <summary>Tokens differ, but only where our logits had the reference token within the tolerance of our own top choice (a genuine near-tie, as FP32 reordering produces).</summary>
    NearTie,
    /// <summary>At least one mismatch where our top choice beat the reference token by more than the tolerance.</summary>
    Diverged,
}

public sealed record ParityOptions
{
    /// <summary>Logit gap (our top-1 minus the reference token's logit) at or below which a mismatch counts as a near-tie.</summary>
    public double NearTieTolerance { get; init; } = 0.02;
    /// <summary>
    /// Reference margin (top-1 minus top-2 log-probability, nats) below which the reference itself was undecided, so a mismatch there is a near-tie. The default matches
    /// the <c>ConfidentMargin</c> the teacher-forced parity classes have always used: a match is demanded only where llama.cpp was confident.
    /// </summary>
    public double ConfidentMargin { get; init; } = 1.5;
    /// <summary>Bound for the stepwise-vs-prefill self-check, following the existing receipts' precedent for the int8 prefill approximation.</summary>
    public float StepwiseMaxAbsDiff { get; init; } = 5.0f;
    /// <summary>
    /// Also require the stepwise and single-pass argmax to agree. Off by default: if two paths differ by at most D everywhere, a flipped argmax means the two
    /// tokens were within 2D of each other, so a flip is fully explained by the measured difference and adds nothing beyond the <see cref="StepwiseMaxAbsDiff"/>
    /// bound (found on SmolLM2-135M, 2026-10-09: max diff 0.96, argmax flipped at a near-tie). The legacy parity classes required agreement; set this to keep that.
    /// </summary>
    public bool RequireStepwiseArgmaxAgreement { get; init; }
}

/// <summary>Precise execution timing recorded during golden verification.</summary>
/// <param name="PrefillDuration">Time spent inside <see cref="IForwardPass.Prefill"/>.</param>
/// <param name="PromptTokens">Number of prompt tokens evaluated during prefill.</param>
/// <param name="DecodeDuration">Cumulative time spent inside subsequent <see cref="IForwardPass.Forward"/> calls.</param>
/// <param name="DecodeSteps">Number of timed <see cref="IForwardPass.Forward"/> steps.</param>
public sealed record GoldenTimingInfo(
    TimeSpan PrefillDuration,
    int PromptTokens,
    TimeSpan DecodeDuration,
    int DecodeSteps)
{
    /// <summary>Pure decode throughput (timed Forward steps divided by accumulated decode duration).</summary>
    public double? DecodeTokensPerSecond =>
        DecodeSteps > 0 && DecodeDuration.TotalSeconds > 0
            ? DecodeSteps / DecodeDuration.TotalSeconds
            : null;

    /// <summary>Prompt processing throughput (prompt tokens evaluated divided by prefill duration).</summary>
    public double? PrefillTokensPerSecond =>
        PromptTokens > 0 && PrefillDuration.TotalSeconds > 0
            ? PromptTokens / PrefillDuration.TotalSeconds
            : null;
}

/// <param name="Index">Position in the continuation.</param>
/// <param name="Expected">Reference token.</param>
/// <param name="Actual">Our top choice.</param>
/// <param name="Gap">Our logit for <paramref name="Actual"/> minus our logit for <paramref name="Expected"/>; 0 means an exact tie.</param>
/// <param name="ReferenceMargin">The reference engine's own top-1 minus top-2 margin at this position (nats), when the golden recorded it.</param>
public sealed record PositionMismatch(int Index, int Expected, int Actual, double Gap, bool NearTie, double? ReferenceMargin = null);

/// <param name="ConfidentMatched">Positions that matched where the reference itself was confident (margin at or above the confident threshold).</param>
public sealed record GoldenCaseResult(
    string Name, string Mode, CaseVerdict Verdict, int Compared, int Matched, IReadOnlyList<PositionMismatch> Mismatches, IReadOnlyList<int> Generated,
    int ConfidentMatched = 0, int MinConfident = 0, GoldenTimingInfo? Timing = null)
{
    public PositionMismatch? FirstMismatch => Mismatches.Count > 0 ? Mismatches[0] : null;
    /// <summary>False when the case asked for a minimum number of confident matches and fewer were seen (too little evidence).</summary>
    public bool EvidenceOk => ConfidentMatched >= MinConfident;
}

/// <param name="ArgmaxGap">The single-pass logit gap between the two argmax tokens (0 when they agree); a flip is always within 2 x MaxAbsDiff.</param>
public sealed record StepwiseCheckResult(bool ArgmaxAgrees, float MaxAbsDiff, int StepwiseArgmax, int PrefillArgmax, bool WithinBound, float ArgmaxGap, bool RequireAgreement)
{
    public bool Passed => WithinBound && (ArgmaxAgrees || !RequireAgreement);
}

public sealed record GoldenRunResult(
    string Architecture, IReadOnlyList<GoldenCaseResult> Cases, StepwiseCheckResult? Stepwise, GoldenTimingInfo? Timing = null)
{
    /// <summary>The worst case verdict, or <see cref="CaseVerdict.Exact"/> when there are none.</summary>
    public CaseVerdict Verdict => Cases.Count == 0 ? CaseVerdict.Exact : Cases.Max(c => c.Verdict);

    /// <summary>True when no case diverged and the self-consistency check (if run) passed.</summary>
    public bool Passed => Verdict != CaseVerdict.Diverged && Cases.All(c => c.EvidenceOk) && (Stepwise is null || Stepwise.Passed);

    public string Format()
    {
        var sb = new StringBuilder();
        sb.Append(Architecture).Append(": ").Append(Passed ? "PASS" : "FAIL").Append(" (").Append(Verdict).AppendLine(")");
        foreach (var c in Cases)
        {
            sb.Append("  case '").Append(c.Name).Append("' [").Append(c.Mode).Append("]: ").Append(c.Verdict)
              .Append(", ").Append(c.Matched).Append('/').Append(c.Compared).Append(" matched")
              .Append(c.MinConfident > 0 ? $", {c.ConfidentMatched} confident (need {c.MinConfident}){(c.EvidenceOk ? "" : " INSUFFICIENT EVIDENCE")}" : "");
            if (c.Timing?.DecodeTokensPerSecond is { } dSpeed)
                sb.Append($", decode {dSpeed.ToString("F1", System.Globalization.CultureInfo.InvariantCulture)} t/s");
            foreach (var m in c.Mismatches.Take(5))
                sb.Append("; @").Append(m.Index).Append(" expected ").Append(m.Expected).Append(" got ").Append(m.Actual)
                  .Append(" gap ").Append(m.Gap.ToString("0.####", System.Globalization.CultureInfo.InvariantCulture))
                  .Append(m.ReferenceMargin is { } rmg ? $" ref-margin {rmg.ToString("0.###", System.Globalization.CultureInfo.InvariantCulture)}" : "")
                  .Append(m.NearTie ? " (near-tie)" : " (DIVERGED)");
            sb.AppendLine();
        }
        if (Stepwise is not null)
            sb.Append("  stepwise-vs-prefill: argmax ").Append(Stepwise.ArgmaxAgrees ? "agrees" : $"DISAGREES ({Stepwise.StepwiseArgmax} vs {Stepwise.PrefillArgmax}, gap {Stepwise.ArgmaxGap.ToString("0.####", System.Globalization.CultureInfo.InvariantCulture)}; {(Stepwise.RequireAgreement ? "REQUIRED" : "informational")})")
              .Append(", max |diff| ").Append(Stepwise.MaxAbsDiff.ToString("0.####", System.Globalization.CultureInfo.InvariantCulture))
              .AppendLine(Stepwise.WithinBound ? "" : " (OVER BOUND)");
        return sb.ToString();
    }
}

/// <summary>
/// Runs a <see cref="GoldenFile"/> against a forward pass and reports structured results; it never asserts, so the same logic serves the
/// heavy tests, <c>admit-arch --golden</c> and tooling. The forward pass is supplied as a factory because each check needs fresh state.
/// </summary>
public static class GoldenParityRunner
{
    public static GoldenRunResult Run(GoldenFile golden, Func<IForwardPass> create, ParityOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(golden);
        ArgumentNullException.ThrowIfNull(create);
        options ??= new ParityOptions();

        var cases = new List<GoldenCaseResult>();
        TimeSpan totalPrefill = TimeSpan.Zero;
        int totalPromptTokens = 0;
        TimeSpan totalDecode = TimeSpan.Zero;
        int totalDecodeSteps = 0;

        foreach (var c in golden.Cases)
        {
            using var fwd = create();
            var caseResult = RunCase(c, fwd, options);
            cases.Add(caseResult);
            if (caseResult.Timing is { } t)
            {
                totalPrefill += t.PrefillDuration;
                totalPromptTokens += t.PromptTokens;
                totalDecode += t.DecodeDuration;
                totalDecodeSteps += t.DecodeSteps;
            }
        }

        StepwiseCheckResult? stepwise = null;
        var basis = golden.Cases.FirstOrDefault(c => c.PromptTokens.Length > 0 && c.Tokens.Length >= 2);
        if (basis is not null)
            stepwise = CheckStepwiseVsPrefill(create, basis.PromptTokens, basis.Tokens[0], basis.Tokens[1], options);

        var aggregateTiming = new GoldenTimingInfo(totalPrefill, totalPromptTokens, totalDecode, totalDecodeSteps);
        return new GoldenRunResult(golden.Architecture, cases, stepwise, aggregateTiming);
    }

    public static GoldenCaseResult RunCase(GoldenCase c, IForwardPass fwd, ParityOptions options)
    {
        if (c.PromptTokens.Length == 0) throw new InvalidDataException($"Golden case '{c.Name}' has no prompt tokens.");
        bool teacher = string.Equals(c.Mode, "teacherForced", StringComparison.OrdinalIgnoreCase);
        // A free-running case that hits a *near-tie* switches to teacher forcing for the rest of the continuation: our own token and the
        // reference's differ there, so the two sequences stop being comparable, but feeding the reference token lets every later position still be
        // checked. (A confident mismatch still ends the comparison: the sequences have genuinely diverged.)
        bool forceReference = teacher;
        int n = c.Tokens.Length;
        var mismatches = new List<PositionMismatch>();
        var generated = new List<int>(n);

        long prefillStart = Stopwatch.GetTimestamp();
        var logits = fwd.Prefill(c.PromptTokens);
        TimeSpan prefillDuration = Stopwatch.GetElapsedTime(prefillStart);
        int promptTokensCount = c.PromptTokens.Length;

        TimeSpan decodeDuration = TimeSpan.Zero;
        int decodeSteps = 0;

        int pos = c.PromptTokens.Length;
        int matched = 0;
        int compared = 0;
        int confidentMatched = 0;
        for (int i = 0; i < n; i++)
        {
            int vocab = Math.Min(logits.Length, fwd.VocabSize);
            int top = ArgMax(logits, vocab);
            generated.Add(top);
            compared++;
            if (top == c.Tokens[i])
            {
                matched++;
                if (c.Margins is { } cm && i < cm.Length && cm[i] >= options.ConfidentMargin) confidentMatched++;
            }
            else
            {
                double gap = c.Tokens[i] >= 0 && c.Tokens[i] < vocab ? logits[top] - logits[c.Tokens[i]] : double.PositiveInfinity;
                double? refMargin = c.Margins is { } ms && i < ms.Length ? ms[i] : null;
                bool nearTie = gap <= options.NearTieTolerance || (refMargin is { } rm && rm < options.ConfidentMargin);
                mismatches.Add(new PositionMismatch(i, c.Tokens[i], top, gap, nearTie, refMargin));
                if (!forceReference)
                {
                    if (!nearTie) break;          // confident mismatch in free mode: genuinely diverged, nothing further is comparable
                    forceReference = true;        // near-tie: carry on, teacher-forced
                }
            }
            if (i + 1 < n)
            {
                long forwardStart = Stopwatch.GetTimestamp();
                logits = fwd.Forward(forceReference ? c.Tokens[i] : top, pos++);
                decodeDuration += Stopwatch.GetElapsedTime(forwardStart);
                decodeSteps++;
            }
        }

        var timing = new GoldenTimingInfo(prefillDuration, promptTokensCount, decodeDuration, decodeSteps);
        var verdict = mismatches.Count == 0 ? CaseVerdict.Exact
            : mismatches.All(m => m.NearTie) ? CaseVerdict.NearTie
            : CaseVerdict.Diverged;
        return new GoldenCaseResult(c.Name, teacher ? "teacherForced" : "free", verdict, compared, matched, mismatches, generated, confidentMatched, c.MinConfident, timing);
    }

    /// <summary>
    /// Model-independent consistency: the last-position logits from a step-by-step decode must agree with a single-pass prefill of the same
    /// tokens (same argmax, and a bounded absolute difference). Mirrors the check each parity class used to carry.
    /// </summary>
    public static StepwiseCheckResult CheckStepwiseVsPrefill(Func<IForwardPass> create, int[] prompt, int next1, int next2, ParityOptions options)
    {
        int[] full = [.. prompt, next1, next2];
        float[] stepwise;
        using (var fwd = create())
        {
            fwd.Prefill(prompt);
            fwd.Forward(next1, prompt.Length);
            var last = fwd.Forward(next2, prompt.Length + 1);
            stepwise = last[..Math.Min(last.Length, fwd.VocabSize)].ToArray();
        }
        float[] single;
        using (var fwd = create())
        {
            var last = fwd.Prefill(full);
            single = last[..Math.Min(last.Length, fwd.VocabSize)].ToArray();
        }

        int a = ArgMax(stepwise, stepwise.Length), b = ArgMax(single, single.Length);
        float maxDiff = 0;
        for (int i = 0; i < Math.Min(stepwise.Length, single.Length); i++)
            maxDiff = Math.Max(maxDiff, Math.Abs(stepwise[i] - single[i]));
        float gap = a == b ? 0f : Math.Abs(single[b] - single[a]);
        return new StepwiseCheckResult(a == b, maxDiff, a, b, maxDiff < options.StepwiseMaxAbsDiff, gap, options.RequireStepwiseArgmaxAgreement);
    }

    private static int ArgMax(ReadOnlySpan<float> logits, int count)
    {
        int best = 0;
        for (int i = 1; i < count; i++)
            if (logits[i] > logits[best]) best = i;
        return best;
    }
}
