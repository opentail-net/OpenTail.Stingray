namespace OpenTail.Stingray.Tests.Cli;

/// <summary>
/// <c>run --auto</c> used to crash: the plan stores its KV dtype as <c>DType.ToString().ToLowerInvariant()</c> ("float32",
/// "bfloat16") and the CLI wrote that straight into STINGRAY_KV_DTYPE, whose parser only accepts "fp32", "bf16", "q8_0".
/// Found while baselining the RunCommand split; <see cref="ExecutionPlan.KvDtypeToEnvValue"/> is the single translation.
/// </summary>
public sealed class ExecutionPlanKvDtypeTests
{
    [Theory]
    [InlineData("float32", "fp32")]
    [InlineData("Float32", "fp32")]
    [InlineData("bfloat16", "bf16")]
    [InlineData("q8_0", "q8_0")]
    [InlineData("fp32", "fp32")]
    [InlineData("bf16", "bf16")]
    [InlineData("q8", "q8_0")]
    public void PlanValuesAreTranslatedToTheEngineVocabulary(string plan, string expected) =>
        Assert.Equal(expected, ExecutionPlan.KvDtypeToEnvValue(plan));

    [Fact]
    public void UnrecognisedValuesPassThroughSoTheEngineStillReportsThem() =>
        Assert.Equal("float64", ExecutionPlan.KvDtypeToEnvValue("float64"));

    /// <summary>
    /// Every dtype the planner can emit (the builder serialises <c>DType.ToString().ToLowerInvariant()</c> for float32,
    /// bfloat16 and q8_0) must, once translated, be accepted by the engine's own parser. This is the check that was missing.
    /// </summary>
    [Fact]
    public void EveryDtypeThePlannerCanEmit_IsAcceptedByTheEnginesParserOnceTranslated()
    {
        string? original = Environment.GetEnvironmentVariable("STINGRAY_KV_DTYPE");
        try
        {
            foreach (var dtype in new[] { DType.Float32, DType.BFloat16, DType.Q8_0 })
            {
                string planValue = dtype.ToString().ToLowerInvariant();            // exactly how ExecutionPlanBuilder serialises it
                Environment.SetEnvironmentVariable("STINGRAY_KV_DTYPE", ExecutionPlan.KvDtypeToEnvValue(planValue));
                Assert.Equal(dtype, CudaForwardPass.ResolveConfiguredKvDTypeOrNull());

                // And the untranslated value is the bug: the parser rejects it (for the two renamed dtypes).
                if (planValue is "float32" or "bfloat16")
                {
                    Environment.SetEnvironmentVariable("STINGRAY_KV_DTYPE", planValue);
                    Assert.Throws<ArgumentException>(() => CudaForwardPass.ResolveConfiguredKvDTypeOrNull());
                }
            }
        }
        finally { Environment.SetEnvironmentVariable("STINGRAY_KV_DTYPE", original); }
    }
}
