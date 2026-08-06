using System.Text.Json;
using OpenTail.Stingray.Cli;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.Cli;

public sealed class ExecutionPlanContractTests
{
    [Fact]
    public void Json_UsesStableStringDecisionEnumsAndKeepsRequestSeparateFromSelection()
    {
        var configuration = EffectiveConfigurationResolver.Resolve(
        [
            new EffectiveConfigurationSetting("backend", "auto",
            [new EffectiveConfigurationCandidate("default", "auto")])
        ]);
        var plan = new ExecutionPlan(
            1,
            new PlanRequest("model.gguf", "auto", -1, 8192, false, "f16", "auto", 1, false),
            "cpu", 0, 24, 8192, true,
            [new ExecutionPlanDecision(ExecutionPlanDecisionCodes.BackendSelection,
                PlanDecisionDisposition.Selected, PlanDiagnosticSeverity.Info,
                "Auto selected CPU.")],
            configuration)
        {
            ModelFormat = ModelFormat.SafeTensors
        };

        string json = JsonSerializer.Serialize(plan, StaticPlanJsonContext.Default.ExecutionPlan);

        Assert.Contains("\"backend\":\"auto\"", json);
        Assert.Contains("\"selected_backend\":\"cpu\"", json);
        Assert.Contains("\"disposition\":\"Selected\"", json);
        Assert.Contains("\"severity\":\"Info\"", json);
        Assert.Contains("\"model_format\":\"SafeTensors\"", json);
    }
}
