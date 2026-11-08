using OpenTail.Stingray.Core;
using OpenTail.Stingray.Core.Grammar;
using OpenTail.Stingray.Server.Endpoints;

namespace OpenTail.Stingray.Server;

/// <summary>
/// Checks parsed tool calls against the tools the request declared. A call to an undeclared tool, or with arguments that
/// break the declared schema, is a problem. The server returns these as errors, not as tool calls (owner decision D3).
/// </summary>
internal static class ToolCallValidation
{
    public static List<string> Errors(IReadOnlyList<ParsedToolCall> calls, OaiTool[]? declared)
    {
        var errors = new List<string>();
        foreach (var call in calls)
        {
            var tool = declared?.FirstOrDefault(t => string.Equals(t.Function?.Name, call.Name, StringComparison.Ordinal));
            if (tool?.Function is null)
            {
                errors.Add($"tool '{call.Name}' is not declared in this request");
                continue;
            }
            var schema = ToolSchema.FromOpenAiFunction(tool.Function.Name, tool.Function.Parameters);
            errors.AddRange(ToolCallArgumentValidator.Validate(call, schema));
        }
        return errors;
    }
}
