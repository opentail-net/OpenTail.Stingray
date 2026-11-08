using System.Text.Json;
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
    public static List<string> Errors(IReadOnlyList<ParsedToolCall> calls, OaiTool[]? declared) =>
        Errors(calls, declared?
            .Where(t => t.Function is not null)
            .Select(t => (t.Function!.Name, t.Function.Parameters)) ?? []);

    public static List<string> Errors(IReadOnlyList<ParsedToolCall> calls, IEnumerable<(string Name, JsonElement? Parameters)> declared)
    {
        var tools = declared.ToList();
        var errors = new List<string>();
        foreach (var call in calls)
        {
            var tool = tools.FirstOrDefault(t => string.Equals(t.Name, call.Name, StringComparison.Ordinal));
            if (tool.Name is null)
            {
                errors.Add($"tool '{call.Name}' is not declared in this request");
                continue;
            }
            var schema = ToolSchema.FromOpenAiFunction(tool.Name, tool.Parameters);
            errors.AddRange(ToolCallArgumentValidator.Validate(call, schema));
        }
        return errors;
    }
}
