using System;
using System.Collections.Generic;
using System.Linq;

namespace OpenTail.Stingray.Core.Tools;

/// <summary>
/// Contextual metadata driving capability filtering for an inference session.
/// </summary>
public sealed record InferenceToolContext(
    string? Task = null,
    string? Mode = null,
    IReadOnlySet<string>? AllowedTools = null);

/// <summary>
/// Abstraction supplying permitted tool capabilities to an inference session.
/// </summary>
public interface IToolProvider
{
    /// <summary>
    /// Returns the active list of permitted tool definitions for the given context.
    /// </summary>
    IReadOnlyList<ToolDefinition> GetTools(InferenceToolContext? context = null);
}

/// <summary>
/// Composite tool provider aggregating multiple IToolProvider sources with deduplication and context filtering.
/// </summary>
public sealed class CompositeToolProvider : IToolProvider
{
    private readonly IReadOnlyList<IToolProvider> _providers;

    public CompositeToolProvider(IEnumerable<IToolProvider> providers)
    {
        ArgumentNullException.ThrowIfNull(providers);
        // Guard against accidental null elements from callers and materialize the list
        _providers = providers.Where(p => p is not null).ToList();
    }

    public CompositeToolProvider(params IToolProvider[] providers)
        : this((IEnumerable<IToolProvider>)providers) { }

    public IReadOnlyList<ToolDefinition> GetTools(InferenceToolContext? context = null)
    {
        var result = new List<ToolDefinition>();
        var seenNames = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        // Normalize allowed-tools to a case-insensitive set for consistent filtering
        HashSet<string>? allowed = context?.AllowedTools is not null
            ? new HashSet<string>(context.AllowedTools, StringComparer.OrdinalIgnoreCase)
            : null;
        var mode = context?.Mode;

        foreach (var provider in _providers)
        {
            if (provider is null)
                continue;

            var tools = provider.GetTools(context) ?? Array.Empty<ToolDefinition>();
            foreach (var tool in tools)
            {
                if (tool is null)
                    continue;

                var name = tool.Name;
                if (string.IsNullOrEmpty(name))
                    continue;

                // Context filtering
                if (allowed is not null && !allowed.Contains(name))
                {
                    continue;
                }

                // Mode filtering heuristics (e.g., CodeReview excludes destructive tools)
                if (string.Equals(mode, "CodeReview", StringComparison.OrdinalIgnoreCase))
                {
                    if (tool.Annotations?.DestructiveHint == true ||
                        name.StartsWith("write_", StringComparison.OrdinalIgnoreCase) ||
                        name.StartsWith("delete_", StringComparison.OrdinalIgnoreCase) ||
                        name.StartsWith("git_push", StringComparison.OrdinalIgnoreCase) ||
                        name.StartsWith("git_commit", StringComparison.OrdinalIgnoreCase))
                    {
                        continue;
                    }
                }

                if (seenNames.Add(name))
                {
                    result.Add(tool);
                }
            }
        }

        return result;
    }
}

/// <summary>
/// Simple in-memory tool provider backed by a static or dynamic collection of tool definitions.
/// </summary>
public sealed class MemoryToolProvider : IToolProvider
{
    private readonly IReadOnlyList<ToolDefinition> _tools;

    public MemoryToolProvider(IEnumerable<ToolDefinition> tools)
    {
        ArgumentNullException.ThrowIfNull(tools);
        // Filter out any accidental null entries and materialize
        _tools = tools.Where(t => t is not null).ToList();
    }

    public IReadOnlyList<ToolDefinition> GetTools(InferenceToolContext? context = null)
    {
        if (context?.AllowedTools is null && string.IsNullOrEmpty(context?.Mode))
        {
            return _tools;
        }

        var allowed = context?.AllowedTools is not null
            ? new HashSet<string>(context.AllowedTools, StringComparer.OrdinalIgnoreCase)
            : null;
        var mode = context?.Mode;

        return _tools.Where(t =>
        {
            if (t is null)
                return false;

            var name = t.Name;
            if (allowed is not null && !allowed.Contains(name))
                return false;

            if (string.Equals(mode, "CodeReview", StringComparison.OrdinalIgnoreCase))
            {
                if (t.Annotations?.DestructiveHint == true ||
                    name.StartsWith("write_", StringComparison.OrdinalIgnoreCase) ||
                    name.StartsWith("delete_", StringComparison.OrdinalIgnoreCase) ||
                    name.StartsWith("git_push", StringComparison.OrdinalIgnoreCase) ||
                    name.StartsWith("git_commit", StringComparison.OrdinalIgnoreCase))
                    return false;
            }

            return true;
        }).ToList();
    }
}
