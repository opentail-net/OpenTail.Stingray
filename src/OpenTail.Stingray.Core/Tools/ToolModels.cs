using System;
using System.Collections.Generic;
using System.Text.Json;

namespace OpenTail.Stingray.Core.Tools;

/// <summary>
/// Annotations providing operational hints about a tool's execution characteristics (MCP 2.0 standard aligned).
/// </summary>
public sealed record ToolAnnotations(
    bool ReadOnlyHint = false,
    bool DestructiveHint = false,
    bool IdempotentHint = false,
    bool OpenWorldHint = false);

/// <summary>
/// Canonical MCP-aligned tool capability definition exposed to model inference sessions.
/// </summary>
public sealed record ToolDefinition(
    string Name,
    string? Description = null,
    JsonElement InputSchema = default,
    JsonElement? OutputSchema = null,
    ToolAnnotations? Annotations = null,
    string? Title = null) : ITool;

/// <summary>
/// Structured tool call emitted during model generation or via API continuation.
/// </summary>
public sealed record ToolCall(
    string Id,
    string Name,
    JsonElement Arguments,
    bool IsAuthorized = true);

/// <summary>
/// Execution result of a tool call returned from host/application execution into Stingray.
/// </summary>
public sealed record ToolResult(
    string ToolCallId,
    JsonElement Content,
    bool IsError = false);
