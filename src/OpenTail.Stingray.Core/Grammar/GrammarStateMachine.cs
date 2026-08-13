using System;
using System.Collections.Generic;

namespace OpenTail.Stingray.Core.Grammar;

/// <summary>
/// Lexical state nodes for JSON Schema grammar state machine.
/// </summary>
public enum JsonLexicalState
{
    RootObjectStart,
    ObjectKeyStart,
    ObjectKeyContent,
    KeyValueDelimiter,
    ValueStart,
    StringValue,
    NumberValue,
    BooleanValue,
    EnumValue,
    ArrayStart,
    ArrayValueStart,
    ArrayValueEnd,
    ValueEnd,
    ObjectEnd,
    Terminal
}

/// <summary>
/// Compact node representing a schema property definition.
/// </summary>
public sealed record SchemaPropertyNode(
    string Name,
    string Type,
    bool IsRequired,
    IReadOnlyList<string>? EnumValues = null,
    IReadOnlyList<SchemaPropertyNode>? ChildProperties = null,
    SchemaPropertyNode? ArrayItemSchema = null);

/// <summary>
/// Stack frame tracking nested object/array scope.
/// </summary>
public sealed class GrammarFrame
{
    public GrammarFrame(IReadOnlyList<SchemaPropertyNode>? properties)
    {
        Properties = properties ?? Array.Empty<SchemaPropertyNode>();
        RequiredProperties = new HashSet<string>(StringComparer.Ordinal);
        foreach (var p in Properties)
        {
            if (p.IsRequired)
            {
                RequiredProperties.Add(p.Name);
            }
        }
    }

    public IReadOnlyList<SchemaPropertyNode> Properties { get; }
    public HashSet<string> RequiredProperties { get; }
    public HashSet<string> EmittedProperties { get; } = new(StringComparer.Ordinal);
    public int ActivePropertyIndex { get; set; } = -1;
}

/// <summary>
/// Compact, token-aware state machine tracking JSON lexical boundaries and nested schema constraints.
/// </summary>
public sealed class GrammarStateMachine
{
    private readonly IReadOnlyList<SchemaPropertyNode> _rootProperties;
    private readonly Stack<GrammarFrame> _frameStack = new();

    public GrammarStateMachine(IReadOnlyList<SchemaPropertyNode> properties)
    {
        _rootProperties = properties ?? Array.Empty<SchemaPropertyNode>();
        _frameStack.Push(new GrammarFrame(_rootProperties));
        CurrentState = JsonLexicalState.RootObjectStart;
    }

    public JsonLexicalState CurrentState { get; private set; }
    public IReadOnlyList<SchemaPropertyNode> Properties => _rootProperties;
    public bool IsEscaped { get; set; }

    public GrammarFrame? CurrentFrame => _frameStack.Count > 0 ? _frameStack.Peek() : null;
    public bool IsTerminal => CurrentState == JsonLexicalState.Terminal;

    public void Reset()
    {
        _frameStack.Clear();
        _frameStack.Push(new GrammarFrame(_rootProperties));
        IsEscaped = false;
        CurrentState = JsonLexicalState.RootObjectStart;
    }

    public bool CanAcceptChar(char ch)
    {
        if (IsEscaped)
        {
            return true; // Any character allowed after backslash
        }

        return CurrentState switch
        {
            JsonLexicalState.RootObjectStart => ch == '{',
            JsonLexicalState.ObjectKeyStart => ch == '"' || ch == '}',
            JsonLexicalState.KeyValueDelimiter => ch == ':',
            JsonLexicalState.ValueStart => ch == '"' || ch == '{' || ch == '[' || char.IsDigit(ch) || ch == '-' || ch == 't' || ch == 'f' || ch == 'n',
            JsonLexicalState.ValueEnd => ch == ',' || ch == '}',
            JsonLexicalState.ObjectEnd => ch == '}' || ch == ',' || ch == ']',
            JsonLexicalState.Terminal => false,
            _ => true
        };
    }

    public void AdvanceLexicalState(JsonLexicalState newState)
    {
        CurrentState = newState;
    }

    /// <summary>Creates an independent lexical-state snapshot for validating a complete token.</summary>
    public GrammarStateMachine Clone()
    {
        var clone = new GrammarStateMachine(_rootProperties)
        {
            CurrentState = CurrentState,
            IsEscaped = IsEscaped
        };
        return clone;
    }

    /// <summary>Validates and consumes one character of the compact JSON lexical grammar.</summary>
    public bool TryAcceptChar(char c)
    {
        if (!CanAcceptChar(c)) return false;

        if (IsEscaped)
        {
            IsEscaped = false;
            return true;
        }

        if (c == '\\' && (CurrentState == JsonLexicalState.ObjectKeyContent || CurrentState == JsonLexicalState.StringValue))
        {
            IsEscaped = true;
            return true;
        }

        switch (CurrentState)
        {
            case JsonLexicalState.RootObjectStart when c == '{': AdvanceLexicalState(JsonLexicalState.ObjectKeyStart); break;
            case JsonLexicalState.ObjectKeyStart when c == '"': AdvanceLexicalState(JsonLexicalState.ObjectKeyContent); break;
            case JsonLexicalState.ObjectKeyStart when c == '}': AdvanceLexicalState(JsonLexicalState.Terminal); break;
            case JsonLexicalState.ObjectKeyContent when c == '"': AdvanceLexicalState(JsonLexicalState.KeyValueDelimiter); break;
            case JsonLexicalState.KeyValueDelimiter when c == ':': AdvanceLexicalState(JsonLexicalState.ValueStart); break;
            case JsonLexicalState.ValueStart when c == '"': AdvanceLexicalState(JsonLexicalState.StringValue); break;
            case JsonLexicalState.ValueStart when char.IsDigit(c) || c == '-': AdvanceLexicalState(JsonLexicalState.NumberValue); break;
            case JsonLexicalState.ValueStart when c == 't' || c == 'f': AdvanceLexicalState(JsonLexicalState.BooleanValue); break;
            case JsonLexicalState.StringValue when c == '"': AdvanceLexicalState(JsonLexicalState.ValueEnd); break;
            case JsonLexicalState.NumberValue or JsonLexicalState.BooleanValue when c == ',': AdvanceLexicalState(JsonLexicalState.ObjectKeyStart); break;
            case JsonLexicalState.NumberValue or JsonLexicalState.BooleanValue when c == '}': AdvanceLexicalState(JsonLexicalState.Terminal); break;
            case JsonLexicalState.ValueEnd when c == ',': AdvanceLexicalState(JsonLexicalState.ObjectKeyStart); break;
            case JsonLexicalState.ValueEnd when c == '}': AdvanceLexicalState(JsonLexicalState.Terminal); break;
        }
        return true;
    }

    public void PushFrame(IReadOnlyList<SchemaPropertyNode>? childProps)
    {
        _frameStack.Push(new GrammarFrame(childProps));
    }

    public bool PopFrame()
    {
        if (_frameStack.Count > 1)
        {
            _frameStack.Pop();
            return true;
        }
        return false;
    }

    public void RecordPropertyEmitted(string propName)
    {
        if (CurrentFrame is { } frame)
        {
            frame.EmittedProperties.Add(propName);
            for (int i = 0; i < frame.Properties.Count; i++)
            {
                if (string.Equals(frame.Properties[i].Name, propName, StringComparison.Ordinal))
                {
                    frame.ActivePropertyIndex = i;
                    break;
                }
            }
        }
    }

    public bool AreAllRequiredPropertiesEmitted()
    {
        if (CurrentFrame is { } frame)
        {
            foreach (var req in frame.RequiredProperties)
            {
                if (!frame.EmittedProperties.Contains(req))
                {
                    return false;
                }
            }
        }
        return true;
    }
}
