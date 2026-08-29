
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
    NullValue,
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
/// Stack frame tracking nested object/array scope. A lightweight value type: pushed/popped
/// during grammar validation, including once per candidate-token evaluation in
/// <see cref="JsonSchemaGrammarMasker"/>'s per-vocabulary-entry loop (every decode step), so it
/// must stay cheaply copyable without its own heap allocations -- required/emitted property
/// tracking uses a bitmask instead of a <c>HashSet&lt;string&gt;</c>, which bounds schemas to 64
/// tracked properties per nesting level. That matches this grammar's stated scope (fast, narrow,
/// inference-focused -- not a general schema engine, see the class doc comment below); a
/// property beyond index 63 is simply not required-tracked rather than causing an error.
/// </summary>
public struct GrammarFrame
{
    public IReadOnlyList<SchemaPropertyNode> Properties;
    public long RequiredMask;
    public long EmittedMask;
    public int ActivePropertyIndex;
    public SchemaPropertyNode? ArrayItemSchema;
    public JsonLexicalState ReturnState;

    public static GrammarFrame ForObject(IReadOnlyList<SchemaPropertyNode>? properties, JsonLexicalState returnState)
    {
        properties ??= Array.Empty<SchemaPropertyNode>();
        long required = 0;
        int limit = Math.Min(properties.Count, 64);
        for (int i = 0; i < limit; i++)
        {
            if (properties[i].IsRequired)
            {
                required |= 1L << i;
            }
        }
        return new GrammarFrame
        {
            Properties = properties,
            RequiredMask = required,
            EmittedMask = 0,
            ActivePropertyIndex = -1,
            ArrayItemSchema = null,
            ReturnState = returnState
        };
    }

    public static GrammarFrame ForArray(SchemaPropertyNode? itemSchema, JsonLexicalState returnState)
    {
        return new GrammarFrame
        {
            Properties = Array.Empty<SchemaPropertyNode>(),
            RequiredMask = 0,
            EmittedMask = 0,
            ActivePropertyIndex = -1,
            ArrayItemSchema = itemSchema,
            ReturnState = returnState
        };
    }

    public readonly bool IsArray => ArrayItemSchema != null;

    public readonly bool AreAllRequiredEmitted => (RequiredMask & ~EmittedMask) == 0;

    /// <summary>
    /// The schema node governing the value currently being written in this frame: the active
    /// object property (by key), or -- for an array frame -- the array's own per-element schema.
    /// </summary>
    public readonly SchemaPropertyNode? ActiveValueSchema =>
        ArrayItemSchema is { } item ? item
        : ActivePropertyIndex >= 0 && ActivePropertyIndex < Properties.Count ? Properties[ActivePropertyIndex]
        : null;
}

/// <summary>
/// Compact, token-aware state machine tracking JSON lexical boundaries and nested schema constraints.
/// </summary>
public sealed class GrammarStateMachine
{
    private readonly IReadOnlyList<SchemaPropertyNode> _rootProperties;
    private readonly List<GrammarFrame> _frames = new();
    private readonly StringBuilder _keyBuilder = new();
    private readonly StringBuilder _valueBuilder = new();

    public GrammarStateMachine(IReadOnlyList<SchemaPropertyNode> properties)
    {
        _rootProperties = properties ?? Array.Empty<SchemaPropertyNode>();
        _frames.Add(GrammarFrame.ForObject(_rootProperties, JsonLexicalState.Terminal));
        CurrentState = JsonLexicalState.RootObjectStart;
    }

    public JsonLexicalState CurrentState { get; private set; }
    public IReadOnlyList<SchemaPropertyNode> Properties => _rootProperties;
    public bool IsEscaped { get; set; }

    public GrammarFrame? CurrentFrame => _frames.Count > 0 ? _frames[^1] : null;
    public bool IsTerminal => CurrentState == JsonLexicalState.Terminal;

    private bool CurrentIsArray => _frames.Count > 0 && _frames[^1].IsArray;

    /// <summary>State to return to once the value currently being opened (by a '{' or '[') closes,
    /// based on the ENCLOSING frame active right now -- object values return to
    /// <see cref="JsonLexicalState.ValueEnd"/>, array elements to <see cref="JsonLexicalState.ArrayValueEnd"/>.</summary>
    private JsonLexicalState ValueCompleteState => CurrentIsArray ? JsonLexicalState.ArrayValueEnd : JsonLexicalState.ValueEnd;

    /// <summary>State to move to after a ',' following a scalar value -- next object key, or next array element.</summary>
    private JsonLexicalState NextItemOrKeyState => CurrentIsArray ? JsonLexicalState.ArrayValueStart : JsonLexicalState.ObjectKeyStart;

    public void Reset()
    {
        _frames.Clear();
        _frames.Add(GrammarFrame.ForObject(_rootProperties, JsonLexicalState.Terminal));
        IsEscaped = false;
        CurrentState = JsonLexicalState.RootObjectStart;
        _keyBuilder.Clear();
        _valueBuilder.Clear();
    }

    public bool CanAcceptChar(char ch)
    {
        if (IsEscaped)
        {
            return true; // Any character allowed after backslash
        }

        bool inArray = CurrentIsArray;

        return CurrentState switch
        {
            JsonLexicalState.RootObjectStart => ch == '{',
            JsonLexicalState.ObjectKeyStart => ch == '"' || (ch == '}' && AreAllRequiredPropertiesEmitted()),
            JsonLexicalState.KeyValueDelimiter => ch == ':',
            JsonLexicalState.ValueStart or JsonLexicalState.ArrayValueStart =>
                ch == '"' || ch == '{' || ch == '[' || char.IsDigit(ch) || ch == '-' || ch == 't' || ch == 'f' || ch == 'n',
            JsonLexicalState.ArrayStart =>
                ch == ']' || ch == '"' || ch == '{' || ch == '[' || char.IsDigit(ch) || ch == '-' || ch == 't' || ch == 'f' || ch == 'n',
            // Loose scalar spelling (matches the pre-existing boolean behavior this mirrors): any
            // character continues the literal except the two bracket closers, and ',' is always a
            // boundary. '}' only legal outside an array and only once required properties are in;
            // ']' only legal inside an array.
            JsonLexicalState.NumberValue or JsonLexicalState.BooleanValue or JsonLexicalState.NullValue =>
                inArray ? ch != '}' : ch != ']' && (ch != '}' || AreAllRequiredPropertiesEmitted()),
            JsonLexicalState.ValueEnd => ch == ',' || (ch == '}' && AreAllRequiredPropertiesEmitted()),
            JsonLexicalState.ArrayValueEnd => ch == ',' || ch == ']',
            JsonLexicalState.ObjectEnd => ch == '}' || ch == ',' || ch == ']',
            JsonLexicalState.Terminal => false,
            _ => true
        };
    }

    public void AdvanceLexicalState(JsonLexicalState newState)
    {
        CurrentState = newState;
    }

    /// <summary>
    /// Creates an independent lexical-state snapshot for validating a complete token. Allocates --
    /// prefer <see cref="CopyFrom"/> against a single reused instance in hot loops (see
    /// <see cref="JsonSchemaGrammarMasker"/>'s per-vocab-token candidate evaluation, which used to
    /// call this once per vocabulary entry per decode step).
    /// </summary>
    public GrammarStateMachine Clone()
    {
        var clone = new GrammarStateMachine(_rootProperties);
        clone.CopyFrom(this);
        return clone;
    }

    /// <summary>
    /// Overwrites this instance's mutable state from <paramref name="source"/> without allocating
    /// -- reuses this instance's own frame list and string builders rather than creating new ones.
    /// <paramref name="source"/> must have been built from the same root properties as this
    /// instance. Intended for a single reused "scratch" instance validating many candidate tokens
    /// per decode step.
    /// </summary>
    public void CopyFrom(GrammarStateMachine source)
    {
        ArgumentNullException.ThrowIfNull(source);

        CurrentState = source.CurrentState;
        IsEscaped = source.IsEscaped;

        _frames.Clear();
        _frames.AddRange(source._frames); // GrammarFrame is a struct: value-copied, no aliasing.

        _keyBuilder.Clear();
        _keyBuilder.Append(source._keyBuilder);
        _valueBuilder.Clear();
        _valueBuilder.Append(source._valueBuilder);
    }

    /// <summary>Validates and consumes one character of the compact JSON lexical grammar.</summary>
    public bool TryAcceptChar(char c)
    {
        if (!CanAcceptChar(c)) return false;

        if (IsEscaped)
        {
            IsEscaped = false;
            if (CurrentState == JsonLexicalState.ObjectKeyContent) _keyBuilder.Append(c);
            else if (CurrentState == JsonLexicalState.StringValue) _valueBuilder.Append(c);
            return true;
        }

        if (c == '\\' && (CurrentState == JsonLexicalState.ObjectKeyContent || CurrentState == JsonLexicalState.StringValue))
        {
            IsEscaped = true;
            return true;
        }

        // Ordinary key/string content (not the delimiting quote) -- accumulate for
        // RecordPropertyEmitted (keys) / enum exact-match validation (values). Not part of the
        // structural switch below since these are the two states with unbounded "any char until
        // the closing quote" runs.
        if (CurrentState == JsonLexicalState.ObjectKeyContent && c != '"')
        {
            _keyBuilder.Append(c);
            return true;
        }

        if (CurrentState == JsonLexicalState.StringValue && c != '"')
        {
            var enumValues = _frames.Count > 0 ? _frames[^1].ActiveValueSchema?.EnumValues : null;
            if (enumValues is { Count: > 0 })
            {
                _valueBuilder.Append(c);
                if (!HasEnumPrefixMatch(enumValues, _valueBuilder))
                {
                    _valueBuilder.Length--; // undo the speculative append; this char doesn't lead anywhere legal
                    return false;
                }
            }
            else
            {
                _valueBuilder.Append(c);
            }
            return true;
        }

        switch (CurrentState)
        {
            case JsonLexicalState.RootObjectStart when c == '{':
                AdvanceLexicalState(JsonLexicalState.ObjectKeyStart);
                break;

            case JsonLexicalState.ObjectKeyStart when c == '"':
                _keyBuilder.Clear();
                AdvanceLexicalState(JsonLexicalState.ObjectKeyContent);
                break;

            case JsonLexicalState.ObjectKeyStart when c == '}':
                CloseObject();
                break;

            case JsonLexicalState.ObjectKeyContent when c == '"':
                RecordPropertyEmitted(_keyBuilder.ToString());
                AdvanceLexicalState(JsonLexicalState.KeyValueDelimiter);
                break;

            case JsonLexicalState.KeyValueDelimiter when c == ':':
                AdvanceLexicalState(JsonLexicalState.ValueStart);
                break;

            case JsonLexicalState.ValueStart or JsonLexicalState.ArrayValueStart or JsonLexicalState.ArrayStart when c == '"':
                _valueBuilder.Clear();
                AdvanceLexicalState(JsonLexicalState.StringValue);
                break;

            case JsonLexicalState.ValueStart or JsonLexicalState.ArrayValueStart or JsonLexicalState.ArrayStart
                when char.IsDigit(c) || c == '-':
                AdvanceLexicalState(JsonLexicalState.NumberValue);
                break;

            case JsonLexicalState.ValueStart or JsonLexicalState.ArrayValueStart or JsonLexicalState.ArrayStart
                when c == 't' || c == 'f':
                AdvanceLexicalState(JsonLexicalState.BooleanValue);
                break;

            case JsonLexicalState.ValueStart or JsonLexicalState.ArrayValueStart or JsonLexicalState.ArrayStart when c == 'n':
                AdvanceLexicalState(JsonLexicalState.NullValue);
                break;

            case JsonLexicalState.ValueStart or JsonLexicalState.ArrayValueStart or JsonLexicalState.ArrayStart when c == '{':
                PushObjectFrameForValue();
                AdvanceLexicalState(JsonLexicalState.ObjectKeyStart);
                break;

            case JsonLexicalState.ValueStart or JsonLexicalState.ArrayValueStart or JsonLexicalState.ArrayStart when c == '[':
                PushArrayFrameForValue();
                AdvanceLexicalState(JsonLexicalState.ArrayStart);
                break;

            case JsonLexicalState.ArrayStart when c == ']':
                CloseArray();
                break;

            case JsonLexicalState.StringValue when c == '"':
                if (!CanCloseStringValue()) return false;
                AdvanceLexicalState(ValueCompleteState);
                break;

            case JsonLexicalState.NumberValue or JsonLexicalState.BooleanValue or JsonLexicalState.NullValue when c == ',':
                AdvanceLexicalState(NextItemOrKeyState);
                break;

            case JsonLexicalState.NumberValue or JsonLexicalState.BooleanValue or JsonLexicalState.NullValue when c == '}':
                CloseObject();
                break;

            case JsonLexicalState.NumberValue or JsonLexicalState.BooleanValue or JsonLexicalState.NullValue when c == ']':
                CloseArray();
                break;

            case JsonLexicalState.ValueEnd when c == ',':
                AdvanceLexicalState(JsonLexicalState.ObjectKeyStart);
                break;

            case JsonLexicalState.ValueEnd when c == '}':
                CloseObject();
                break;

            case JsonLexicalState.ArrayValueEnd when c == ',':
                AdvanceLexicalState(JsonLexicalState.ArrayValueStart);
                break;

            case JsonLexicalState.ArrayValueEnd when c == ']':
                CloseArray();
                break;
        }
        return true;
    }

    /// <summary>Whether the accumulated string content is a legal value to close right now --
    /// always true unless the active property/array-item schema has enum values, in which case the
    /// closing quote is only legal once the accumulated text is an EXACT match for one of them.</summary>
    private bool CanCloseStringValue()
    {
        var enumValues = _frames.Count > 0 ? _frames[^1].ActiveValueSchema?.EnumValues : null;
        return enumValues is not { Count: > 0 } || ContainsExact(enumValues, _valueBuilder);
    }

    private void PushObjectFrameForValue()
    {
        var schema = _frames.Count > 0 ? _frames[^1].ActiveValueSchema : null;
        var returnState = ValueCompleteState;
        _frames.Add(GrammarFrame.ForObject(schema?.ChildProperties, returnState));
    }

    private void PushArrayFrameForValue()
    {
        var schema = _frames.Count > 0 ? _frames[^1].ActiveValueSchema : null;
        var returnState = ValueCompleteState;
        _frames.Add(GrammarFrame.ForArray(schema?.ArrayItemSchema, returnState));
    }

    /// <summary>Closes an OBJECT: pops back to the enclosing value's return state, or -- at the
    /// root -- ends generation. Callers must have already confirmed (via CanAcceptChar's
    /// AreAllRequiredPropertiesEmitted gate) that closing is legal.</summary>
    private void CloseObject()
    {
        if (_frames.Count > 1)
        {
            var returnState = _frames[^1].ReturnState;
            _frames.RemoveAt(_frames.Count - 1);
            AdvanceLexicalState(returnState);
        }
        else
        {
            AdvanceLexicalState(JsonLexicalState.Terminal);
        }
    }

    /// <summary>Closes an ARRAY: pops its frame (arrays are never the root, so this always has a
    /// parent to return to) back to the enclosing value's return state.</summary>
    private void CloseArray()
    {
        if (_frames.Count > 1)
        {
            var returnState = _frames[^1].ReturnState;
            _frames.RemoveAt(_frames.Count - 1);
            AdvanceLexicalState(returnState);
        }
        else
        {
            // Should not happen (root is always an object, never an array) -- fail closed rather
            // than underflow the frame stack.
            AdvanceLexicalState(JsonLexicalState.Terminal);
        }
    }

    public void PushFrame(IReadOnlyList<SchemaPropertyNode>? childProps)
    {
        _frames.Add(GrammarFrame.ForObject(childProps, ValueCompleteState));
    }

    public bool PopFrame()
    {
        if (_frames.Count > 1)
        {
            _frames.RemoveAt(_frames.Count - 1);
            return true;
        }
        return false;
    }

    public void RecordPropertyEmitted(string propName)
    {
        if (_frames.Count == 0) return;

        var span = CollectionsMarshal.AsSpan(_frames);
        ref var frame = ref span[^1];
        int limit = Math.Min(frame.Properties.Count, 64);
        for (int i = 0; i < limit; i++)
        {
            if (string.Equals(frame.Properties[i].Name, propName, StringComparison.Ordinal))
            {
                frame.ActivePropertyIndex = i;
                frame.EmittedMask |= 1L << i;
                return;
            }
        }
        // Not found within the tracked-property bound (undeclared key, or index >= 64): still
        // record it as the active property if it's a genuine schema property beyond the bitmask
        // limit, so ActiveValueSchema keeps working for nested/enum lookups on it.
        for (int i = limit; i < frame.Properties.Count; i++)
        {
            if (string.Equals(frame.Properties[i].Name, propName, StringComparison.Ordinal))
            {
                frame.ActivePropertyIndex = i;
                return;
            }
        }
    }

    public bool AreAllRequiredPropertiesEmitted()
    {
        return _frames.Count == 0 || _frames[^1].AreAllRequiredEmitted;
    }

    private static bool HasEnumPrefixMatch(IReadOnlyList<string> enumValues, StringBuilder candidate)
    {
        for (int i = 0; i < enumValues.Count; i++)
        {
            if (IsPrefix(enumValues[i], candidate)) return true;
        }
        return false;
    }

    private static bool IsPrefix(string value, StringBuilder prefix)
    {
        if (prefix.Length > value.Length) return false;
        for (int i = 0; i < prefix.Length; i++)
        {
            if (value[i] != prefix[i]) return false;
        }
        return true;
    }

    private static bool ContainsExact(IReadOnlyList<string> enumValues, StringBuilder candidate)
    {
        for (int i = 0; i < enumValues.Count; i++)
        {
            if (EqualsBuilder(enumValues[i], candidate)) return true;
        }
        return false;
    }

    private static bool EqualsBuilder(string value, StringBuilder candidate)
    {
        if (value.Length != candidate.Length) return false;
        for (int i = 0; i < value.Length; i++)
        {
            if (value[i] != candidate[i]) return false;
        }
        return true;
    }
}
