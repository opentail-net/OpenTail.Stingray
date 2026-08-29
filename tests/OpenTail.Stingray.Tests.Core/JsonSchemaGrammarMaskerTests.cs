
namespace OpenTail.Stingray.Tests.Core;

public enum TestRole { User, Admin, Guest }

public class SampleDto
{
    public string Name { get; set; } = "";
    public int Age { get; set; }
    public TestRole Role { get; set; }
}

public class JsonSchemaGrammarMaskerTests
{
    [Fact]
    public void Test01_Compiler_SubMillisecondLatency()
    {
        using var doc = JsonDocument.Parse("""
        {
          "type": "object",
          "properties": {
            "name": { "type": "string" },
            "age": { "type": "integer" },
            "role": { "type": "string", "enum": ["admin", "user", "guest"] }
          },
          "required": ["name", "role"]
        }
        """);

        var sm = JsonSchemaGrammarCompiler.CompileWithBenchmark(doc.RootElement, out var compilationDuration);

        Assert.NotNull(sm);
        Assert.Equal(3, sm.Properties.Count);
        Assert.True(compilationDuration.TotalMilliseconds < 50.0, $"Compilation took {compilationDuration.TotalMilliseconds}ms");
    }

    [Fact]
    public void Test02_SimpleObjectSyntax()
    {
        using var doc = JsonDocument.Parse("""
        {
          "type": "object",
          "properties": {
            "greeting": { "type": "string" }
          },
          "required": ["greeting"]
        }
        """);

        var sm = JsonSchemaGrammarCompiler.Compile(doc.RootElement);

        Assert.Equal(JsonLexicalState.RootObjectStart, sm.CurrentState);
        Assert.True(sm.CanAcceptChar('{'));
        Assert.False(sm.CanAcceptChar(']'));
    }

    [Fact]
    public void Test03_EnumConstraintMasking()
    {
        using var doc = JsonDocument.Parse("""
        {
          "type": "object",
          "properties": {
            "status": { "type": "string", "enum": ["active", "pending"] }
          },
          "required": ["status"]
        }
        """);

        var sm = JsonSchemaGrammarCompiler.Compile(doc.RootElement);
        var statusProp = sm.Properties[0];

        Assert.NotNull(statusProp.EnumValues);
        Assert.Contains("active", statusProp.EnumValues);
        Assert.Contains("pending", statusProp.EnumValues);
    }

    [Fact]
    public void Test04_RequiredPropertiesCheck()
    {
        using var doc = JsonDocument.Parse("""
        {
          "type": "object",
          "properties": {
            "id": { "type": "string" },
            "token": { "type": "string" }
          },
          "required": ["id"]
        }
        """);

        var sm = JsonSchemaGrammarCompiler.Compile(doc.RootElement);
        Assert.False(sm.AreAllRequiredPropertiesEmitted());

        sm.RecordPropertyEmitted("id");
        Assert.True(sm.AreAllRequiredPropertiesEmitted());
    }

    [Fact]
    public void Test05_BpeTokenBoundaryHandling()
    {
        var tok = new FakeJsonTokenizer();
        var vocab = new GrammarVocabulary(tok);

        using var doc = JsonDocument.Parse("""
        {
          "type": "object",
          "properties": {
            "action": { "type": "string" }
          },
          "required": ["action"]
        }
        """);

        var sm = JsonSchemaGrammarCompiler.Compile(doc.RootElement);
        var masker = new JsonSchemaGrammarMasker(vocab, sm);

        Assert.True(masker.IsConstraining);

        Span<float> logits = new float[vocab.VocabSize];
        var masked = masker.Filter(logits);

        Assert.Equal(vocab.VocabSize, masked.Length);
    }

    [Fact]
    public void Test06_CompilerForType_Dto()
    {
        var sm = JsonSchemaGrammarCompiler.CompileForType<SampleDto>();

        Assert.NotNull(sm);
        Assert.Equal(3, sm.Properties.Count);
        Assert.Equal("Name", sm.Properties[0].Name);
        Assert.Equal("string", sm.Properties[0].Type);
        Assert.Equal("Role", sm.Properties[2].Name);
        Assert.NotNull(sm.Properties[2].EnumValues);
        Assert.Contains("User", sm.Properties[2].EnumValues!);
    }

    [Fact]
    public void Test07_NestedObjectProperties()
    {
        using var doc = JsonDocument.Parse("""
        {
          "type": "object",
          "properties": {
            "user": {
              "type": "object",
              "properties": {
                "id": { "type": "integer" }
              }
            }
          }
        }
        """);

        var sm = JsonSchemaGrammarCompiler.Compile(doc.RootElement);

        Assert.Single(sm.Properties);
        Assert.Equal("object", sm.Properties[0].Type);
        Assert.NotNull(sm.Properties[0].ChildProperties);
        Assert.Single(sm.Properties[0].ChildProperties!);
        Assert.Equal("id", sm.Properties[0].ChildProperties![0].Name);
    }

    [Fact]
    public void Test08_StringEscapeHandling()
    {
        var sm = new GrammarStateMachine(Array.Empty<SchemaPropertyNode>());
        sm.AdvanceLexicalState(JsonLexicalState.StringValue);

        Assert.False(sm.IsEscaped);
        sm.IsEscaped = true;

        Assert.True(sm.CanAcceptChar('"'));
    }

    [Fact]
    public void Test09_ArrayItemSchema()
    {
        using var doc = JsonDocument.Parse("""
        {
          "type": "object",
          "properties": {
            "tags": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "tag": { "type": "string" }
                }
              }
            }
          }
        }
        """);

        var sm = JsonSchemaGrammarCompiler.Compile(doc.RootElement);

        Assert.Single(sm.Properties);
        Assert.Equal("array", sm.Properties[0].Type);
        Assert.NotNull(sm.Properties[0].ArrayItemSchema);
    }

    [Fact]
    public void Test10_UnicodeInsideStrings()
    {
        var tokens = new List<byte[]>
        {
            Encoding.UTF8.GetBytes("{\"text\":\""),
            Encoding.UTF8.GetBytes("café"),
            Encoding.UTF8.GetBytes("日本語"),
            Encoding.UTF8.GetBytes("🎉"),
            Encoding.UTF8.GetBytes("\"}")
        };
        var tok = new CustomByteTokenizer(tokens);
        var vocab = new GrammarVocabulary(tok);

        using var doc = JsonDocument.Parse("""
        {
          "type": "object",
          "properties": {
            "text": { "type": "string" }
          },
          "required": ["text"]
        }
        """);

        var sm = JsonSchemaGrammarCompiler.Compile(doc.RootElement);
        var masker = new JsonSchemaGrammarMasker(vocab, sm);

        // Accept {"text":"
        masker.Accept(3);
        Assert.Equal(JsonLexicalState.StringValue, sm.CurrentState);

        // Accept "café", "日本語", "🎉"
        masker.Accept(4);
        masker.Accept(5);
        masker.Accept(6);
        Assert.Equal(JsonLexicalState.StringValue, sm.CurrentState);

        // Filter logits before closing
        Span<float> logits = new float[vocab.VocabSize];
        var masked = masker.Filter(logits);
        Assert.True(masked[7] > float.NegativeInfinity, "Token 7 ('\"}') must be allowed to close string and object.");

        // Accept "}
        masker.Accept(7);
        Assert.True(sm.IsTerminal);
    }

    [Fact]
    public void Test11_MultibyteCharacterSplitAcrossTokens()
    {
        // '日' = 0xE6, 0x97, 0xA5 (3 bytes)
        // '🎉' = 0xF0, 0x9F, 0x8E, 0x89 (4 bytes)
        var tokens = new List<byte[]>
        {
            Encoding.UTF8.GetBytes("{\"val\":\""),   // Token 3
            new byte[] { 0xE6, 0x97 },              // Token 4: first 2 bytes of '日'
            new byte[] { 0xA5 },                    // Token 5: last byte of '日'
            new byte[] { 0xF0, 0x9F },              // Token 6: first 2 bytes of '🎉'
            new byte[] { 0x8E },                    // Token 7: 3rd byte of '🎉'
            new byte[] { 0x89 },                    // Token 8: 4th byte of '🎉'
            Encoding.UTF8.GetBytes("\"}")           // Token 9: "}
        };
        var tok = new CustomByteTokenizer(tokens);
        var vocab = new GrammarVocabulary(tok);

        using var doc = JsonDocument.Parse("""
        {
          "type": "object",
          "properties": {
            "val": { "type": "string" }
          },
          "required": ["val"]
        }
        """);

        var sm = JsonSchemaGrammarCompiler.Compile(doc.RootElement);
        var masker = new JsonSchemaGrammarMasker(vocab, sm);

        masker.Accept(3); // {"val":"

        // Filter logits: Token 4 (partial UTF-8 bytes) must be allowed inside string
        Span<float> logits = new float[vocab.VocabSize];
        var masked1 = masker.Filter(logits);
        Assert.True(masked1[4] > float.NegativeInfinity, "Token 4 (first 2 bytes of '日') must be allowed inside string literal.");

        masker.Accept(4); // Accept 0xE6, 0x97 (pending 2 bytes)

        // Filter logits while 2 bytes are pending: Token 5 (0xA5) must complete '日' and be allowed
        logits.Clear();
        var masked2 = masker.Filter(logits);
        Assert.True(masked2[5] > float.NegativeInfinity, "Token 5 (0xA5) completing '日' must be allowed.");
        Assert.Equal(float.NegativeInfinity, masked2[9]); // Token 9 ("}") cannot be allowed with incomplete UTF-8 bytes pending

        masker.Accept(5); // Accept 0xA5 (completes '日')
        masker.Accept(6); // Accept 0xF0, 0x9F (first 2 bytes of 🎉)
        masker.Accept(7); // Accept 0x8E (3rd byte of 🎉)
        masker.Accept(8); // Accept 0x89 (4th byte of 🎉, completes 🎉)

        Assert.Equal(JsonLexicalState.StringValue, sm.CurrentState);

        masker.Accept(9); // "}
        Assert.True(sm.IsTerminal);
    }

    [Fact]
    public void Test12_EscapedUnicode()
    {
        var tokens = new List<byte[]>
        {
            Encoding.UTF8.GetBytes("{\"code\":\""), // Token 3
            Encoding.UTF8.GetBytes("\\u65"),       // Token 4
            Encoding.UTF8.GetBytes("e5"),          // Token 5
            Encoding.UTF8.GetBytes("\\n"),         // Token 6
            Encoding.UTF8.GetBytes("\\\""),        // Token 7
            Encoding.UTF8.GetBytes("\\\\"),        // Token 8
            Encoding.UTF8.GetBytes("\"}")          // Token 9
        };
        var tok = new CustomByteTokenizer(tokens);
        var vocab = new GrammarVocabulary(tok);

        using var doc = JsonDocument.Parse("""
        {
          "type": "object",
          "properties": {
            "code": { "type": "string" }
          },
          "required": ["code"]
        }
        """);

        var sm = JsonSchemaGrammarCompiler.Compile(doc.RootElement);
        var masker = new JsonSchemaGrammarMasker(vocab, sm);

        masker.Accept(3); // {"code":"
        masker.Accept(4); // \u65
        masker.Accept(5); // e5
        masker.Accept(6); // \n
        masker.Accept(7); // \"
        masker.Accept(8); // \\
        Assert.Equal(JsonLexicalState.StringValue, sm.CurrentState);

        masker.Accept(9); // "}
        Assert.True(sm.IsTerminal);
    }

    [Fact]
    public void Test13_PunctuationAdjacentToUnicode()
    {
        var tokens = new List<byte[]>
        {
            Encoding.UTF8.GetBytes("{\""),          // Token 3
            Encoding.UTF8.GetBytes("日"),           // Token 4: '日' in key
            Encoding.UTF8.GetBytes("\":\""),        // Token 5: ":"
            Encoding.UTF8.GetBytes("val"),         // Token 6
            Encoding.UTF8.GetBytes("\"}")           // Token 7
        };
        var tok = new CustomByteTokenizer(tokens);
        var vocab = new GrammarVocabulary(tok);

        using var doc = JsonDocument.Parse("""
        {
          "type": "object",
          "properties": {
            "日": { "type": "string" }
          },
          "required": ["日"]
        }
        """);

        var sm = JsonSchemaGrammarCompiler.Compile(doc.RootElement);
        var masker = new JsonSchemaGrammarMasker(vocab, sm);

        masker.Accept(3); // {"
        Assert.Equal(JsonLexicalState.ObjectKeyContent, sm.CurrentState);

        masker.Accept(4); // 日
        Assert.Equal(JsonLexicalState.ObjectKeyContent, sm.CurrentState);

        masker.Accept(5); // ":"
        Assert.Equal(JsonLexicalState.StringValue, sm.CurrentState);

        masker.Accept(6); // val
        masker.Accept(7); // "}
        Assert.True(sm.IsTerminal);
    }

    /// <summary>
    /// Was: an opener token replayed as its own "illegal continuation" probe from ValueStart. That
    /// stopped being a genuine dead end once nested-object support was added -- '{' now legitimately
    /// opens a nested value at ANY value position (that's the point of the fix), so replaying an
    /// opener like {"key": now validates as "open a new nested object, key, colon" and no longer
    /// proves the state is dead. Redesigned around an enum-prefix mismatch instead: inside a string
    /// value constrained to enum ["active","pending"], a character that isn't a prefix of either is
    /// still a genuine, un-escapable dead end regardless of nesting -- '{'/'[' are ordinary string
    /// content there (a string CAN contain a literal brace), not structural, so this reproduces the
    /// same class of defect docs/bugstofix.md described without depending on a byte sequence nesting
    /// support has since made legal. See ChoiceConstraint.cs/GrammarStateMachine.cs for the sibling
    /// "narrow grammar must fail closed rather than silently accept" pattern this guards.
    /// </summary>
    [Fact]
    public void Test14_DeadState_ThrowsJsonGrammarDeadStateException()
    {
        var tokens = new List<byte[]>
        {
            Encoding.UTF8.GetBytes("{\"status\":\""), // Token 3: opens through StringValue (enum-constrained)
            Encoding.UTF8.GetBytes("z"),               // Token 4: not a prefix of "active" or "pending"
        };
        var tok = new CustomByteTokenizer(tokens);
        var vocab = new GrammarVocabulary(tok);

        using var doc = JsonDocument.Parse("""
        {
          "type": "object",
          "properties": { "status": { "type": "string", "enum": ["active", "pending"] } },
          "required": ["status"]
        }
        """);

        var sm = JsonSchemaGrammarCompiler.Compile(doc.RootElement);
        var masker = new JsonSchemaGrammarMasker(vocab, sm);

        masker.Accept(3); // {"status":"
        Assert.Equal(JsonLexicalState.StringValue, sm.CurrentState);

        Span<float> logits = new float[vocab.VocabSize];
        JsonGrammarDeadStateException? caught = null;
        try
        {
            masker.Filter(logits);
        }
        catch (JsonGrammarDeadStateException ex)
        {
            caught = ex;
        }

        Assert.NotNull(caught);
        Assert.Equal(JsonLexicalState.StringValue, caught.State);
    }

    [Fact]
    public void Test15_TerminalState_ForcesEndOfGenerationOnly()
    {
        var tokens = new List<byte[]>
        {
            Encoding.UTF8.GetBytes("{\"k\":\"v\"}"), // Token 3: closes the whole object in one token
            Encoding.UTF8.GetBytes("more"),           // Token 4: plausible-looking garbage continuation
        };
        var tok = new CustomByteTokenizer(tokens);
        var vocab = new GrammarVocabulary(tok);

        using var doc = JsonDocument.Parse("""
        {
          "type": "object",
          "properties": { "k": { "type": "string" } },
          "required": ["k"]
        }
        """);

        var sm = JsonSchemaGrammarCompiler.Compile(doc.RootElement);
        var masker = new JsonSchemaGrammarMasker(vocab, sm);

        masker.Accept(3); // {"k":"v"}
        Assert.True(sm.IsTerminal);
        Assert.True(masker.IsConstraining, "must keep constraining post-completion to force a stop.");

        Span<float> logits = new float[vocab.VocabSize];
        logits.Fill(5.0f); // every token looks equally attractive to the sampler
        var masked = masker.Filter(logits);

        // Only the EOS id (2) may survive; everything else -- including the plausible-looking
        // continuation token 4 -- must be masked to -inf so the sampler is forced to stop.
        Assert.Equal(5.0f, masked[CustomByteTokenizer.Eos]);
        for (int i = 0; i < masked.Length; i++)
        {
            if (i == CustomByteTokenizer.Eos) continue;
            Assert.True(float.IsNegativeInfinity(masked[i]), $"token {i} must be masked once the JSON is complete.");
        }

        // A token sampled after completion (forced EOS or otherwise) must not throw when accepted.
        masker.Accept(CustomByteTokenizer.Eos);
        masker.Accept(4);
        Assert.True(sm.IsTerminal);
    }

    [Fact]
    public void Test16_EmptyByteTokensForbiddenMidGeneration()
    {
        var tokens = new List<byte[]>
        {
            Encoding.UTF8.GetBytes("{\"k\":\""), // Token 3: opens into a string value (StringValue state)
            Encoding.UTF8.GetBytes("hello"),       // Token 4: plain string content, stays in StringValue
        };
        var tok = new CustomByteTokenizer(tokens);
        var vocab = new GrammarVocabulary(tok);

        using var doc = JsonDocument.Parse("""
        {
          "type": "object",
          "properties": { "k": { "type": "string" } },
          "required": ["k"]
        }
        """);

        var sm = JsonSchemaGrammarCompiler.Compile(doc.RootElement);
        var masker = new JsonSchemaGrammarMasker(vocab, sm);

        masker.Accept(3); // {"k":"
        Assert.Equal(JsonLexicalState.StringValue, sm.CurrentState);

        Span<float> logits = new float[vocab.VocabSize];
        var masked = masker.Filter(logits);

        // Pad(0)/Bos(1)/Eos(2) carry no bytes and must never be selectable mid-string -- letting
        // one through would truncate the call before the object closes.
        Assert.True(float.IsNegativeInfinity(masked[0]));
        Assert.True(float.IsNegativeInfinity(masked[1]));
        Assert.True(float.IsNegativeInfinity(masked[CustomByteTokenizer.Eos]));
        Assert.True(masked[4] > float.NegativeInfinity, "plain string content must remain allowed.");
    }

    /// <summary>
    /// Regression for GrammarStateMachine.cs:146 (docs/bugstofix.md): '{' at a value position used
    /// to have no transition at all, leaving the state stuck at ValueStart with no way to enter or
    /// correctly close a nested object. Also exercises required-property enforcement at a NESTED
    /// level (not just the root), via the same PushFrame/RecordPropertyEmitted/CanEvict-style
    /// machinery this file's Test04 only exercised manually.
    /// </summary>
    [Fact]
    public void Test17_NestedObjectRequiredPropertyEnforced()
    {
        using var doc = JsonDocument.Parse("""
        {
          "type": "object",
          "properties": {
            "user": {
              "type": "object",
              "properties": { "id": { "type": "string" } },
              "required": ["id"]
            }
          },
          "required": ["user"]
        }
        """);

        var sm = JsonSchemaGrammarCompiler.Compile(doc.RootElement);

        foreach (char c in "{\"user\":{") Assert.True(sm.TryAcceptChar(c));
        Assert.Equal(JsonLexicalState.ObjectKeyStart, sm.CurrentState);

        // Nested object's own required "id" hasn't been emitted yet -- closing early is illegal.
        Assert.False(sm.CanAcceptChar('}'));

        foreach (char c in "\"id\":\"x\"") Assert.True(sm.TryAcceptChar(c));
        Assert.True(sm.TryAcceptChar('}')); // closes the nested object (its own "id" satisfied)
        Assert.True(sm.TryAcceptChar('}')); // closes the root object ("user" satisfied)
        Assert.True(sm.IsTerminal);
    }

    /// <summary>
    /// Regression for GrammarStateMachine.cs:146: '[' had no transition either, and ArrayStart/
    /// ArrayValueStart/ArrayValueEnd existed in the enum with zero reachable code. Covers array of
    /// objects specifically (not just scalars) so PushFrame's ArrayItemSchema resolution and the
    /// per-element required-property check both get exercised, and that closing one element
    /// correctly returns to ArrayValueEnd (not the object-context ValueEnd) so a second element or
    /// ']' is offered next rather than another object key.
    /// </summary>
    [Fact]
    public void Test18_ArrayOfObjectsRoundTrip()
    {
        using var doc = JsonDocument.Parse("""
        {
          "type": "object",
          "properties": {
            "tags": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": { "tag": { "type": "string" } },
                "required": ["tag"]
              }
            }
          },
          "required": ["tags"]
        }
        """);

        var sm = JsonSchemaGrammarCompiler.Compile(doc.RootElement);

        foreach (char c in "{\"tags\":[{") Assert.True(sm.TryAcceptChar(c));
        Assert.Equal(JsonLexicalState.ObjectKeyStart, sm.CurrentState);

        // First element's own required "tag" hasn't been emitted -- can't close it yet.
        Assert.False(sm.CanAcceptChar('}'));

        foreach (char c in "\"tag\":\"a\"}") Assert.True(sm.TryAcceptChar(c));
        Assert.Equal(JsonLexicalState.ArrayValueEnd, sm.CurrentState);

        foreach (char c in ",{\"tag\":\"b\"}") Assert.True(sm.TryAcceptChar(c));
        Assert.Equal(JsonLexicalState.ArrayValueEnd, sm.CurrentState);

        Assert.True(sm.TryAcceptChar(']'));
        Assert.Equal(JsonLexicalState.ValueEnd, sm.CurrentState);

        Assert.True(sm.TryAcceptChar('}'));
        Assert.True(sm.IsTerminal);
    }

    /// <summary>Regression for GrammarStateMachine.cs:146: 'n' at a value position had no
    /// transition, so null-valued properties could never be produced.</summary>
    [Fact]
    public void Test19_NullLiteralAccepted()
    {
        using var doc = JsonDocument.Parse("""
        {
          "type": "object",
          "properties": { "maybe": { "type": "string" } },
          "required": ["maybe"]
        }
        """);

        var sm = JsonSchemaGrammarCompiler.Compile(doc.RootElement);

        foreach (char c in "{\"maybe\":null}") Assert.True(sm.TryAcceptChar(c));
        Assert.True(sm.IsTerminal);
    }

    /// <summary>
    /// The exact failure scenario from docs/bugstofix.md: "letting a model emit {} for a schema
    /// requiring fields." Required-property enforcement (PushFrame/RecordPropertyEmitted/
    /// AreAllRequiredPropertiesEmitted) existed with zero callers before this fix -- nothing
    /// actually gated the closing '}' on it.
    /// </summary>
    [Fact]
    public void Test20_RootObjectCannotCloseEmpty_WhenRequiredPropertyMissing()
    {
        using var doc = JsonDocument.Parse("""
        {
          "type": "object",
          "properties": { "name": { "type": "string" } },
          "required": ["name"]
        }
        """);

        var sm = JsonSchemaGrammarCompiler.Compile(doc.RootElement);

        Assert.True(sm.TryAcceptChar('{'));
        Assert.Equal(JsonLexicalState.ObjectKeyStart, sm.CurrentState);

        Assert.False(sm.CanAcceptChar('}'));
        Assert.False(sm.TryAcceptChar('}'));
        Assert.Equal(JsonLexicalState.ObjectKeyStart, sm.CurrentState); // rejected: no state change

        foreach (char c in "\"name\":\"x\"") Assert.True(sm.TryAcceptChar(c));
        Assert.True(sm.TryAcceptChar('}'));
        Assert.True(sm.IsTerminal);
    }

    /// <summary>
    /// Enum enforcement through the ACTUAL masker path (Filter()'s per-candidate-token masking),
    /// not just the state machine directly -- docs/bugstofix.md specifically called out that no
    /// test exercised this. A token that cannot possibly complete to an allowed enum value must be
    /// masked; one that can (even mid-spelling) must not be.
    /// </summary>
    [Fact]
    public void Test21_EnumConstraintRejectsInvalidValueThroughMasker()
    {
        var tokens = new List<byte[]>
        {
            Encoding.UTF8.GetBytes("{\"status\":\""), // Token 3
            Encoding.UTF8.GetBytes("active"),          // Token 4: a valid enum value
            Encoding.UTF8.GetBytes("banana"),          // Token 5: not a prefix of any allowed value
            Encoding.UTF8.GetBytes("\"}"),              // Token 6: closes string + object
        };
        var tok = new CustomByteTokenizer(tokens);
        var vocab = new GrammarVocabulary(tok);

        using var doc = JsonDocument.Parse("""
        {
          "type": "object",
          "properties": { "status": { "type": "string", "enum": ["active", "pending"] } },
          "required": ["status"]
        }
        """);

        var sm = JsonSchemaGrammarCompiler.Compile(doc.RootElement);
        var masker = new JsonSchemaGrammarMasker(vocab, sm);

        masker.Accept(3); // {"status":"

        Span<float> logits = new float[vocab.VocabSize];
        var masked = masker.Filter(logits);

        Assert.True(masked[4] > float.NegativeInfinity, "'active' is a legal prefix toward an allowed enum value.");
        Assert.True(float.IsNegativeInfinity(masked[5]), "'banana' cannot lead to any allowed enum value.");

        masker.Accept(4); // active
        masker.Accept(6); // "}
        Assert.True(sm.IsTerminal);
    }

    /// <summary>
    /// Regression for JsonSchemaGrammarMasker.cs:78 (docs/bugstofix.md): Filter() used to
    /// GrammarStateMachine.Clone() once PER VOCABULARY ENTRY -- 100k+ allocations per decode step
    /// on a real tokenizer. With a single reused scratch instance (GrammarStateMachine.CopyFrom),
    /// allocated bytes should stay roughly flat regardless of vocab size rather than scaling
    /// linearly with it.
    /// </summary>
    [Fact]
    public void Test22_FilterDoesNotAllocateProportionallyToVocabSize()
    {
        var tokens = new List<byte[]> { Encoding.UTF8.GetBytes("{\"k\":\"") }; // Token 3: opens into StringValue
        for (int i = 0; i < 2000; i++)
        {
            tokens.Add(Encoding.UTF8.GetBytes(((char)('a' + (i % 26))).ToString()));
        }
        var tok = new CustomByteTokenizer(tokens);
        var vocab = new GrammarVocabulary(tok);

        using var doc = JsonDocument.Parse("""
        {
          "type": "object",
          "properties": { "k": { "type": "string" } },
          "required": ["k"]
        }
        """);

        var sm = JsonSchemaGrammarCompiler.Compile(doc.RootElement);
        var masker = new JsonSchemaGrammarMasker(vocab, sm);
        masker.Accept(3); // {"k":" -- now inside StringValue, where the single-letter tokens are plausible continuations
        Assert.Equal(JsonLexicalState.StringValue, sm.CurrentState);

        // Warm up so the scratch instance's internal List/StringBuilder capacities stabilize
        // before the measured call -- otherwise one-time growth would be charged to it.
        Span<float> warmup = new float[vocab.VocabSize];
        masker.Filter(warmup);

        long before = GC.GetAllocatedBytesForCurrentThread();
        Span<float> logits = new float[vocab.VocabSize];
        masker.Filter(logits);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;

        // The old per-token Clone() allocated a whole GrammarStateMachine (frame stack, string
        // builders, backing object) per vocabulary entry -- far more than 32 bytes/token. Budget
        // generously to catch a real regression without being brittle to incidental bookkeeping.
        Assert.True(allocated < vocab.VocabSize * 32,
            $"Filter() allocated {allocated} bytes for {vocab.VocabSize} vocab entries " +
            $"({(double)allocated / vocab.VocabSize:F1} bytes/token) -- expected roughly flat, not scaling with vocab size.");
    }
}

internal sealed class CustomByteTokenizer : OpenTail.Stingray.Core.ITokenizer
{
    public const int Pad = 0, Bos = 1, Eos = 2;

    private readonly List<byte[]> _tokenBytes = new();
    private readonly Dictionary<string, int> _specials = new(StringComparer.Ordinal);

    public CustomByteTokenizer(IEnumerable<byte[]> tokens)
    {
        _tokenBytes.Add([]); // Pad = 0
        _tokenBytes.Add([]); // Bos = 1
        _tokenBytes.Add([]); // Eos = 2
        foreach (var t in tokens)
        {
            _tokenBytes.Add(t);
        }
    }

    public int VocabSize => _tokenBytes.Count;
    public int BosTokenId => 1;
    public int EosTokenId => 2;
    public int UnknownTokenId => 0;
    public int PadTokenId => 0;
    public bool AddBosToken => false;
    public System.Collections.Immutable.ImmutableArray<int> EogTokenIds => [2];
    public IReadOnlyDictionary<string, int> SpecialTokens => _specials;

    public byte[] DecodeBytes(int token) => token >= 0 && token < _tokenBytes.Count ? _tokenBytes[token] : [];
    public IReadOnlyList<int> Encode(string text) => throw new NotImplementedException();
    public string Decode(IEnumerable<int> tokens) => throw new NotImplementedException();
}


