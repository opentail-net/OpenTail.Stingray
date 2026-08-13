using System;
using System.Text.Json;
using System.Threading.Tasks;
using OpenTail.Stingray.Core.Grammar;
using Xunit;

namespace OpenTail.Stingray.Tests.Core;

public class Plan007JsonConstraintInvariantsTests
{
    private static (ITokenConstraint c, FakeJsonTokenizer tok, int vocab) Build(string schemaJson, bool ordered = false)
    {
        var tok = new FakeJsonTokenizer();
        var vocab = new GrammarVocabulary(tok);
        using var doc = JsonDocument.Parse(schemaJson);
        var schemaObj = ToolSchema.FromOpenAiFunction("_", doc.RootElement.Clone()).Arguments;
        var c = JsonConstraint.FromSchema(vocab, schemaObj, ordered);
        return (c, tok, vocab.VocabSize);
    }

    private static void Feed(ITokenConstraint c, FakeJsonTokenizer tok, string text)
    {
        foreach (int id in tok.Encode(text)) c.Accept(id);
    }

    private static bool Allowed(ITokenConstraint c, int vocab, int tokenId)
    {
        Span<float> logits = new float[vocab];
        var masked = c.Filter(logits);
        return !float.IsNegativeInfinity(masked[tokenId]);
    }

    [Fact]
    public void Test1_SimpleObject_IsValid()
    {
        var (c, tok, vocab) = Build("""{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}""");
        Feed(c, tok, "{\"name\":\"Alice\"}");
        Assert.True(Allowed(c, vocab, FakeJsonTokenizer.Eos));
    }

    [Fact]
    public void Test2_InvalidPunctuation_IsMasked()
    {
        var (c, tok, vocab) = Build("""{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}""");
        Feed(c, tok, "{");
        Assert.False(Allowed(c, vocab, tok.Char('}')));
    }

    [Fact]
    public void Test3_NestedObjectsAndArrays()
    {
        var (c, tok, vocab) = Build("""{"type":"object","properties":{"items":{"type":"array","items":{"type":"object","properties":{"id":{"type":"integer"}},"required":["id"]}}},"required":["items"]}""");
        Feed(c, tok, "{\"items\":[{\"id\":1},{\"id\":2}]}");
        Assert.True(Allowed(c, vocab, FakeJsonTokenizer.Eos));
    }

    [Fact]
    public void Test4_StringEscapes()
    {
        var (c, tok, vocab) = Build("""{"type":"object","properties":{"val":{"type":"string"}},"required":["val"]}""");
        Feed(c, tok, "{\"val\":\"hello\\\"world\"}");
        Assert.True(Allowed(c, vocab, FakeJsonTokenizer.Eos));
    }

    [Fact]
    public void Test5_UnicodeEscape()
    {
        var (c, tok, vocab) = Build("""{"type":"object","properties":{"val":{"type":"string"}},"required":["val"]}""");
        Feed(c, tok, "{\"val\":\"\\u0041\"}");
        Assert.True(Allowed(c, vocab, FakeJsonTokenizer.Eos));
    }

    [Fact]
    public void Test6_NumberStateMachine()
    {
        var (c, tok, vocab) = Build("""{"type":"object","properties":{"num":{"type":"number"}},"required":["num"]}""");
        Feed(c, tok, "{\"num\":125");
        Assert.True(Allowed(c, vocab, tok.Char('}')));
    }

    [Fact]
    public void Test7_LiteralStateMachine()
    {
        var (c, tok, vocab) = Build("""{"type":"object","properties":{"flag":{"type":"boolean"}},"required":["flag"]}""");
        Feed(c, tok, "{\"flag\":tr");
        Assert.True(Allowed(c, vocab, tok.Char('u')));
        Assert.False(Allowed(c, vocab, tok.Char('x')));
    }

    [Fact]
    public void Test8_MultiCharacterToken()
    {
        var (c, tok, vocab) = Build("""{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}""");
        Assert.True(Allowed(c, vocab, tok.Merged("{\"")));
        Assert.False(Allowed(c, vocab, tok.Merged("{}")));
    }

    [Fact]
    public void Test9_CompleteRootValue()
    {
        var (c, tok, vocab) = Build("""{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}""");
        Feed(c, tok, "{\"name\":\"Alice\"}");
        Assert.True(Allowed(c, vocab, FakeJsonTokenizer.Eos));
        Assert.False(Allowed(c, vocab, tok.Char('x')));
    }

    [Fact]
    public void Test10_RequiredProperty()
    {
        var (c, tok, vocab) = Build("""{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"]}""");
        Feed(c, tok, "{\"name\":\"Alice\"");
        Assert.False(Allowed(c, vocab, tok.Char('}')));
        Assert.True(Allowed(c, vocab, tok.Char(',')));
    }

    [Fact]
    public void Test11_AdditionalPropertiesFalse()
    {
        var (c, tok, vocab) = Build("""{"type":"object","properties":{"name":{"type":"string"}},"required":["name"],"additionalProperties":false}""");
        Feed(c, tok, "{\"name\":\"Alice\"");
        Assert.False(Allowed(c, vocab, tok.Char(',')));
        Assert.True(Allowed(c, vocab, tok.Char('}')));
    }

    [Fact]
    public void Test12_Enum()
    {
        var (c, tok, vocab) = Build("""{"type":"object","properties":{"color":{"type":"string","enum":["red","green","blue"]}},"required":["color"]}""");
        Feed(c, tok, "{\"color\":\"");
        Assert.True(Allowed(c, vocab, tok.Char('r')));
        Assert.False(Allowed(c, vocab, tok.Char('y')));
    }

    [Fact]
    public void Test13_SchemaType()
    {
        var (c, tok, vocab) = Build("""{"type":"object","properties":{"count":{"type":"integer"}},"required":["count"]}""");
        Feed(c, tok, "{\"count\":");
        Assert.True(Allowed(c, vocab, tok.Char('1')));
        Assert.False(Allowed(c, vocab, tok.Char('"')));
    }

    [Fact]
    public void Test14_AllTokensRejected()
    {
        var (c, _, vocab) = Build("""{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}""");
        Span<float> logits = new float[vocab];
        var masked = c.Filter(logits);
        Assert.True(masked.Length > 0);
    }

    [Fact]
    public void Test15_ConstraintStateIsolated()
    {
        var (c1, tok1, v1) = Build("""{"type":"object","properties":{"a":{"type":"string"}},"required":["a"]}""");
        var (c2, tok2, v2) = Build("""{"type":"object","properties":{"b":{"type":"string"}},"required":["b"]}""");

        Feed(c1, tok1, "{\"a\":");
        Assert.True(c1.IsConstraining);
        Assert.True(c2.IsConstraining); // JsonSchemaOutputConstraint is active from token 1 for c2 independently
    }
}
