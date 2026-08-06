using OpenTail.Stingray.Cli.Terminal;

namespace OpenTail.Stingray.Tests.Cli;

public sealed class MarkupRendererTests
{
    private static readonly string Esc = ((char)27).ToString();

    private static string Plain(string markup) => MarkupRenderer.Render(markup, ansi: false);
    private static string Ansi(string markup) => MarkupRenderer.Render(markup, ansi: true);

    // ── Escaping ────────────────────────────────────────────────────────────

    [Fact]
    public void DoubledBrackets_RenderAsLiteralBrackets()
    {
        Assert.Equal("[", Plain("[["));
        Assert.Equal("]", Plain("]]"));
        Assert.Equal("[dim]", Plain("[[dim]]"));
    }

    [Fact]
    public void Escape_ThenRender_RoundTripsArbitraryText()
    {
        const string raw = """{"type":"function","required":["location"]}""";
        Assert.Equal(raw, Plain(Markup.Escape(raw)));
    }

    [Fact]
    public void EscapeMarkupExtension_MatchesMarkupEscape()
    {
        const string raw = "a [b] c";
        Assert.Equal(Markup.Escape(raw), raw.EscapeMarkup());
    }

    // ── The regression that broke `--help` ──────────────────────────────────
    // Spectre.Console throws "Could not find color or style" on an unparseable tag, which
    // took down `--help` entirely because a [Description] embedded a JSON example. Rendering
    // is cosmetic and must degrade to literal text instead of killing the process.

    [Fact]
    public void UnknownTag_RendersLiterally_InsteadOfThrowing()
    {
        const string json = """([{type:"function"}], or a {"tools":[...]} wrapper)""";
        string rendered = Plain(json);
        Assert.Equal(json, rendered);
    }

    [Fact]
    public void UnknownTag_DoesNotThrow_EvenWithAnsiEnabled()
    {
        var ex = Record.Exception(() => Ansi("""{"required":[...]}"""));
        Assert.Null(ex);
    }

    [Fact]
    public void UnterminatedBracket_IsTreatedAsLiteral()
    {
        Assert.Equal("value [", Plain("value ["));
    }

    // ── Styling ─────────────────────────────────────────────────────────────

    [Fact]
    public void KnownStyle_StripsToPlainTextWhenAnsiDisabled()
    {
        Assert.Equal("Error: bad", Plain("[red]Error:[/] bad"));
    }

    [Fact]
    public void KnownStyle_EmitsSgrAndResetWhenAnsiEnabled()
    {
        string rendered = Ansi("[red]Error:[/]");
        Assert.Equal($"{Esc}[31mError:{Esc}[0m", rendered);
    }

    [Theory]
    [InlineData("dim")]
    [InlineData("red")]
    [InlineData("yellow")]
    [InlineData("green")]
    [InlineData("bold")]
    [InlineData("i")]
    [InlineData("cyan")]
    [InlineData("blue")]
    public void EveryStyleUsedByTheCli_IsRecognised(string style)
    {
        // Recognised styles vanish under plain rendering; unrecognised ones would survive
        // as literal text, so this asserts the CLI's whole vocabulary is covered.
        Assert.Equal("x", Plain($"[{style}]x[/]"));
    }

    [Fact]
    public void NestedStyles_RestoreOuterStyleOnClose()
    {
        string rendered = Ansi("[red]a[bold]b[/]c[/]");
        // Closing the inner style resets, then replays the still-open outer style.
        Assert.Equal($"{Esc}[31ma{Esc}[1mb{Esc}[0m{Esc}[31mc{Esc}[0m", rendered);
    }

    [Fact]
    public void MultiTokenStyle_CombinesSgrParameters()
    {
        Assert.Equal($"{Esc}[1;31mx{Esc}[0m", Ansi("[bold red]x[/]"));
    }

    [Fact]
    public void TextWithoutMarkup_IsReturnedUnchanged()
    {
        const string s = "plain text, no brackets";
        Assert.Equal(s, Plain(s));
        Assert.Equal(s, Ansi(s));
    }

    [Fact]
    public void StrayCloseTag_DoesNotThrow()
    {
        var ex = Record.Exception(() => Ansi("no open [/] here"));
        Assert.Null(ex);
    }
}
