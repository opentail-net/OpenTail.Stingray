namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// strftime_now() as llama.cpp's minja provides it. gpt-oss renders "Current date: %Y-%m-%d" and Llama 3.1
/// "Today Date: %d %b %Y"; before this the call returned null and the date line rendered empty.
/// </summary>
public sealed class JinjaStrftimeNowTests
{
    private static readonly DateTime s_fixed = new(2026, 3, 7, 14, 5, 9);

    [Theory]
    [InlineData("%Y-%m-%d", "2026-03-07")]
    [InlineData("%d %b %Y", "07 Mar 2026")]
    [InlineData("%A, %B %e", "Saturday, March  7")]
    [InlineData("%H:%M:%S %I%p", "14:05:09 02PM")]
    [InlineData("%j %y %F %T", "066 26 2026-03-07 14:05:09")]
    [InlineData("100%% %Q", "100% %Q")]
    public void Strftime_FormatsCCodes(string format, string expected) =>
        Assert.Equal(expected, JinjaChatTemplate.Strftime(s_fixed, format));

    [Fact]
    public void StrftimeNow_RendersTheDateInATemplate()
    {
        var saved = JinjaChatTemplate.Now;
        try
        {
            JinjaChatTemplate.Now = () => s_fixed;
            var tmpl = new JinjaChatTemplate("{{- \"Current date: \" + strftime_now(\"%Y-%m-%d\") + \"\n\" }}");
            Assert.Equal("Current date: 2026-03-07\n", tmpl.Render(new Dictionary<string, object?>()));
        }
        finally
        {
            JinjaChatTemplate.Now = saved;
        }
    }
}
