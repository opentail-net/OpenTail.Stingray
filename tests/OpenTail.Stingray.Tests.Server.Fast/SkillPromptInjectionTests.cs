using System.Net;
using System.Net.Http.Json;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Server;

namespace OpenTail.Stingray.Tests.Server.Fast;

/// <summary>
/// docs/051: a <c>skills</c> field on <c>/v1/chat/completions</c> and <c>/v1/messages</c> folds
/// each skill's instructions into a prepended system-message segment and its tools into the
/// effective declared tools, so a skill (e.g. fetched client-side from a registry such as
/// skills.sh) "becomes a prompt" without the caller hand-building a system message. See
/// <c>OpenAiEndpoints.ApplySkills</c> / <c>AnthropicEndpoints.ApplySkills</c>.
/// </summary>
public sealed class SkillPromptInjectionTests : IDisposable
{
    private readonly List<WebApplicationFactory<Program>> _factories = new();

    public void Dispose()
    {
        foreach (var factory in _factories) factory.Dispose();
    }

    // Records enable_thinking, every message's content, and every declared tool's function name,
    // so a test can assert exactly what reached the template — the same technique ServerTests.cs
    // uses for its own prompt-probing tests.
    private const string ProbeTemplate =
        "{% for m in messages %}[{{ m.role }}:{{ m.content }}]{% endfor %}" +
        "{% for t in tools %}(TOOL:{{ t.function.name }}){% endfor %}";

    private (HttpClient client, FakeInferenceEngine fake) ProbeClient()
    {
        var fake = new FakeInferenceEngine("m");
        var renderer = new ChatTemplateRenderer("test", new JinjaChatTemplate(ProbeTemplate));
        var factory = new WebApplicationFactory<Program>().WithWebHostBuilder(b =>
            b.ConfigureServices(s =>
            {
                s.AddSingleton(renderer);
                s.AddSingleton<IInferenceEngine>(fake);
            }));
        _factories.Add(factory);
        return (factory.CreateClient(), fake);
    }

    [Fact]
    public async Task OpenAi_Skills_PrependsInstructionsAndMergesTools()
    {
        var (client, fake) = ProbeClient();
        var req = new
        {
            model = "m",
            messages = new[] { new { role = "user", content = "hi" } },
            skills = new[]
            {
                new
                {
                    name = "weather",
                    instructions = new[] { new { content = "You can check the weather." } },
                    tools = new[] { new { name = "get_forecast", description = "Fetches a forecast" } },
                },
            },
        };

        var resp = await client.PostAsJsonAsync("/v1/chat/completions", req);
        Assert.Equal(HttpStatusCode.OK, resp.StatusCode);
        Assert.NotNull(fake.LastPrompt);
        Assert.Contains("[system:You can check the weather.]", fake.LastPrompt);
        Assert.Contains("[user:hi]", fake.LastPrompt);
        // System segment precedes the user turn.
        Assert.True(fake.LastPrompt.IndexOf("[system:", StringComparison.Ordinal)
            < fake.LastPrompt.IndexOf("[user:", StringComparison.Ordinal));
        Assert.Contains("(TOOL:get_forecast)", fake.LastPrompt);
    }

    [Fact]
    public async Task OpenAi_Skills_MergesWithExplicitToolsAndTools()
    {
        var (client, fake) = ProbeClient();
        var req = new
        {
            model = "m",
            messages = new[] { new { role = "user", content = "hi" } },
            tools = new[] { new { type = "function", function = new { name = "explicit_tool" } } },
            skills = new[]
            {
                new { name = "weather", tools = new[] { new { name = "get_forecast" } } },
            },
        };

        var resp = await client.PostAsJsonAsync("/v1/chat/completions", req);
        Assert.Equal(HttpStatusCode.OK, resp.StatusCode);
        Assert.NotNull(fake.LastPrompt);
        Assert.Contains("(TOOL:explicit_tool)", fake.LastPrompt);
        Assert.Contains("(TOOL:get_forecast)", fake.LastPrompt);
    }

    [Fact]
    public async Task OpenAi_NoSkills_IsUnaffected()
    {
        var (client, fake) = ProbeClient();
        var req = new { model = "m", messages = new[] { new { role = "user", content = "hi" } } };

        var resp = await client.PostAsJsonAsync("/v1/chat/completions", req);
        Assert.Equal(HttpStatusCode.OK, resp.StatusCode);
        Assert.NotNull(fake.LastPrompt);
        Assert.DoesNotContain("[system:", fake.LastPrompt);
    }

    [Fact]
    public async Task Anthropic_Skills_PrependsInstructionsAheadOfExistingSystem_AndMergesTools()
    {
        var (client, fake) = ProbeClient();
        var req = new
        {
            model = "m",
            max_tokens = 16,
            system = "Be terse.",
            messages = new[] { new { role = "user", content = "hi" } },
            skills = new[]
            {
                new
                {
                    name = "weather",
                    instructions = new[] { new { content = "You can check the weather." } },
                    tools = new[] { new { name = "get_forecast" } },
                },
            },
        };

        var resp = await client.PostAsJsonAsync("/v1/messages", req);
        Assert.Equal(HttpStatusCode.OK, resp.StatusCode);
        Assert.NotNull(fake.LastPrompt);
        Assert.Contains("[system:You can check the weather.\n\nBe terse.]", fake.LastPrompt);
        Assert.Contains("(TOOL:get_forecast)", fake.LastPrompt);
    }
}
