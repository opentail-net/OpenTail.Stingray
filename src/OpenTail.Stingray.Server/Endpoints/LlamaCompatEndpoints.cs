using System.Text.Json.Serialization;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Options;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Server.Endpoints;

/// <summary>
/// llama-server API on-ramp: the three cheap endpoint shapes that are useful out of the box
/// and require no new inference capability.
///
/// <list type="bullet">
///   <item><c>POST /tokenize</c> — encode text to token IDs.</item>
///   <item><c>POST /detokenize</c> — decode token IDs to text.</item>
///   <item><c>GET  /props</c> — model hyperparameters and effective chat-template name.</item>
/// </list>
///
/// Deliberately deferred: <c>/completion</c> (llama-server field and streaming shapes),
/// <c>/embedding</c> (needs pooled embeddings), <c>/infill</c> (FIM tokens),
/// <c>/slots</c> (continuous-batching only). See <c>docs/llamacpp-onramp-plan.md</c> Tier 2.
/// </summary>
public static class LlamaCompatEndpoints
{
    public static IEndpointRouteBuilder MapLlamaCompatEndpoints(this IEndpointRouteBuilder app)
    {
        app.MapPost("/tokenize",   HandleTokenize);
        app.MapPost("/detokenize", HandleDetokenize);
        app.MapGet("/props",       HandleProps);
        return app;
    }

    // ── /tokenize ──────────────────────────────────────────────────────────────

    private static IResult HandleTokenize(
        TokenizeRequest req,
        IInferenceEngine engine,
        HttpContext ctx)
    {
        // Resolve the tokenizer from DI (populated by the engine loader).
        var tokenizer = ctx.RequestServices.GetService<ITokenizer>();
        if (tokenizer is null)
            return Results.Problem("Tokenizer not available.", statusCode: 503);

        if (req.Content is null)
            return Results.BadRequest(new ErrorResponse("invalid_request_error", "content field is required."));

        IReadOnlyList<int> ids = tokenizer.Encode(req.Content);

        // add_special=true: prepend BOS by token ID, not by string concatenation.
        // String round-tripping (prepend "<s>" then re-encode) gives wrong IDs if the
        // model's BOS isn't that literal, or if the encoder doesn't mark it as special.
        if (req.AddSpecial == true && tokenizer.AddBosToken)
        {
            var withBos = new int[ids.Count + 1];
            withBos[0] = tokenizer.BosTokenId;
            for (int i = 0; i < ids.Count; i++) withBos[i + 1] = ids[i];
            ids = withBos;
        }

        return Results.Json(
            new TokenizeResponse(ids.Count, ids is int[] arr ? arr : ids.ToArray()),
            OpenTailStingrayJsonContext.Default.TokenizeResponse);
    }


    // ── /detokenize ────────────────────────────────────────────────────────────

    private static IResult HandleDetokenize(
        DetokenizeRequest req,
        IInferenceEngine engine,   // load-bearing: resolving IInferenceEngine populates TokenizerRelay
        HttpContext ctx)
    {
        var tokenizer = ctx.RequestServices.GetService<ITokenizer>();
        if (tokenizer is null)
            return Results.Problem("Tokenizer not available.", statusCode: 503);

        if (req.Tokens is null)
            return Results.BadRequest(new ErrorResponse("invalid_request_error", "tokens field is required."));

        string text = tokenizer.Decode(req.Tokens);
        return Results.Json(
            new DetokenizeResponse(text),
            OpenTailStingrayJsonContext.Default.DetokenizeResponse);
    }

    // ── /props ─────────────────────────────────────────────────────────────────

    private static IResult HandleProps(
        IInferenceEngine engine,
        ChatTemplateRenderer chatTemplate,
        IOptions<OpenTailStingrayServerOptions> options,
        HttpContext ctx)
    {
        var tokenizer = ctx.RequestServices.GetService<ITokenizer>();

        // Derive a human-readable template name from the architecture. llama-server's /props
        // returns the full Jinja source; we return the name (or "custom" for an unknown arch).
        string tmplName = chatTemplate.Architecture switch
        {
            "llama"  => "llama3",
            "llama4" => "llama4",
            "gemma"  or "gemma2" or "gemma4" => "gemma",
            _        => "chatml",   // qwen2, smollm, default
        };
        if (chatTemplate.JinjaTemplate is not null && tmplName == "chatml"
                && chatTemplate.Architecture is not ("qwen2" or "qwen3"))
            tmplName = "custom";

        var props = new PropsResponse(
            ModelId:     engine.ModelId,
            Architecture: chatTemplate.Architecture,
            ChatTemplate: tmplName,
            BosToken:    chatTemplate.JinjaTemplate?.BosToken,
            VocabSize:   tokenizer?.VocabSize,
            ContextSize: options.Value.ContextSize > 0 ? options.Value.ContextSize : (int?)null,
            ThinkingEnabled: !options.Value.DisableThinking);

        return Results.Json(props, OpenTailStingrayJsonContext.Default.PropsResponse);
    }
}

// ── Request / response shapes ─────────────────────────────────────────────────

/// <summary>Request body for <c>POST /tokenize</c>.</summary>
public sealed record TokenizeRequest(
    [property: JsonPropertyName("content")] string? Content,
    [property: JsonPropertyName("add_special")] bool? AddSpecial);

/// <summary>Response body for <c>POST /tokenize</c>.</summary>
public sealed record TokenizeResponse(
    [property: JsonPropertyName("n_tokens")] int NTokens,
    [property: JsonPropertyName("tokens")]   int[] Tokens);

/// <summary>Request body for <c>POST /detokenize</c>.</summary>
public sealed record DetokenizeRequest(
    [property: JsonPropertyName("tokens")] int[]? Tokens);

/// <summary>Response body for <c>POST /detokenize</c>.</summary>
public sealed record DetokenizeResponse(
    [property: JsonPropertyName("content")] string Content);

/// <summary>Response body for <c>GET /props</c>.</summary>
public sealed record PropsResponse(
    [property: JsonPropertyName("model")]            string ModelId,
    [property: JsonPropertyName("architecture")]     string Architecture,
    [property: JsonPropertyName("chat_template")]    string ChatTemplate,
    [property: JsonPropertyName("bos_token")]        string? BosToken,
    [property: JsonPropertyName("vocab_size")]       int? VocabSize,
    [property: JsonPropertyName("context_size")]     int? ContextSize,
    [property: JsonPropertyName("thinking_enabled")] bool ThinkingEnabled);
