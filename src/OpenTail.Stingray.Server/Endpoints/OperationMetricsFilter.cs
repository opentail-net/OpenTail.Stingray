using System.Diagnostics;

namespace OpenTail.Stingray.Server.Endpoints;

/// <summary>
/// Records <c>gen_ai.server.request.duration</c> for endpoints that are not chat generation
/// (embeddings, images, audio, rerank) without touching their handler bodies. Operation and
/// endpoint are fixed strings chosen where the route is mapped; outcome is a closed set.
/// These handlers are plain <see cref="RequestDelegate"/>s, which endpoint filters do not cover,
/// so the delegate itself is wrapped when the endpoint is built.
/// </summary>
internal static class OperationMetricsFilterExtensions
{
    /// <summary><c>HttpContext.Items</c> key where an audio handler stores the audio seconds it produced (speech) or consumed (transcription), as a double.</summary>
    internal const string AudioSecondsItem = "opentail.audio_seconds";

    public static TBuilder WithOperationMetrics<TBuilder>(this TBuilder builder, string operation, string endpoint)
        where TBuilder : IEndpointConventionBuilder
    {
        builder.Add(endpointBuilder =>
        {
            var inner = endpointBuilder.RequestDelegate;
            if (inner is null) return;
            endpointBuilder.RequestDelegate = async ctx =>
            {
                var metrics = ctx.RequestServices.GetService<ServerMetrics>();
                if (metrics is null || !metrics.OperationMetricsEnabled)
                {
                    await inner(ctx);
                    return;
                }

                long started = Stopwatch.GetTimestamp();
                string outcome = "success";
                string? errorType = null;
                try
                {
                    await inner(ctx);
                    int status = ctx.Response.StatusCode;
                    if (status >= 500) outcome = "error";
                    else if (status >= 400) outcome = "client_error";
                }
                catch (OperationCanceledException)
                {
                    outcome = "cancelled";
                    throw;
                }
                catch (Exception ex)
                {
                    outcome = "error";
                    errorType = ex.GetType().Name; // the type name only: messages can echo request content
                    throw;
                }
                finally
                {
                    var elapsed = Stopwatch.GetElapsedTime(started);
                    metrics.RecordOperation(operation, endpoint, elapsed, outcome, errorType);
                    if (outcome == "success" && ctx.Items.TryGetValue(AudioSecondsItem, out var seconds) && seconds is double audioSeconds && audioSeconds > 0)
                        metrics.RecordAudio(operation, endpoint, audioSeconds, elapsed);
                }
            };
        });
        return builder;
    }
}
