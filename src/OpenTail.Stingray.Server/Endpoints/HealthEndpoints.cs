
namespace OpenTail.Stingray.Server.Endpoints;

public static class HealthEndpoints
{
    public static IEndpointRouteBuilder MapHealthEndpoints(this IEndpointRouteBuilder app)
    {
        // Diagnostics never load or pin a model in a multi-model deployment: see ServerDiagnosticsTarget.
        app.MapGet("/health", (IServiceProvider services, ServerMetrics metrics) =>
        {
            var target = ServerDiagnosticsTarget.Resolve(services);
            return Results.Ok(new HealthStatus("ok", target.Model,
                (long)metrics.Uptime.TotalSeconds, target.ConfiguredModels));
        });

        app.MapGet("/metrics", HandleMetrics);
        return app;
    }

    private static Task HandleMetrics(HttpContext ctx, IServiceProvider services, ServerMetrics metrics,
        RequestConcurrencyGate admissionGate)
    {
        var engine = ServerDiagnosticsTarget.Resolve(services).Engine;
        ctx.Response.ContentType = "text/plain; version=0.0.4";
        double uptime = metrics.Uptime.TotalSeconds;
        long totalRequests = metrics.TotalRequests;
        long totalTokens = metrics.TotalTokens;
        long overloadRejections = metrics.OverloadRejections;
        double tps = uptime > 0 ? totalTokens / uptime : 0;
        var batching = engine as IContinuousBatchingObservability;
        string batchingMetrics = batching is { IsContinuousBatching: true }
            ? RenderBatchingMetrics(batching)
            : string.Empty;
        return ctx.Response.WriteAsync(
            $"# HELP opentail_llm_requests_total Total inference requests served\n" +
            $"# TYPE opentail_llm_requests_total counter\n" +
            $"opentail_llm_requests_total {totalRequests}\n" +
            $"# HELP opentail_llm_tokens_generated_total Total tokens generated\n" +
            $"# TYPE opentail_llm_tokens_generated_total counter\n" +
            $"opentail_llm_tokens_generated_total {totalTokens}\n" +
            $"# HELP opentail_llm_uptime_seconds Server uptime in seconds\n" +
            $"# TYPE opentail_llm_uptime_seconds gauge\n" +
            $"opentail_llm_uptime_seconds {(long)uptime}\n" +
            $"# HELP opentail_llm_tokens_per_second Lifetime-average tokens generated per second\n" +
            $"# TYPE opentail_llm_tokens_per_second gauge\n" +
            $"opentail_llm_tokens_per_second {tps:F2}\n" +
            $"# HELP opentail_llm_queue_depth Number of requests waiting to start generation\n" +
            $"# TYPE opentail_llm_queue_depth gauge\n" +
            $"opentail_llm_queue_depth {engine.QueueDepth}\n" +
            $"# HELP opentail_llm_active_requests Number of requests currently generating tokens\n" +
            $"# TYPE opentail_llm_active_requests gauge\n" +
            $"opentail_llm_active_requests {engine.ActiveRequests}\n" +
            $"# HELP opentail_llm_admission_capacity Maximum active-plus-waiting generation requests (0 = unbounded)\n" +
            $"# TYPE opentail_llm_admission_capacity gauge\n" +
            $"opentail_llm_admission_capacity {admissionGate.Limit}\n" +
            $"# HELP opentail_llm_overload_rejections_total Requests rejected because the bounded inference queue was full\n" +
            $"# TYPE opentail_llm_overload_rejections_total counter\n" +
            $"opentail_llm_overload_rejections_total {overloadRejections}\n" +
            $"# HELP opentail_llm_prefix_cache_enabled 1 if the engine's prefix-cache reuse path is active, 0 if disabled (e.g. GDN hybrid models)\n" +
            $"# TYPE opentail_llm_prefix_cache_enabled gauge\n" +
            $"opentail_llm_prefix_cache_enabled {(engine.PrefixCacheEnabled ? 1 : 0)}\n" +
            $"# HELP opentail_llm_prefill_tokens_reused_total Total prompt tokens skipped via the prefix-cache fast path\n" +
            $"# TYPE opentail_llm_prefill_tokens_reused_total counter\n" +
            $"opentail_llm_prefill_tokens_reused_total {engine.PrefillTokensReused}\n" +
            batchingMetrics +
            metrics.RenderServingLatencyMetrics(),
            ctx.RequestAborted);
    }

    private static string RenderBatchingMetrics(IContinuousBatchingObservability m) =>
        $"# HELP opentail_llm_batch_prefill_chunk_tokens Maximum tokens in one incremental prefill work item\n" +
        $"# TYPE opentail_llm_batch_prefill_chunk_tokens gauge\n" +
        $"opentail_llm_batch_prefill_chunk_tokens {m.PrefillChunkTokens}\n" +
        $"# HELP opentail_llm_batch_kv_committed_tokens KV tokens reserved by admitted requests\n" +
        $"# TYPE opentail_llm_batch_kv_committed_tokens gauge\n" +
        $"opentail_llm_batch_kv_committed_tokens {m.CommittedKvTokens}\n" +
        $"# HELP opentail_llm_batch_kv_token_budget KV-token admission budget; 9223372036854775807 means unlimited\n" +
        $"# TYPE opentail_llm_batch_kv_token_budget gauge\n" +
        $"opentail_llm_batch_kv_token_budget {m.KvTokenBudget}\n" +
        $"# HELP opentail_llm_prefix_cache_entries Immutable shared prefix snapshots retained\n" +
        $"# TYPE opentail_llm_prefix_cache_entries gauge\n" +
        $"opentail_llm_prefix_cache_entries {m.PrefixCacheEntries}\n" +
        $"# HELP opentail_llm_prefix_cache_bytes Bytes occupied by retained prefix snapshots\n" +
        $"# TYPE opentail_llm_prefix_cache_bytes gauge\n" +
        $"opentail_llm_prefix_cache_bytes {m.PrefixCacheUsedBytes}\n" +
        $"# HELP opentail_llm_prefix_cache_capacity_bytes Configured prefix-cache capacity; 9223372036854775807 means unlimited\n" +
        $"# TYPE opentail_llm_prefix_cache_capacity_bytes gauge\n" +
        $"opentail_llm_prefix_cache_capacity_bytes {m.PrefixCacheBudgetBytes}\n" +
        $"# HELP opentail_llm_prefix_cache_hits_total Successful prefix-cache lookups\n" +
        $"# TYPE opentail_llm_prefix_cache_hits_total counter\n" +
        $"opentail_llm_prefix_cache_hits_total {m.PrefixCacheHits}\n" +
        $"# HELP opentail_llm_prefix_cache_misses_total Prefix-cache lookups with no reusable prefix\n" +
        $"# TYPE opentail_llm_prefix_cache_misses_total counter\n" +
        $"opentail_llm_prefix_cache_misses_total {m.PrefixCacheMisses}\n" +
        $"# HELP opentail_llm_prefix_cache_evictions_total Prefix snapshots removed to honour cache capacity\n" +
        $"# TYPE opentail_llm_prefix_cache_evictions_total counter\n" +
        $"opentail_llm_prefix_cache_evictions_total {m.PrefixCacheEvictions}\n" +
        $"# HELP opentail_llm_batched_argmax_steps_total Batched decode steps using argmax-only sampling\n" +
        $"# TYPE opentail_llm_batched_argmax_steps_total counter\n" +
        $"opentail_llm_batched_argmax_steps_total {m.BatchedArgmaxSteps}\n" +
        $"# HELP opentail_llm_batched_full_logits_steps_total Batched decode steps requiring full logits\n" +
        $"# TYPE opentail_llm_batched_full_logits_steps_total counter\n" +
        $"opentail_llm_batched_full_logits_steps_total {m.BatchedFullLogitsSteps}\n" +
        $"# HELP opentail_llm_batched_argmax_sequences_total Sequences served by argmax-only batched decode\n" +
        $"# TYPE opentail_llm_batched_argmax_sequences_total counter\n" +
        $"opentail_llm_batched_argmax_sequences_total {m.BatchedArgmaxSequences}\n" +
        $"# HELP opentail_llm_batched_full_logits_sequences_total Sequences served by full-logits batched decode\n" +
        $"# TYPE opentail_llm_batched_full_logits_sequences_total counter\n" +
        $"opentail_llm_batched_full_logits_sequences_total {m.BatchedFullLogitsSequences}\n";
}

/// <summary><c>/health</c> body. In a multi-model deployment <see cref="Model"/> is the default alias
/// (where a request without <c>model</c> goes) and <see cref="Models"/> lists every configured alias;
/// single-model deployments omit <see cref="Models"/>.</summary>
public sealed record HealthStatus(string Status, string Model, long UptimeSeconds,
    IReadOnlyList<string>? Models = null);
