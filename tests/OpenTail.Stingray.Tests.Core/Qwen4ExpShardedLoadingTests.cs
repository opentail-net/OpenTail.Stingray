using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Engine.Verification;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Qwen 3.8 plan, Phase 6.1, real checkpoint: a sharded qwen4exp GGUF loads through memory-mapped pointers, and the per-layer
/// token table (<c>per_layer_token_embd</c>) takes no managed heap. Skipped when the UD-IQ1_S trunk is absent.
/// </summary>
public sealed class Qwen4ExpShardedLoadingTests
{
    private static string? FindModel() => ModelLocator.FindDeep("Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf");

    [Fact]
    public void ShardedCheckpoint_MapsEveryShard_AndKeepsThePerLayerTableOffTheManagedHeap()
    {
        string? path = FindModel();
        Assert.SkipUnless(path is not null, "Qwen3.8-Flash-Next UD-IQ1_S checkpoint is required (paged, 72.5 GB).");

        string[] shardPaths = GgufModel.ResolveShardPaths(path!);
        Assert.True(shardPaths.Length > 1, "the UD-IQ1_S checkpoint is three shards; the test must see more than one");

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        long managedBefore = GC.GetTotalMemory(forceFullCollection: true);

        using var model = GgufModel.Open(path!);
        int totalLayers = (int)Convert.ToInt64(model.Metadata["qwen4exp.block_count"]);
        var hp = Qwen4ExpHyperparams.FromMetadata(model.Metadata, totalLayers);
        var tensors = Qwen4ExpTensorSet.Load(model, hp, totalLayers);
        using var pass = new Qwen4ExpForwardPass(model, hp, tensors);

        long managedAfter = GC.GetTotalMemory(forceFullCollection: true);
        long tableBytes = pass.PleTableBytes;
        long managedGrowth = managedAfter - managedBefore;

        int shardsUsed = model.Tensors.Select(t => t.ShardIndex).Distinct().Count();
        Console.WriteLine($"SHARDS files={shardPaths.Length} shards_with_tensors={shardsUsed} tensors={model.Tensors.Count}");
        Console.WriteLine("SHARD_COUNTS " + string.Join(",", model.Tensors.GroupBy(t => t.ShardIndex).OrderBy(g => g.Key).Select(g => $"{g.Key}:{g.Count()}")));
        Console.WriteLine($"MANAGED_HEAP before={managedBefore} after={managedAfter} growth={managedGrowth} per_layer_table_bytes={tableBytes}");

        // Shard 1 is metadata only (10.9 MB); shard 2 holds the 28.8 GB per-layer table as its single tensor. Only the table's
        // shard is checked here: the metadata shard is expected to hold no tensors.
        var table = model.Tensors.Single(t => t.Name == "per_layer_token_embd.weight");
        Assert.Equal(shardPaths.Length - 1, table.ShardIndex);
        Assert.True(shardsUsed >= 2, "tensor data must span more than one shard");
        Assert.True(tableBytes > 0, "per_layer_token_embd.weight must be present");
        Assert.True(managedGrowth < 512L * 1024 * 1024,
            $"loading grew the managed heap by {managedGrowth} bytes; the {tableBytes}-byte table must not be on it");
    }
}
