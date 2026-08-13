using System;

namespace OpenTail.Stingray.Engine;

/// <summary>
/// Model and KV configuration namespace isolating prefix tree root paths.
/// Prevents invalid cross-model or cross-quantization page sharing.
/// </summary>
public readonly record struct PrefixCacheNamespace(
    string ModelId,
    string KvConfigHash);

/// <summary>
/// Compound lookup key combining namespace identity and token ID sequence.
/// </summary>
public readonly record struct PrefixCacheKey(
    PrefixCacheNamespace Namespace,
    ReadOnlyMemory<int> TokenIds);

/// <summary>
/// Result of a prefix cache lookup operation.
/// </summary>
public readonly record struct PrefixMatchResult(
    int MatchedTokenCount,
    int MatchedPageCount,
    ReadOnlyMemory<KvPageId> SharedPages);

/// <summary>
/// Authoritative Radix Tree index mapping token sequences to shared physical KV pages.
/// Decouples physical memory residency from logical session inference scheduling.
/// </summary>
public interface IPrefixCacheIndex : IDisposable
{
    /// <summary>
    /// Looks up the longest page-aligned prefix match for the given namespace and token sequence.
    /// Atomically retains ref-counts on matched physical pages inside the tree lock before returning.
    /// </summary>
    PrefixMatchResult MatchPrefix(PrefixCacheNamespace ns, ReadOnlySpan<int> tokens);

    /// <summary>
    /// Publishes a completed, committed sequence of full physical pages from an IKvSequence into the prefix index.
    /// </summary>
    void Publish(PrefixCacheNamespace ns, ReadOnlySpan<int> tokens, IKvSequence sequence, int committedTokenCount);

    /// <summary>
    /// Removes a specific prefix entry and releases its page ref-counts.
    /// </summary>
    void Remove(PrefixCacheNamespace ns, ReadOnlySpan<int> tokens);

    /// <summary>
    /// Evicts unreferenced LRU prefix entries until target memory is reclaimed.
    /// </summary>
    int EvictLruEntries(int targetPagesToFree);
}
