namespace OpenTail.Stingray.Sessions;


public sealed record CowBranchInfo(
    SessionId ParentSessionId,
    SessionId ChildSessionId,
    SessionRevision ForkRevision,
    ImmutableArray<SegmentBlockRef> SharedParentBlocks);

/// <summary>
/// Manifest-level bookkeeping for branched sessions: which sealed parent blocks a child inherits,
/// and how a child's delta blocks concatenate onto them.
/// </summary>
/// <remarks>
/// <para><b>This is a scaffold, not the fork feature.</b> It has no production caller — only tests.
/// It performs no reference counting, allocates nothing, shares nothing, and is not connected to any
/// KV state. Both methods are pure operations on <see cref="ImmutableArray{T}"/> of block
/// descriptors. Do not read the type name as evidence that copy-on-write forking exists.</para>
///
/// <para><b>Where the real thing lives, and what is still missing.</b> Genuine copy-on-write does
/// exist one layer down: <c>PagedKvCache.ForkSharedPrefix</c> shares pages by reference through a
/// reference-counted <c>NativePagePool</c> and copies a block only on first write. It is called from
/// <c>ForwardPass</c> only. <b>Nothing connects a session-level fork to it</b> — there is no
/// <c>ForkAsync</c>, and this type does not invoke it. Branching a session and having parent and
/// child share pages is therefore NOT implemented.</para>
///
/// <para>The plan tracks that work as <b>Optional Milestone 5</b> (§15, "current-HEAD fork and
/// copy-on-write"), whose checklist is entirely open: global block pool keyed by persistence block
/// id, reference-counted sealed blocks, partial-tail copy on branch continuation, <c>EnsureWritable</c>
/// on every mutation path, safe decrement on disposal, current-HEAD-only forking,
/// <c>RevisionNotRetained</c> for unretained revisions, and no KV-layer branch merging. Its gate —
/// eight branches from a common 4K prefix copying zero historical payload — has never been run.</para>
///
/// <para>Note this type was previously labelled "Milestone 4". That is the wrong milestone: §14 is
/// multimodal durability. Fork/copy-on-write is §15.</para>
/// </remarks>
public static class CowSessionSnapshot
{
    public static CowBranchInfo CreateBranch(
        SessionId parentSessionId,
        SessionId childSessionId,
        SessionRevision forkRevision,
        ImmutableArray<SegmentBlockRef> parentBlocks)
    {
        var sharedBlocks = parentBlocks.IsDefault ? ImmutableArray<SegmentBlockRef>.Empty : parentBlocks;

        return new CowBranchInfo(
            parentSessionId,
            childSessionId,
            forkRevision,
            sharedBlocks);
    }

    public static ImmutableArray<SegmentBlockRef> MergeBranch(
        CowBranchInfo branchInfo,
        ImmutableArray<SegmentBlockRef> childDeltaBlocks)
    {
        ArgumentNullException.ThrowIfNull(branchInfo);

        var deltas = childDeltaBlocks.IsDefault ? ImmutableArray<SegmentBlockRef>.Empty : childDeltaBlocks;
        var builder = ImmutableArray.CreateBuilder<SegmentBlockRef>(branchInfo.SharedParentBlocks.Length + deltas.Length);

        builder.AddRange(branchInfo.SharedParentBlocks);
        builder.AddRange(deltas);

        return builder.ToImmutable();
    }
}
