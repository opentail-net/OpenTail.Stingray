using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Sessions;
using Xunit;

namespace OpenTail.Stingray.Tests.Sessions;

public sealed class SessionMetadataTests
{
    [Fact]
    public void Test1_SetAndGet()
    {
        ISessionMetadata metadata = new SessionMetadata();
        metadata.Set("workflow", "review");

        Assert.Equal("review", metadata.Get("workflow"));
        Assert.Equal("review", metadata.Get<string>("workflow"));
    }

    [Fact]
    public void Test2_MissingKeyReturnsNull()
    {
        ISessionMetadata metadata = new SessionMetadata();

        Assert.Null(metadata.Get("nonexistent"));
        Assert.Null(metadata.Get<string>("nonexistent"));
        Assert.Equal(0, metadata.Get<int>("nonexistent"));
    }

    [Fact]
    public void Test3_TryGetMissingReturnsFalse()
    {
        ISessionMetadata metadata = new SessionMetadata();

        bool found = metadata.TryGet<string>("missing", out var val);
        Assert.False(found);
        Assert.Null(val);
    }

    [Fact]
    public void Test4_Remove()
    {
        ISessionMetadata metadata = new SessionMetadata();
        metadata.Set("temp", 123);

        Assert.True(metadata.Remove("temp"));
        Assert.False(metadata.Remove("temp")); // Second remove returns false
        Assert.Null(metadata.Get("temp"));
    }

    [Fact]
    public void Test5_TypedValue()
    {
        ISessionMetadata metadata = new SessionMetadata();
        metadata.Set("count", 42);
        metadata.Set("ratio", 3.14);
        metadata.Set("enabled", true);
        metadata["indexer"] = "hello";

        Assert.True(metadata.TryGet<int>("count", out int countVal));
        Assert.Equal(42, countVal);

        Assert.True(metadata.TryGet<double>("ratio", out double ratioVal));
        Assert.Equal(3.14, ratioVal);

        Assert.True(metadata.TryGet<bool>("enabled", out bool enabledVal));
        Assert.True(enabledVal);

        Assert.Equal("hello", metadata["indexer"]);
    }

    [Fact]
    public void Test6_WrongTypeDoesNotSilentlyConvert()
    {
        ISessionMetadata metadata = new SessionMetadata();
        metadata.Set("count", 42);

        // Retrieving as string should return false / null without throwing or silent conversion
        Assert.False(metadata.TryGet<string>("count", out var stringVal));
        Assert.Null(stringVal);
        Assert.Null(metadata.Get<string>("count"));
    }

    [Fact]
    public async Task Test7_MetadataContainerIsIndependentAfterFork()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var parent = new InferenceSession(cache);

        parent.Metadata.Set("workflow", "code-review");
        parent.Metadata.Set("user_id", "user_123");

        await using var child = parent.Fork();

        Assert.NotSame(parent.Metadata, child.Metadata);
        Assert.Equal("code-review", child.Metadata.Get<string>("workflow"));
        Assert.Equal("user_123", child.Metadata.Get<string>("user_id"));
    }

    [Fact]
    public async Task Test8_ParentMetadataIsolation()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var parent = new InferenceSession(cache);

        parent.Metadata.Set("workflow", "review");
        await using var child = parent.Fork();

        // Mutating child metadata leaves parent untouched
        child.Metadata.Set("workflow", "testing");
        child.Metadata.Set("new_child_key", 999);

        Assert.Equal("review", parent.Metadata.Get<string>("workflow"));
        Assert.Null(parent.Metadata.Get("new_child_key"));
        Assert.Equal("testing", child.Metadata.Get<string>("workflow"));
    }

    [Fact]
    public async Task Test9_ChildMetadataIsolation()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var parent = new InferenceSession(cache);

        parent.Metadata.Set("workflow", "initial");
        await using var child = parent.Fork();

        // Mutating parent metadata after fork leaves child untouched
        parent.Metadata.Set("workflow", "updated_parent");
        parent.Metadata.Set("parent_only_key", "secret");

        Assert.Equal("initial", child.Metadata.Get<string>("workflow"));
        Assert.Null(child.Metadata.Get("parent_only_key"));
    }

    [Fact]
    public async Task Test10_NestedForkMetadataIsolation()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var root = new InferenceSession(cache);
        root.Metadata.Set("tier", "root");

        await using var branchA = root.Fork();
        branchA.Metadata.Set("tier", "branchA");

        await using var branchA1 = branchA.Fork();
        branchA1.Metadata.Set("tier", "branchA1");

        Assert.Equal("root", root.Metadata.Get<string>("tier"));
        Assert.Equal("branchA", branchA.Metadata.Get<string>("tier"));
        Assert.Equal("branchA1", branchA1.Metadata.Get<string>("tier"));
    }

    [Fact]
    public async Task Test11_SuspensionPreservesMetadata()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var session = new InferenceSession(cache);

        session.Metadata.Set("persistent_key", "still_here");
        await session.AppendAsync(new int[] { 1, 2, 3 });

        await session.SuspendAsync();
        Assert.Equal(SessionState.Suspended, session.State);
        Assert.Equal("still_here", session.Metadata.Get<string>("persistent_key"));

        await session.ResumeAsync();
        Assert.Equal(SessionState.Ready, session.State);
        Assert.Equal("still_here", session.Metadata.Get<string>("persistent_key"));
    }

    [Fact]
    public void Test12_ConcurrentMetadataAccess()
    {
        ISessionMetadata metadata = new SessionMetadata();

        Parallel.For(0, 1000, i =>
        {
            metadata.Set($"key_{i % 50}", i);
            metadata.Get($"key_{i % 50}");
            if (i % 3 == 0)
            {
                metadata.Remove($"key_{i % 50}");
            }
        });

        // Verifies no thread corruption or unhandled concurrency exception occurred
        metadata.Set("final", "done");
        Assert.Equal("done", metadata.Get<string>("final"));
    }

    [Fact]
    public async Task Test13_OpenTailContextIntegration()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        await using var session = new InferenceSession(cache);

        session.Metadata.Set("workflow", "code-review");
        session.Metadata.Set("current_file", "Foo.cs");
        session.Metadata.Set("permission_mode", "read-only");

        await using var branch = session.Fork();
        branch.Metadata.Set("current_file", "Bar.cs");

        Assert.Equal("Foo.cs", session.Metadata.Get<string>("current_file"));
        Assert.Equal("Bar.cs", branch.Metadata.Get<string>("current_file"));
        Assert.Equal("code-review", branch.Metadata.Get<string>("workflow"));
    }
}
