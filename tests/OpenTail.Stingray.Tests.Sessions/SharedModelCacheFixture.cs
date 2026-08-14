using OpenTail.Stingray.Core;

[assembly: Xunit.AssemblyFixture(typeof(OpenTail.Stingray.Tests.Sessions.SharedModelCacheFixture))]

namespace OpenTail.Stingray.Tests.Sessions;

/// <summary>
/// Constructed once by xUnit before any test in this assembly runs, disposed once after all
/// tests finish (xUnit v3's <c>[assembly: AssemblyFixture]</c> mechanism — a real per-assembly
/// lifecycle hook, not a bare static with no teardown). Test files use the shared cache via
/// <see cref="Instance"/> directly rather than a constructor-injected fixture parameter, so
/// migrating call sites doesn't require touching every test class's constructor — see docs/025.
/// Separate instance from Tests.ForwardPass's fixture of the same name: each test project is
/// its own process, so caches cannot be shared across them regardless.
/// </summary>
public sealed class SharedModelCacheFixture : IDisposable
{
    /// <summary>The process-wide cache every test file's <c>GgufModel.Open</c> call site was
    /// migrated to go through instead. Valid only between xUnit constructing this fixture and
    /// disposing it — i.e., for the lifetime of the test run.</summary>
    public static SharedModelCache Instance { get; private set; } = null!;

    public SharedModelCacheFixture() => Instance = new SharedModelCache();

    public void Dispose() => Instance.Dispose();
}
