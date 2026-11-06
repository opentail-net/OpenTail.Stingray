using OpenTail.Stingray.Mlx;

namespace OpenTail.Stingray.Tests.Cuda;

/// <summary>
/// <see cref="MlxBackend.IsAvailable"/> is the only thing this skeleton can actually be checked
/// against here: no Mac exists on this development machine, so <c>libmlxc</c> is never present and
/// this must return <c>false</c> gracefully rather than throw. See
/// docs/4-performance/gpu/2026-11-06-mlx-metal-backend-plan.md.
/// </summary>
public sealed class MlxBackendTests
{
    [Fact]
    public void IsAvailable_WithNoNativeLibraryPresent_ReturnsFalse_DoesNotThrow()
    {
        // This asserts the honest, expected-here outcome, not a mocked one: this machine genuinely
        // has no libmlxc. If a future CI runner or dev machine ever has one, this test's assumption
        // changes and it should be revisited, not silently left green for the wrong reason.
        Assert.False(MlxBackend.IsAvailable());
    }
}
