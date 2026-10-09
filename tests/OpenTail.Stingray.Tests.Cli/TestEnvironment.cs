using System.Runtime.CompilerServices;

namespace OpenTail.Stingray.Tests.Cli;

/// <summary>
/// Keeps the whole Cli test assembly away from the developer's real per-user configuration. The resolver reads favourites.json from the config
/// directory, so without this any machine that has a favourite set would change the result of tests that never mention favourites (found by running
/// the suite with a deliberately poisoned config directory: two FrontDoorTaskCommandsTests failed). Tests that need a specific directory pass their own.
/// </summary>
internal static class TestEnvironment
{
    [ModuleInitializer]
    internal static void IsolateUserConfig() =>
        Environment.SetEnvironmentVariable("STINGRAY_CONFIG_DIR", Path.Combine(Path.GetTempPath(), "stingray-cli-tests-config-" + Guid.NewGuid().ToString("N")));
}
