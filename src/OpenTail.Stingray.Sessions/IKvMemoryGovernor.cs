using System;
using System.Threading;
using System.Threading.Tasks;

namespace OpenTail.Stingray.Sessions;

/// <summary>
/// Observability statistics for the adaptive physical KV memory governor.
/// </summary>
public readonly record struct KvGovernorStatistics(
    long GovernorCycles,
    long PressureEvents,
    long SessionsSuspended,
    long PagesReclaimed,
    long SuspensionFailures,
    long SkippedBusySessions);

/// <summary>
/// Interface for the adaptive physical KV memory governor managing memory pressure via transient idle session suspension.
/// <para>
/// <b>Transient Reclamation Semantics:</b> Under memory pressure, the governor releases physical KV cache pages
/// while retaining logical token history in memory. Subsequent interaction with a suspended session transparently
/// reconstructs KV state on demand. Durable disk persistence is managed separately by <c>FileSessionStore</c>.
/// </para>
/// </summary>
public interface IKvMemoryGovernor : IAsyncDisposable
{
    KvGovernorOptions Options { get; }
    KvGovernorStatistics Statistics { get; }
    Task ReclaimMemoryIfUnderPressureAsync(CancellationToken cancellationToken = default);
}
