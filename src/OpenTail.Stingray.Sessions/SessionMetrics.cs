using System;
using System.Threading;

namespace OpenTail.Stingray.Sessions;

internal sealed class SessionMetrics : ISessionMetrics
{
    private readonly Func<int> _getKvPagesHeld;
    private long _promptTokens;
    private long _generatedTokens;
    private long _prefillTicks;
    private long _generationTicks;

    public SessionMetrics(Func<int> getKvPagesHeld)
    {
        _getKvPagesHeld = getKvPagesHeld ?? throw new ArgumentNullException(nameof(getKvPagesHeld));
    }

    private SessionMetrics(long promptTokens, long generatedTokens, long prefillTicks, long generationTicks, Func<int> getKvPagesHeld)
    {
        _promptTokens = promptTokens;
        _generatedTokens = generatedTokens;
        _prefillTicks = prefillTicks;
        _generationTicks = generationTicks;
        _getKvPagesHeld = getKvPagesHeld;
    }

    public long PromptTokens => Interlocked.Read(ref _promptTokens);
    public long GeneratedTokens => Interlocked.Read(ref _generatedTokens);

    public TimeSpan TotalPrefillTime => TimeSpan.FromTicks(Interlocked.Read(ref _prefillTicks));
    public TimeSpan TotalGenerationTime => TimeSpan.FromTicks(Interlocked.Read(ref _generationTicks));

    public double TokensPerSecond
    {
        get
        {
            long gen = GeneratedTokens;
            double seconds = TotalGenerationTime.TotalSeconds;
            return (gen > 0 && seconds > 0) ? gen / seconds : 0.0;
        }
    }

    public int KvPagesHeld => _getKvPagesHeld();

    public void AddPromptTokens(long count, TimeSpan duration)
    {
        if (count > 0)
        {
            Interlocked.Add(ref _promptTokens, count);
        }
        if (duration > TimeSpan.Zero)
        {
            Interlocked.Add(ref _prefillTicks, duration.Ticks);
        }
    }

    public void AddGeneratedTokens(long count, TimeSpan duration)
    {
        if (count > 0)
        {
            Interlocked.Add(ref _generatedTokens, count);
        }
        if (duration > TimeSpan.Zero)
        {
            Interlocked.Add(ref _generationTicks, duration.Ticks);
        }
    }

    public SessionMetrics CloneForFork(Func<int> childGetKvPagesHeld)
    {
        return new SessionMetrics(
            PromptTokens,
            0, // Child branch starts generation metrics fresh
            TotalPrefillTime.Ticks,
            0,
            childGetKvPagesHeld);
    }
}
