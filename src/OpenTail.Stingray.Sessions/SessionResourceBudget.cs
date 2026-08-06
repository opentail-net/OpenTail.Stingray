namespace OpenTail.Stingray.Sessions;

/// <summary>Hot-state capacity limits for one <see cref="HotSessionRuntime"/>.</summary>
public sealed record HotSessionRuntimeOptions
{
    public long MaxResidentBytes { get; }
    public long MaxSessionBytes { get; }
    public int MaxCapturedOutputChunks { get; }
    public int MaxCompletedOperationRecords { get; }
    public TimeSpan CompletedOperationRetention { get; }

    /// <summary>
    /// Sessions the budget is expected to serve concurrently. When set and
    /// <paramref name="maxSessionBytes"/> is left unspecified, the per-session cap is derived as
    /// <c>maxResidentBytes / expectedConcurrentSessions</c>.
    ///
    /// <para><b>Why this exists.</b> Leaving both limits at their defaults makes the per-session
    /// cap equal to the whole global budget, which entitles ONE session to all of it: a session
    /// that starts first grows into the entire budget through rolling renewal and a later arrival
    /// is refused at admission. That is the configuration behaving as asked, but it is a footgun —
    /// the starving setup is the one you get by not thinking about it. Stating the concurrency you
    /// expect is easier to get right than computing a byte share by hand.</para>
    /// </summary>
    public int? ExpectedConcurrentSessions { get; }
    public IReadOnlyDictionary<string, long>? ModelBudgets { get; }

    public HotSessionRuntimeOptions(
        long maxResidentBytes = long.MaxValue,
        long maxSessionBytes = long.MaxValue,
        int maxCapturedOutputChunks = 4_096,
        int maxCompletedOperationRecords = 256,
        TimeSpan? completedOperationRetention = null,
        int? expectedConcurrentSessions = null,
        IReadOnlyDictionary<string, long>? modelBudgets = null)
    {
        if (expectedConcurrentSessions is <= 0)
            throw new ArgumentOutOfRangeException(nameof(expectedConcurrentSessions),
                "Expected concurrency must be at least 1.");
        // Derive the per-session share only when the caller did not state one explicitly: an
        // explicit cap is a deliberate decision and must not be silently overridden.
        if (expectedConcurrentSessions is { } concurrency
            && maxSessionBytes == long.MaxValue
            && maxResidentBytes != long.MaxValue)
        {
            maxSessionBytes = Math.Max(1, maxResidentBytes / concurrency);
        }
        if (maxResidentBytes <= 0) throw new ArgumentOutOfRangeException(nameof(maxResidentBytes));
        if (maxSessionBytes <= 0) throw new ArgumentOutOfRangeException(nameof(maxSessionBytes));
        if (maxCapturedOutputChunks < 4)
            throw new ArgumentOutOfRangeException(nameof(maxCapturedOutputChunks),
                "A captured response needs room for usage, stop and up to two decoder flush chunks.");
        if (maxCompletedOperationRecords <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxCompletedOperationRecords));
        var retention = completedOperationRetention ?? TimeSpan.FromHours(1);
        if (retention <= TimeSpan.Zero)
            throw new ArgumentOutOfRangeException(nameof(completedOperationRetention));
        MaxResidentBytes = maxResidentBytes;
        MaxSessionBytes = maxSessionBytes;
        ExpectedConcurrentSessions = expectedConcurrentSessions;
        MaxCapturedOutputChunks = maxCapturedOutputChunks;
        MaxCompletedOperationRecords = maxCompletedOperationRecords;
        CompletedOperationRetention = retention;
        ModelBudgets = modelBudgets;
    }
}

/// <summary>Raised before admission when projected retained state exceeds a configured byte limit.</summary>
public sealed class SessionResourceBudgetExceededException(long requestedBytes, long availableBytes, long sessionLimit)
    : InvalidOperationException(
        $"Session state requires {requestedBytes} bytes; available global capacity is {availableBytes} bytes and the per-session limit is {sessionLimit} bytes.")
{
    public long RequestedBytes { get; } = requestedBytes;
    public long AvailableBytes { get; } = availableBytes;
    public long SessionLimit { get; } = sessionLimit;
}

/// <summary>
/// Tracks committed hot-state bytes plus temporary turn reservations. It does not evict: a session
/// with running work keeps its reservation until the operation commits, rolls back, or fails.
/// </summary>
internal sealed class SessionResourceBudget
{
    private readonly object _gate = new();
    private readonly long _maxResidentBytes;
    private readonly long _maxSessionBytes;
    private readonly Dictionary<SessionId, long> _resident = [];
    private readonly Dictionary<string, long> _modelBudgets = [];
    private readonly Dictionary<string, long> _modelResidentBytes = [];
    private readonly Dictionary<string, long> _modelReservedBytes = [];
    private readonly Dictionary<SessionId, string> _sessionModel = [];
    private long _residentBytes;
    private long _reservedBytes;

    public SessionResourceBudget(HotSessionRuntimeOptions options)
    {
        _maxResidentBytes = options.MaxResidentBytes;
        _maxSessionBytes = options.MaxSessionBytes;
        if (options.ModelBudgets is { } modelBudgets)
        {
            foreach (var (modelKey, maxBytes) in modelBudgets)
            {
                if (maxBytes > 0) _modelBudgets[modelKey] = maxBytes;
            }
        }
    }

    public long ResidentBytes
    {
        get { lock (_gate) return _residentBytes; }
    }

    public void SetModelBudget(string modelKey, long maxBytes)
    {
        ArgumentException.ThrowIfNullOrEmpty(modelKey);
        if (maxBytes <= 0) throw new ArgumentOutOfRangeException(nameof(maxBytes));
        lock (_gate)
        {
            _modelBudgets[modelKey] = maxBytes;
        }
    }

    public long GetModelResidentBytes(string modelKey)
    {
        ArgumentException.ThrowIfNullOrEmpty(modelKey);
        lock (_gate)
        {
            return _modelResidentBytes.GetValueOrDefault(modelKey);
        }
    }

    public SessionResourceReservation Reserve(SessionId sessionId, long projectedBytes, string? modelKey = null)
    {
        if (projectedBytes < 0) throw new ArgumentOutOfRangeException(nameof(projectedBytes));
        lock (_gate)
        {
            long current = _resident.GetValueOrDefault(sessionId);
            long available = checked(_maxResidentBytes - _residentBytes - _reservedBytes + current);
            if (projectedBytes > _maxSessionBytes || projectedBytes > available)
                throw new SessionResourceBudgetExceededException(projectedBytes, Math.Max(0, available), _maxSessionBytes);

            string key = modelKey ?? _sessionModel.GetValueOrDefault(sessionId) ?? "default";

            // Reattribute resident bytes if model key changes (#4)
            if (_sessionModel.TryGetValue(sessionId, out var priorKey) && priorKey != key && current > 0)
            {
                long oldRes = _modelResidentBytes.GetValueOrDefault(priorKey);
                long newOldRes = Math.Max(0, checked(oldRes - current));
                if (newOldRes == 0) _modelResidentBytes.Remove(priorKey);
                else _modelResidentBytes[priorKey] = newOldRes;

                long newRes = _modelResidentBytes.GetValueOrDefault(key);
                _modelResidentBytes[key] = checked(newRes + current);
            }

            if (_modelBudgets.TryGetValue(key, out var modelMax))
            {
                long modelResident = _modelResidentBytes.GetValueOrDefault(key);
                long modelReserved = _modelReservedBytes.GetValueOrDefault(key);
                long modelAvailable = checked(modelMax - modelResident - modelReserved + current);
                if (projectedBytes > modelAvailable)
                    throw new SessionResourceBudgetExceededException(projectedBytes, Math.Max(0, modelAvailable), modelMax);
            }

            _sessionModel[sessionId] = key;
            long reservation = checked(projectedBytes - current);
            _reservedBytes = checked(_reservedBytes + reservation);

            long currentModelReserved = _modelReservedBytes.GetValueOrDefault(key);
            _modelReservedBytes[key] = checked(currentModelReserved + reservation);

            return new SessionResourceReservation(this, sessionId, current, reservation, key);
        }
    }

    public void Remove(SessionId sessionId)
    {
        lock (_gate)
        {
            if (_resident.Remove(sessionId, out var bytes))
            {
                _residentBytes = checked(_residentBytes - bytes);
                if (_sessionModel.Remove(sessionId, out var modelKey) && bytes > 0)
                {
                    long currentModelBytes = _modelResidentBytes.GetValueOrDefault(modelKey);
                    long updatedModelBytes = Math.Max(0, checked(currentModelBytes - bytes));
                    if (updatedModelBytes == 0) _modelResidentBytes.Remove(modelKey);
                    else _modelResidentBytes[modelKey] = updatedModelBytes;
                }
            }
        }
    }

    /// <summary>Reconciles committed residency after a compensated turn rollback.</summary>
    public void SetResidentBytes(SessionId sessionId, long actualBytes)
    {
        lock (_gate)
        {
            if (actualBytes < 0 || actualBytes > _maxSessionBytes)
                throw new ArgumentOutOfRangeException(nameof(actualBytes));
            long prior = _resident.GetValueOrDefault(sessionId);
            _residentBytes = checked(_residentBytes - prior + actualBytes);
            if (actualBytes == 0) _resident.Remove(sessionId);
            else _resident[sessionId] = actualBytes;

            if (_sessionModel.TryGetValue(sessionId, out var modelKey))
            {
                long priorModel = _modelResidentBytes.GetValueOrDefault(modelKey);
                long newModel = Math.Max(0, checked(priorModel - prior + actualBytes));
                if (newModel == 0) _modelResidentBytes.Remove(modelKey);
                else _modelResidentBytes[modelKey] = newModel;
            }
        }
    }

    private void Complete(SessionId sessionId, long priorBytes, long reservationBytes, long actualBytes, string modelKey)
    {
        lock (_gate)
        {
            if (actualBytes < 0 || actualBytes > _maxSessionBytes)
                throw new ArgumentOutOfRangeException(nameof(actualBytes));

            _reservedBytes = checked(_reservedBytes - reservationBytes);
            _residentBytes = checked(_residentBytes - priorBytes + actualBytes);
            if (actualBytes == 0) _resident.Remove(sessionId);
            else _resident[sessionId] = actualBytes;

            long priorModelReserved = _modelReservedBytes.GetValueOrDefault(modelKey);
            long newModelReserved = Math.Max(0, checked(priorModelReserved - reservationBytes));
            if (newModelReserved == 0) _modelReservedBytes.Remove(modelKey);
            else _modelReservedBytes[modelKey] = newModelReserved;

            long priorModelResident = _modelResidentBytes.GetValueOrDefault(modelKey);
            long newModelResident = Math.Max(0, checked(priorModelResident - priorBytes + actualBytes));
            if (newModelResident == 0) _modelResidentBytes.Remove(modelKey);
            else _modelResidentBytes[modelKey] = newModelResident;
        }
    }

    internal bool TryRenewReservation(SessionId sessionId, long priorBytes, ref long reservationBytes, long newProjectedBytes, string modelKey)
    {
        lock (_gate)
        {
            if (newProjectedBytes < 0) return false;
            long current = _resident.GetValueOrDefault(sessionId);
            long available = checked(_maxResidentBytes - _residentBytes - _reservedBytes + current);
            long neededReservation = checked(newProjectedBytes - current);
            long additionalNeeded = checked(neededReservation - reservationBytes);

            if (_modelBudgets.TryGetValue(modelKey, out var modelMax))
            {
                // NOTE the asymmetry with Reserve, which adds `current` back: Reserve compares an
                // ABSOLUTE projection against its headroom, so it must credit what this session
                // already holds. Here the comparison is a DELTA (additionalNeeded), and
                // _modelReservedBytes ALREADY contains this session's reservationBytes — crediting
                // it back a second time would let a renewing session exceed the model cap by the
                // size of its own reservation. Rolling renewal runs every decode step, so that is a
                // continuous leak of the guarantee, not a corner case.
                long modelResident = _modelResidentBytes.GetValueOrDefault(modelKey);
                long modelReserved = _modelReservedBytes.GetValueOrDefault(modelKey);
                long modelAvailable = checked(modelMax - modelResident - modelReserved);
                if (additionalNeeded > modelAvailable) return false;
            }

            if (additionalNeeded <= 0)
            {
                // Shrinking. This path deliberately skips the _maxSessionBytes ceiling check
                // below: a reservation that is getting SMALLER cannot newly breach a ceiling it
                // already satisfied. Asserted rather than assumed, because the day someone calls
                // TryRenew to shrink from an already-over-limit state, silently accepting it would
                // launder a budget violation into a valid reservation.
                if (newProjectedBytes > _maxSessionBytes) return false;
                _reservedBytes = checked(_reservedBytes + additionalNeeded);

                long currentModelRes = _modelReservedBytes.GetValueOrDefault(modelKey);
                _modelReservedBytes[modelKey] = Math.Max(0, checked(currentModelRes + additionalNeeded));

                reservationBytes = neededReservation;
                return true;
            }
            if (newProjectedBytes > _maxSessionBytes || additionalNeeded > available)
                return false;

            _reservedBytes = checked(_reservedBytes + additionalNeeded);

            long modelRes = _modelReservedBytes.GetValueOrDefault(modelKey);
            _modelReservedBytes[modelKey] = checked(modelRes + additionalNeeded);

            reservationBytes = neededReservation;
            return true;
        }
    }

    internal sealed class SessionResourceReservation : IDisposable
    {
        private SessionResourceBudget? _owner;
        private readonly SessionId _sessionId;
        private readonly long _priorBytes;
        private long _reservationBytes;
        private readonly string _modelKey;

        internal SessionResourceReservation(SessionResourceBudget owner, SessionId sessionId, long priorBytes, long reservationBytes, string modelKey)
        {
            _owner = owner;
            _sessionId = sessionId;
            _priorBytes = priorBytes;
            _reservationBytes = reservationBytes;
            _modelKey = modelKey;
        }

        public bool TryRenew(long newProjectedBytes)
        {
            var owner = _owner;
            if (owner is null) return false;
            return owner.TryRenewReservation(_sessionId, _priorBytes, ref _reservationBytes, newProjectedBytes, _modelKey);
        }

        public void Complete(long actualBytes)
        {
            var owner = Interlocked.Exchange(ref _owner, null)
                ?? throw new InvalidOperationException("The resource reservation has already completed.");
            owner.Complete(_sessionId, _priorBytes, _reservationBytes, actualBytes, _modelKey);
        }

        public void Dispose()
        {
            var owner = Interlocked.Exchange(ref _owner, null);
            owner?.Complete(_sessionId, _priorBytes, _reservationBytes, _priorBytes, _modelKey);
        }
    }
}
