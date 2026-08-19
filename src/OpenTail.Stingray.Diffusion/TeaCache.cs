using System;
using System.Collections.Generic;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

namespace OpenTail.Stingray.Diffusion;

/// <summary>
/// Mode of operation for TeaCache / TaylorSeer acceleration.
/// </summary>
public enum TeaCacheMode
{
    /// <summary>
    /// Computes full forward passes at fixed intervals, extrapolating in between via Taylor series expansion.
    /// </summary>
    FixedInterval,

    /// <summary>
    /// Measures relative velocity in conditioning/hidden representations and skips passes when cumulative change is below threshold.
    /// </summary>
    RelativeVelocityThreshold,

    /// <summary>
    /// Dynamic hybrid: checks relative velocity threshold while capping the maximum consecutive skipped steps.
    /// </summary>
    Hybrid
}

/// <summary>
/// Configuration for TeaCache / TaylorSeer caching.
/// Reference: "TeaCache: Timestep Residual Velocity Caching for Accelerated Diffusion" &amp; "TaylorSeer" (diffusers.hooks.taylorseer_cache)
/// </summary>
public sealed record TeaCacheConfig
{
    /// <summary>
    /// Operating mode (default: Hybrid for optimal quality-speed balance).
    /// </summary>
    public TeaCacheMode Mode { get; init; } = TeaCacheMode.Hybrid;

    /// <summary>
    /// Interval between full computation steps (in FixedInterval or Hybrid mode). Default: 3.
    /// </summary>
    public int CacheInterval { get; init; } = 3;

    /// <summary>
    /// Relative L1 difference threshold to trigger full computation. Default: 0.15 (15% relative change).
    /// </summary>
    public float Threshold { get; init; } = 0.15f;

    /// <summary>
    /// Initial warmup steps that always execute full forward passes to gather baseline trajectory. Default: 2.
    /// </summary>
    public int WarmupSteps { get; init; } = 2;

    /// <summary>
    /// Optional step index after which caching is disabled (e.g. for final detail refinement).
    /// </summary>
    public int? CooldownStep { get; init; } = null;

    /// <summary>
    /// Maximum Taylor series expansion order (0 = zero-order hold, 1 = velocity, 2 = acceleration). Default: 1.
    /// </summary>
    public int MaxOrder { get; init; } = 1;
}

/// <summary>
/// State tracking and Taylor series extrapolation engine for a single cached module/layer or full transformer output.
/// </summary>
public sealed class TeaCacheState
{
    private readonly int _maxOrder;
    private readonly List<float[]> _taylorFactors = []; // order 0, 1, 2...
    private float[]? _lastIndicator;
    private float _accumulatedDelta;
    private int? _lastUpdateStep;

    public int? LastUpdateStep => _lastUpdateStep;
    public float AccumulatedDelta => _accumulatedDelta;
    public bool HasFactors => _taylorFactors.Count > 0;

    public TeaCacheState(int maxOrder = 1)
    {
        _maxOrder = Math.Clamp(maxOrder, 0, 2);
    }

    /// <summary>
    /// Resets all cached derivatives and timestep state between generation runs.
    /// </summary>
    public void Reset()
    {
        _taylorFactors.Clear();
        _lastIndicator = null;
        _accumulatedDelta = 0.0f;
        _lastUpdateStep = null;
    }

    /// <summary>
    /// Evaluates whether a full computation is required for the given step and optional indicator representation.
    /// </summary>
    public bool ShouldCompute(int currentStep, TeaCacheConfig config, ReadOnlySpan<float> indicator = default)
    {
        // 1. Warmup phase: always compute
        if (currentStep < config.WarmupSteps)
            return true;

        // 2. Cooldown phase: always compute if past cooldown threshold
        if (config.CooldownStep.HasValue && currentStep >= config.CooldownStep.Value)
            return true;

        // 3. If we have no cached state yet, must compute
        if (_lastUpdateStep == null || _taylorFactors.Count == 0)
            return true;

        int stepDelta = currentStep - _lastUpdateStep.Value;

        switch (config.Mode)
        {
            case TeaCacheMode.FixedInterval:
                return stepDelta >= config.CacheInterval;

            case TeaCacheMode.RelativeVelocityThreshold:
                if (indicator.IsEmpty || _lastIndicator == null)
                    return stepDelta >= config.CacheInterval;

                float relDiff = ComputeRelativeDifference(indicator, _lastIndicator);
                _accumulatedDelta += relDiff;
                return _accumulatedDelta >= config.Threshold;

            case TeaCacheMode.Hybrid:
            default:
                if (stepDelta >= config.CacheInterval)
                    return true;

                if (!indicator.IsEmpty && _lastIndicator != null)
                {
                    float diff = ComputeRelativeDifference(indicator, _lastIndicator);
                    _accumulatedDelta += diff;
                    return _accumulatedDelta >= config.Threshold;
                }
                return false;
        }
    }

    /// <summary>
    /// Updates Taylor series derivative factors with the new full computation output.
    /// </summary>
    public void Update(int currentStep, ReadOnlySpan<float> output, ReadOnlySpan<float> indicator = default)
    {
        int length = output.Length;
        int deltaStep = _lastUpdateStep.HasValue ? Math.Max(1, currentStep - _lastUpdateStep.Value) : 1;

        if (_taylorFactors.Count == 0 || _taylorFactors[0].Length != length)
        {
            _taylorFactors.Clear();
            var order0 = new float[length];
            output.CopyTo(order0);
            _taylorFactors.Add(order0);
        }
        else
        {
            var prevOrder0 = _taylorFactors[0];
            var newOrder0 = new float[length];
            output.CopyTo(newOrder0);

            // Compute 1st order divided difference: (new - prev) / deltaStep
            if (_maxOrder >= 1)
            {
                var order1 = new float[length];
                float invDelta = 1.0f / deltaStep;

                Parallel.For(0, length, i =>
                {
                    order1[i] = (newOrder0[i] - prevOrder0[i]) * invDelta;
                });

                // Compute 2nd order divided difference if maxOrder >= 2
                if (_maxOrder >= 2 && _taylorFactors.Count >= 2)
                {
                    var prevOrder1 = _taylorFactors[1];
                    var order2 = new float[length];

                    Parallel.For(0, length, i =>
                    {
                        order2[i] = (order1[i] - prevOrder1[i]) * invDelta;
                    });

                    if (_taylorFactors.Count < 3)
                        _taylorFactors.Add(order2);
                    else
                        _taylorFactors[2] = order2;
                }

                if (_taylorFactors.Count < 2)
                    _taylorFactors.Add(order1);
                else
                    _taylorFactors[1] = order1;
            }

            _taylorFactors[0] = newOrder0;
        }

        if (!indicator.IsEmpty)
        {
            _lastIndicator ??= new float[indicator.Length];
            if (_lastIndicator.Length != indicator.Length)
                _lastIndicator = new float[indicator.Length];
            indicator.CopyTo(_lastIndicator);
        }

        _accumulatedDelta = 0.0f;
        _lastUpdateStep = currentStep;
    }

    /// <summary>
    /// Predicts output tensor at the specified step using Taylor series polynomial expansion.
    /// </summary>
    public void Predict(int currentStep, Span<float> destination)
    {
        if (_lastUpdateStep == null || _taylorFactors.Count == 0)
            throw new InvalidOperationException("Cannot predict without prior initialization/update.");

        int length = _taylorFactors[0].Length;
        if (destination.Length < length)
            throw new ArgumentException("Destination span too small.", nameof(destination));

        float stepOffset = currentStep - _lastUpdateStep.Value;

        // Order 0: Base output
        var order0 = _taylorFactors[0];
        order0.AsSpan().CopyTo(destination);

        // Order 1: + (Factor1 * dt)
        if (_maxOrder >= 1 && _taylorFactors.Count >= 2)
        {
            var order1 = _taylorFactors[1];
            for (int i = 0; i < length; i++)
            {
                destination[i] += order1[i] * stepOffset;
            }
        }

        // Order 2: + (Factor2 * dt^2 / 2)
        if (_maxOrder >= 2 && _taylorFactors.Count >= 3)
        {
            var order2 = _taylorFactors[2];
            float coeff2 = (stepOffset * stepOffset) * 0.5f;
            for (int i = 0; i < length; i++)
            {
                destination[i] += order2[i] * coeff2;
            }
        }
    }

    /// <summary>
    /// Predicts output tensor at the specified step and returns a newly allocated array.
    /// </summary>
    public float[] Predict(int currentStep)
    {
        if (_lastUpdateStep == null || _taylorFactors.Count == 0)
            throw new InvalidOperationException("Cannot predict without prior initialization/update.");

        var result = new float[_taylorFactors[0].Length];
        Predict(currentStep, result);
        return result;
    }

    private static float ComputeRelativeDifference(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        int length = Math.Min(a.Length, b.Length);
        if (length == 0) return 0.0f;

        float sumDiff = 0.0f;
        float sumBase = 0.0f;

        for (int i = 0; i < length; i++)
        {
            sumDiff += MathF.Abs(a[i] - b[i]);
            sumBase += MathF.Abs(b[i]);
        }

        float meanDiff = sumDiff / length;
        float meanBase = sumBase / length;

        return meanBase > 1e-6f ? (meanDiff / meanBase) : meanDiff;
    }
}

/// <summary>
/// High-level TeaCache hook and manager for orchestrating timestep residual caching across diffusion sampling loops.
/// </summary>
public sealed class TeaCacheHook
{
    private readonly TeaCacheConfig _config;
    private readonly Dictionary<string, TeaCacheState> _states = new(StringComparer.Ordinal);

    public TeaCacheConfig Config => _config;

    public TeaCacheHook(TeaCacheConfig? config = null)
    {
        _config = config ?? new TeaCacheConfig();
    }

    /// <summary>
    /// Resets all registered module/layer cache states.
    /// </summary>
    public void Reset()
    {
        foreach (var state in _states.Values)
        {
            state.Reset();
        }
    }

    /// <summary>
    /// Gets or creates a cache state for a named module/layer (e.g. "transformer", "block.12").
    /// </summary>
    public TeaCacheState GetState(string moduleName = "default")
    {
        if (!_states.TryGetValue(moduleName, out var state))
        {
            state = new TeaCacheState(_config.MaxOrder);
            _states[moduleName] = state;
        }
        return state;
    }

    /// <summary>
    /// Checks whether the specified module should compute its full forward pass.
    /// </summary>
    public bool ShouldCompute(int currentStep, string moduleName = "default", ReadOnlySpan<float> indicator = default)
    {
        return GetState(moduleName).ShouldCompute(currentStep, _config, indicator);
    }

    /// <summary>
    /// Updates the cached state after a full forward pass.
    /// </summary>
    public void Update(int currentStep, ReadOnlySpan<float> output, string moduleName = "default", ReadOnlySpan<float> indicator = default)
    {
        GetState(moduleName).Update(currentStep, output, indicator);
    }

    /// <summary>
    /// Predicts the module output without running the full forward pass.
    /// </summary>
    public float[] Predict(int currentStep, string moduleName = "default")
    {
        return GetState(moduleName).Predict(currentStep);
    }

    /// <summary>
    /// Predicts the module output directly into a destination span.
    /// </summary>
    public void Predict(int currentStep, Span<float> destination, string moduleName = "default")
    {
        GetState(moduleName).Predict(currentStep, destination);
    }
}
