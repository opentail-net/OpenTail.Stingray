using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// Unit tests for <see cref="IForwardPass"/> default interface implementations and execution context isolation
/// (docs/3-product-and-runtime/010-forward-pass-context-isolation-for-session-forking-plan.md).
/// </summary>
public sealed class ForwardPassTests
{
    private sealed class DefaultStatelessForwardPass : IForwardPass
    {
        public int VocabSize => 5;
        public int MaxSeqLen => 2048;
        public int LastPosition { get; set; }
        public int LastToken { get; set; }
        public float[] LogitsToReturn { get; set; } = [1f, 2f, 3f, 4f, 5f];

        public ReadOnlySpan<float> Forward(int token, int position)
        {
            LastToken = token;
            LastPosition = position;
            return LogitsToReturn;
        }

        public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0)
        {
            LastPosition = startPos + tokens.Count;
            return LogitsToReturn;
        }

        public void TruncateTo(int position) => LastPosition = position;
        public void ResetCache() => LastPosition = 0;
        public void Dispose() { }
    }

    private sealed class IsolatedStatefulForwardPass : IForwardPass
    {
        // Shared immutable weights
        private readonly float[] _sharedWeights;

        public int VocabSize => _sharedWeights.Length;
        public int MaxSeqLen => 2048;

        // Isolated mutable decode state
        public int Position { get; private set; }
        public int Token { get; private set; }

        public IsolatedStatefulForwardPass(float[] sharedWeights)
        {
            _sharedWeights = sharedWeights;
        }

        public IForwardPass CreateContext() => new IsolatedStatefulForwardPass(_sharedWeights);

        public ReadOnlySpan<float> Forward(int token, int position)
        {
            Token = token;
            Position = position;
            return _sharedWeights;
        }

        public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0)
        {
            Position = startPos + tokens.Count;
            return _sharedWeights;
        }

        public void TruncateTo(int position) => Position = position;
        public void ResetCache() => Position = 0;
        public void Dispose() { }
    }


    [Fact]
    public void IForwardPass_OverriddenCreateContext_ProducesIsolatedExecutionContext()
    {
        // Issue #010: Stateful implementations override CreateContext() to return an isolated execution context.
        var weights = new float[] { 0.1f, 0.5f, 0.9f };
        var parent = new IsolatedStatefulForwardPass(weights);
        var child = (IsolatedStatefulForwardPass)parent.CreateContext();

        Assert.NotSame(parent, child);

        // Generating on child updates child state without affecting parent
        child.Forward(token: 99, position: 5);
        Assert.Equal(99, child.Token);
        Assert.Equal(5, child.Position);

        Assert.Equal(0, parent.Token);
        Assert.Equal(0, parent.Position);
    }

    [Fact]
    public void IForwardPass_ForwardArgmax_StrictTieBreakingAndPeakLogitContract()
    {
        // Issue #219 / IForwardPass.ForwardArgmax default implementation:
        // Must perform a left-to-right strict-> host scan: highest value wins, lowest index on tie.
        var impl = new DefaultStatelessForwardPass
        {
            LogitsToReturn = [-5.0f, 10.0f, -2.0f, 10.0f, 3.0f]
        };
        IForwardPass fwd = impl;

        // Ties: indices 1 and 3 both have peak value 10.0f. Lowest index (1) must win.
        var (token, logit) = fwd.ForwardArgmax(token: 1, position: 0);
        Assert.Equal(1, token);
        Assert.Equal(10.0f, logit);

        // All negative logits
        impl.LogitsToReturn = [-12.5f, -3.2f, -8.7f, -3.2f];
        var (negToken, negLogit) = fwd.ForwardArgmax(token: 2, position: 1);
        Assert.Equal(1, negToken); // lowest index among ties at -3.2f
        Assert.Equal(-3.2f, negLogit);
    }
}
