namespace OpenTail.Stingray.Tests.Core;

// EngineEvents is process-wide static state; xunit runs classes in parallel, so every test here uses only its own sink and filters
// by a unique Name, and no test asserts on the global Enabled flag while another could be subscribed.
public sealed class EngineEventsTests
{
    private sealed class Recorder : IEngineEventSink
    {
        public readonly List<EngineEvent> Seen = [];
        public void OnEvent(in EngineEvent e) { lock (Seen) Seen.Add(e); }
        public int Count(string name) { lock (Seen) return Seen.Count(x => x.Name == name); }
    }

    private sealed class Throwing : IEngineEventSink
    {
        public void OnEvent(in EngineEvent e) => throw new InvalidOperationException("sink bug");
    }

    [Fact]
    public void Subscribed_SinkReceivesEvent_UnsubscribedDoesNot()
    {
        string name = "t-" + Guid.NewGuid();
        var rec = new Recorder();
        using (EngineEvents.Subscribe(rec))
        {
            Assert.True(EngineEvents.Enabled);
            EngineEvents.Emit(new(EngineEventKind.PrefillCompleted, name, A: 12, Ms: 3.5));
        }
        EngineEvents.Emit(new(EngineEventKind.PrefillCompleted, name, A: 99));   // after Dispose
        Assert.Equal(1, rec.Count(name));
        Assert.Equal(12, rec.Seen.Single(e => e.Name == name).A);
    }

    [Fact]
    public void ThrowingSink_NeverReachesTheCaller_AndOthersStillReceive()
    {
        string name = "t-" + Guid.NewGuid();
        var rec = new Recorder();
        using var bad = EngineEvents.Subscribe(new Throwing());
        using var good = EngineEvents.Subscribe(rec);
        EngineEvents.Emit(new(EngineEventKind.MtpStep, name, A: 1));   // must not throw
        Assert.Equal(1, rec.Count(name));
    }

    [Fact]
    public void DoubleDispose_IsHarmless_AndConcurrentEmitsAreAllDelivered()
    {
        string name = "t-" + Guid.NewGuid();
        var rec = new Recorder();
        var sub = EngineEvents.Subscribe(rec);
        Parallel.For(0, 200, i => EngineEvents.Emit(new(EngineEventKind.DecodeCompleted, name, A: i)));
        sub.Dispose();
        sub.Dispose();
        Assert.Equal(200, rec.Count(name));
    }

    [Fact]
    public void JsonLine_IsInvariantEscapedAndOmitsZeroFields()
    {
        string line = EngineEvents.ToJsonLine(new(EngineEventKind.ModelLoaded, "ar\"ch\n", Ms: 1234.5678, Detail: "forwardPass=CpuDense"));
        Assert.Equal("{\"event\":\"ModelLoaded\",\"name\":\"ar\\\"ch\\n\",\"ms\":1234.568,\"detail\":\"forwardPass=CpuDense\"}", line);
        Assert.Equal("{\"event\":\"MtpStep\",\"a\":20,\"b\":1,\"c\":1}", EngineEvents.ToJsonLine(new(EngineEventKind.MtpStep, A: 20, B: 1, C: 1)));
    }
}
