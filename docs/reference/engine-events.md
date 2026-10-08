# Engine events (telemetry bus)

`OpenTail.Stingray.Core.EngineEvents` is a process-wide, in-process event bus. The engine reports what it did; it never decides who listens. The CLI, a server, a benchmark or a profiler subscribe independently, so no logging code lives in the hot paths. Static and reflection-free (NativeAOT-safe). With no subscriber, `Emit` is a single volatile read; guard any work done only to build an event with `EngineEvents.Enabled`. A throwing sink is isolated: telemetry can never affect inference.

```csharp
using var sub = EngineEvents.Subscribe(new MySink());   // IEngineEventSink.OnEvent(in EngineEvent)
```

| Kind | Emitted by | Payload |
|---|---|---|
| `ModelLoaded` | `RuntimeInstance.Create` | `Name` architecture, `Ms` build time, `Detail` `forwardPass=<kind>` |
| `PrefillCompleted` | CLI single-prompt paths | `A` prompt tokens, `Ms` |
| `DecodeCompleted` | CLI single-prompt paths | `A` tokens generated (incl. hidden thinking), `B` visible tokens, `Ms` |
| `MtpStep` | `MtpDecoder` (batched verify) | `A` position, `B` drafts proposed, `C` drafts accepted |
| `GcStats` | CLI process exit | `A`/`B`/`C` gen0/1/2 collections, `Ms` total GC pause, `Detail` sizes |

**CLI:** `STINGRAY_EVENTS=1` prints one JSON object per event to **stderr** (`EngineEvents.ToJsonLine`). Redirect stderr (`2> events.jsonl`) to consume it: events can start mid-line in a terminal because stdout streams tokens without newlines. `STINGRAY_GC_STATS=1` additionally prints the human-readable `[gc]` line.

Not yet instrumented (add when a consumer needs them): batch started/finished in `ContinuousBatchingEngine`, server request timing (`ServerMetrics` already exports metrics), cache hits/misses. Add a new `EngineEventKind` member and document its payload in the enum; keep payloads to the existing generic slots so emitting stays allocation-free.
