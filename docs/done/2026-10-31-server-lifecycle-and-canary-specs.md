# Specs: HTTP disconnect test (plan 2 P8) and server telemetry canary (plan 2 P5)

Status: **both specs are implemented** (Spec A, the disconnect test: `ClientDisconnectTests`, PR 92; Spec B, the telemetry canary: PR 80). Written against main at b9b00e1 (2026-10-31) as specifications; kept here as the record of what was asked for.
Parent plan: [2026-10-31-autonomous-plan-2.md](../2026-10-31-autonomous-plan-2.md); rows P5 and P8 are now in its [done half](2026-10-31-autonomous-plan-2-completed-items-2026-10.md).

## Contracts this must satisfy (read 2026-10-31)

| Member | Promise | What the spec relies on |
|---|---|---|
| `IInferenceService.AcquireAsync(ModelId, CancellationToken)` (`src/OpenTail.Stingray.Server/InferenceService.cs:38`) | Returns a `ModelRuntimeHandle`; throws `InsufficientResourcesException` when it cannot admit | `OpenAiEndpoints.cs:85` maps the throw to 503 |
| `ModelRuntimeHandle.Dispose` (`src/OpenTail.Stingray.Server/ModelRuntime.cs:283`) | Releases the runtime reference once (`Interlocked.Exchange`), then notifies residency | `OpenAiEndpoints.cs:111` `using var handleGuard = handle;` |
| `ConcurrencyLimit` filter (`src/OpenTail.Stingray.Server/Endpoints/ConcurrencyLimit.cs:60-85`) | `TryEnter` admits or returns 429; `Exit()` in `finally` releases the slot on normal return, exception, or mid-stream abort | The streaming handler returns normally on abort, so the slot is released |
| `IInferenceEngine.GenerateChunksAsync` (`src/OpenTail.Stingray.Engine/IInferenceEngine.cs:67`) | Async enumerable that honours its cancellation token | `HandleStreaming` iterates with `ctx.RequestAborted` (`OpenAiEndpoints.cs:~646`) |
| Streaming catch-all (`OpenAiEndpoints.cs:~682`) | `catch (Exception ex) when (ex is not OperationCanceledException)` | A client disconnect is not logged as a generation failure |
| Streaming `finally` (`OpenAiEndpoints.cs:~690-712`) | Writes after disconnect are swallowed (`OperationCanceledException`, `IOException`) | No exception escapes after the response is committed |
| `ChatCompletionRequest` (`OpenAiEndpoints.cs:985`) | Fields: `Model`, `Messages`, `MaxTokens`, `Temperature`, `TopP`, `Stream`, `LogitBias`, `ResponseFormat`, `EnableThinking`, `MaxThinkingTokens`, `Tools`, `ToolChoice`, and more. **No `user` field** | An unknown `user` property is ignored by deserialisation, so it never reaches telemetry |
| `FakeInferenceEngine` (`tests/OpenTail.Stingray.Tests.Server.Fast/ServerTests.cs:1567`) | `Hold` blocks generation; `Entered` signals start; scripted output | Reusable, but a mid-stream disconnect needs a new per-token gate option (see Spec A) |
| `ConcurrencyLimitTests` (`tests/OpenTail.Stingray.Tests.Server.Fast/ConcurrencyLimitTests.cs`) | `WebApplicationFactory<Program>` with `MaxConcurrentRequests`, `MaxQueuedRequests`, and `IInferenceEngine` replaced by the fake | Same host construction is reused |
| `ServerTelemetryContractTests` (`tests/OpenTail.Stingray.Tests.Server.Fast/ServerTelemetryContractTests.cs:62`) | Uses `MeterListener` for metrics | Spans need an `ActivityListener`, as in `ModelLoadWaitTelemetryTests.cs:41` |

## Spec A: client disconnect stops generation and frees resources (P8)

File (proposed): `tests/OpenTail.Stingray.Tests.Server.Fast/ClientDisconnectTests.cs`

Setup:
- Host built like `ConcurrencyLimitTests.CreateClient`, with `MaxConcurrentRequests = 1`, so a leaked slot shows up as a 429.
- A variant of `FakeInferenceEngine` whose stream yields one token per step, each step awaiting a gate the test controls. The engine records how many tokens it produced and whether its cancellation token fired. The existing `Hold` blocks before any output, so it cannot test a mid-stream disconnect.

Steps:
1. Open a streaming `POST /v1/chat/completions` (`stream: true`) with `HttpCompletionOption.ResponseHeadersRead`.
2. Read the first SSE line (the role delta). This proves generation is in flight.
3. Assert the engine's `Entered` signal is set and at least one token was produced.
4. Dispose the response and cancel the client request. This is a real disconnect, not a flag.
5. Release the engine's gate once, so the operation already in progress can finish.
6. Within a bounded wait (5 s), assert:
   - the engine's cancellation token reports cancelled;
   - the token count is frozen after the cancellation point;
   - the enumerator completes; nothing hangs.
7. Assert the slot is free: a second request returns 200, not 429.
8. Assert the runtime handle is released. Preferred: an existing residency or status counter (check `ServerStatusSnapshot`). Otherwise, a test-only counter incremented in the fake's `Dispose`. Do not assert on private fields.
9. Send a normal non-streaming request. It must succeed.

Logging assertion: capture logger output. "Generation failed after the streaming response was committed" must not appear. This pins the `when (ex is not OperationCanceledException)` filter.

Mutation checks (each must make the test fail; apply on a scratch branch, record the failure, revert, do not commit):
- **M1:** pass `CancellationToken.None` instead of `ctx.RequestAborted` to `ImageContent.Generate` (`OpenAiEndpoints.cs:~646`). Expected failure: step 6.
- **M2:** remove the handle dispose (`OpenAiEndpoints.cs:111`). Expected failure: step 8.
- **M3:** remove the `when (ex is not OperationCanceledException)` filter. Expected failure: the logging assertion.
- **M4 (slot):** remove `gate.Exit()` from the `finally` in `ConcurrencyLimit.cs:80`. Expected failure: step 7.

Scope: OpenAI chat streaming only. Anthropic (`AnthropicEndpoints.cs`, which also passes `ctx.RequestAborted`) and Responses need the same test as follow-ups.

Timing: no sleeps. Use the `Entered` signal and bounded waits. No real model and no `STINGRAY_RUN_HEAVY_TESTS`, so the class belongs in the `.Fast` project.

## Spec B: server-side telemetry canary (P5, server half)

Existing coverage, not duplicated:
- `tests/OpenTail.Stingray.Tests.ForwardPass.Fast/EngineTelemetryPrivacyTests.cs` sends `CANARY-7f3a91-private` and a Windows path canary through both engines. It covers the prompt only, at the engine level.
- `ServerTelemetryContractTests.cs` and `ServerTelemetryFailureTests.cs` contain no canary.

File (proposed): `tests/OpenTail.Stingray.Tests.Server.Fast/ServerTelemetryCanaryTests.cs`

Canaries (each unique, greppable):
- prompt: `CANARY-PROMPT-<guid>`
- system message: `CANARY-SYSTEM-<guid>`
- completion: the fake emits `CANARY-COMPLETION-<guid>`
- tool arguments: a tool definition plus a tool-call output containing `CANARY-TOOLARG-<guid>`
- **Not included:** a `user` identifier. `ChatCompletionRequest` has no such field, so the value is dropped before it reaches anything. A test for that would assert nothing useful. If the team wants a guarantee that unknown fields never reach telemetry, that is a separate test.
- **Not included:** a model path such as `C:\Users\canary-user\...`. The request's `model` field is a model id, so the value would fail model resolution before telemetry is emitted. Add a negative case only if the endpoint's error response is itself in scope for the privacy claim.

Steps:
1. Start a `MeterListener` and an `ActivityListener` (metrics and spans; the existing contract test has only the meter half).
2. Send one streaming and one non-streaming chat request containing the canaries, including a tool call.
3. Send one failing request (malformed JSON or unknown model) containing a prompt canary, to cover error-path telemetry.
4. Collect metric attributes, span tags, span events, span links, and exception messages on error spans.
5. Assert no canary appears in any collected string (case-insensitive). Assert nothing contains `canary`.
6. Positive control: assert the capture is non-empty and includes the request-count metric. Without this, the negative assertions can pass on an empty capture, which is the failure mode CLAUDE.md rule 12 warns about.

Mutation checks (apply on a scratch branch, record, revert, do not commit):
- **M5:** add `activity.SetTag("prompt", prompt)` to the chat-completion span (`OpenAiEndpoints.cs`, where `BeginServingRequest` is called, around lines 423 and 542). Expected failure: step 5, prompt canary.
- **M6:** add the model id or a message content value as a metric tag. Expected failure: step 5.

Scope: chat endpoint only.

## Explicitly not in these specs
- The global allow-list test (every instrument, attribute, event and link against the privacy list). It needs the full inventory from `docs/3-product-and-runtime/2026-10-12-opentelemetry-observability-plan.md` and is a separate item under P5.
- Anthropic and Responses canaries.

## Open after this draft
1. Write both tests, and run each mutation.
2. Confirm the `ServerStatusSnapshot` counter for Spec A step 8, or decide on the test-only counter.
