# Request parameter audit, 2026-10-31 (plan 2, P7)

**Scope.** The request fields that the OpenAI chat endpoint (`POST /v1/chat/completions`) and the Anthropic messages
endpoint (`POST /v1/messages`) deserialise: the `ChatCompletionRequest` record in
`src/OpenTail.Stingray.Server/Endpoints/OpenAiEndpoints.cs` and the `AnthropicMessageRequest` record in
`src/OpenTail.Stingray.Server/Endpoints/AnthropicEndpoints.cs`. These DTOs are the only place the server declares what
it reads, so they are the surface this audit covers. Responses, embeddings, images and audio endpoints are out of scope
and need their own audit.

**Method.** Read each record, grep every use of each field in its endpoint, and count the test files that send the JSON
name (`grep` over `tests/OpenTail.Stingray.Tests.Server.Fast` and `tests/OpenTail.Stingray.Tests.Server`). Classification uses the three outcomes from plan 2 P7:
**implemented** (read and applied, with a test that sends it), **rejected** (refused with a stable error), or **ignored**
(read and dropped, or never declared, with a stated reason). A field that is never declared is dropped by
System.Text.Json without an error, so it is classed as **silently ignored**. That is the outcome the plan most wants to
avoid, because a client believes it has set something the server does not do.

Unknown JSON properties are accepted and ignored, and the server must keep doing so: clients send vendor fields.

## OpenAI chat (`ChatCompletionRequest`)

| Field | Read at | Tests that send it | Outcome |
|---|---|---|---|
| `model` | `src/OpenTail.Stingray.Server/Endpoints/OpenAiEndpoints.cs` (`ResolveModel`, multi-model routing) | 20 files | implemented |
| `messages` | `BuildMessageList` | 17 files | implemented |
| `max_tokens` | `SamplingParamsBuilder` → `MaxNewTokens` | 19 files | implemented; no range check (see F3) |
| `temperature` | `SamplingParamsBuilder` (default from server options) | 5 files | implemented; no range check (see F3) |
| `top_p` | `SamplingParamsBuilder` (default from server options) | 0 by JSON name (configuration only) | implemented, **untested as a request field** |
| `stream` | `req.Stream == true` selects `HandleStreaming` | 14 files | implemented |
| `logit_bias` | converted to a token-id dictionary (`src/OpenTail.Stingray.Server/Endpoints/OpenAiEndpoints.cs:208`) | 1 file | implemented |
| `response_format` | `json_schema` / `json_object`, schema-constrained decoding | 4 files | implemented |
| `tools` | tool-call prompt rendering and parsing | 7 files | implemented |
| `tool_choice` | `required`, named function, `none` | 2 files | implemented |
| `enable_thinking` (vendor) | thinking on or off | 2 files | implemented |
| `max_thinking_tokens` (vendor) | `SamplingParamsBuilder` → `maxThinking` | 1 file | implemented |
| `preserve_thinking` (vendor) | prior-turn `<think>` handling | 1 file | implemented |
| `opentail_timing` (vendor) | timing extension, non-streaming only | 2 files | implemented; **streaming ignores it** by design (documented in the record) |
| `presence_penalty` | `SamplingParamsBuilder` | 1 file | implemented |
| `frequency_penalty` | `SamplingParamsBuilder` | 1 file | implemented |
| `skills` (vendor) | `ApplySkills` folds instructions and tools into the request | 2 files | implemented |

Fields a client may send that the record **does not declare**, so they are silently ignored:

| Field | Why it matters | Outcome |
|---|---|---|
| `stop` | Stop sequences are common. A client that sends `stop` gets generation past its stop string, with no error. | **silently ignored (F1)** |
| `seed` | Reproducibility. The client believes sampling is seeded. | **silently ignored (F1)** |
| `n` | Multiple choices. The server always returns one. | **silently ignored (F1)** |
| `user` | End-user identifier. Dropping it is correct for privacy (plan section 7), but the drop is not documented. | silently ignored, intentional (F4) |
| `max_completion_tokens` | The newer name for `max_tokens`. A client that sends only this gets the server default budget. | **silently ignored (F1)** |
| `stream_options` | Usage in streaming. The streaming handler says it is unsupported, but nothing rejects the field. | silently ignored; the code comment at `src/OpenTail.Stingray.Server/Endpoints/OpenAiEndpoints.cs:~649` documents it (F2) |
| `parallel_tool_calls` | Tool-call parallelism. The server's behaviour is fixed. | **silently ignored (F1)** |
| `top_k` | Not an OpenAI field, but common in clients that also talk to llama.cpp. | **silently ignored (F1)** |

## Anthropic messages (`AnthropicMessageRequest`)

| Field | Read at | Tests that send it | Outcome |
|---|---|---|---|
| `model` | routing | (shared harness) | implemented |
| `messages` | message list build | (shared harness) | implemented |
| `max_tokens` (required) | `SamplingParamsBuilder` | (shared harness) | implemented; no range check (F3) |
| `system` (string or content blocks) | flattened by `ExtractTextContent` | (shared harness) | implemented |
| `stream` | streaming handler | (shared harness) | implemented |
| `temperature`, `top_p` | `SamplingParamsBuilder` | not sent by name | implemented, **untested as a request field** |
| `top_k` | `topK: req.TopK` (`src/OpenTail.Stingray.Server/Endpoints/AnthropicEndpoints.cs:204`) | none | implemented, **untested** |
| `thinking` | thinking configuration | 3 files | implemented |
| `tools` | tool rendering | (shared harness) | implemented |
| `preserve_thinking`, `opentail_timing`, `skills` (vendor) | as on the OpenAI side | — | implemented |

Fields a client may send that the record **does not declare**:

| Field | Why it matters | Outcome |
|---|---|---|
| `stop_sequences` | The Anthropic name for stop strings. Same risk as `stop` (F1). | **silently ignored (F1)** |
| `tool_choice` | Anthropic's tool-choice field. No code reads it. The OpenAI side honours it, the Anthropic side does not. | **silently ignored (F1)** |
| `metadata` | Caller metadata. Dropping it is correct for privacy. | silently ignored, intentional (F4) |

## Findings

**F1. Common fields are dropped without an error (the main risk).** `stop`, `seed`, `n`, `max_completion_tokens`,
`parallel_tool_calls`, `top_k` (OpenAI side), and `stop_sequences` and `tool_choice` (Anthropic side) are not declared in
either record, so System.Text.Json discards them. No test sends `stop`, and none sends `seed`, `max_completion_tokens` or
`stop_sequences`. A client that relies on any of them is silently misled. Options for the owner:
- implement the field, with a test (`stop` and `stop_sequences` are the cheapest to implement; `seed` depends on the
  sampler); or
- reject it with a stable 400 error that names the field. This is a decision, because a reject would also break clients
  that send the field only as a hint. Clients that pass unknown vendor fields must still work, so a reject can only apply
  to a known list of standard fields, not to unknown JSON properties; or
- keep it ignored and say so in the docs, with the reason.

**F2. Streaming usage.** `stream_options` is not declared. The streaming handler notes that
`stream_options.include_usage` is unsupported, but the code does not reject or report it, so the clear-error option
applies here too.

**F3. No range checks on sampling values.** Neither endpoint validates `temperature`, `top_p`, `max_tokens`, `top_k`,
`presence_penalty` or `frequency_penalty`. A negative `max_tokens`, a `top_p` above 1, or a `temperature` below 0 is passed
to the engine, and the behaviour for those values is not tested. Each needs a defined outcome (clamp with a documented
rule, or reject with a 400), and a test per row.

**F4. Intentional drops are undocumented.** `user` and `metadata` are dropped on purpose, to keep client identifiers out
of telemetry (plan section 7). That is the right behaviour, but no document says so. This table is where that reason
should be recorded.

**F5. Several supported fields have no request-level test.** `top_p`, `top_k` (Anthropic), `stream_options` (N/A) and the
Anthropic `temperature`/`top_p` are never sent by name in a test. They are implemented, but the claim is untested.

## Not verified

- This is a static audit: each row comes from reading the record and grepping its uses. Nothing here is a runtime test
  of an ignored field. A runtime check of F1 would send `stop` and compare the output with and without it; the fake
  engine used by the server tests does not apply stop strings, so it cannot show the difference, and a real engine is
  needed.
- The shared-harness test counts for Anthropic are not broken down per file.
- Responses, embeddings, images and audio endpoints are not audited.

## Recommended next step (owner decision)

Choose one of the three options for F1 per field, then add the test for each row in this table. That closes P7 for
the chat and messages endpoints.

## Resolution of F1 and F3 (P7b, 2026-10-31)

Decisions taken under the plan's standing authority; each is reversible. Tests: `RequestParameterTests` (30 cases, mutation-checked:
without the OpenAI validation 12 fail, without `max_completion_tokens` 2, without `seed` 1, without the Anthropic validation 5).

| Field | Decision | Where |
|---|---|---|
| `max_completion_tokens` | implemented: sets the token budget and wins over `max_tokens` when both are sent | `OpenAiEndpoints` |
| `seed` | implemented: `SamplingParams.Seed` seeds the sampler RNG | `SamplingParamsBuilder` |
| `top_k` (OpenAI side) | implemented: `SamplingParams.TopK` | `OpenAiEndpoints` |
| `n` | refused with a 400 unless it is 1: the server returns exactly one choice | `RequestParameterValidation` |
| `max_tokens` / `max_completion_tokens` below 1 | refused with a 400 | `RequestParameterValidation` |
| `temperature` | refused outside 0 to 2 (OpenAI) or 0 to 1 (Anthropic) | `RequestParameterValidation` |
| `top_p` | refused outside 0 to 1 | `RequestParameterValidation` |
| `top_k` | refused below 0 | `RequestParameterValidation` |
| `presence_penalty`, `frequency_penalty` | refused outside -2 to 2 | `RequestParameterValidation` |
| `parallel_tool_calls` | stays ignored on purpose: the server's tool-call behaviour is fixed, and the field is a hint | this table |
| `user`, `metadata` | stay ignored on purpose (privacy, plan section 7); this row is the record of that (F4) | this table |
| unknown JSON properties | still accepted and ignored: clients send vendor fields | `RequestParameterTests` |
| `stop`, `stop_sequences` | **not yet**: needs text-level matching with holdback in four handlers; tracked as P7c in the plan | open |
| `stream_options` (`include_usage`) | **not yet**: common in OpenAI SDK clients, so refusing it would break them; needs a final usage chunk; tracked as P7d | open |
| Anthropic `tool_choice` | **not yet**: the OpenAI side honours it, the Anthropic side does not; tracked as P7d | open |

Finding while testing: the error body this server returns is flat, `{"type": "...", "message": "..."}`. The OpenAI API wraps it as
`{"error": {"message": ..., "type": ...}}` and Anthropic as `{"type": "error", "error": {...}}`, so clients that parse the nested
envelope get nothing useful from a 400. This predates P7 and is not changed here; it is listed in the plan as P7e.