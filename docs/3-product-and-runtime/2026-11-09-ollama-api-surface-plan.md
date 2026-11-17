# Ollama API surface for Stingray — plan (draft)

Status 2026-11-09. **Nothing is built and nothing is committed.** One system: the Ollama-compatible HTTP surface in `OpenTail.Stingray.Server`. Model management (pull, create, copy, push, delete, blobs) is out of scope.

**Owner decisions (2026-11-09):**
- Route: `POST /api/chat`.
- First cut: `/api/chat` only.
- Match TensorSharp's behaviour "more or less" (this revision).

## 1. Matching principle

The default for every behaviour is **TensorSharp at HEAD** (`C:\Git-Public\example\TensorSharp`, commit `9d18a64d`). Deviate only in the cases listed in section 4, and say why in the matrix. Three kinds of deviation:

- **Fix:** TensorSharp crashes or loses data. Do not copy the defect.
- **Divergence:** TensorSharp differs from Ollama's pinned source in a way a client can see. The default is TensorSharp's; the owner confirms (section 7).
- **Not applicable:** the feature depends on TensorSharp-only machinery (knowledge retrieval, code execution, workspaces, sub-agents). Excluded.

## 2. TensorSharp behaviour matched, by route

Sources are TensorSharp HEAD. Methods are cited by name because HEAD line numbers differ from the stale copy. Helper files live in `TensorSharp.Chat`.

**Routes** (`TensorSharp.Server/Endpoints/OllamaEndpoints.cs`, lines 29-37)

| Route | TensorSharp | Plan |
|---|---|---|
| `GET /api/version` | yes | Phase 3 (see divergence D5) |
| `GET /api/tags` | yes | Phase 3 |
| `POST /api/show` | yes | Phase 3 |
| `POST /api/embed` | yes | Already in Stingray; unchanged |
| `POST /api/generate` | yes | Phase 2 |
| `POST /api/chat/ollama` | yes | **Here `/api/chat`** (owner decision; divergence D1) |
| `GET /api/ps`, `POST /api/embeddings`, model management | not mapped | **Not mapped** (ASP.NET answers 404) |

**Model resolution** (`HostedModelGuard`, `TensorSharp.Chat/Hosting/HostedModelGuard.cs`)
- TensorSharp hosts exactly one model. A request's `model` must match the hosted file name, the file name without extension, or the absolute path (lines 29-58, 159-183). Anything else is rejected: `model '<x>' is not hosted by this server. Restart ... to change it.`
- Consequence: a client sending `llama3.2:latest` gets an error, because `:tag` is not matched. Match this in single-model mode.
- Multi-model mode keeps Stingray's alias resolution (`IInferenceService.ResolveModel`). That is a Stingray extension, not a TensorSharp behaviour.

**Chat request** (`OllamaAdapter.ChatAsync`, `ChatMessageParser.ParseOllama`, `SamplingConfigParser.ParseOllama`, `ToolFunctionParser.ParseOllama`)

- `model` missing or empty: 400 `{"error": "model is required"}`.
- `messages` missing or not an array: 400 `{"error": "messages is required"}`.
- `stream` defaults to **true**. TensorSharp throws on a non-boolean `stream` (500). See fix F2.
- Message fields: `role` defaults to `"user"`, `content` defaults to `""`, `images` is a base64 array. Each image is decoded and stored under the SHA-256 of its bytes, so a resent image is not duplicated (`WriteContentAddressed`).
- `tool_calls`, `tool_name` and `thinking` on messages are **ignored** by `ParseOllama`. See fix F1.
- `options` (`SamplingConfigParser.ParseOllama`, lines 88-111) are read as: `temperature`, `top_k`, `top_p`, `min_p`, `repeat_penalty`, `repeat_last_n`, `presence_penalty`, `frequency_penalty`, `seed`, and `stop` (array only). Absent keys take the server's `SamplingDefaults`.
- `options.num_predict`: absent, zero or negative (including Ollama's `-1`) uses the server default. A pinned `--max-tokens` caps larger requests (`ServerHostingOptions.ResolveMaxTokens`, lines 547-554). The documented default is **20000** (USAGE.md:763).
- `options.num_ctx`, `keep_alive`, and `format` are **not read**. See divergence D3 for `format`.
- `think`: read as a boolean (`GetBoolean`). A string level such as `"high"` throws (500). See fix F2.
- `tools`: `ToolFunctionParser.ParseOllama` (lines 48-61). Entries without a `function` object are skipped. The entry's `type` is **not** checked. Nothing throws. A property with no `type` defaults to `"string"`.
- `tool_choice: "none"` disables server-side tools (`ChatAsync`, HEAD).
- Client-run tool loops: when the client supplies its own tools, the server does not add its own unless skills are requested (`SkillRequestPlan.ClientRunsToolLoop`, HEAD).
- Diffusion model plus tools: 400 with an explanatory message (`ChatAsync`).
- `skills`: `SkillSelectionParser.Parse` (lines 75-118). Names are read from an array, a single string, or `{name}`/`{id}` objects. Absent means none. An empty array means "none, deliberately". `skills_discovery` is read by `ParseDiscovery`. If the server is started with `--no-skills`, a request naming skills gets 400 `Agent skills are disabled on this server (--no-skills).`. An unknown name gets 400 `No skill called '<x>' is installed ...`.

**Chat response** (`OllamaResponseFactory`, `NdJsonWriter`, `JsonOptions`, `FinishReasonMapper`)

- Streaming: `Content-Type: application/x-ndjson`, `Cache-Control: no-cache`, one JSON object per line, flushed after each line (`NdJsonWriter.ApplyHeaders` and `WriteLineAsync`, lines 25-46).
- Serialisation omits null properties (`JsonOptions.IgnoreNulls`, lines 24-27). So `thinking` and `tool_calls` are absent when empty.
- Content chunk: `{model, created_at, message: {role: "assistant", content}, done: false}`. A thinking chunk adds `message.thinking`.
- Final chunk: `{model, created_at, message, done: true, done_reason, total_duration, prompt_eval_count, prompt_eval_duration, eval_count, eval_duration, prompt_cache_hit_tokens, prompt_cache_hit_ratio}`. Tool calls, when present, go in `message.tool_calls`. `load_duration` is not emitted.
- Non-streaming: one object with the same final fields and a full `message`.
- `created_at`: UTC, `DateTime.ToString("o")`.
- `model`: the loaded model's file name (`LoadedModelName`), not the name the client sent.
- `done_reason` (`FinishReasonMapper.ToOllamaDoneReason`, lines 119-124): `length` when truncated, `tool_calls` when the turn ends in tool calls, otherwise `stop`. Truncation is `max_tokens`, `thinking_budget` or `repetition` (`IsTruncated`, lines 166-169). See divergence D2.
- `prompt_cache_hit_tokens` (int) and `prompt_cache_hit_ratio` (0..1) are on the final chunk and the non-streamed response. `USAGE.md:2555` documents them. This is a TensorSharp extension.
- Keep-alive: when a stream has been idle for `StreamKeepAlive.DefaultInterval` (**15 s**, `StreamKeepAlive.cs:40`), an ordinary empty chunk (not done) is written. Non-streamed responses get no keep-alive, because the status code is decided at the end.
- Errors: `{"error": "<message>"}`. Validation errors are 400. A hosted-model or load failure is 404 for non-streamed requests (`TryEnsureHostedModelLoaded`, lines 112-152). Upload-limit errors use the exception's status.
- Streaming errors: once headers are sent, the error is one NDJSON line: `{model, created_at, message: {role, content: ""}, done: true, done_reason: "error", error}`, at HTTP 200. See divergence D4 for the model-not-found case.

**Generate** (`OllamaAdapter.GenerateAsync`)

- `model` required, as for chat. `stream` defaults to true. `prompt` defaults to `""`. `images` are read as for chat. `options` and `num_predict` are read as for chat.
- `system`, `template`, `raw`, `suffix` and `context` are **not read**. Silently ignored. See divergence D6.
- Stream chunk: `{model, created_at, response, done: false}`. Final chunk: `{model, created_at, response: "", done: true, done_reason, timings, prompt_cache_hit_tokens, prompt_cache_hit_ratio}`. Non-streamed: the same final fields with the full `response`.
- Stream error: `{model, created_at, response: "", done: true, done_reason: "error", error}`.

**Tags** (`OllamaAdapter.GetTags`)

- Lists the one hosted model, or an empty list when none is hosted. Item: `name` (file name without extension), `model` (file name), `size` (bytes), `modified_at` (file mtime, UTC). **No `digest` and no `details`.** See divergence D7.

**Show** (`OllamaAdapter.ShowAsync`)

- `model` required (400). A model that is not hosted: 404.
- Response: `modelfile: ""`, `parameters: ""`, `template: ""`, `details: {format: "gguf", family: <architecture if this model is loaded, else "">}`, `model_info: {file, size}`. No `capabilities`.

**Version** (`OllamaEndpoints.cs`, `GET /api/version`)

- Returns `{"version": "0.1.0"}`, a literal placeholder. See divergence D5.

**Embed**

- `POST /api/embed` already exists in Stingray. It stays byte-compatible. Replace its untyped serializer (`OpenAiEmbeddingEndpoints.cs:74`) in Phase 0.

## 3. Fixes: TensorSharp defects not copied

| Id | TensorSharp behaviour | Plan |
|---|---|---|
| F1 | History `tool_calls`, `tool_name` and `thinking` are dropped (`ParseOllama` reads only `role`, `content` and `images`). A multi-turn tool conversation loses the call that preceded each result. | Read `tool_calls` (`arguments` object converted to the string `OaiToolCall` expects), `tool_name` (for the tool result's name), and `thinking` for assistant messages only. That last rule is the Web UI's own (`ParseWebUi`). |
| F2 | A non-boolean `stream` or `think` throws, giving a 500 (`GetBoolean`). | Return 400 `{"error": "..."}` naming the field. |
| F3 | `ChatAsync` reads `think` as a boolean only, so a level string such as `"high"` (which Ollama documents) fails. | Accept booleans. Reject level strings with 400 until a level is mapped. Ollama's spec documents `think` levels (`docs/api.md`). How the source handles them is not yet checked, so do not invent a mapping. |
| F4 | A stream for a model that is not hosted returns HTTP 200 with an error line. | Return 404 before streaming (see D4). |

## 4. Divergences from TensorSharp

Each needs an owner decision (section 7). Defaults are TensorSharp's unless noted.

| Id | Topic | TensorSharp | Ollama at pin `d63b702` | Plan default |
|---|---|---|---|---|
| D1 | Chat path | `/api/chat/ollama` (its `/api/chat` is the Web UI) | `/api/chat` | **`/api/chat`** (owner decided) |
| D2 | `done_reason: "tool_calls"` | Emitted | Not in the enum (`llm/server.go:251-270` has `stop`, `length`). So a tool-call turn can only report `stop` or `length`; inferred from the enum, not traced through the handler. | TensorSharp's value. Recommend Ollama's, for clients that switch on `done_reason`. |
| D3 | `format` (`"json"` or a schema) | Ignored | Documented in `docs/api.md` | Ignored by default. Stingray already has `json_object` and `json_schema` (`OpenAiEndpoints.cs:338-365`), so honouring it is cheap. Recommend honouring it. |
| D4 | Model not hosted, on a streamed request | 200 with an error line | `GenerateHandler` returns 404 before streaming (`routes.go:362`). The chat handler is not yet checked. | **404 before streaming** (matches Ollama's generate). Reason: status codes are what clients check. |
| D5 | `/api/version` value | `"0.1.0"` placeholder | Real version (`routes.go:2393`) | Keep the shape. Report Stingray's version, **1.0.9** (`Directory.Build.props`). Owner may choose `0.1.0`. |
| D6 | Generate's `system`, `template`, `raw`, `suffix`, `context` | Silently ignored | Documented as honoured | Silently ignored, as TensorSharp does. Recommend 400 for `raw` and `template`, because they change the prompt. |
| D7 | Tag items `digest`, `details` | Omitted | Present (`docs/api.md`) | Omitted, as TensorSharp does. `digest` needs a SHA-256 of the GGUF; that cost is an owner call. |
| D8 | `tool_calls[].function.index` | Omitted | Present (`types.go:231-235`) | **Include `index`.** Additive, and Ollama's shape. |
| D9 | `load_duration` on the final chunk | Omitted | Present (`docs/api.md`) | Send `0`. Additive. |
| D10 | `skills`, `multi_agent`, knowledge, code execution, workspace | TensorSharp features | Not in Ollama | `skills`: see open item O4. The rest are **not implemented** and are ignored with a log line, as TensorSharp ignores unknown fields. |

## 5. Contracts this must satisfy

- **`IInferenceService`**: `ResolveModel` returns a `ModelId` or throws `ModelNotFoundException`. `AcquireAsync` returns a `ModelRuntimeHandle` that the caller disposes. `AvailableModelAliases` and `IsMultiModel` drive `/api/tags` in multi-model mode. The single-model path keeps resolving `IInferenceEngine` from DI, because tests replace it (`OpenAiEndpoints.cs:61-67`).
- **`IInferenceEngine` and `GenerateChunk`** (`GenerateChunk.cs`): Text, Usage, Stop (once, `TruncatedByMaxTokens`), ToolCall (`ToolCalls`). Read Thinking kind and `GenerationStats` before coding.
- **Source-generated JSON**: new code uses `OpenTailStingrayJsonContext` only, so the new surface stays on the AOT-clean side.
- **`WithConcurrencyLimit()`** on every generation route.
- **`ServerCompatibilitySnapshot`** reports only mapped routes, so new routes appear in `/capabilities` automatically (`CompatibilityEndpoints.cs:17-29`; the snapshot class itself is not yet read).
- **`/api/embed`** stays unchanged.
- **Corpus (product plan A4) and docs-as-tests (A7)**: every route and case gets a recorded status and error type.

## 6. Stingray branch map (required before coding)

Branches of Stingray's `HandleChatCompletion` seen so far. Each needs a row in the Ollama adapter.

| Branch | Where | Ollama counterpart |
|---|---|---|
| JSON parse failure → 400 | `OpenAiEndpoints.cs:26-43` | 400 `{"error": ...}` |
| `messages` missing or empty → 400 | `:45-53` | 400 `{"error": "messages is required"}` |
| `RequestParameterValidation.ValidateOpenAi` → 400 | `:54-58` | Map each rule; some have no Ollama field |
| `ApplySkills` | `:59` | Open item O4 |
| Multi-model `ResolveModel` and `AcquireAsync` | `:61-71` | Same |
| Single-model DI path | `:61-67` | Same, with the TensorSharp model-name rule |
| `ModelLoadRefusal` → status and type | `:103-106` | 404 non-streamed; error line streamed |
| `response_format` schema vs `json_object` | `:338-365`, `:835` | `format` (D3) |
| Tools active → `ToolGrammarHelper` | `:313-322` | `tools` |
| Stop sequences | `:452`, `:690` | `options.stop` |
| Images via `ImageContent.GenerateMedia` | `:452` | `messages[].images` |
| Stop chunk with truncation → `finish_reason: length` | `:730-763` | `done_reason` |

**Incomplete:** `OpenAiEndpoints.cs` lines 73-790 are not yet read branch by branch. Completing that is Phase 0, step 1.

## 7. Open decisions (owner)

1. **Named consumer** (product plan A4). Needed before Phase 1 starts.
2. **Divergences D2, D3, D4, D5, D6, D8, D9.** The defaults above are TensorSharp's, except D4 (Ollama's), D8 and D9 (additive, Ollama's). Confirm or change.
3. **Fixes F1-F4.** The plan does not copy TensorSharp's defects. Confirm.
4. **O4, skills.** Stingray's `ChatCompletionRequest` takes inline skill definitions (`WireSkill[]`). Whether a named registry also exists is not yet verified. If there is none, `skills` should be rejected with 400, not ignored. Check before deciding.
5. **O5, `:tag` names.** TensorSharp refuses them. Keep that, which will break clients that send `name:tag`?

## 8. Phases

**Phase 0: prerequisite, no Ollama route yet**
1. Complete the section 6 table. Read `OpenAiEndpoints.cs` lines 73-790, `ServerCompatibilitySnapshot`, `SamplingParams`, `GenerationStats`, and Stingray's default `max_tokens`. If it is not 20000, record the difference.
2. Extract the protocol-neutral generation core from `HandleChatCompletion`, with no behaviour change. The existing OpenAI tests must pass unchanged.
3. Replace the untyped `JsonSerializer` in `/api/embed` (`OpenAiEmbeddingEndpoints.cs:74`) with a typed context, no output change.

**Phase 1: `POST /api/chat` (first cut)**
4. `OllamaEndpoints.cs` (routes only), an Ollama request translator, and an Ollama response factory. Source-generated JSON throughout.
5. Implement section 2 (chat), with fixes F1-F4 and the divergences the owner confirms.
6. Tests (section 9). Update `docs/STATUS.md` (rule 10) only after a real client is verified against a real model.

**Phase 2: `POST /api/generate`**, reusing the Phase 1 core. Section 2 (generate).

**Phase 3: `GET /api/tags`, `POST /api/show`, `GET /api/version`.** Section 2 (tags, show, version). Read-only routes, cheap.

Not planned, matching TensorSharp: `/api/ps`, `/api/embeddings`, model management.

## 9. Tests

- Fake-engine tests in `OpenTail.Stingray.Tests.Server.Fast`, using the DI seam the OpenAI tests use.
- Corpus cases (A4) for each route, streamed and non-streamed, each with a recorded status and error type. Include the cases TensorSharp's own tests cover: `FinishReasonMappingTests.cs` and `EmbeddingEndpointTests.cs` in `InferenceWeb.Tests`. Not yet read.
- Explicit tests for F1 (tool history round-trips), F2 (400, not 500), and F4 (404 before streaming).
- A real-GGUF smoke test. Compare field names and `done_reason` against a real Ollama server at the pinned revision, if one can run here.
- A test is verification only if it runs against real weights or the real Ollama server. Check timing, not just "passed".

## 10. Verified sources

Ollama, pinned `ollama/ollama` `d63b702609c4300a2ca07be2bdb0adb9834725e5` (2026-11-08), read from source:
- `server/routes.go`: routes (2393-2430), model-not-found 404 (362), NDJSON stream (2625-2648; `application/x-ndjson` at 2626), error bodies `gin.H{"error": ...}`, stream check `req.Stream != nil && !*req.Stream` (430, 844, 1547, 1602, 2967, 3146).
- `api/types.go`: `ToolCallFunction` (231-235), `ToolCallFunctionArguments` as an ordered JSON object (237-301), `keep_alive` default 5 minutes (1283), `Options` keys (~579-595).
- `llm/server.go`: `DoneReason` enum `stop`/`length` and the empty `ConnectionClosed` value (251-270).
- `docs/api.md`: field lists for chat, generate, tags, ps, show, embed. Not yet checked against structs.

TensorSharp HEAD `9d18a64d`: the files cited in section 2, by name and line as listed.

Stingray: `OpenAiEndpoints.cs` (lines cited), `InferenceService.cs:19-59`, `GenerateChunk.cs`, `EndpointRouteBuilderExtensions.cs:17-30`, `CompatibilityEndpoints.cs:17-29`, `OpenAiEmbeddingEndpoints.cs:54-75`, `Directory.Build.props` (version 1.0.9).

## 11. Not verified

- Ollama's chat and generate response struct JSON names, and `/api/tags` and `/api/show` detail objects. Only `docs/api.md` covers them.
- How Ollama handles `think` levels in source.
- `FinishReasonMappingTests.cs` and `EmbeddingEndpointTests.cs`: not read.
- `ReasoningEffortParser`, which TensorSharp's chat path calls: not read.
- `OpenAiEndpoints.cs` lines 73-790, `SamplingParams`, `GenerationStats`, `ServerCompatibilitySnapshot`: not read branch by branch.
- Whether Stingray has a named skill registry (O4), a repetition penalty (`repeat_penalty`), and KV reuse counts (for `prompt_cache_hit_*`).
- Whether data-URL and base64 images reach the model through Stingray's `ImageContent` path.
- No code was changed or run for this plan, and nothing is committed.
