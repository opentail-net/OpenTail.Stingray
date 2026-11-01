# Request parameter audit of the other endpoints, 2026-11-01 ([plan 3, Q5](../2026-10-31-autonomous-plan-3.md))

Follows [the chat and messages audit](request-parameter-audit-2026-10-31.md). Same method and the same three outcomes: **implemented**
(read, applied, with a test that sends it), **rejected** (refused with a stable error that names the field), **ignored** (read and dropped, or
never declared, with a stated reason). A field a record does not declare is dropped by System.Text.Json without an error, so it counts as
silently ignored. Unknown JSON properties stay accepted: clients send vendor fields.

**Method.** For each endpoint: read the request record or the form reads in `src/OpenTail.Stingray.Server/Endpoints/`, grep every use of each
field in its handler, and read the 4xx paths. Tests added with this audit are in `OtherEndpointParameterTests` (Server.Fast, 18 tests).

**Covered:** `/v1/responses`, `/v1/audio/speech`, `/v1/images/generations`, `/v1/images/edits`, `/v1/embeddings`, `/v1/rerank`.
**Not covered, and so not claimed:** `/v1/audio/transcriptions` and `/v1/audio/translations` beyond listing the form fields they read, `/v1/sessions/*`,
`/tokenize`, `/detokenize`, `/props`, `/status`, `/api/embed`. They still need their own pass.

## Findings

**F1. The HTTP image API was a stub that returned a fake image with a 200 (fixed here).** `POST /v1/images/generations` and `/v1/images/edits`
never called a diffusion pipeline. The handlers called a private `SynthesizeImageRgb(prompt, width, height, seed)`, a colour gradient seeded from
`prompt.GetHashCode()` (which .NET randomises per process, so the "image" was not even stable across restarts), encoded it as PNG and returned
it as `b64_json` or a temp-file `url`. The Server README listed the routes as serving SD 1.5, SDXL, SD 3/3.5, FLUX and Z-Image and as doing
Img2Img and inpainting; two guides and `WHAT-YOU-CAN-DO.md` pointed at them; no test covered either route; and the guides also named a
`/v1/images/variations` route that was never mapped. A client that read the 200 believed it had a generated image. Decision (owner, 2026-11-01):
return `501 Not Implemented` after the request validates, with an error that points at `stingray image`, and correct the docs. Connecting the real
pipelines is open work (plan 3). The 501 body carries `type: not_implemented`; a missing prompt, bad JSON or a non-multipart edit stay 400.

**F2. `/v1/responses` dropped features and answered errors with an empty body (fixed here).** Its record declared `model`, `input`,
`instructions`, `max_output_tokens`, `temperature`, `top_p`, `stream` and `opentail_timing` only. A client sending `tools` or `previous_response_id`
got a plain text answer with no sign that either was ignored, and the three sampling fields had no range check (the chat and messages endpoints
got theirs in the previous audit). Every 400 on the route had no body at all. Now: non-empty `tools` and any `previous_response_id` are refused with a
400 that names the field; `max_output_tokens` below 1, `temperature` outside 0 to 2 and `top_p` outside 0 to 1 are refused; an empty `tools` array is
accepted; and every 400 on the route carries the same error body as the chat endpoints (flat `type` and `message` plus the nested `error` object).

**F3. `/v1/audio/speech` ignored `response_format` and always answered WAV (fixed here).** The field was declared and never read, so a client asking
for `mp3`, `opus`, `aac`, `flac` or `pcm` received WAV bytes. Now anything other than `wav` (case-insensitive) is a 400 naming the field; omitting it is
accepted, which keeps the default behaviour (the OpenAI default format is mp3, so a client that sends nothing still gets WAV, as before).

**F4. Silent clamps, documented not changed.** `/v1/images/*` clamp `n` to 1..10 and `size` to 64..2048 per side, and ignore `quality`, `style`
and `user`. `speed` on speech is replaced by 1.0 when it is not positive. With the image routes now answering 501, the clamps are moot until the
pipelines are connected; the speech `speed` fallback is a smaller instance of the same pattern and stays open.

**F5. Not changed, still open.** `/v1/rerank` does not range-check `top_n` at the endpoint (the downstream behaviour for zero or negative was not
tested here), and Cohere or Jina fields such as `max_chunks_per_doc` are dropped. `/v1/embeddings` ignores `user` (intentional, privacy).
`/v1/audio/speech` does not read `instructions` or `stream_format`.

## `/v1/responses`

| Field | Outcome |
|---|---|
| `model` | implemented (routing) |
| `input`, `instructions` | implemented (`BuildMessageList`) |
| `max_output_tokens` | implemented; **range checked now** (at least 1) |
| `temperature`, `top_p` | implemented; **range checked now** (0 to 2, 0 to 1) |
| `stream` | implemented |
| `opentail_timing` (vendor) | implemented |
| `tools` | **rejected when non-empty** (empty array accepted) |
| `previous_response_id` | **rejected** |
| `tool_choice`, `store`, `reasoning`, `text`, `metadata`, `parallel_tool_calls`, `stop`, `seed` | silently ignored: not declared. `tools` is the one that changes what the model can do, hence the 400; the others are accepted so clients that send defaults keep working |

## `/v1/audio/speech`

| Field | Outcome |
|---|---|
| `model` | read |
| `input` | implemented; required (400 when blank) |
| `voice` | implemented; default `af_heart` when blank |
| `speed` | implemented; falls back to 1.0 when not positive (F4) |
| `response_format` | **rejected unless `wav`** (F3) |
| `instructions`, `stream_format` | silently ignored: not declared |

## `/v1/images/generations` and `/v1/images/edits`

Both answer `501` after validation (F1). Declared on generations: `prompt` (required), `model`, `n`, `quality`, `response_format`, `size`, `style`, `user`.
On edits (multipart): `prompt` (required), `size`; the other OpenAI fields are not read.

## `/v1/embeddings`

| Field | Outcome |
|---|---|
| `model` | implemented (encoder resolution) |
| `input` / `inputs` | implemented (`TryParseInput`) |
| `dimensions` | implemented; 400 when not positive |
| `encoding_format` | implemented; 400 unless `float` or `base64` |
| `user` | ignored, intentional |

## `/v1/rerank`

| Field | Outcome |
|---|---|
| `model` | implemented |
| `query` | implemented; required (400) |
| `documents` | implemented; required and non-empty (400) |
| `top_n` | passed through; **no range check at the endpoint** (F5) |
| `return_documents` | implemented |

## Transcriptions and translations (form fields only)

The handler reads `file`, `language`, `prompt` (or `initial_prompt`), `response_format` (default `json`, also accepted as a query parameter),
`temperature`, `vad` (or `use_vad`) and `stream`. How an unknown `response_format` or an out-of-range `temperature` is handled was not checked here.

## Tests

`OtherEndpointParameterTests` (18): the three `/v1/responses` sampling ranges, `tools` and `previous_response_id`, an accepted empty `tools`, malformed
JSON with a body; five speech formats refused and three accepted forms; the two image routes (501 with the right error, no image data, and the 400
paths unchanged). Mutation checks: removing the Responses validation, allowing `tools`, allowing `previous_response_id`, removing the speech format
check, dropping the error body, and turning the image 501 back into a 200 each fail tests.
