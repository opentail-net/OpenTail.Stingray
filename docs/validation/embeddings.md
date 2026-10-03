# Embeddings validation: GGUF `bert`-architecture encoders

Written 2026-10-03 (Phase 1 of the TensorSharp selective-port plan). Scope: GGUF BERT / XLM-R embedding models loaded
through `HfEncoderEmbeddingPipeline.LoadGguf` (routed by `EncoderPipelineFactory.IsGgufBertEncoder`), and the
`/v1/embeddings` + `/api/embed` wire formats. HF-safetensors encoders were already verified; see `docs/STATUS.md`.

## Checkpoints used

| Model | File | Source | Size | SHA-256 |
|---|---|---|---|---|
| all-MiniLM-L6-v2 Q8_0 | `all-MiniLM-L6-v2.Q8_0.gguf` | `leliuga/all-MiniLM-L6-v2-GGUF` | 25,008,064 | `e5ec722e8c82dc4ffaf965175ca472f5da3f97b695590b5b0780bdbfa29bcaf3` |
| Snowflake Arctic Embed L v2.0 Q8_0 | `snowflake-arctic-embed-l-v2.0-q8_0.gguf` | `Savyasaachin/snowflake-arctic-embed-l-v2.0-Q8_0-GGUF` | 634,554,784 | `30b0d43dd5825809d2999fa2f4013a1d788c1539ad9ec163d1d305fa3b586ed8` |
| Arctic Embed L v2.0 F16 (reference only) | `snowflake-arctic-embed-l-v2.0-f16.gguf` | `embedme/snowflake-arctic-embed-l-v2.0-f16.gguf` | 1,157,672,256 | not recorded; deleted after the run |

Community conversions, pinned only by the hashes above (no revision pin; see the model-manifest phase).

## Reference

llama.cpp `bed0a8566` (CPU build, `examples/llama.cpp/llama.cpp/build-ref`), `llama-embedding --embd-output-format json`,
pooling and normalisation taken from the GGUF (MiniLM mean, Arctic CLS; L2-normalised). Vectors are committed as
fixtures in `tests/OpenTail.Stingray.Tests.Embeddings/Fixtures/`.

## Results (CPU, our F32-accumulate path over dequantised Q8_0 weights)

| Model | Check | Result |
|---|---|---|
| MiniLM Q8_0 | cosine, ours vs llama.cpp Q8_0 (3 texts incl. accents) | 0.9997 - 0.9998 |
| MiniLM Q8_0 | token ids vs the verified HF WordPiece path | identical on the one text compared (12 ids); the 3-text cosine above is the broader check |
| Arctic L v2 Q8_0 | cosine, ours vs llama.cpp Q8_0 (4 texts incl. French and Chinese) | 0.9991 - 0.9996 |
| Arctic L v2 Q8_0 | cosine, ours vs llama.cpp **F16** | 0.9995 - 0.9997 |
| Arctic L v2 Q8_0 | cosine, llama.cpp Q8_0 vs llama.cpp **F16** | 0.9987 - 0.9992 |
| Arctic L v2 Q8_0 | pairwise text similarity, ours vs F16 | within 0.009 |

Reading the Arctic rows: llama.cpp also quantises activations to Q8_0 inside its matmuls; ours accumulates in F32.
Against the near-exact F16 run, ours is *closer* than llama.cpp's own Q8_0 path, so the gap between ours and llama.cpp Q8_0
is llama.cpp's quantisation noise, not an encoder error. The test asserts exactly that ordering
(`GgufBertEncoderTests.ArcticEmbedLV2Q8Gguf_MatchesLlamaEmbedding`).

A first run scored cosine 0.17 against llama.cpp: llama.cpp's converter stores BERT WordPiece vocabularies in a
"phantom space" form (`▁word` for word starts, bare continuations, `[SPECIAL]` verbatim), which read directly as a WordPiece
vocab turned most words into `[UNK]`. `BertWordPieceTokenizer.FromGgufVocab` inverts it (`BertGgufVocabTests`).

## Wire format (`EmbeddingEndpointTests`, no model required)

- `input` as a string or an array of strings on both routes; empty strings, empty arrays and non-string items are
  rejected with 400 instead of being dropped (the array form used to fail as "Invalid JSON").
- OpenAI error shape `{"error":{"message","type":"invalid_request_error"}}` on `/v1/embeddings`; Ollama's `{"error":"..."}`
  on `/api/embed`; 404 when no encoder is configured (never synthetic vectors).

## Not validated

- Batch-versus-single and repeated-input determinism: tested only for MiniLM Q8_0 (`MiniLmQ8Gguf_BatchEqualsSingle_...`).
- `/v1/embeddings` and `/api/embed` against a real encoder over HTTP (only validation and 404 paths are covered).
- `encoding_format=base64`, `dimensions` (Matryoshka truncation) over HTTP; Ollama `truncate`, `options`, `keep_alive`.
- Concurrency, bounded batch size, cancellation, model unload (the server serialises calls with a lock).
- GGUF `nomic-bert` and other non-`bert` encoder architectures; long inputs near the 8192-token limit.
- Speed: the encoder dequantises weights to F32 at load (a Q8_0 model uses about 4x its file size in RAM).
- GPU backends.
