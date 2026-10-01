# Guide: search by meaning (embeddings and reranking)

**You get:** two building blocks for search and retrieval.

- **Embeddings** turn text into a vector of numbers so similar texts are close together.
- **Reranking** scores candidate documents against a query, so you can put the best first.

The usual pattern: embed your documents once, find the nearest few to a query, then rerank those
few for a better order.

## Embeddings

```bash
stingray embed -m models/qwen3-embedding-0.6b/qwen3-embedding-0.6b-q8_0.gguf -p "Hello world"
stingray embed -m <model> -f lines.txt -o vectors.json     # one text per line, vectors to JSON
```

- The vectors are length-normalised by default (`--no-norm` turns that off).
- `-d 512` shortens the vector (Matryoshka-style models such as Qwen3-Embedding support it).
- `--pooling mean|cls|last` overrides the model's own choice, which is the right default.
- On the reference machine one short text takes about 0.3 s (1024 dimensions).

## Reranking

```bash
stingray rerank -m models/qwen3-embedding-0.6b/qwen3-embedding-0.6b-q8_0.gguf \
                -q "how do I reset my password" \
                -d "To reset your password, open Settings and choose Security." \
                -d "Our office is closed on public holidays."
```

`-d` is repeated per candidate, or give `-f docs.txt` with one per line. `-k 5` keeps the top 5.
`-o results.json` writes the ranking.

## Which models

Checked against independent references (see [STATUS.md](../STATUS.md)):

- **Embeddings:** BERT, MiniLM, BGE, E5, XLM-R, MPNet and NomicBERT families; BGE-M3 (dense, sparse
  and ColBERT scores); Qwen3-Embedding.
- **Rerankers:** `ms-marco-MiniLM` and `bge-reranker-v2-m3` cross-encoders.
- **CLIP:** image and text embeddings in one space, for searching images by text (library API).

The CLI takes a model directory or GGUF as shown in the examples above; the model each command
accepts is in `stingray embed --help` and `stingray rerank --help`.

## Over HTTP

`POST /v1/embeddings` (OpenAI format) and `POST /v1/rerank` (Cohere format). See the
[serving guide](serve-an-api.md).

## Limits

- Stingray gives you the vectors and the scores. **It does not include a vector database.** Store
  and search the vectors with your own code or store.
- Use the same model for the documents and the query. Vectors from different models are not
  comparable.
