# Text Embeddings, Rerankers & Encoders (BERT, BGE, MiniLM, E5, NomicBERT, BGE-M3, T5)

[← Back to Architecture Cards](README.md)

Related: [llama](llama.md), [qwen](qwen.md). Verification: [STATUS.md](../STATUS.md); measured commands: [RUNNING.md](../RUNNING.md).

| Property | Value |
|---|---|
| **Architectures** | `bert`, `xlm-roberta`, `nomic-bert`, `mpnet`, `bge-m3`, `t5` |
| **Engine Implementations** | `TransformerEncoder`, `BgeM3Pipeline`, `T5Model`, `BertWordPieceTokenizer`, `UnigramTokenizer` |
| **Model Formats** | Hugging Face SafeTensors (`model.safetensors` + `tokenizer.json`) and GGUF |
| **Pooling Modes** | Mean pooling, CLS token pooling, and ColBERT multi-vector representation |
| **Numerical Parity** | 🔬 **Golden-Verified** (Cosine similarity $\ge 0.9999997$ against reference ONNX models) |
| **Status & Confidence** | 🟢 **Admitted & Level 2 Proven** (Verified in server `/v1/embeddings`, `/v1/rerank` and CLI `embed`/`rerank`) |

---

## 1. Overview & Architectural Highlights

In addition to generative autoregressive LLMs, OpenTail.Stingray provides native, in-process evaluation of bidirectional transformer encoders for text embeddings, semantic search, cross-encoder reranking, and sequence-to-sequence translation.

### Key Architectural Characteristics
* **Native Bidirectional `TransformerEncoder`:**
  * Runs standard bidirectional self-attention without causal masking.
  * Evaluates post-encoder sentence-transformers pooling strategies (Mean, CLS, normalized embeddings).
* **High-Precision Numeric Parity:**
  * Verified to 7 decimal places against official ONNX exports across `bge-small-en-v1.5`, `bge-large-en-v1.5`, `all-MiniLM-L6-v2`, `multilingual-e5-small`, `all-mpnet-base-v2`, and `nomic-embed-text-v1.5`.
* **BGE-M3 (Multi-Functionality Embedding):**
  * Evaluates **Dense** representations via shared XLM-R backbone.
  * Evaluates **Sparse lexical** representations (`sparse_linear.pt`).
  * Evaluates **Multi-Vector ColBERT** representations (`colbert_linear.pt`) using PyTorch ZIP `TorchCheckpointReader`.
* **Cross-Encoder Rerankers:**
  * Evaluates pair-wise relevance scores for `ms-marco-MiniLM-L6-v2` and `bge-reranker-v2-m3` directly inside the engine.
* **T5 Sequence-to-Sequence (`T5Model`):**
  * Full encoder-decoder with gated-GELU, relative position biases, and KV-cached greedy generation.

---

## 2. Checkpoints & Recommended Models

| Model | Task | Parameters | Typical Size | Hugging Face Repository |
|---|---|---|---|---|
| **all-MiniLM-L6-v2** | General Text Embedding | 22M | ~80 MB | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| **bge-small-en-v1.5** | High-Accuracy Retrieval | 33M | ~130 MB | [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) |
| **bge-reranker-v2-m3** | Cross-Encoder Reranking | 560M | ~2.1 GB | [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) |
| **BGE-M3** | Dense + Sparse + ColBERT | 560M | ~2.2 GB | [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) |
| **nomic-embed-text-v1.5** | Long-Context Embedding (8k) | 137M | ~540 MB | [nomic-ai/nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) |
| **flan-t5-small** | Seq2Seq Translation / Reasoning | 77M | ~300 MB | [google/flan-t5-small](https://huggingface.co/google/flan-t5-small) |

---

## 3. Usage & Code Examples

### C# Text Embeddings

```csharp
using OpenTail.Stingray.Core.Embeddings;
using OpenTail.Stingray.Engine.Encoders;

// 1. Initialize text embedding pipeline from local SafeTensors folder
using var pipeline = HfEncoderEmbeddingPipeline.Load("models/bge-small-en-v1.5");

// 2. Generate normalized semantic embedding vector
var result = pipeline.Embed(new EmbeddingRequest
{
    Inputs = ["How do I evaluate embedding models in .NET?"],
    Normalize = true
});

float[] embedding = result.Data[0].Vector;
Console.WriteLine($"Generated embedding vector with {embedding.Length} dimensions.");
```

### CLI Command & Server Endpoints

```bash
# Generate embeddings from CLI
stingray embed -m models/bge-small-en-v1.5 -i "Semantic search in managed C#"

# Rerank search candidates
stingray rerank -m models/bge-reranker-v2-m3 -q "What is SIMD?" -c "SIMD stands for Single Instruction Multiple Data" -c "A CPU is a central processing unit"
```

In the OpenAI-compatible server (`OpenTail.Stingray.Server`), set `STINGRAY_EMBEDDING_MODEL` to automatically expose `/v1/embeddings`.
