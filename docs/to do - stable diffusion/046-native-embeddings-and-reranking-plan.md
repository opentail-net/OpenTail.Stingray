# Plan â Native Text Embeddings & Cross-Encoder Reranking Support for OpenTail.Stingray

**Reference:** `examples/llama.cpp/llama.cpp/examples/embedding/embedding.cpp`, `examples/llama.cpp/llama.cpp/examples/retrieval/retrieval.cpp`  
**Target:** `opentail-net/OpenTail.Stingray` (`src/OpenTail.Stingray.Core/`, `src/OpenTail.Stingray.Engine/`, `src/OpenTail.Stingray.Server/`)  
**Execution:** **100% native managed C# (.NET 10) â zero external binaries, Python, P/Invoke, or sidecar process**

---

# Status

**COMPLETED (100% Native C#)**

OpenTail.Stingray supports Text Generation (LLMs), Multimodal Vision, Native Diffusion (Images & Video), a 5-Engine Text-to-Speech (TTS) Suite, native OpenAI Whisper Speech-to-Text (ASR), and native Text Embeddings & Cross-Encoder Reranking (`POST /v1/embeddings` & `POST /v1/rerank`).

---

# 1. Architectural Analysis of Embeddings & Reranking

### 1.1 Embedding & Reranking Pipeline Graphs

#### Dense Text Embeddings
```text
Input Text Prompts / Documents
        â
        â¼
GgufTokenizer (BPE / SentencePiece)
        â
        â¼ Token IDs [B, N]
Transformer Forward Pass (CPU / Vulkan / CUDA)
        â
        â¼ Hidden States [B, N, d_model]
Pooling Layer (Mean, CLS / BOS, Last Token, or Attention-Weighted)
        â
        â¼ Pooled Representation [B, d_model]
L2 Normalization: v / ||v||_2 (or Matryoshka dimension truncation)
        â
        â¼
Dense Embedding Vectors [B, d_emb] (e.g. 768, 1024, 1536, 4096 dims)
```

#### Cross-Encoder Reranker
```text
Query + Candidate Documents List
        â
        â¼
Pair Formatting: "[CLS] Query [SEP] Document [SEP]" or Jinja Rerank Template
        â
        â¼
Cross-Attention Transformer Forward Pass
        â
        â¼ CLS / Last Token Representation
Classification Head (Linear + Sigmoid / Softmax)
        â
        â¼
Relevance Scores [N_docs] + Ranked Results
```

---

# 2. Design & Implementation Structure

Target layout across projects:

```text
src/OpenTail.Stingray.Core
âââ Embeddings
â   âââ IEmbeddingPipeline.cs      // Embedding request/result interfaces
â   âââ IRerankerPipeline.cs       // Cross-encoder reranker interfaces
â   âââ PoolingType.cs             // Mean, CLS, LastToken, None, Rank
â   âââ EmbeddingNormalizer.cs     // L2 normalization & Matryoshka dimension reduction

src/OpenTail.Stingray.Engine
âââ EmbeddingEngine.cs             // GGUF/Safetensors forward pass hidden state extraction

src/OpenTail.Stingray.Server/Endpoints
âââ OpenAiEmbeddingEndpoints.cs    // POST /v1/embeddings (OpenAI compatible)
âââ RerankEndpoints.cs             // POST /v1/rerank (Cohere/BGE compatible)

src/OpenTail.Stingray.Cli
âââ EmbedCommand.cs                // stingray embed -m <model.gguf> -p "text"
âââ RerankCommand.cs               // stingray rerank -m <reranker.gguf> -q "query" -d "doc1" "doc2"
```

---

# 3. Key Specifications

* **Supported Pooling Types:**
  * `Mean` â Average across all non-padding token hidden states (BGE, ModernBERT, GTE, Nomic, MiniLM).
  * `CLS` / `FirstToken` â First token embedding (BERT, RoBERTa, DeBERTa).
  * `LastToken` â Last token representation (Qwen2-Embed, Llama-3-Embed, Snowflake Arctic, Mistral-Embed).
  * `Rank` â Linear scalar classification output for Cross-Encoder Rerankers (BGE-Reranker, Cohere-Rerank, Jina-Reranker).
* **Matryoshka Representation Learning (MRL):** Support truncation of embedding dimensions (e.g. 1536 $\rightarrow$ 512 or 256) with dynamic re-normalization.
* **OpenAI API Compatibility:** Exact schema match for `POST /v1/embeddings` (`model`, `input`, `encoding_format: "float" | "base64"`, `dimensions`).
* **Cohere / BGE API Compatibility:** Exact schema match for `POST /v1/rerank` (`model`, `query`, `documents`, `top_n`, `return_documents`).

---

# 4. Phased Implementation Plan

### Phase 1: Core Interfaces & Pooling Algorithms [COMPLETED]
* Implemented `PoolingType.cs`, `IEmbeddingPipeline.cs`, `IRerankerPipeline.cs`, and `EmbeddingNormalizer.cs` in `OpenTail.Stingray.Core.Embeddings`.

### Phase 2: Engine Hidden-State Extraction & Execution [COMPLETED]
* Implemented `EmbeddingEngine.cs` in `OpenTail.Stingray.Engine` extracting last-layer hidden states with Mean, CLS, LastToken pooling and Matryoshka support.

### Phase 3: CLI Commands (`stingray embed` & `stingray rerank`) [COMPLETED]
* Implemented `EmbedCommand.cs` and `RerankCommand.cs` in `OpenTail.Stingray.Cli`.
* Registered commands in `Program.cs` and updated `docs/cli-option-inventory.md` (189 options across 16 commands).

### Phase 4: Server HTTP Endpoints [COMPLETED]
* Implemented `POST /v1/embeddings` in `OpenAiEmbeddingEndpoints.cs`.
* Implemented `POST /v1/rerank` in `RerankEndpoints.cs`.
* Registered endpoints in `EndpointRouteBuilderExtensions.cs`.

### Phase 5: Automated Testing & Verification [COMPLETED]
* Implemented unit tests in `EmbeddingTests.cs` in `OpenTail.Stingray.Tests.Core`.
* Verified full solution build across `OpenTail.Stingray.slnx` (36 projects built with 0 errors and 0 warnings; all 528 Core tests, 367 CLI tests, and 12 Server tests pass).
