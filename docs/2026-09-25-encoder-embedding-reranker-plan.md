# Plan: encoder / embedding / reranker models + decoder re-verification (written 2026-09-25)

Queued to start **after** the audio re-check (`docs/2026-09-24-audio-recheck-plan.md`).

Target checkpoints (16), grouped by architecture family rather than treated as 16 engines:

| Family | Checkpoints | HF architecture |
|---|---|---|
| **A. BERT encoder** | google-bert/bert-base-uncased, sentence-transformers/all-MiniLM-L6-v2, BAAI/bge-small-en-v1.5, BAAI/bge-large-en-v1.5, intfloat/multilingual-e5-small, sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2, google/electra-base-discriminator | `BertModel` / `ElectraForPreTraining` |
| **B. RoBERTa-style (XLM-R)** | FacebookAI/xlm-roberta-base | `XLMRobertaModel` |
| **C. MPNet** | sentence-transformers/all-mpnet-base-v2 | `MPNetForMaskedLM` (encoder only used) |
| **D. NomicBERT** | nomic-ai/nomic-embed-text-v1.5 | `NomicBertModel` |
| **E. Cross-encoders** (A/B + classification head) | cross-encoder/ms-marco-MiniLM-L6-v2 (BERT), BAAI/bge-reranker-v2-m3 (XLM-R large) | `*ForSequenceClassification` |
| **F. Decoder-only** (existing engine) | openai-community/gpt2, Qwen/Qwen3-0.6B, Qwen/Qwen3-8B, trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 | `GPT2LMHeadModel`, `Qwen3ForCausalLM`, `Qwen2ForCausalLM` |

## 1. What already exists (surveyed 2026-09-25, not assumed)

- **Decoder-only: largely done.** `gpt2` and `qwen3` are in `ModelCompatibility`'s admitted list with
  greedy-parity receipts against llama.cpp (`Gpt2GreedyParityTests`; Qwen3 has tests including
  `Qwen3CudaGraphParityTests`). GGUFs on disk: `gpt2.Q8_0.gguf`, `Qwen3-0.6B-Q8_0.gguf`,
  `Qwen3-8B-Q4_K_M.gguf`. `ModelGraph` already handles GPT-2's learned absolute positions (no RoPE)
  and Qwen3's `key_length` ≠ `hidden/heads` (head_dim 128 at hidden 1024). What's missing is
  loading the **HF safetensors** checkpoints themselves (the task names HF repos), and the tiny Qwen2
  test checkpoint, which only exists as safetensors.
- **Encoder embeddings: the existing native path is FAKE and must be replaced, not extended.**
  `Core/Embeddings/BertGgufEmbeddingPipeline.cs` opens the GGUF but never reads a weight: the
  "embeddings" are `sin(tid*17 + d*0.2)`, the "attention" is a sigmoid of a 32-dim dot product,
  and the tokenizer is `string.GetHashCode()` (randomized per process in .NET, so not even
  deterministic). `EmbeddingEngine.Rerank` (`Engine/EmbeddingEngine.cs`) is a bi-encoder cosine
  over that, with a `MathF.Sin(hash...)` fallback: not a cross-encoder. The README must not claim
  embedding/rerank support today.
- **Useful, real pieces to reuse:**
  - `EmbedCommand` / `RerankCommand` (CLI), `OpenAiEmbeddingEndpoints` / `RerankEndpoints`
    (server), `IEmbeddingPipeline` / `IRerankerPipeline`, `PoolingType`, `EmbeddingNormalizer`
    (the plumbing is fine; only the compute behind it is fake).
  - `BertWordPieceTokenizer` (Core): documented as a faithful port of HF `BasicTokenizer` +
    `WordpieceTokenizer` (lowercase, accent strip, CJK, punctuation, `##`). Loads `vocab.txt`.
  - `UnigramTokenizer` (Core): SentencePiece/HF Unigram Viterbi, built for Parler's T5. Needed for
    the XLM-R vocabulary (xlm-roberta-base, bge-reranker-v2-m3, multilingual-e5-small,
    paraphrase-multilingual-MiniLM-L12-v2, which are all 250k SentencePiece).
  - `SafetensorsLoader` and `HuggingFaceTokenizerSource` (Core). **Note:** `HuggingFaceTokenizerSource`
    currently accepts **BPE only**, so WordPiece/Unigram `tokenizer.json` loading needs adding.
  - The audio project's `*SafetensorsTensorSource` bridges (e.g. `QwenAsrLlmSafetensorsTensorSource`)
    already map HF Qwen2/Qwen3 tensor names onto the GGUF naming `ForwardPass` expects, which is the
    pattern for loading HF LLM checkpoints directly.
  - `OnnxModelSession` (onnxruntime) exists. **Only as a test oracle**, never as the engine.
  - Kernels: `SimdKernels` / `PackedSgemmF32` / `QuantizedWeightCache` for the GEMMs;
    `DiffusionOps`/`WanAttention` tiled attention; `Primitives/*Kernels.cs` (LayerNorm, GELU,
    DenseKernels). Vulkan GEMM/attention via `IComputeBackend` for the GPU pass.
  - **Vendored C++ reference for the BERT math**: `examples/onnxruntime/onnxruntime/contrib_ops/cpu/bert/`
    (`embed_layer_norm.cc`: fused word+position+token_type embedding + LayerNorm, including mask
    index handling; `attention.cc`/`attention_cpu_base.h`/`attention_helper.h`: masked MHA with
    key-padding masks; `bias_gelu.cc`/`fast_gelu.cc`: erf vs tanh GELU; `skip_layer_norm`;
    `rotary_embedding` for Nomic). Diff ported math against these (CLAUDE.md rule 8).
  - **Vendored llama.cpp encoder graphs**: `examples/llama.cpp/llama.cpp/src/models/bert.cpp`
    (one graph for BERT / RoBERTa / XLM-R / NomicBERT / Jina, with per-arch branches for token types,
    positions, fused QKV, SwiGLU and RoPE), `nomic-bert.cpp`, and `conversion/bert.py` (the HF→GGUF
    tensor-name maps for `BertModel`/`BertForSequenceClassification`/`RobertaModel`/
    `XLMRobertaModel`/`XLMRobertaForSequenceClassification`/`NomicBertModel`, which is also exactly the
    HF-name ↔ GGUF-name table the weight-source layer needs). Pooling and rerank:
    `examples/embedding/embedding.cpp` plus the server's `tools/server/tests/unit/test_rerank.py`.
    **Tokenizer goldens**: `models/ggml-vocab-bert-bge.gguf` with `.inp`/`.out` (input strings and
    expected ids) and `ggml-vocab-nomic-bert-moe.gguf`, for free WordPiece/Unigram id-parity tests.
    llama.cpp does **not** cover MPNet or ELECTRA; for those the oracle is ONNX (MPNet) or see §4.
  - `examples/CrispASR/src/bert_encoder.cpp`: a standalone ggml BERT encoder (bert-base-uncased for
    MeloTTS: WordPiece, embeddings + LN, per-layer output taps), a second C++ read.
  - **vLLM model sources** (readable references; no Python gets added to this repo):
    `examples/vllm/vllm/vllm/model_executor/models/bert.py` (BERT + pooler + sequence
    classification), `roberta.py` (RoBERTa/XLM-R position-id offset and classification head), and
    `bert_with_rope.py` (NomicBERT and other RoPE+SwiGLU BERT variants).
  - No vendored source covers **MPNet** or **ELECTRA** (searched `examples/` 2026-09-25): port those
    deltas from the HF model cards/configs, with ONNX as the oracle.
  - `KokoroBertEncoder` (Audio) is an ALBERT encoder: check it for reusable layer code (DRY), but
    don't bend it into the general encoder.

## 2. Model facts that change the implementation (from each repo's real config, fetched 2026-09-25)

| Checkpoint | Arch facts | Tokenizer | Pooling / output (from `modules.json` + `1_Pooling/config.json`) |
|---|---|---|---|
| bert-base-uncased | 12L/768, post-LN, learned abs pos, token_type 2 | WordPiece lower | base model: last_hidden_state (+ `pooler_output` = tanh(dense(CLS))) |
| all-MiniLM-L6-v2 | BERT 6L/384 | WordPiece lower | **mean** + Normalize |
| bge-small-en-v1.5 / bge-large-en-v1.5 | BERT 12L/384, 24L/1024 | WordPiece lower | **CLS** + Normalize. Retrieval queries take the prefix "Represent this sentence for searching relevant passages: " (docs) |
| multilingual-e5-small | `BertModel` 12L/384 but **vocab 250037, SentencePiece** | Unigram (XLM-R style) | **mean** + Normalize; inputs must be prefixed `query: ` / `passage: ` |
| paraphrase-multilingual-MiniLM-L12-v2 | BERT 12L/384, 250k SentencePiece | Unigram | **mean, NO Normalize module** |
| electra-base-discriminator | BERT-identical encoder (`embedding_size` 768 = hidden, so no embeddings projection) + discriminator head (dense→GELU→dense_prediction) | WordPiece lower | base model: last_hidden_state; head: per-token real/fake logits. **Weights: no safetensors** (only `pytorch_model.bin`, `tf_model.h5`, `flax_model.msgpack`, `rust_model.ot`) |
| xlm-roberta-base | RoBERTa: **positions start at padding_idx+1 = 2** and skip pad tokens, token_type 1 | Unigram 250k (fairseq id offset baked into `tokenizer.json`) | base model: last_hidden_state |
| all-mpnet-base-v2 | MPNet 12L/768: RoBERTa-style positions (pad_token_id 1) + **T5-style bucketed relative position bias (32 buckets) computed once and added to every layer's attention scores**; no token_type | WordPiece-style (`MPNetTokenizer`: BERT basic+wordpiece with `<s>`/`</s>`) | **mean** + Normalize |
| nomic-embed-text-v1.5 | NomicBERT 12L/768: **RoPE** (base 1000, full head dim 64), **SwiGLU** MLP, fused Wqkv, **no biases** on qkv / fc1 / fc2, **post-norm** (`prenorm: false`), token_type 2, n_positions 8192 | WordPiece lower (bert-base vocab) | **mean, NO Normalize module in ST**; docs: task prefixes (`search_query: ` / `search_document: ` …) are mandatory; Matryoshka = layer_norm → truncate → L2 normalize |
| ms-marco-MiniLM-L6-v2 | BERT 6L/384 + `BertForSequenceClassification` (pooler tanh(dense(CLS)) → classifier 384→1) | WordPiece, **pair input** `[CLS] q [SEP] d [SEP]` with token_type 0/1 | score = raw logit (verify the ST CrossEncoder default activation in the card before choosing sigmoid vs identity) |
| bge-reranker-v2-m3 | XLM-R **large** 24L/1024, 8194 positions, `RobertaClassificationHead` (dense→tanh→out_proj on `<s>`) | Unigram 250k, pair input `<s> q </s></s> d </s>` | score = raw logit; `sigmoid` for the documented normalized score |
| tiny-Qwen2ForCausalLM-2.5 | Qwen2 2L, hidden 8, 4 heads / 2 KV, untied, vocab 152064 | Qwen BPE (existing) | logits |
| Qwen3-0.6B / 8B | Qwen3 (QK-norm, head_dim 128), 0.6B tied embeddings | Qwen BPE (existing) | logits / greedy tokens |

Verify every row against the downloaded `config.json` / `tokenizer_config.json` / model card
before coding; the table is a starting map, not a substitute.

## 3. Design (one encoder, configured per family)

1. **`TransformerEncoder`** (new, Engine project, CPU first): embeddings (word + position +
   optional token_type, LayerNorm) → N layers → final hidden states. Per-family switches, not
   separate engines:
   - position scheme: learned absolute from 0 (BERT/ELECTRA) · learned absolute from
     padding_idx+1, skipping pads (RoBERTa/XLM-R/MPNet) · RoPE (Nomic);
   - attention bias: none · T5-bucketed relative bias shared across layers (MPNet);
   - norm placement: post-LN (all six families here) with pre-LN kept possible;
   - MLP: GELU (erf) with biases · SwiGLU without biases (Nomic);
   - fused vs separate QKV, with or without biases.
   - **Attention mask**: key-padding mask for batched inputs (all padded positions masked out of
     every softmax), and the same mask drives mean pooling. Batched (padded) and single inputs must
     give identical embeddings: that's a test, not an assumption.
2. **Weight sources** behind one name map: (a) HF safetensors (primary; the real checkpoints the
   task names), (b) GGUF from llama.cpp's converter (`token_embd`, `blk.N.attn_q`, …) where one is
   already on disk. ELECTRA needs a new reader: a **Flax msgpack** reader (msgpack + numpy ext
   types, simple and AOT-safe) is preferred over a restricted PyTorch-pickle reader. First check
   whether the HF hub has a `refs/pr/*` safetensors conversion for the repo.
3. **Heads**: pooling (CLS / mean / last, mask-aware) + optional LayerNorm (Nomic Matryoshka) +
   optional L2; BERT pooler; sequence-classification heads (BERT: pooler → linear; RoBERTa:
   dense → tanh → out_proj); ELECTRA discriminator head.
4. **Model profile** loaded from the checkpoint directory: `config.json` → family + dims;
   `modules.json` / `1_Pooling/config.json` → pooling + normalize; the model card's documented
   prefixes → a per-model prefix table (E5, BGE, Nomic). Nothing hard-coded by guess.
5. **Tokenizers**: teach `HuggingFaceTokenizerSource` WordPiece and Unigram models plus their
   normalizers/pre-tokenizers/post-processors (`BertProcessing`/`RobertaProcessing`/
   `TemplateProcessing` for special tokens and pair inputs, `token_type_ids`, truncation), reusing
   `BertWordPieceTokenizer` and `UnigramTokenizer` for segmentation.
6. **Replace the fakes**: `BertGgufEmbeddingPipeline` → the real encoder; `EmbeddingEngine.Rerank`
   → a real cross-encoder when the checkpoint has a classification head (keep bi-encoder cosine
   only as an explicit mode for embedding models). Delete the sine/hash fallbacks.
7. **Decoder-only**: add an HF-safetensors `IModelTensorSource` for `Qwen2ForCausalLM` /
   `Qwen3ForCausalLM` / `GPT2LMHeadModel` (generalising the audio project's Qwen bridges, whose
   name maps already exist), so the named HF checkpoints load directly into the existing
   `ForwardPass`. GPT-2 note: HF stores `c_attn`/`c_fc` as Conv1D ([in, out]), so transpose on load.

## 4. Correctness plan (per the task: not "working" until validated on real checkpoints)

**Oracle coverage by family**: A/B/D/E → llama.cpp (`llama-server --embedding` / `--rerank`,
same graph source as above) **and** ONNX; C (MPNet) → ONNX only; ELECTRA → neither (see below).

**Oracle** (no Python in this repo): each repo's own **`onnx/model.onnx`** run through
onnxruntime in tests (available for 12 of 16; confirmed via the HF API 2026-09-25), plus the vendored
`tools/llama.cpp/llama-server.exe` (`--embedding` / `--rerank`) where a GGUF exists. Gaps:
- **bge-reranker-v2-m3** has no ONNX export: use llama.cpp (it supports XLM-R rerankers) on a
  GGUF conversion from the hub, and cross-check the ranking plus the documented example scores in
  the model card.
- **electra-base-discriminator** has no ONNX: check the model card's documented discriminator
  example; otherwise validate the shared BERT encoder through the other six BERT checkpoints, plus a
  weight-load checksum.
- **Decoder**: token-exact greedy parity vs llama.cpp on the GGUFs (existing receipts), plus logit
  comparison of our HF-safetensors load vs our GGUF load of the same model (F16/BF16 GGUF for exact
  parity, per the GPT-2 Q6_K lesson in `ModelCompatibility`).

**Tests** (in a new `tests/OpenTail.Stingray.Tests.Embeddings` or the existing fast project,
following conventions; each real-weight test must fail visibly when its checkpoint is missing, not
silently no-op, per CLAUDE.md rule 12):
- tokenizer: ids vs the ONNX/HF reference ids for a fixed multilingual sentence set (including
  accents, CJK, emoji, long words, pairs, truncation);
- per family: final hidden states vs ONNX (max abs diff and cosine; start at cosine ≥ 0.9999 and
  maxAbs ≤ 1e-3 for F32, then tighten to what's measured);
- per model: pooled + normalized embedding vs ONNX + ST pooling rules (catches pooling/normalize/
  prefix mistakes);
- padding: a batch of different-length inputs == the same inputs one at a time;
- rerankers: scores vs oracle for several query/doc pairs, including ranking order;
- decoder: greedy tokens / logits as above.

## 5. Performance (after correctness)

Route every Linear through `QuantizedWeightCache` / `PackedSgemmF32` (packed-once F32 or GemmQuant),
use the shared tiled attention with the key-padding mask, and avoid per-layer allocations (reuse
workspace buffers). Batch many short inputs into one padded GEMM. Benchmark: embeddings/s for
batch 1 and 32 at 128 tokens (MiniLM, bge-large, nomic), and pairs/s for both rerankers, CPU and
Vulkan (CLAUDE.md rule 13: iGPU numbers only describe this machine). Record in `PerformanceLeague.md`
next to the ONNX Runtime numbers from the same machine as a baseline.

## 6. Disk / downloads

Free space on 2026-09-25: C: 9.9 GB, F: (`models/_models`) 28 GB, K: 166 GB. Download all
encoder checkpoints (safetensors + tokenizer/config/pooling files + `onnx/model.onnx`), ~6 GB
total, the largest being bge-reranker-v2-m3 (~2.3 GB) and bge-large (~1.3 GB). Qwen3-8B safetensors
is ~16 GB: move idle checkpoints to `K:\_other_models` first (don't delete them), or keep Qwen3-8B
on the existing GGUF plus the 0.6B safetensors path. Moves get logged in the progress doc.

## 7. Phases and honest effort estimate

| Phase | Work | Size |
|---|---|---|
| 0 | Download checkpoints + oracles; record exact configs | small |
| 1 | Tokenizers (WordPiece + Unigram via `tokenizer.json`, pair inputs, token types) with id-parity tests | **medium, the main risk** |
| 2 | `TransformerEncoder` for family A (BERT) + pooling/normalize + ONNX parity for the 5 sentence/embedding BERT models and bert-base | medium |
| 3 | Family B (XLM-R positions) + E (both classification heads) → xlm-roberta-base, both rerankers | small (config deltas) |
| 4 | Family C (MPNet relative bias) and D (Nomic RoPE/SwiGLU/no-bias) | small each |
| 5 | ELECTRA weights reader (Flax msgpack) + discriminator head | small-medium |
| 6 | HF-safetensors decoder loading (Qwen2/Qwen3/GPT-2) + the tiny Qwen2 + logit parity | small-medium |
| 7 | Replace fake pipelines in CLI/server; README matrix rows (sourced) | small |
| 8 | Perf pass (CPU, then Vulkan) + DRY pass (CLAUDE.md rule 7) | medium |

"Relatively easy" is mostly true for the **architectures**: families B–E are small deltas on one
encoder, and the decoder side is already admitted. The real work is (1) that the current embedding
and rerank compute is a placeholder, so the encoder is new code, not a tweak; (2) tokenizer
exactness for WordPiece and 250k-vocab Unigram; (3) ELECTRA's missing safetensors.

## 8. Final report (at the end, per the task brief)

Report which models are fully working, partial or blocked; which implementations are shared; the new
capabilities (encoder, tokenizer modes, heads, weight readers); the validation numbers; the
benchmarks; and the remaining issues. No model is claimed until its real checkpoint passes against
the oracle.

## Status

- [x] Phase 0 (2026-09-25): all 15 repos downloaded to `models/_models/hf/<owner>__<name>/` (17 GB; F: 11 GB free after). Each has weights (ELECTRA: `flax_model.msgpack` only, as expected), `tokenizer.json` + `config.json`, the ST `modules.json`/`1_Pooling` where published, and `onnx/model.onnx` for 11 repos (none for bge-reranker-v2-m3, GPT-2, Qwen3-0.6B, tiny-Qwen2). Qwen3-8B stays on the existing GGUF.
- [x] Phase 1 tokenizers (2026-09-25, see progress below): WordPiece via vocab.txt + `tokenizer.json` normalizer flags; Unigram needs the **Precompiled charsmap** normalizer (port from `examples/llama.cpp/llama.cpp/src/llama-vocab.cpp` UGM `precompiled_charsmap`/xcda); id oracles = llama.cpp `models/ggml-vocab-bert-bge.gguf.inp/.out` + `llama-server /tokenize` on small GGUFs.
- [ ] Phases 2-8

### Phase 1 progress (2026-09-25)
- **WordPiece: golden-verified.** New `Tests.Core/BertWordPieceTokenizerGoldenTests` runs llama.cpp's BERT/BGE
  vocab test vectors (`ggml-vocab-bert-bge.gguf.inp/.out`, 46 cases: whitespace, accents, punctuation, emoji,
  CJK) through `BertWordPieceTokenizer` with the real bge-small `vocab.txt`. **Found a real bug:** accent
  stripping used `string.Normalize(FormD)`, which under this repo's `InvariantGlobalization` does not decompose
  non-ASCII, so "Äpfel" → [UNK] instead of `apfel` (same for "Cửa Việt"). Fixed with a culture-free
  `UnicodeAccentStrip` (generated 1,493-entry canonical-decomposition table + algorithmic Hangul). Now 46/46.
- **Same pitfall elsewhere (follow-up):** `UnigramTokenizer` and `SentencePieceBpeTokenizer` call
  `Normalize(FormKC)`, a no-op on non-ASCII under invariant mode. For XLM-R-family Unigram the real fix is the
  tokenizer.json **Precompiled charsmap** (next); `SentencePieceBpeTokenizer`'s users should be checked too.
- **Unigram (XLM-R family): golden-verified, 58/58 on all four checkpoints.** New `PrecompiledCharsmap`
  ports llama.cpp's UGM XCDA/`normalize_prefix` (longest-prefix charsmap replacement, as in
  SentencePiece's `normalizer.cc`). `UnigramTokenizer.FromTokenizerJson` now follows the file's own
  `normalizer` (`Precompiled`/`Strip`/`Replace`, alone or in a `Sequence`) and `pre_tokenizer`
  (`WhitespaceSplit`, `Metaspace` with `add_prefix_space`/`prepend_scheme`/`split`), accepts the older
  tokenizer.json with no `model.type` (xlm-roberta-base), and merges runs of consecutive UNK pieces as
  SentencePiece does. The GGUF path reads `tokenizer.ggml.precompiled_charsmap` too. Oracle: llama.cpp
  `llama-tokenize --ids` on `gpustack/bge-reranker-v2-m3-GGUF` Q8_0 (downloaded to `models/_models`, also
  the Phase 2 rerank oracle), over llama.cpp's SPM test inputs plus multilingual/charsmap cases, stored as
  `Tests.Core/Fixtures/xlm-roberta-ugm.inp/.out`. `XlmRobertaUnigramGoldenTests` runs it against bge-reranker-v2-m3,
  xlm-roberta-base, paraphrase-multilingual-MiniLM and multilingual-e5-small. The only config-driven difference is
  e5 (`Replace " {2,}"` without Strip), which keeps trailing whitespace as a `▁` piece where llama.cpp drops
  it, so the test compares e5 on right-trimmed input. Parler's T5 goldens still pass.
- **WordPiece from tokenizer.json + templates/pairs: done.** `BertWordPieceTokenizer.FromTokenizerJson` reads the
  vocab dict, `unk_token`/`continuing_subword_prefix`/`max_input_chars_per_word` and the `BertNormalizer` flags
  (`strip_accents: null` follows `lowercase`, as in HF). New `EncoderTokenizer` picks WordPiece or Unigram from the
  file and applies the `post_processor`: `TemplateProcessing` as written (BERT `[CLS] A [SEP] B [SEP]`, types 0/1;
  XLM-R `<s> A </s></s> B </s>`) or `RobertaProcessing` (MPNet), pad id from `padding.pad_id` or `[PAD]`/`<pad>`.
  It truncates with HF's `LongestFirst` arithmetic. `EncoderTokenizerTests` reruns llama.cpp's BGE golden through the
  JSON path for all 7 uncased-BERT-vocab checkpoints (bge-small/large, all-MiniLM, ms-marco-MiniLM,
  bert-base-uncased, electra-base, nomic-embed v1.5): 46/46 each. The pair/template/truncation cases are structural
  tests. Their real check is Phase 2+ rerank score parity against llama.cpp `--rerank` and ONNX.
- **`SentencePieceBpeTokenizer` (MOSS-TTS) now applies the model's own charsmap** (`NormalizerSpec` field 2 in the
  `.model` protobuf) instead of the `FormKC` no-op. The MOSS tokenizer and reference-parity real-weight tests still pass.
- **Not handled (noted):** added-token splitting inside the text (e.g. a literal `[MASK]` or `<mask>` in the input is
  word-pieced rather than mapped to its id). Embedding/rerank inputs don't need it.
- **Phase 1 is complete.** Next: Phase 2, the TransformerEncoder (BERT first: bge-small/all-MiniLM vs ONNX + llama.cpp
  `--embedding`).

### Phase 2 progress (2026-09-25): BERT encoder done, ONNX-verified
- New `Engine/Encoders/TransformerEncoder` (post-LN BERT, CPU F32 from HF safetensors): fused QKV, one packed GEMM
  per Linear over **all tokens of all inputs** (no padding; attention runs per sequence), exact-erf GELU
  (`Cpu/ErfGelu`), LayerNorm eps from config, `bert.`/`roberta.`/… prefix and `gamma`/`beta` names handled.
  `EncoderConfig` reads `config.json` (BERT/ELECTRA and RoBERTa/XLM-R position offset). New shared
  `Cpu/PackedLinearF32` (pack once, keep only panels); Audio's `DenseKernels` batched path now reuses it (DRY;
  Audio.Fast 72/72 and RVC HuBERT real-weight forward pass still pass).
- `HfEncoderEmbeddingPipeline` (implements `IEmbeddingPipeline`): pooling/Normalize/max_seq_length from the
  sentence-transformers files.
- **Parity vs each checkpoint's own ONNX** (new `tests/OpenTail.Stingray.Tests.Embeddings`, `BertEncoderOnnxParityTests`,
  5 texts incl. multilingual and a ~150-token passage), last_hidden_state per token:

  | checkpoint | tokens | maxAbs | min cosine | ours / onnx ms |
  |---|---|---|---|---|
  | bge-small-en-v1.5 | 191 | 3.0e-6 | 0.9999998 | 234 / 118 |
  | all-MiniLM-L6-v2 | 191 | 3.5e-6 | 0.9999998 | 92 / 20 |
  | multilingual-e5-small | 229 | 1.6e-6 | 0.9999998 | 496 / 48 |
  | paraphrase-multilingual-MiniLM-L12-v2 | 203 | 6.2e-6 | 0.9999998 | 46 / 43 |
  | bge-large-en-v1.5 | 191 | 3.2e-5 | 0.9999997 | 360 / 384 |

  (Timings are first-call, unoptimized; the perf pass is Phase 8.) Batched output equals one-at-a-time bit for bit.
- **bert-base-uncased**: its `model.onnx` is `BertForMaskedLM` (logits only), so `BertMlmOnnxParityTests` applies the
  checkpoint's MLM head to our hidden states: logits maxAbs 1.8e-4, min cosine 0.9999997 over 6 inputs; the model
  card's fill-mask prompt ("Hello I'm a [MASK] model.") gives top id 4827 "fashion", as documented.
- **Pooling independently checked** against `llama-server --embedding` on `bge-small-en-v1.5-q8_0.gguf`
  (`SentenceEmbeddingLlamaCppTests`, via a reusable `LlamaServerOracle` test helper): cos 0.9996-0.9999 per text
  (Q8_0-bound), where mean pooling instead of the configured CLS would give 0.895-0.959.
- Next: Phase 3 (XLM-R positions + classification heads → xlm-roberta-base, ms-marco-MiniLM, bge-reranker-v2-m3).

### Phase 3 progress (2026-09-25): XLM-R + cross-encoders done
- **xlm-roberta-base** (RoBERTa positions from pad+1): its root `model.onnx` is `XLMRobertaForMaskedLM`, checked through
  the checkpoint's `lm_head` on our hidden states: logits maxAbs 3.6e-4, min cosine 0.9999998 over 6 inputs; the card's
  fill-mask prompt gives `▁fashion` (54543) as documented. (`BertMlmOnnxParityTests` is now a Theory over both.)
- New `HfCrossEncoderPipeline` (implements `IRerankerPipeline`): pair input via `EncoderTokenizer.EncodePair`, head picked
  from the checkpoint's tensors (RoBERTa `classifier.dense`→tanh→`out_proj`; BERT `pooler`→tanh→`classifier`); raw
  logit score, optional sigmoid.
- **ms-marco-MiniLM-L6-v2** vs its ONNX: identical to 5 decimals on 4 pairs (8.84585, -11.24556, 0.66096, -11.19189).
- **bge-reranker-v2-m3** (no ONNX): matches the model card's documented `compute_score` values:
  `[query, passage]` -5.6501 (card -5.6523), panda batch -8.1838 / 5.2650 (card -8.1875 / 5.2617, card computed in fp16).
  `llama-server --rerank` on the gpustack FP16 and Q8_0 GGUFs gives the **same ranking** but scores the card's own
  `[query, passage]` example -6.37 / -6.33, i.e. a systematic llama.cpp-side difference from the HF reference
  (both quantizations agree with each other, so it isn't precision). The model card is used as the tight oracle; llama.cpp
  is checked for ranking order only. `bge-reranker-v2-m3-FP16.gguf` (1.1 GB) was added to `models/_models` for this.
- Next: Phase 4 (MPNet relative position bias; NomicBERT RoPE/SwiGLU/no-bias).

### Phase 4 progress (2026-09-25): MPNet + NomicBERT done, ONNX-verified
- `EncoderFamily` switch in `EncoderConfig`/`TransformerEncoder`:
  - **MPNet**: `attention.attn.{q,k,v,o}` names, positions from pad+1, no token types, T5 bidirectional bucketed
    relative bias (`encoder.relative_attention_bias`, 32 buckets, max distance 128, float32 log like torch) added to
    the scaled scores in every layer.
  - **NomicBERT** (per vLLM `bert_with_rope.py` + llama.cpp tensor map): no position table, `emb_ln`, fused bias-free
    `Wqkv`, NeoX RoPE base 1000 over the full 64-dim head, FFN `fc2(silu(fc12·h) ⊙ fc11·h)` (fc12 = gate, fc11 = up,
    packed as one [gate|up] GEMM), post-LN `norm1`/`norm2`. Unsupported Nomic variants (prenorm, biases, partial or
    scaled rotary, parallel block) are rejected from config rather than mis-run.
- `BertEncoderOnnxParityTests` last_hidden_state vs ONNX: **all-mpnet-base-v2** maxAbs 3.8e-6, min cos 0.9999998
  (191 tokens); **nomic-embed-text-v1.5** maxAbs 1.4e-5, min cos 0.9999998. The 5 BERT-family rows are unchanged.
- Pooling from the ST files: MPNet mean + Normalize (max 384 tokens), Nomic mean, no Normalize module (max 8192).
  Nomic's task prefixes (`search_query: ` etc.) and Matryoshka (layer_norm → truncate → L2) are caller/pipeline
  options still to wire in Phase 7.
- Next: Phase 5 (ELECTRA: Flax msgpack weights + discriminator head).

### Phase 5 progress (2026-09-25): ELECTRA done
- Instead of an engine-side Flax reader: the hub's official **safetensors-convert bot PR** (`refs/pr/6`, author
  `SFconvertbot`, 440,318,828 bytes, SHA-256 `7f7b1605…f8ef` = the hub's linked ETag) was downloaded as
  `model.safetensors`. It is **verified bit-identical** to the repo's own `flax_model.msgpack`: a test-only
  `FlaxMsgpackReader` (msgpack + flax ndarray ext type 1) compares all 201 tensors (Linear kernels transposed).
  The stray `electra.embeddings_project` is ignored, as HF does when `embedding_size == hidden_size`.
- New `ElectraDiscriminator` (`dense_prediction(gelu(dense(h)))` per token). Model-card example "The quick brown fox
  fake over the lazy dog": only "fake" gets a positive replaced-token logit (+0.27; all others -1.65 to -5.74).
  The encoder is the ONNX-verified BERT path (same `electra.` prefix handling).
- Next: Phase 6 (HF-safetensors decoder loading: Qwen2/Qwen3/GPT-2 + tiny Qwen2 + logit parity).
