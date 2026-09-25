# Plan: the remaining "most downloaded on HF" families (written 2026-09-25)

Follows `docs/2026-09-25-encoder-embedding-reranker-plan.md` (done). The user's list of most-downloaded HF families was
checked against the code base on 2026-09-25 (source, `ModelCompatibility`, README matrix). Already covered: MiniLM, BGE
small/large, MPNet, E5, Nomic, BGE reranker, ms-marco MiniLM, BERT, ELECTRA, XLM-R, GPT-2, Qwen3-0.6B/8B, Qwen3-Coder MoE
(`qwen3moe`), Qwen3.8-27B (`qwen35`, 24/24 greedy vs llama.cpp), Qwen3-VL, SD1.5 + derivatives, Kokoro.

**Out of scope (user decision 2026-09-25): MiniMax-H3.** It's a text/image-to-video+audio diffusion system with a 33B DiT
(66 GB BF16), a 66 GB text encoder and a 10 GB video VAE; not realistic here.

## Targets

| # | Family | Checkpoint(s) (HF downloads, 2026-09-25) | Gap today | Weights | Oracle (no Python in this repo) |
|---|---|---|---|---|---|
| 1 | **BGE-M3** | BAAI/bge-m3 (36.7M) | Dense probably works (XLM-R-large, CLS + norm) but untested; sparse (`sparse_linear.pt`) and ColBERT multi-vector (`colbert_linear.pt`) heads missing | main repo: `pytorch_model.bin` only; SFconvertbot safetensors PR; the two heads are tiny PyTorch zip pickles (3.5 KB, 2.1 MB) | llama.cpp `--embedding` on a bge-m3 GGUF (dense); model card's documented dense/sparse/ColBERT example scores; head math from the FlagEmbedding source (`BGEM3FlagModel`), read, not run |
| 2 | **T5** (encoder-decoder) | google-t5/t5-small (24.5M), t5-base, google/flan-t5-base | Only the T5 *encoder* exists (Parler, MusicGen, AudioGen conditioning); no decoder / text-to-text generation | safetensors | t5-small's own `onnx/encoder_model.onnx` + `decoder_model.onnx`; llama.cpp `src/models/t5.cpp` (llama.cpp runs T5 GGUFs, greedy receipt) |
| 3 | **Chronos** (time series) | amazon/chronos-2 (21.9M), amazon/chronos-bolt-small (6.9M), amazon/chronos-t5-small (1.1M) | Nothing | safetensors | Bolt and chronos-t5 are T5 underneath (`model_type: t5`), so #2 validates the backbone. Bolt adds patch-embedding input + quantile head; Chronos-2 is its own `Chronos2Model`. **Open question:** no ONNX or C++ reference found; candidates are model-card example forecasts or vendoring `chronos-forecasting` as a read-only reference |
| 4 | **CLIP** (standalone) | openai/clip-vit-base-patch32 (22.3M), clip-vit-large-patch14 (8.7M) | CLIP text encoder exists inside SD; CLIP vision towers inside LLaVA-style models; no standalone image/text embedding + zero-shot pipeline | large: safetensors; base: `.bin` only, SFconvertbot PR | Xenova/clip-vit-base-patch32 `onnx/model.onnx` (text + image + logits); llama.cpp `llama-mtmd` CLIP for the vision tower |
| 5 | **Wav2Vec2** CTC ASR | facebook/wav2vec2-base-960h (1.5M), jonatasgrosman/wav2vec2-large-xlsr-53-english | Related encoders only (RVC HuBERT, OmniVoice semantic encoder); no Wav2Vec2 CTC ASR | safetensors | Xenova/wav2vec2-base-960h `onnx/model.onnx` (logits); vendored `examples/CrispASR/src/wav2vec2-ggml.cpp`; LibriSpeech clips already used for the ASR re-check |
| 6 | **MobileNetV3** | timm/mobilenetv3_small_100.lamb_in1k (19.6M), mobilenetv3_large_100 | Nothing (the MobileNetV5 adapter is unrelated) | safetensors (timm) | onnx-community/mobilenetv3_small_100.lamb_in1k `onnx/model.onnx` (logits); ImageNet top-5 on sample images |
| 7 | **EfficientNet** | timm/efficientnet_b0.ra_in1k (0.8M) | Nothing | safetensors (timm) | No ONNX export found yet: find one or check the timm model card's documented top-5 on its example image |
| 8 | **Qwen3.6-35B-A3B** (FP8 safetensors + CPU perf) | Qwen/Qwen3.6-35B-A3B(-FP8) | Runs from GGUF (`qwen35moe` admitted) but CPU is 0.05x llama.cpp prefill / 0.17x decode (`PerformanceLeague.md`, 2026-09-10); the HF safetensors path only covers dense Qwen2/Qwen3 | GGUF on disk; FP8 safetensors ~35 GB | llama.cpp on the same GGUF (existing receipts) |

## Shared building blocks (build once, reuse)

- **Weights**: `SafetensorsLoader` (F32/F16/BF16/FP8 already). New: a minimal, AOT-safe **PyTorch zip-pickle reader**
  (`data.pkl` opcodes for an `OrderedDict` of `torch._utils._rebuild_tensor_v2` + raw `data/N` storages; refuse anything
  else) for BGE-M3's two heads and any `.bin`-only checkpoint without a convert-bot PR. The convert-bot PR is preferred
  where it exists (as for ELECTRA), verified against the `.bin` with the reader.
- **Encoder**: `TransformerEncoder` (BGE-M3 = XLM-R large; T5/CLIP/Wav2Vec2 get their own blocks but share
  `PackedLinearF32`, LayerNorm/RMSNorm, the transposed-K/V attention and erf-GELU).
- **T5 relative position bias**: MPNet already has the bucketed bias; T5's is the same bucket function (uni- vs
  bidirectional), so share it (DRY) rather than copy it.
- **Image preprocessing** (CLIP, MobileNetV3, EfficientNet): resize/center-crop/normalize from `preprocessor_config.json`
  or timm's `pretrained_cfg` (bicubic vs bilinear, crop_pct, mean/std). One shared implementation.
- **Conv2d / depthwise conv / BatchNorm fold / hard-swish / squeeze-excite**: check `Diffusion` (im2col conv) and Audio CNN
  ports (MarbleNet/Citrinet depthwise-separable) for reuse before writing new kernels.
- **Tests**: each real-weight test uses `Assert.SkipUnless` so a missing checkpoint shows as skipped (CLAUDE.md rule 12).

## Order and estimates

| Phase | Work | Size |
|---|---|---|
| 0 | Download checkpoints + oracles (F: has ~6 GB free; skip bge-m3's 2.3 GB ONNX) | small |
| 1 | BGE-M3 dense: load via convert-bot safetensors, parity vs llama.cpp GGUF + model card | small |
| 2 | PyTorch zip-pickle reader + BGE-M3 sparse and ColBERT heads (+ `compute_score` combination) | small-medium |
| 3 | T5 encoder-decoder (t5-small, flan-t5-base): decoder with cross-attention + KV cache, greedy generate; SentencePiece tokenizer (Unigram, existing) | medium |
| 4 | Chronos-Bolt / chronos-t5 on the T5 backbone; Chronos-2 (after settling the oracle question) | medium |
| 5 | Wav2Vec2 CTC (conv feature extractor, positional conv embedding, encoder, CTC greedy) | medium |
| 6 | CLIP standalone (ViT + text transformer + projections; zero-shot) | medium |
| 7 | MobileNetV3 + EfficientNet (shared CNN blocks + image preprocessing) | medium |
| 8 | Qwen3.6-35B-A3B CPU perf on the hybrid GDN MoE path (separate perf investigation; FP8 safetensors after) | large |
| 9 | CLI/server wiring where it fits (embeddings/rerank for BGE-M3, `transcribe` for Wav2Vec2), README rows, perf + DRY pass (CLAUDE.md rule 7) | small each |

## Status

- [x] Phase 0 (2026-09-25): bge-m3 (safetensors from SFconvertbot PR #130, identical hash to PRs #116/#118; heads `sparse_linear.pt`/`colbert_linear.pt`), `bge-m3-Q8_0.gguf` (gpustack, llama.cpp oracle), t5-small (+ ONNX encoder/decoder), wav2vec2-base-960h (+ Xenova ONNX), timm mobilenetv3_small_100 (+ onnx-community ONNX), timm efficientnet_b0, chronos-bolt-small, all under `models/_models/hf/`. F: now has 2.1 GB free: CLIP, Chronos-2 and flan-t5 wait for space (the user declined moving the encoder benchmark GGUFs to K:).
- [x] Phase 1 (2026-09-25): **BGE-M3 dense works unchanged** through `HfEncoderEmbeddingPipeline` (XLM-R path, CLS + Normalize from the ST files). `BgeM3Tests`: the card's four dense scores 0.62590/0.34750/0.34987/0.67825 vs 0.62598/0.34741/0.34985/0.67822 (fp16 card, within 1e-4); llama.cpp Q8_0 cos 0.9992-0.9996 over 9 texts.
- [x] Phase 2 (2026-09-25): new `Core/TorchCheckpointReader` (PyTorch zip `state_dict` reader: allowlisted opcodes, only `OrderedDict` + `_rebuild_tensor_v2` callable, Float/Half/BFloat16/Long/Int storages, contiguous only; refuses anything else). Verified on timm mobilenetv3_small_100: `pytorch_model.bin` gives all 210 weight tensors bit-identical to its `model.safetensors` (`TorchCheckpointReaderTests`). New `Engine/Encoders/BgeM3Pipeline`: dense + sparse (`relu(sparse_linear)`, max per token id, cls/eos/pad/unk dropped) + ColBERT (`normalize(colbert_linear(h[1:]))`) and `compute_score`'s weighted modes. **All 20 card `compute_score` values match within 1.1e-4** (e.g. colbert 0.77967/0.46212/0.45242/0.78987 vs card 0.77965/0.46215/0.45238/0.78986; sparse 0.19554/0.00880/0/0.18041 vs 0.19556/0.00880/0/0.18030), and the card's 7 lexical weights for "What is BGE M3?" within 2e-4. Not wired into CLI/server yet (dense already works there through the generic embedding path) — Phase 9.
- [x] Phase 3 (2026-09-25): new `Engine/Encoders/T5Model` (encoder-decoder, CPU F32 from safetensors, ReLU and gated-GELU FFN, tied or untied LM head, decoder self-attention K/V cache + cross K/V computed once, greedy generate; `EncodeEmbeddings`/`StepEmbeddings` for models that feed their own embeddings, i.e. Chronos). The MPNet bucket helper now also does T5's unidirectional (decoder) buckets. `T5Tests` vs each checkpoint's ONNX (3 prompts, teacher-forced decoder): **t5-small** encoder maxAbs ≤ 6.7e-7, logits cos 0.9999999, argmax 49/49; **flan-t5-small** (gated-GELU, untied; Xenova ONNX) encoder ≤ 5.5e-7, logits cos ≥ 0.9999998, argmax 56/56; cached decoding bit-identical to the full pass. Greedy receipt: t5-small "translate English to German: The house is wonderful." → "Das Haus ist wunderbar." (the HF docs example). flan-t5-small downloaded (653 MB incl. ONNX); F: now 1.4 GB free.
- [~] Phase 4 (2026-09-25): **Chronos-Bolt done, behaviour-verified (no numeric oracle yet).** Reference source vendored read-only at `examples/chronos-forecasting/` (`chronos_bolt.py`, Apache-2.0; `examples/` is gitignored, local only, fetched from amazon-science/chronos-forecasting main). New `Engine/Encoders/ChronosBoltModel` (instance norm, NaN-left-padded patches + mask features, ResidualBlock in/out, [REG] token, masked T5 encoder, one decoder step, quantile head, long-horizon quantile-path heuristic incl. `torch.quantile` linear interpolation); `T5Model` gained encoder/cross-attention key masks and a max-subtracted softmax (T5 scores are unscaled; T5 ONNX parity unchanged). `ChronosBoltTests` on chronos-bolt-small: seasonal+trend median MAE 0.20 vs seasonal-naive 0.91, 0 quantile-order violations over 64 steps, affine equivariance 1.6e-7, NaN input finite, 150-step horizon MAE 0.20. **Open: numeric oracle.** No ONNX export or C++ port exists on the hub; options are running the vendored reference once outside the repo to capture golden quantiles (needs the user's OK given the no-Python rule) or published example forecasts. Chronos-2 (`Chronos2Model`, a different architecture) not started: needs its source vendored and 0.48 GB of disk.
- [x] Phase 5 (2026-09-25): new `Audio/Wav2Vec2/Wav2Vec2CtcModel` (config-driven HF `Wav2Vec2ForCTC`: group- or layer-norm conv feature extractor, feature projection, weight-normalized grouped positional conv, post-LN or stable pre-LN encoder, CTC greedy; convolutions as im2col + `PackedLinearF32`). `Wav2Vec2CtcRealWeightsTests` on facebook/wav2vec2-base-960h vs Xenova's ONNX of the same weights over the 4 audio.cpp LibriSpeech clips: logits maxAbs ≤ 3.3e-3, cos 1.000000, argmax 1115/1115 frames, transcripts identical to the ONNX's; **WER 1.4 % (1/69 words;** the one error "FOTED" for "FORWARDED" is the model's own, ONNX gives the same). 14.2 s of audio in 1.6 s CPU. Only the base (group norm, post-LN) layout is checkpoint-verified; the large/XLS-R (layer norm, stable LN) branch is untested. DRY follow-up: RVC's `RvcHubertEncoder` is the same architecture with fixed dims and could run on this model (not done: its verified numerics weren't touched).
