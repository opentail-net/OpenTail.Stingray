# LFM3-Omni-2B (`lfm3`, text path) port plan

**Status (2026-11-24): PORTED, SMOKE-TESTED ON THE REAL CHECKPOINT ONLY. NOT ADMITTED.** Policy: CLAUDE.md rule 14. Parent: [2026-11-24-recent-models-support-plan.md](2026-11-24-recent-models-support-plan.md), item T1.3.

## Reference and checkpoint

- Reference: llama.cpp `src/models/lfm3.cpp` (211 lines) and `src/models/lfm-base.cpp` (`build_lfm_dense_ffn`), read from upstream master on 2026-11-24 (scratch clone `33aeb17f85869279e3821f746bc645357139bc2c`). Not pinned in this repo; not built here (see the Spark plan §7).
- Checkpoint: `LiquidAI/LFM3-Omni-2B-GGUF`, revision `d508b88f08a5b4286f2d2805a6a8d69a0ff44c72`, file `LFM3-Omni-2B-Q4_K_M.gguf` (1,099,630,080 bytes; SHA-256 `9f8d30e9d4fd7a9349d586e077988229b3ca0a0e4ad730897563d1eccc9893e2`, verified by `stingray pull` against the published hash). Stored at `E:\_models\lfm3-omni-2b\`. Licence per the GGUF: `other` (LFM 1.0). Not a gate: the project only runs inference and ships neither weights nor binaries (owner, 2026-11-24).

## Architecture (from the GGUF and lfm3.cpp)

- 24 layers, embedding 2048, FFN 7680 (SwiGLU, SiLU, parallel gate/up), vocab 128000.
- Hybrid: `attention.head_count_kv` is an array `[0, 0, 4, 0, …]`: layers with 0 KV heads are **short-conv** layers (`shortconv.l_cache` 7), the others are attention (16 heads, 4 KV heads, head dim 128).
- Attention layers: QK-norm (`attn_q_norm`, `attn_k_norm`), NEOX RoPE on every layer (`rope.dimension_count` 128 = head dim, `rope.freq_base` 1e6), `rope_freqs.weight` [64] masks the non-SWA layers as in Gemma 4 (ForwardPass.cs:540-545), and a **per-head sigmoid attention output gate** (`attn_gate.weight` [2048, 16], `sigmoid(wqkv_gate · attn_norm_out)` multiplied into the attention output before `wo`; lfm3.cpp:124-130). That gate is the same shape as Spark-X's (`AttentionOutputGatePerHead`).
- Sandwich norms: `post_attention_norm` and `post_ffw_norm` (both present on every layer; lfm3.cpp:172-192). Presence is detected from the tensors.
- `sliding_window` 256 with `sliding_window_pattern` (24-item bool array); `rope.freq_base_swa` 10000.
- No embedding scale (lfm3.cpp uses the plain attention scale `1/sqrt(head_dim)`).
- Tokenizer: `gpt2`, pre-type `lfm2` (llama-vocab.cpp:2474; already implemented). `add_bos_token` is **false** in the GGUF, but the chat template starts with `{{- bos_token -}}` (BOS id 124894), so the chat template must be used (see Verification 1).

## Changes in this port (2026-11-24)

- `src/OpenTail.Stingray.Engine/Architectures/Lfm3Architecture.cs` (new): `Id = "lfm3"`, `NotAdmitted`, `Traits.Hybrid = ShortConv`, `UsesNeoxRope`, `ApplyModelSemantics = GatedSwa(muse: false)` with `RopeOnlySwaLayers = false`, `EmbeddingScale = 1f`, `AttentionOutputGatePerHead = true`. `CPU` only, no continuous batching.
- `Architectures/BuiltInArchitectures.cs`: one manifest line.
- No new engine code: the per-head gate and the sandwich-norm/rope-freqs paths were already present (the gate was added for Spark-X on 2026-11-24, see the Spark plan).
- No new tokenizer literal (the pre-type `lfm2` is already in `PreTokenizerPatterns`).

## Verification so far

1. **Real checkpoint smoke (not parity).** `stingray admit-arch` (bypassed forward pass) ran cleanly on the raw prompt. The CLI in raw mode (`-m … -p "…" -g -1`, with `STINGRAY_DIAGNOSTIC_ALLOW_UNSUPPORTED_ARCH=1`) without BOS produced a degenerate repeat (token 355 eight times, and `<|audio_start|>` in the text). The same run with the model's own chat format (`<|startoftext|><|im_start|>user…<|im_end|><|im_start|>assistant…`, the BOS is `<|startoftext|>`) answered **"The capital of France is Paris."** Prefill 25 tokens at 30.5 t/s, decode at 28.9 t/s, one run.
2. **Control.** The admitted LFM2-1.2B-Q8_0 (`F:\_models`), raw prompt on the CPU path (`-g 0`), answered "The capital of France is Paris. It's a city…", so the shared short-conv path produces sane text. The LFM3 raw-prompt result is attributed to the missing BOS, not to the port; that is an inference from one run on each model, not a reference comparison. (The LFM2 run with `-g -1` failed on the GPU path with a missing `attn_q` tensor in a shortconv layer. That is a pre-existing GPU-path issue, not part of this port.)
3. **Contract tests**: see the Spark plan §Verification 3. LFM3 adds one facts-baseline line.

## Not verified (blocking admission)

- [ ] **Independent reference** (same blocker as Spark: the C:\Git llama.cpp build (`c0bc8591e`) does not register `lfm3`; a llama.cpp at or after master `33aeb17f` is needed).
- [ ] **Golden parity** with teacher forcing, including a prompt past the 256-token sliding window and a case with an image or audio token (out of scope for the text path, see below).
- [ ] **Synthetic forward test**: none yet for LFM3 (the short-conv reference would have to be written; the Spark test pattern can be reused for the attention part).
- [ ] **Timed runs** (one run only so far).
- [ ] **Omni inputs** (image, audio): not ported. Text path only (owner decision O6).
- [ ] **GPU paths**: refuse the gate (CPU only), as Spark.
- [x] **Licence** (`other`, LFM 1.0): not a gate (owner, 2026-11-24: inference only, nothing shipped).
- [ ] **STATUS.md row**: not added.

## Notes

- `rope_freqs.weight` is applied only through the existing non-SWA global mask. If llama.cpp's frequency factors differ from that mask for this checkpoint, the golden will show it. Unchecked.
- The raw-prompt degenerate output is a property of how the CLI sends prompts (no BOS). Anyone testing LFM3 should use the chat template.
