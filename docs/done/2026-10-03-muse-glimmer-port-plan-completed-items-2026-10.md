# Completed items split from 2026-10-03-muse-glimmer-port-plan.md (2026-10-30)

Finished items moved verbatim out of [../2-coverage/2026-10-03-muse-glimmer-port-plan.md](../2-coverage/2026-10-03-muse-glimmer-port-plan.md). Open items stay there. Each block notes the section it came from.

> from: ## New work
- [x] `ModelGraph`: the `muse-glimmer`/`muse_glimmer` arch.
  - SWA period (default 4) and `RopeOnlySwaLayers`; optional `rope.freq_base_swa`.
  - Raw logit scale (0.19611613513818404); softcap (20.0) only when key is present.
  - Fields `PostNormEps` (1e-8), `InputEmbeddingRmsNorm`, `AttentionOutputGate`.

> from: ## New work
- [x] `ForwardPass` decode:
  - load `attn_gate` per layer;
  - gate projection from `_normBuf` next to Q/K/V, `attnOut *= sigmoid(g)` before `Wo`;
  - `PostNormEps` (1e-8) at both post-norm sites;
  - unweighted embedding RMSNorm in `Forward`.

> from: ## New work
- [x] `ModelCompatibility`: a `// muse-glimmer — NOT admitted` block (not in allowlist).
  `STINGRAY_DIAGNOSTIC_ALLOW_UNSUPPORTED_ARCH=1` runs it for experiments.

> from: ## Verification (levels as in [ported-families-todo](ported-families-todo.md))
- [x] **Specification test (level 2):** created a synthetic tiny `muse-glimmer` GGUF (4 layers, P = 2,
   window 3, small dims, random F32 weights; `MuseGlimmerSyntheticTests`). Models converted GGUF semantics:
   `attn_q_norm.weight = 3.87f`, `attn_k_norm.weight = 1.0f`, converted per-layer norm weights, unshifted
   final output norm, SWA masking past the window, NoPE on full layers, attention output gate, 1e-8 post-norms,
   embedding norm and output multiplier scale-then-softcap; verified max |Δlogit| < 1e-3 (F32). Passed 2026-10-03.

> from: ## Verification (levels as in [ported-families-todo](ported-families-todo.md))
- [x] **Real weights, text, CPU (level 4), 2026-10-03:** `unsloth/Muse-Glimmer-30B-GGUF` `UD-Q4_K_XL` (14.8 GB), greedy,
   `--temp 0 --repeat-penalty 1.0`, raw prompt via a pass-through `--chat-template`, against llama.cpp `bed0a8566`
   (`llama-completion -no-cnv`, CPU):
   - "The capital of France is": identical for 20 tokens, then a **0.006-nat near-tie** (" in" NLL 0.7108 vs " proper"
     0.7169 in our engine), i.e. quantised-accumulation noise;
   - "Q: What is 17 times 23? Think step by step. A:": identical through "340+51=391. So";
   - "def fibonacci(n):": identical through `else:`.
   Decode about 1.1 t/s on this CPU. **Not covered:** prompts were under 40 tokens, so the sliding-window mask on real weights
   is not exercised; no `admit-arch` verdict; no timed runs; vision and DFlash untouched. Harness notes: the CLI's default
   `--repeat-penalty 1.1` breaks greedy parity (always pass 1.0); the model's real chat template crashed our Jinja evaluator
   (list `+` list), fixed 2026-10-03 with a regression test but not compared against a reference rendering. The checkpoint was
   deleted afterwards.

> from: ## Verification (levels as in [ported-families-todo](ported-families-todo.md))
- [x] **`admit-arch` verdict, 2026-10-03:** reference ids from `llama-server /completion` (`return_tokens`), prompt "The capital of
   France is": **ADMIT, 8-of-8 exact**; our prompt tokenisation is identical to llama.cpp's (6 tokens, BOS 200000).

> from: ## Verification (levels as in [ported-families-todo](ported-families-todo.md))
- [x] **Sliding-window mask on real weights, 2026-10-03.** The model's window is 2048 (`muse-glimmer.attention.sliding_window`).
   - A 3,748-token prompt (past the real window), 24-token greedy continuation: **identical** (llama.cpp `-c 4096`).
   - Window **patched to 64** in the GGUF header (value restored to 2048 afterwards) with three ~300-token prompts, so most
     layers' attention is truncated: identical for 6 / ~20 / ~16 tokens, then a divergence on each. The first was measured: our
     engine scores `..` at NLL 1.5085 and `+` at 1.5602 (a 0.05-nat near-tie); the other two margins were not measured.

> from: ## Verification (levels as in [ported-families-todo](ported-families-todo.md))
- [x] **Speed, same machine, CPU (Ryzen 5700G, default threads):** llama.cpp prefill **15.2 t/s** (3,746 tokens, batched) and decode
   **2.42-2.48 t/s**; ours prefill **1.4-1.7 t/s** (per-token path for gated models) and decode **1.1-1.7 t/s**. Prefill is about
   10x slower, decode about 0.6x. Single runs, no thread tuning on either side.

> from: ## Verification (levels as in [ported-families-todo](ported-families-todo.md))
- [x] **Admission (level 5), 2026-10-03: text only, CPU only.** `muse-glimmer` / `muse_glimmer` are in the `ModelCompatibility`
   allowlist with the evidence and limits; STATUS row added. The CPU server path now keeps gated models off the continuous batcher
   (`PrefillWithCache` and `BatchForwardMulti` throw for them). Caveats: the checkpoint's license bucket was not reviewed, so no
   real-weight parity test is committed; no vision, no DFlash, no GPU.

