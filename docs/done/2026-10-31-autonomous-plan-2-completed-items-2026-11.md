# Plan 2 (2026-10-31): carry-on work, done without owner input - completed items

Ticked items moved from [docs/2026-10-31-autonomous-plan-2.md](../00-current-work.md#open-work-as-of-2026-11-03) on 2026-11-03. Each keeps its section heading and nested lines.

## Checklist

  - [x] Test the slot-cache candidates: placement timing confirmed (8 of 8 runs bit-identical with prefetcher and warm-up off); expert warm-up alone and predictive prefetch alone are not the cause

  - [x] Record a decision about the bar: 0.97 per-step floor plus a median above 0.999 for this row only (PR 83)

- [x] **P2** Handoff receipts on the Vulkan hybrid (Qwen2.5 admitted; Phi-3, gemma-2 and Granite recorded as not admissible, with the reasons; P2b and P2c fixed; P2d blocked on hardware) (one passing permanent row each; a failing family stays Unverified with its cosine recorded)
  - [x] Qwen2.5 (receipt on the Vulkan hybrid; three permanent rows)
  - [x] Phi-3: tried, cannot run on the hybrid (fused `attn_qkv`, full offload only), stays Unverified; the full-GPU handoff path for it is untested
  - [x] gemma-2: tried, fails (min cosine 0.90 handoff vs sequential); the cause is the Vulkan hybrid and full GPU passes, not the handoff (see P2b); stays Unverified
  - [x] Granite: a dense checkpoint was on `H:\_models` all along (`ibm-granite_granite-3.3-2b-instruct`, missed by my first search because its name starts with `ibm-`). It is not a receipt candidate: the Vulkan hybrid ignores Granite's scale fields and produces garbage, see P2c. The earlier vision-LM result was a property of that file
  - [x] **P2c** The Vulkan hybrid applies none of the model scale fields (embedding, residual, logit, attention-scale override). Found with the dense Granite: `-g 8` answered `C. to - > and \ C、` instead of `Paris` (sequential hybrid against CPU: min cosine -0.36; CPU perplexity 7.81, full GPU 7.77); gemma-3-4b the same (0.47); ERNIE-0.3B (no scales) 0.998. **Superseded the same day by [plan 3, Q2](../00-current-work.md#open-work-as-of-2026-11-03): the hybrid now applies the scale fields and the guard covers soft-capping only.** Original fix: Fixed: `GpuForwardPass.VulkanHybridUnsupportedReason` refuses such models, and `run` prints a clear error with exit code 1 instead of a stack trace. Tests: 6 unit tests, 2 real-model refusals, a 13-checkpoint guard that no receipt checkpoint is caught; three production mutations each fail tests. The clean CLI error was verified by hand only
  - [x] **P2b** Silent GPU degradation for CPU-only architectures. Found: real runs were already safe (the gemma2 descriptor declares `SupportedBackends = Cpu` and the planner falls back to the CPU for `-g 8` and `-g -1`), but the override was silent, `stingray plan` reported 26 GPU layers (it did not consult the descriptor), and `perplexity -g -1` built the GPU pass directly and measured a 22 percent worse gemma-2 without saying so. Fixed: the planner raises a `cpu_only_architecture` warning when explicit GPU layers are overridden, `run` prints it as one note, `plan` consults the descriptor (and shows the reason), and `perplexity -g -1` refuses a CPU-only architecture. Tests: 3 planner and 3 plan tests, four production mutations each fail tests; the `run` note and the perplexity refusal were checked by hand only. Not changed: code that constructs the GPU passes directly (tests, library users) still accepts gemma2, and the dense GPU path still lacks soft-capping
  - Note: SmolLM2 and Mistral-7B were no longer in `models/` at 17:27 and are on no `_models` drive; re-pull before A/B work that needs them
