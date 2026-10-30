# Completed items split from 2026-10-03-minimax-h3-port-plan.md (2026-10-30)

Finished items moved verbatim out of [../2-coverage/2026-10-03-minimax-h3-port-plan.md](../2-coverage/2026-10-03-minimax-h3-port-plan.md). Open items stay there. Each block notes the section it came from.

> from: ## Implementation Phases
- [x] **6. Experimental Gate:**
  - [x] Unadvertised in CLI/STATUS until real checkpoint verified (CLAUDE.md Rule 14).

> from: ## Verification Ladder
- [x] **Exact Shape & Layout Validation:** synthetic tests for `MiniMaxH3Layout`, coordinate bounds, and channel-major patch-minor ordering.

> from: ## Verification Ladder
- [x] **Synthetic Component Tests:**
   - [x] Check dual flow schedule values for video ($shift=12$) and audio ($shift=3$).
   - [x] Check AdaLN curve table interpolation and modulation projections ($8 \to 96768$).
   - [x] Test an isolated single DiT block forward pass with packed video+audio tokens and full 5376d/7168d/96768d dimensions.
   - [x] Test Audio VAE decode (32-channel to 32 kHz stereo).
   - [x] Test Video VAE 3D temporal chunking (<=22 frames vs >22 frames with seam cross-fading).
   - [x] Test multimodal RoPE 3D spatio-temporal axis decomposition and norm preservation.
   - [x] Test external `vocab.json` + `merges.txt` BPE tokenizer.

> from: ## Verification Ladder
- [x] **Synthetic End-to-End Tiny H3:** test a video + audio packed denoising step with pipeline Euler integration.

