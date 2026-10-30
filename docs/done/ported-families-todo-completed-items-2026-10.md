# Completed items split from ported-families-todo.md (2026-10-30)

Finished items moved verbatim out of [../2-coverage/ported-families-todo.md](../2-coverage/ported-families-todo.md). Open items stay there. Each block notes the section it came from.

> from: ### 1. Qwen 3.8 Flash Next (`qwen4exp`)
- [x] Port

> from: ### 1. Qwen 3.8 Flash Next (`qwen4exp`)
- [x] Not-admitted block

> from: ### 1. Qwen 3.8 Flash Next (`qwen4exp`)
- [x] Table row

> from: ### 1. Qwen 3.8 Flash Next (`qwen4exp`)
- [x] Specification & synthetic tests passing (24 tests in `Qwen4ExpAlphaTests`, `Qwen4ExpRoutedMoeTests`, `Qwen4ExpMoeRoutingTests`, `Qwen4ExpGuardTests`)

> from: ### 2. GLM-5.x (`glm-dsa`, alias `glm_dsa`; GLM-5.3-Flash is `glm5next`)
- [x] Port `glm-dsa`

> from: ### 2. GLM-5.x (`glm-dsa`, alias `glm_dsa`; GLM-5.3-Flash is `glm5next`)
- [x] Port `glm5next`

> from: ### 2. GLM-5.x (`glm-dsa`, alias `glm_dsa`; GLM-5.3-Flash is `glm5next`)
- [x] Add not-admitted blocks

> from: ### 2. GLM-5.x (`glm-dsa`, alias `glm_dsa`; GLM-5.3-Flash is `glm5next`)
- [x] Add table rows

> from: ### 2. GLM-5.x (`glm-dsa`, alias `glm_dsa`; GLM-5.3-Flash is `glm5next`)
- [x] Specification tests passing (`GlmDsaSyntheticTests`, `Glm5NextSyntheticTests`)

> from: ### 3. Muse-Glimmer (`muse-glimmer`, alias `muse_glimmer`)
- [x] Port text tower

> from: ### 3. Muse-Glimmer (`muse-glimmer`, alias `muse_glimmer`)
- [x] Add not-admitted block

> from: ### 3. Muse-Glimmer (`muse-glimmer`, alias `muse_glimmer`)
- [x] Add table row

> from: ### 3. Muse-Glimmer (`muse-glimmer`, alias `muse_glimmer`)
- [x] Specification test (level 2) passing (`MuseGlimmerSyntheticTests`)

> from: ### 4. DiffusionGemma (`diffusion-gemma`, alias `diffusion_gemma`)
- [x] Port (`src/OpenTail.Stingray.Diffusion/DiffusionGemma/*.cs`: Gemma-4 MoE backbone, multi-block causal prefill, persistent KV, bidirectional canvas, self-conditioning MLP, EntropyBound sampler, and strict tensor loader)

> from: ### 4. DiffusionGemma (`diffusion-gemma`, alias `diffusion_gemma`)
- [x] Add not-admitted block in `ModelCompatibility.cs` (real checkpoint guarded by `DiffusionGemmaRealCheckpointGuardTests` per Rule 14)

> from: ### 4. DiffusionGemma (`diffusion-gemma`, alias `diffusion_gemma`)
- [x] Add table row

> from: ### 4. DiffusionGemma (`diffusion-gemma`, alias `diffusion_gemma`)
- [x] Specification & synthetic tests passing (32 tests in `DiffusionGemmaTests`)

> from: ### 5. MiniMax-H3 (video + native 32 kHz stereo audio)
- [x] Port (`src/OpenTail.Stingray.Diffusion/MiniMaxH3/*.cs`)

> from: ### 5. MiniMax-H3 (video + native 32 kHz stereo audio)
- [x] Add a not-advertised entry; diffusion has no `ModelCompatibility` allowlist, so keep it off
      the CLI's model list until verified

> from: ### 5. MiniMax-H3 (video + native 32 kHz stereo audio)
- [x] Add table row

> from: ### 5. MiniMax-H3 (video + native 32 kHz stereo audio)
- [x] 13 unit / pipeline tests passing (`MiniMaxH3Tests.cs`)

> from: ### 6. DeepSeek V4 / V4.1 Flash (V4 review + V4.1 distinct architecture)
- [x] Review and fix existing V4 alpha (CSA ratio 4, 8-group output LoRA, Hadamard on indexer)

> from: ### 6. DeepSeek V4 / V4.1 Flash (V4 review + V4.1 distinct architecture)
- [x] Implement V4.1 distinct architecture (`deepseek41`): ratios 0/1/2, 384+1 MoE, Engram subsystem (`DeepSeek41*.cs`)

> from: ### 6. DeepSeek V4 / V4.1 Flash (V4 review + V4.1 distinct architecture)
- [x] Add not-admitted blocks and update table rows

> from: ### 6. DeepSeek V4 / V4.1 Flash (V4 review + V4.1 distinct architecture)
- [x] Specification tests passing (`DeepSeek4AlphaTests`, `DeepSeek41AlphaTests`)

