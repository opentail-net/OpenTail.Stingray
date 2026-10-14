# Architecture descriptors, semantics and golden parity: open follow-ups

Three 2026-10-08/09 plans are finished and archived verbatim in `docs/done/`; this file keeps only what they left open.

| Archived plan | Result |
|---|---|
| [semantics admission](../done/2026-10-08-architecture-semantics-admission-plan.md) | Phases 1-8 done: one resolver, `FromGgufMetadata` removed, equivalence gate zero diff over 172 real headers |
| [descriptors as plugins (capabilities)](../done/2026-10-08-architecture-capabilities-plan.md) | Phases 0-5 done: capabilities/integrations/traits records, literal ratchet, AOT publish checked |
| [golden parity and admission tooling](../done/2026-10-08-golden-parity-and-admission-tooling-plan.md) | Phases 1-7 done: `ModelLocator`, golden runner, `capture-golden`, hash pins, `verify-goldens`, bulk migration |

## Open items

### Verification gaps (from semantics plan phase 9, 2026-10-08 record; re-check against the later golden migrations before acting)
1. **DeepSeek2 greedy parity fails 1/2** (token 9 divergence after the Q3_K kernel rewrite `e7b7aa8a`; bugstofix #24; later characterised as a near-tie and receipted in `DeepSeek2GreedyParityTests`, see `docs/00-current-work.md` "Closed 2026-10-09 #24"). Confirm nothing remains.
2. **Llama 4 heavy parity deferred by the user** (about 44 GB mapped against about 23 GB free RAM at the time; not a result). Related: the `LlamaFour` legacy parity class was deliberately kept (golden deferred).
3. **Heavy suites never run:** Cuda, Vulkan, Sessions and Server heavy projects, and the other heavy ForwardPass classes outside the migrated architectures.
4. **Full heavy `GoldenParityTests` run on an idle machine** (the memory-pressure killer stopped the 2026-10-09 attempt; batch B7/B8 rest on per-model `admit-arch --golden` runs).
5. **UNVERIFIED on 2026-10-08 (no checkpoint or reference on disk, tests skipped in about 0.1 s):** Apertus, Maincoder, Falcon, Olmo, Olmo2, GraniteMoe, Gpt2, Glm4, Granite, 3 of 4 `Gemma4Cpu` tests. Several have since been pulled and migrated to goldens (batches B7, B8); Apertus is still blocked (below).
6. **Legacy parity classes deliberately kept after golden migration:** `PhiMoeGreedyParityTests` (LongRoPE factor selection depends on context size, 4096 vs 8192; the golden runner uses one context), `ApertusGreedyParityTests` (llama-server 10306 cannot load the Apertus GGUF, chat-template parse error; needs a `llama-completion` capture path), `OlmoeGreedyParityTests` (also holds the top-candidates-at-divergence diagnostic and a stricter stepwise bound of 1.0), `GraniteHybrid`, `DeepSeek2` (Q8 prefill setting plus bug-specific assertions; decision D4: keep until `engineSettings` covers it), `Rwkv6/7` (recurrent state checks), `LlamaFour`.

### Remaining architecture-string leaks (semantics plan "Exercise results", capabilities plan Phase 3)
7. **Small Gemma 4 leaks:** `ModelDescription.cs` `IsGemma4` raw-arch fallback (capabilities plan 1.2 removed some; re-grep), `StaticPlanCommand.cs`, `LlamaCompatEndpoints.cs` display mapping, `InferenceEngineLoader.cs`, `GgufModel.cs` / `GgufTokenizer.cs` Gemma 4 pre-type handling.
8. **Core structural tables still keyed on arch strings:** hybrid layer topology (GDN / Mamba-2 / short-conv), RWKV head size, the NEOX table for descriptor-less archs, and the by-name shim `ModelArchitectureTraits.Legacy` (12 literals; `qwen3vlmoe` lives only there). A new hybrid-recurrent family can still need a Core edit until the shim is deleted.
9. **Separate registries kept out of scope, edited per family:** tokenizer family switches (`GgufTokenizer.cs`, `PreTokenizerPatterns.cs`), `ChatProtocolRegistry.cs`, `ToolCallAdapter.cs` (deliberately not moved), `PrefillHandoffFamilies.cs`, vision projector dispatch (`UnifiedVisionPipeline.cs`).
10. **`FallbackChat` vs `ChatProtocolId`** disagree for six architectures by design (CLI/server no-template fallback vs engine API); retire or keep as derived convenience is decision D3, not taken.
11. **Duplicated admitted-set snapshot** in `ArchitectureDescriptorContractTests` and `ArchitectureRegistryTests`: remove one.

### Small and optional
12. Capabilities plan 0.3 (deferred, low value): test that every key in `PrefillHandoffFamilies` names a registered architecture.
13. Golden plan 1.b: point `Tests.Cuda` and other `FindModelPath` copies at `ModelLocator`.
14. Golden plan 4.3 untested paths: busy-port race and Ctrl-C during `capture-golden` (not automatable here).
15. Golden plan decisions D1-D3 were recommendations (goldens under `tests/.../Goldens/`, near-tie tolerance 0.02 logit to revisit with data from the migrated receipts, SHA-256 mandatory for new goldens); confirm they match what shipped before closing.

### Out of scope (unchanged)
Dynamic assembly discovery, moving execution policy into descriptors, changing admission policy (CLAUDE.md rule 14), tokenizer `pre` patterns, audio/diffusion/vision verification, GPU golden runs (CPU first), automating STATUS rows.
