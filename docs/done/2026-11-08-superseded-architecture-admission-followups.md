# Architecture descriptors, semantics and golden parity: open follow-ups
> Ticked items moved to [done/architecture-admission-followups-completed-items-2026-11.md](../done/architecture-admission-followups-completed-items-2026-11.md) on 2026-11-08.

Three 2026-10-08/09 plans are finished. Their completed items are archived as extracts in `docs/done/` (not verbatim copies; the originals stay in `2-coverage/` or `done/`); this file keeps only what they left open.

| Archived plan | Result |
|---|---|
| [semantics admission](2026-10-08-architecture-semantics-admission-plan.md) | Phases 1-8 done: one resolver, `FromGgufMetadata` removed, equivalence gate zero diff over 172 real headers |
| [descriptors as plugins (capabilities)](../done/2026-10-08-architecture-capabilities-plan.md) | Phases 0-5 done: capabilities/integrations/traits records, literal ratchet, AOT publish checked |
| [golden parity and admission tooling](2026-10-08-golden-parity-and-admission-tooling-plan.md) | Phases 1-7 done: `ModelLocator`, golden runner, `capture-golden`, hash pins, `verify-goldens`, bulk migration |

## Open items

Ticked = closed with evidence in the item's own text. Unticked = still open, or a deliberate standing state noted in the item.

### Verification gaps (from semantics plan phase 9, 2026-10-08 record; re-check against the later golden migrations before acting)

- [ ] **1. DeepSeek2 greedy parity fails 1/2** (token 9 divergence after the Q3_K kernel rewrite `e7b7aa8a`; bugstofix #24; later characterised as a near-tie and receipted in `DeepSeek2GreedyParityTests`, see `docs/00-current-work.md` "Closed 2026-10-09 #24"). Still to confirm: nothing remains.
- [ ] **2. Llama 4 heavy parity deferred by the user** (about 44 GB mapped against about 23 GB free RAM at the time; not a result). Related: the `LlamaFour` legacy parity class was deliberately kept (golden deferred).
- [ ] **3. Heavy suites never run:** Cuda, Vulkan, Sessions and Server heavy projects, and the other heavy ForwardPass classes outside the migrated architectures.
- [ ] **5. UNVERIFIED on 2026-10-08 (no checkpoint or reference on disk).** Confirmed 2026-11-06 for Maincoder, Falcon, Olmo, Olmo2, GraniteMoe, Gpt2, Glm4 and Granite (real weights, passing golden). Still open: Apertus (blocked, see item 6) and the 3 `Gemma4Cpu` tests, which were not re-checked (not a golden-migrated class).
- [ ] **6. Legacy parity classes deliberately kept after golden migration:** `PhiMoeGreedyParityTests` (LongRoPE factor selection depends on context size, 4096 vs 8192; the golden runner uses one context), `ApertusGreedyParityTests` (llama-server 10306 cannot load the Apertus GGUF, chat-template parse error; needs a `llama-completion` capture path), `OlmoeGreedyParityTests` (also holds the top-candidates-at-divergence diagnostic and a stricter stepwise bound of 1.0), `GraniteHybrid`, `DeepSeek2` (Q8 prefill setting plus bug-specific assertions; decision D4: keep until `engineSettings` covers it), `Rwkv6/7` (recurrent state checks), `LlamaFour`. Stays open until each class moves to a golden or is formally retired.

### Remaining architecture-string leaks (semantics plan "Exercise results", capabilities plan Phase 3)

- [ ] **8. Core structural tables still keyed on arch strings** (hybrid layer topology, RWKV head size, the NEOX table, the by-name shim `ModelArchitectureTraits.Legacy`): see item 16.
- [ ] **9. Separate registries kept out of scope, edited per family:** tokenizer family switches (`GgufTokenizer.cs`, `PreTokenizerPatterns.cs`), `ChatProtocolRegistry.cs`, `ToolCallAdapter.cs` (deliberately not moved), `PrefillHandoffFamilies.cs`, vision projector dispatch (`UnifiedVisionPipeline.cs`). Standing scope decision, not work to finish.
- [ ] **10. `FallbackChat` vs `ChatProtocolId`** disagree for six architectures by design (CLI/server no-template fallback vs engine API); retire or keep as derived convenience is decision D3, not taken.

### Small and optional

- [ ] **13. Golden plan 1.b** (`FindModelPath` copies -> `ModelLocator`): tracked at item 17.
- [ ] **14. Golden plan 4.3 untested paths:** busy-port race and Ctrl-C during `capture-golden` (not automatable here).

### Large

- [ ] **16. `ModelArchitectureTraits.Legacy` deletion** (ex-item 8): the by-name shim (12 literals; `qwen3vlmoe` lives only there) behind hybrid layer topology (GDN / Mamba-2 / short-conv), RWKV head size and the NEOX table for descriptor-less archs. Deleting it is gated behind migrating its 13 real call sites off the `FromGgufMetadata` bypass (several in not-yet-admitted families and several audio pipelines) and separately admitting `qwen3vlmoe` with its own descriptor, which is new architecture work. `Legacy` is deliberately kept and pinned by a drift test.
- [ ] **17. `FindModelPath` -> `ModelLocator`** (ex-item 13, golden plan 1.b): 208 matches across the test tree, including Vulkan, Cuda, Diffusion and Audio projects, several of which cannot be built or verified on this machine. A bounded but large mechanical migration.

### Out of scope (unchanged)

Dynamic assembly discovery, moving execution policy into descriptors, changing admission policy (CLAUDE.md rule 14), tokenizer `pre` patterns, audio/diffusion/vision verification, GPU golden runs (CPU first), automating STATUS rows.
