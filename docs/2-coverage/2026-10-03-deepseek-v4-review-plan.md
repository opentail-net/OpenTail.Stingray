# DeepSeek V4 / V4.1 review plan (`deepseek4`, `deepseek41`)

**Status:** Stingray has V4 **alpha code, never run** ([058](058-deepseek-full-lineage-implementation-plan.md)).
This plan is a review against two references plus V4.1 deltas, not a new port. **Policy:** stays
not admitted, not advertised (CLAUDE.md rule 14).

## References

- **llama.cpp `src/models/deepseek4.cpp`**: present locally; the vendored b10306 also knows
  `deepseek4`.
- **TensorSharp `Models/DeepSeek4/`**: notably the **pure-C# `DeepSeek4CpuExecutor`**
  (`.cs`, `.Attention.cs`, `.V41.cs`), "held to the PyTorch oracle", plus `DeepSeek41Model.cs`
  and `Dsv41EngramData.cs`. Its V4.1 validation protocol:
  `docs/deepseek41_validation.md` (103 cases).

## V4.1 vs V4 (from TensorSharp's card)

V4.1 has its own GGUF architecture name (`deepseek41`; renaming to `deepseek4` is invalid). Its
attention, **Engram** (embedded metadata validated against tensor dims), residual connections
and chat handling differ. 40 layers, hidden 5120, 64 Q heads x 512, 384 experts top-6 + 1 shared,
expert width 2304, 1M context.

## Checkpoints

V4 Flash (about 340 GB) and V4.1 Flash don't fit this machine. Review and synthetic checks only.

## Work (phases)

- [ ] **1. Diff review:** read our `DeepSeek4Alpha.cs` / `DeepSeek4ForwardPass.cs` /
  `DeepSeek4CompressedState.cs` / `DeepSeek4TensorSet.cs` side by side with `deepseek4.cpp` and
  `DeepSeek4CpuExecutor`. Cover the Sinkhorn mHC, HCA/CSA compression (ratios 0 / 4 / 128), the
  indexer, the sqrt-softplus gate, and the rope-slice handling flagged as unusual in plan 058.
  Record each discrepancy and fix with a citation.
- [ ] **2. Synthetic run:** build a tiny `deepseek4` GGUF and run it through **both** our alpha pass
  and the vendored llama.cpp b10306 (which knows `deepseek4`): an independent mechanical check
  without the real checkpoint.
- [ ] **3. V4.1 deltas:** port the `deepseek41` differences from TensorSharp's `.V41.cs` /
  `DeepSeek41Model.cs` (Engram, residuals, attention). Synthetic test vs a spec-written reference.
- [ ] **4. Docs:** update plan 058's status lines and the ported-families table.

## Verification

Phase 2 is the strongest check available locally. Full admission needs a large-memory host
(`stingray admit-arch` against `llama-server` with the real GGUF).

**Effort:** about half a day for the review + synthetic llama.cpp check; about a day more for V4.1.
