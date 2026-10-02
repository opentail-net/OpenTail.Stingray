# DeepSeek V4 / V4.1 review plan (`deepseek4`, `deepseek41`)

**Status:** Stingray has V4 **alpha code, never run** ([058](058-deepseek-full-lineage-implementation-plan.md)).
This plan is a review against two references plus V4.1 deltas, not a new port. **Policy:** stays
not admitted, not advertised (CLAUDE.md rule 14).

## References

- **llama.cpp `src/models/deepseek4.cpp`**: present locally (source pulled to `bed0a8566` on
  2026-10-03); the vendored b10306 binaries also know `deepseek4`. Upstream has no `deepseek41`
  graph, so llama.cpp does **not** independently validate V4.1.
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

## Work (phases): V4 and V4.1 are separate architectures

- [ ] **A. Review the existing `deepseek4` alpha:** read our `DeepSeek4Alpha.cs` /
  `DeepSeek4ForwardPass.cs` / `DeepSeek4CompressedState.cs` / `DeepSeek4TensorSet.cs` side by side
  with `deepseek4.cpp` (local source, now at `bed0a8566`) and TensorSharp's `DeepSeek4CpuExecutor`.
  Cover the Sinkhorn mHC, HCA/CSA compression (ratios 0 / 4 / 128), the indexer, the sqrt-softplus
  gate, and the rope-slice handling flagged as unusual in plan 058. Record each discrepancy and fix
  with a citation.
- [ ] **B. Prove V4 mechanics synthetically:** build a tiny `deepseek4` GGUF and run it through
  **both** our alpha pass and the vendored llama.cpp b10306, which knows `deepseek4`: a level-3
  independent check without the real checkpoint.
- [ ] **C. Separate `deepseek41` architecture review:** read TensorSharp's `DeepSeek41Model.cs`,
  `DeepSeek4CpuExecutor.V41.cs` and `Dsv41EngramData.cs`, plus its 103-case validation protocol.
  Write the V4.1 spec as its own architecture, not as "V4 deltas". No llama.cpp implementation
  exists, so TensorSharp (held to a PyTorch oracle) is the only implementation reference.
- [ ] **D. Only then add `deepseek41` code,** with specification tests (level 2). Stop there unless
  a PyTorch/HF fixture can be produced (level 3).
- [ ] **E. Docs:** plan 058's status note (done 2026-10-03) and the ported-families table.

## Deferred

V4.1 vision, DSpark/MTP speculative decoding, GPU paths, and tensor parallelism.

## Verification

Phase B is the strongest check available locally for V4. For V4.1, only specification tests are
possible here. Full admission of either needs a large-memory host (`stingray admit-arch` against
`llama-server` for V4).

**Effort:** A + B about half a day; C + D about a day. Real-weight verification (large host),
closeout and admission are separate.
