# OpenTail.Stingray.Sample.HotRouting

> **Classification:** source-level contributor example and **demonstration**, not a benchmark. It uses engine internals (`SessionRegistry`, `HotSessionRuntime`, `ContinuousBatchingEngine`, `ForwardPass`) that are not part of the public package contract.
> **Task:** show that several roles (Planner, Coder, Reviewer, Cutter) can share one model while each keeps its own retained state, so a returning role does not re-process its earlier tokens.

## Prerequisites

- The .NET 10 SDK, a 64-bit x86 CPU with AVX2.
- Optional: `models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf`, found by walking up from the current directory (up to eight levels). **Without that exact file the sample silently falls back to a simulation** driven by a fake forward pass and a 64-token toy tokenizer, so check the first `[INFO]` line it prints to see which mode ran.

## Run

```bash
dotnet run --project samples/OpenTail.Stingray.Sample.HotRouting -c Release
```

## Expected output

With the model present, it prints the five steps of one planning round and a diagnostics block. The run behind this README (Ryzen 7 5700G, CPU, 4.3 s in total):

```text
[INFO] Using model file: ...\models\SmolLM2-1.7B-Instruct-Q4_K_M.gguf
--> [Step 1: Planner] ... Materialized: 13 tokens.
... Coder, Reviewer, Planner return, Cutter ...
    Active Roles Co-Hosted               : 4 (Planner, Coder, Reviewer, Cutter)
    Review Rounds Completed              : 1 / 3
    Retained Tokens Prefilled            : 0 (Exact State Reuse Verified)
[SUCCESS] 4-Role Hot multi-model routing demo completed successfully!
```

## Known limitations

- Each role generates only about a dozen tokens, and "convergence" is decided by a fixed rule in the sample, not by judging the model's answers. It demonstrates **state retention and routing**, not output quality or speed; do not cite its timings as inference benchmarks.
- The model search is hard-coded to one file name.
- In simulation mode (no model file) nothing real is run through an engine.

## Verification

- **Run with real weights, 2026-10-11:** SmolLM2 1.7B Q4_K_M, CPU; completed with "Retained Tokens Prefilled : 0". One run. The reuse claim is the sample's own diagnostic; the repository's session tests are the authoritative evidence for state reuse.
- Not a package-consumer sample and not built against the packed package.
