# 2026-10-02 prefill numerics investigation: experiment code

Reproduction kit for [the investigation record](../../../1-correctness/2026-10-02-prefill-numerics-investigation.md)
behind [ADR-0002](../../../reference/adr-0002-hybrid-gdn-mtp-chunked-prefill.md) and
[ADR-0003](../../../reference/adr-0003-cpu-int8-prefill-default.md). The method itself is written
up in [numerics-investigation-method.md](../../../reference/numerics-investigation-method.md).

Not production code. The engine switches are a patch, so they never ship. The harnesses are
throwaway console apps, kept so every number in the record can be regenerated.

## Engine switches: `hybrid-gdn-experiment-switches.patch`

Apply to `src/OpenTail.Stingray.Engine/HybridGdnForwardPass.cs` with `git apply`. Written
against the tree of 2026-10-02 (before the MTP gate change), so expect fuzz later. Static
switches on `HybridGdnForwardPass`, all off by default:

| Switch | Effect inside `PrefillChunked` |
|---|---|
| `XSeqRec` | GDN recurrence via per-token `GdnRecurrenceDecode` instead of `GdnRecurrenceChunkedPrefill` |
| `XSeqProj` | projections via per-token `FusedMatVec` instead of `BatchedProjection` (`ProjX` wrapper) |
| `XSeqFfn` | dense FFN via per-token `DenseFfnAt` instead of `DenseFfnChunked` |
| `XDblRec` | GDN recurrence in FP64 with FP64 state per layer (`XDoubleRecurrence`, from `GdnReferenceSweepTests.Reference`) |
| `XAllowMtp` | bypass the `!_hasMtp` prefill gate (lets the harnesses run MTP checkpoints) |
| `XAllLogits` | fill logits for every prompt position (output norm + lm_head per row) |
| `XRecProbe` | with `XDblRec`: per GDN layer, also run FP32 sequential and FP32 chunked on identical inputs/state and record local output/state error |
| `XLayerHidden` | copy the hidden state after every layer |
| `XCheckRow`, `XRowCheck` | at one row, batched GEMM and the per-token decode matvec against an FP64 dot of the dequantized weights, per projection call |

Turning all of `XSeqRec`, `XSeqProj` and `XSeqFfn` on must reproduce the per-token path bit for
bit. That is the harness sanity check (config S).

## Harnesses (`harness/`)

Each is a net10.0 console app referencing `src/` by absolute path (`C:/Git-Public/OpenTail.Stingray`);
adjust the `ProjectReference`s if the repo lives elsewhere. Build with `dotnet build -c Release`
after applying the patch. Run **one at a time** (memory; see the timing-under-contention rule).

| Harness | Args | Produces |
|---|---|---|
| `attrib` | `model.gguf prompts.tsv out.csv [allowMtp]` | Per-position logits for A (per-token), D, S, R, P, F, E; CSV row per (comparison, prompt, position): top-1s, margin, δ, tight certificate terms, cosine, KL, top-5 overlap; first-flip top-5 dumps; prompt-level summary |
| `analyze.cs` | `dotnet run analyze.cs -- out.csv` (file-based app) | A/R disagreements tie-broken by D, paired distance-to-D log-ratios with bootstrap CIs, certificate table |
| `probe` | `model.gguf prompts.tsv` | Local recurrence error per GDN layer, per-layer hidden divergence, prefill->decode continuity (16 teacher-forced steps) |
| `rowchk` | `model.gguf text-file nTokens row...` | Per projection: batched GEMM vs int8 matvec vs FP64 dot at the given rows |
| `q8chk` | `model.gguf` | Repeated-token prompts: chunked F32 vs chunked int8 vs per-token |
| `tokctx` | `model.gguf "file\|pos\|tokA\|tokB"...` | Decoded context around a position |
| `ggmlx` | `model.gguf <tools/llama.cpp dir>` | Calls the vendored ggml directly (`ggml_get_type_traits`/`_cpu`: Q8_K quantizer, `vec_dot`, `to_float`): activation-quantization and projection error, ggml Q8_K vs Stingray Q8_KS, against an FP64 dot; plus `Q1_0`/`Q2_0` dequant vs ggml bit for bit. Needs no engine patch. Q8_K has no `to_float` in ggml's traits (an intermediate-only type), so the harness dequantizes it itself |

`prompts.tsv`: `class<TAB>maxTokens<TAB>chat|raw<TAB>text-or-@file`.
