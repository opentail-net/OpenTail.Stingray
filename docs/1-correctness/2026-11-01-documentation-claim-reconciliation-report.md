# Documentation claim reconciliation: report, 2026-11-01

Output of the evidence pass in [the reconciliation plan](2026-10-09-documentation-claim-reconciliation-plan.md). Checkout: `main` at `1419fa70`
plus a documentation-only branch; machine: Ryzen 7 5700G (Zen 3, AVX2 and FMA, no AVX-512), Windows 11, .NET 10. Each item below states what was
checked, the finding at the strongest level actually tested, and what changed in the documents. **R2 (HunyuanVideo and Wan 2.2) was not re-tested in
this pass and stays open.**

| Item | Finding | Level of evidence | Document change |
|---|---|---|---|
| R1 | The sample compiles against the packed package | compiled in clean consumer projects | none needed |
| R2 | not re-tested | none new | none |
| R3 | AVX-512 is implemented, never executed on AVX-512 hardware | source inspection, hardware inventory | CPU row of the hardware guide and a note in the SIMD guide |
| R4 | The registry is `KnownEnvironmentVariables.cs`; `StingrayEnv.cs` does not exist | source and test inspection, drift tests run | filename corrected in the documentation architecture plan |
| R5 | TurboQuant has two quantizers (Lloyd-Max, KVarN); there is no FP8 or INT4 KV mode | source and CLI inspection | guide description corrected in the documentation architecture plan |

## R1. `ModelHome.ResolveModelPath` in the public sample

- **API found.** `public static string ModelHome.ResolveModelPath(string modelIdOrName)` in `src/OpenTail.Stingray.Core/Catalog/ModelHome.cs`
  (namespace `OpenTail.Stingray.Core.Catalog`), with a non-throwing `TryResolveModelPath`. The sample in the root README and in
  `src/OpenTail.Stingray/README.md` calls exactly this.
- **Compile result.** `scripts/verify-package-consumers.ps1` packed the checkout (version 1.0.9 from `Directory.Build.props`) into a throw-away
  local feed, wrote every README block marked `<!-- verify:consumer -->` into its own console project outside the repository with a
  `PackageReference` only, and built each one. **17 checks, 0 failed**, including the block that calls `ModelHome.ResolveModelPath("qwen2.5-0.5b")`
  and the five samples that claim to be package consumers (Chat, Extract, Speech, Diffusion, ChatServer), which also resolved no project references.
- **Scope.** This proves the code compiles against the public package. It does not run inference (that needs model files).
- **Cause of the original doubt.** A stale source snapshot at review time; nothing needed repair.

## R2. HunyuanVideo and Wan 2.2

Not re-tested. The question needs real checkpoints, a fixed prompt and seed, and an independent reference for HunyuanVideo v1; the existing
[00-current-work.md](../00-current-work.md) items 13 and 14 record what blocks it (no independent v1 output). The box stays open.

## R3. AVX-512: implemented, dispatched, executed, measured

| Question | Answer | Evidence |
|---|---|---|
| Implemented? | Yes | `CpuSgemm.cs` has an explicit AVX-512 kernel (`KernelKind.Avx512`, 8x32 blocking, `Avx512F.FusedMultiplyAdd`); `SimdKernels.cs` uses `Vector512` (124 references) |
| Dispatch guarded? | Yes | `CpuSgemm.HasAvx512` requires `Avx512F`, `Avx512BW`, `Avx512DQ` and `Vector512.IsHardwareAccelerated`; the planner reads `HasAvx512` |
| Unit-tested? | The planner side only | `ExecutionPlannerTests` and `SmartOffloadPlannerTests` set `HasAvx512` true or false as an input; no test executes the 512-bit kernel |
| Executed on AVX-512 hardware? | **No evidence** | The development machine is a Ryzen 7 5700G (Zen 3). The Windows CI runner, as the 2026-11-01 hardware step recorded it, is an AMD EPYC 7763 (also Zen 3); the Linux runner's CPU was not read in this pass. Neither machine that was checked has AVX-512. |
| Performance measured? | No | no AVX-512 receipt exists |

So "AVX2/AVX-512 SIMD kernels" is true as a statement about the code and misleading as a statement about verification. The statements in the README, `STATUS.md`
and `RUNNING.md` describe the implementation and are left as they are. The hardware guide's CPU row called the CPU backend a "verified baseline" for x64
(AVX-512, AVX2) and ARM64; it now says AVX2 is verified, AVX-512 is implemented and never run on AVX-512 hardware, and ARM64 is implemented and not run. The SIMD
guide gets a short note saying the same. The external-hardware queue already lists the ARM64 work; an AVX-512 receipt needs a machine that reports the ISA, which
no machine here does.

## R4. Environment registry

- **Names.** `StingrayEnv.cs` does not exist anywhere in the tree. The authoritative list is `src/OpenTail.Stingray.Core/KnownEnvironmentVariables.cs`
  (`KnownEnvironmentVariables.All`, 287 names per the inventory header).
- **What the tests enforce.** `KnownEnvironmentVariablesTests.ListMatchesSource` keeps the registry in step with the names read in the source;
  `KnownEnvironmentVariablesInventoryTests.InventoryTable_CoversTheRegistry_ExceptTheListedGap` fails if a registered variable has no row in
  `docs/reference/env-var-inventory.md` or a row names something the registry does not contain. Both are in `Tests.Core`, which passed on 2026-11-01 (1170 tests, 0 failed).
- **Limit of the checks.** They compare names. Descriptions, defaults and classifications in the inventory are written by hand from each read site and are not mechanically checked.
- **Change.** The documentation architecture plan named `StingrayEnv.cs`; it now names `KnownEnvironmentVariables.cs`.

## R5. TurboQuant KV-cache modes

| Mode | Activation | Key / value representation | Notes |
|---|---|---|---|
| Lloyd-Max (`TqQuantizer.LloydMax`) | `--tq --tq-mode lloydmax` | 3-bit codebooks in FastScan tiles (3 to 4 bit in the code comment) | the original path; documented as severely degrading quality on QK-norm models such as Qwen3 |
| KVarN (`TqQuantizer.KVarN`) | `--tq --tq-mode kvarn` | 4-bit keys, 2-bit values, 128-token tiles | needs an FP32 window of at least 128 tokens |
| `auto` (default for `--tq`) | `--tq` | KVarN where supported, otherwise Lloyd-Max with a quality warning | |

- **No FP8 or INT4 KV mode exists.** A search of the KV cache code and the TurboQuant project for FP8, E4M3 and float8 found nothing, and the guide
  `docs/guides/turboquant-kv-cache.md` does not make that claim. The claim was in the documentation architecture plan's description of the guide only.
- **Separate mechanisms.** The BF16 and F32 KV stores (`STINGRAY_KV_STORE`, `STINGRAY_KV_DTYPE`, BF16 from 1024 tokens under `auto`), paged allocation and prefix reuse are not
  quantization modes of TurboQuant and are described apart from it now.
- **Not done in this pass.** The depth of the TurboQuant tests (round-trip error, quality on a real model) was not audited; only the mode matrix was traced.
- **Change.** The architecture plan's description of the guide now follows this matrix.
