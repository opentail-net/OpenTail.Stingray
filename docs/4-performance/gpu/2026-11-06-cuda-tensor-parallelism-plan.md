# CUDA tensor parallelism (flag-gated, unmeasured) — plan

**Date:** 2026-11-06. **Status:** proposed, not started. **Owner decision on record:** port this even
though it cannot be measured on this machine (no NVIDIA hardware here at all — not even single-GPU
CUDA, per the standing "CUDA and discrete-GPU work: no NVIDIA hardware here" note in
`docs/00-current-work.md`). This plan explicitly overrides that standing position for this one feature,
by the owner's direct instruction ("port for A40/A100/RTX PRO 6000 clusters, even unmeasured. Just gate
by a flag"). Nothing else in that standing position changes: this is not a reason to start measuring or
claiming other CUDA work works.

**Source:** `C:\Git-Public\example\TensorSharp\TENSOR_PARALLELISM_PLAN.md` and `FEATURES.md`'s
"Tensor Parallelism & Distributed Inference" section (BSD 3-Clause, zhongkaifu/TensorSharp — same
licence family already in `THIRD_PARTY_NOTICES.md`'s TensorSharp section). That source describes three
stages: **Stage 1** local (single-node) tensor parallelism on its direct `cuda` backend, **Stage 1b/1c**
the same on its GGML CUDA/Vulkan backends plus fused-execution kernels, **Stage 2** multi-node TCP-mesh
distributed TP, **Stage 3** RDMA. Measured gains throughout (e.g. "Gemma 4 E4B `--tp 2` 1.39x on 2x RTX
2000 Ada") are TensorSharp's own numbers on TensorSharp's own backend and code shape; they are not a
claim about Stingray and are not repeated as targets here — rule 13 (an iGPU loss is not evidence
against a GPU path) and the matching converse both apply: a number from someone else's hardware and
code is not evidence for this port either. Treat the source's architecture-specific sharding
strategies (what gets column/row-split vs replicated, per hybrid/MoE family) as the valuable transferable
content; treat its tok/s tables as context only.

## Scope

**In scope (this plan):** TensorSharp Stage-1-equivalent only — local, single-process, single-node
tensor parallelism on Stingray's own direct CUDA backend (`src/OpenTail.Stingray.Cuda`, cuBLAS-based,
not GGML). Weight sharding (column-parallel QKV/gate-up, row-parallel output/down + AllReduce),
P2P/host-fallback collectives, and per-family sharding strategy (dense, MoE expert slicing, hybrid-GDN
block-cyclic head assignment, Mamba2 replicate-and-broadcast) for whichever architectures Stingray's
CUDA forward pass already covers (`CudaForwardPass.cs`, `CudaHybridForwardPass.cs`).

**Out of scope (future plans, not started, not promised):**
- GGML-backend TP — does not apply; Stingray's CUDA backend is direct cuBLAS, not a GGML wrapper, so
  TensorSharp's Stage 1b/1c (which is specifically about its GGML backend's per-op dispatch and fused
  graph capture) has no equivalent surface here. Stage 1's cuBLAS-based design is the relevant one.
- Stage 2 (multi-node TCP mesh) and Stage 3 (RDMA): distributed, cross-machine. Not attempted until
  Stage-1-equivalent exists and the owner asks for it explicitly; a second plan.
- Fused/CUDA-graph-captured TP execution, continuous batching under TP: TensorSharp's Stage 1c. Only
  per-sequence forward under TP is in scope here, matching what TensorSharp itself falls back to for
  every family except its own Mistral3 (see its FEATURES.md: "Batched/continuous-batching forward
  under TP is implemented for Mistral 3; the other families ... fall back to per-sequence forward").
- Lifting `ModelParams.MainGpu`/`TensorSplit`'s existing refusal for any case OTHER than the new,
  explicitly-opted-into TP path (see Contracts below) — single-GPU behaviour must not change.

## Contracts this must satisfy

- **`Model.Load` (`src/OpenTail.Stingray.Engine/Api/Model.cs:89-92`):** currently throws
  `NotSupportedException` unconditionally when `MainGpu != 0` or `TensorSplit.Count > 1`. This plan adds
  a new, separate opt-in (a `TensorParallelDegree` on `ModelParams` or a dedicated parameter, name TBD
  in the detailed design — not reusing `MainGpu`/`TensorSplit`, which are llama.cpp-shaped concepts that
  do not match this design) rather than changing what those two existing fields mean. Existing
  single-GPU callers see no behaviour change.
- **`IComputeBackend` / `CudaBackend` (`src/OpenTail.Stingray.Cuda/CudaBackend.cs`):** today implicitly
  single-device (no device-index parameter anywhere in the class — confirmed by grep, not assumed).
  Every call site that constructs a `CudaBackend` assumes one GPU. TP needs N `CudaBackend` instances
  (one per rank) plus a coordinator; nothing in the current contract supports that, so this is additive
  (a new `TensorParallelGroup`-equivalent type owning N backends), not a change to `CudaBackend`'s own
  contract.
- **`ArchitectureDescriptor` / the generic `CudaForwardPass`:** Stingray does not have TensorSharp's
  one-class-per-architecture structure (`Mistral3Model.cs`, `Gemma4Model.cs`, ...); it has one generic,
  descriptor-driven forward pass. The port should follow Stingray's own shape — TP column/row-parallel
  linear ops and AllReduce points added generically to `CudaForwardPass`/`CudaHybridForwardPass`, keyed
  off the descriptor's existing structural fields (head counts, FFN width, hybrid kind, MoE expert
  count) — rather than copying TensorSharp's per-model duplication. This is the single biggest design
  deviation from the source and needs its own short design note before coding starts (not written yet).
- **`ForwardPassSelection` / the CPU-vs-GPU-vs-hybrid planner:** needs a new branch for "TP requested,
  architecture's sharding strategy implemented" vs "TP requested, not implemented for this family" (the
  latter refuses at load with a clear message, per the existing pattern for e.g. Gemma 2's CPU-only
  refusal — `Model.cs` and the planner already have this shape for other reasons).

## Flag design

A single, explicit opt-in, off by default, loudly labeled as unverified wherever it surfaces:
- A new `ModelParams`/`ContextParams` field (public API) plus a CLI `--tp <N>` option and
  `STINGRAY_TP_DEGREE` env var (naming to match this project's existing `STINGRAY_*` convention, not
  TensorSharp's `TENSORSHARP_*`).
- `N = 1` (the default) is exactly today's single-GPU path, byte-for-byte — no new code on that path.
- `N > 1` requires `--backend cuda`/CUDA to be selected explicitly; it is refused (clear message, not a
  silent fallback) for every other backend and for any architecture without a ported sharding strategy.
- Every log line, STATUS row, and doc mention of this feature says **"compiles, never run on real
  hardware"** until an owner with access to a CUDA host says otherwise — the same labeling discipline
  CLAUDE.md rule 14 already uses for "ported, not verified" model families, applied here to a whole
  feature rather than one checkpoint.

## Order (first slice, before any other architecture)

Port one architecture end-to-end first to validate the generic-dispatch design against real code,
rather than writing the sharding abstraction and five forward passes blind in one pass:
1. **Dense `llama`-architecture** (plain QKV + gate/up/down FFN, no MoE, no hybrid state) — the
   simplest case, and Stingray's CUDA backend already has a plain dense forward path to extend.
2. Then MoE expert slicing (one admitted CUDA MoE family) and hybrid-GDN block-cyclic head assignment,
   each as its own follow-up slice once the dense case's generic dispatch shape is proven.

## What "done" means here (deliberately not the usual bar)

Because there is no hardware to run this on, the project's usual "real weights, real reference, real
numbers" admission bar (CLAUDE.md rules 8/12/14) does not apply to this feature. Instead:
- Compiles clean (warnings as errors, as everywhere else).
- `N=1` path is provably unchanged (existing CUDA tests, which already don't run for lack of hardware,
  still assert the single-GPU shape; no behavioural diff).
- `N>1` path is reviewed for structural correctness against TensorSharp's documented sharding strategy
  (dimension splits, AllReduce placement) and against the math of Megatron-style TP, but is explicitly
  **not** claimed to produce correct output, because that cannot be checked here.
- A STATUS.md row (or a dedicated note) says exactly this: implemented, flag-gated, compiles, never run.

## Risks, stated plainly

- Silent correctness bugs cannot be caught before a real multi-GPU run. This is accepted by the owner,
  not a reason to lower the labeling discipline above.
- The existing single-GPU CUDA backend itself has never been run on real hardware either (`docs/00-
  current-work.md`, "CUDA variant written, never run"). TP is being built on an unverified foundation;
  a bug surfacing on real hardware could be in either layer. Not a reason to stop — just means a first
  real run, whenever it happens, should test `N=1` (the existing single-GPU path) before `N>1`.
