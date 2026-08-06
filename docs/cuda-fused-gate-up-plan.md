# CUDA: fuse `ffn_gate` + `ffn_up` into one GEMM

> **Status: NOT IMPLEMENTED — blocked on hardware, not on design.**
>
> The development machine this was investigated on has no NVIDIA GPU (`nvidia-smi` absent; the only
> adapter is AMD Radeon integrated graphics). The change lives entirely in the CUDA path, so it could
> not be run, and the whole value of the change is a number. Writing unmeasured performance code and
> calling it done is the failure this repo keeps guarding against, so it stopped at a plan.
>
> Everything below is verified against the code, not recalled. Someone with an NVIDIA box can pick it
> up without re-deriving any of it. **Read "Where this will and will not pay" before starting** — the
> honest expected value is much smaller than the idea first sounds, and one of the findings below
> nearly kills the decode case outright.

Written 2026-08-06.

## The idea in one paragraph

`ffn_gate` and `ffn_up` are two GEMMs over the *same* input `x`, producing two `intermDim` outputs that
are immediately combined by `silu_mul`. Concatenating the two weight matrices along the output-row
axis makes them a single GEMM of `2 × intermDim` rows, then `silu_mul` reads the two halves of one
buffer. One weight upload, one launch, one pass over `x`.

## Why it is not "CUTLASS epilogue fusion"

This came out of reviewing NVIDIA CUTLASS as a source of ideas. **Do not confuse the two.** True
epilogue fusion applies the activation in registers before the GEMM writes back, and for SwiGLU it is
effectively unavailable here: cuBLASLt epilogues cover BIAS/RELU/GELU but **not** SwiGLU, which needs
an elementwise product of two *different* GEMM outputs. Fusing it properly means writing a tile GEMM
to replace cuBLAS and then beating cuBLAS at the GEMM to win back the epilogue. That is what CUTLASS
is for, and it is not reachable from NVRTC-compiled kernel strings.

This plan is the cheap, boring 80%: **fuse the two GEMMs, keep `llm_silu_mul` as a separate kernel.**

## What was verified (do not re-derive)

| Fact | Where |
|---|---|
| Gate/up/down uploaded as three separate tensors | `CudaForwardPass.cs:1287-1289` |
| Dense FFN is `MatMul(gate) → MatMul(up) → SiLuMul → MatMul(down)` | `CudaForwardPass.cs:1912-1915`, repeated at `:2409`, `:2653` |
| `GpuMatMul` resolves weight dtype by tensor handle | `CudaForwardPass.cs:5500-5504` — `_weightDTypes.GetValueOrDefault(weights.Handle, DType.Q4_K)` |
| `MatMul` takes an explicit weight dtype | `CudaBackend.cs:2246` |
| `SiLuMul(gate, up)` sizes itself from `gate.ElementCount` | `CudaBackend.cs:3985` |
| `llm_silu_mul` kernel source | `CudaTextKernels.cs:331`; handle cached at `CudaBackend.cs:7438`; launched at `:3992` |
| **`Tensor.Handle` is an opaque ID, not a device pointer** | `CudaBackend.cs:806-808` — `_devPtrs[handle] = (devPtr, allocSize)` |
| Prefill routes to cuBLAS GEMM | `CudaForwardPass.GpuMatMulBatchedCore`, `:3137` |
| A separate shape-dispatch path references `_wGate[0]` | `CudaForwardPass.cs:1511` |

The handle fact is the one that shapes the implementation: you cannot make a "view" tensor by adding
an offset to a handle. But because `_devPtrs` maps `handle → (devPtr, allocSize)`, you *can* register
a second handle pointing at `devPtr + byteOffset` with a smaller size. That is a contained addition,
not a refactor — unlike the Vulkan backend, where `GpuBuffer` carries no offset at all and the same
change would mean reworking its resource model.

## Where this will and will not pay

**Decode: expect nothing, and understand why before spending time.**

The original argument — "`x` is read once instead of twice" — **does not survive contact with the
code.** `x` is `embDim` floats, a few KB, L2-resident. The weight matrices are `intermDim × embDim`
quantized, i.e. megabytes. The fused GEMM streams *exactly the same weight bytes*. Decode FFN is
bandwidth-bound on those reads, so fusing changes the dominant term by zero.

That leaves launch overhead — and **decode is CUDA-graph captured** (issue #136, default ON,
`CudaForwardPass.cs:1550-1566`): capture "collapses ~1k host launches/token into one `cuGraphLaunch`".
So the two FFN matmuls per layer are already not two host launches; they are two graph nodes. Fusing
removes graph nodes, not launches. **The decode case is close to dead on arrival.**

**Prefill: this is the real target.**

`GpuMatMulBatchedCore` routes the trunk matmuls to cuBLAS GEMM. Two GEMMs of `N = intermDim` become
one of `N = 2·intermDim`: better tile/wave occupancy, one launch, `x` read once from L2 — and here the
activations genuinely are `nTokens × intermDim`, so the traffic argument that fails at decode has some
substance. cuBLAS already handles both shapes competently, so expect *modest*, not transformative.

**If you only do one thing, do the prefill path and leave decode alone.**

## Implementation

### 1. Fused upload, with a dtype guard

Add alongside `UploadWeight`:

```csharp
// Returns null when the pair cannot be fused; caller falls back to separate weights.
private Tensor? TryUploadFusedGateUp(int layer)
```

- Look up `blk.{layer}.ffn_gate.weight` and `blk.{layer}.ffn_up.weight`.
- **Refuse unless `info.DType` matches on both.** Mixed-quant GGUFs usually give gate and up the same
  type, but it is not guaranteed, and silently concatenating two dtypes corrupts the second half with
  no error. This is the single most important guard in the change.
- Refuse unless both have identical `cols` (`embDim`).
- Concatenate the raw bytes in row order — gate rows first, then up rows — and upload once.
- Register `_weightDTypes[result.Handle] = dtype`.

**Why row-order concatenation is valid:** GGUF block-quantized formats quantize per row, with blocks
running along K. Each row is self-contained, so appending up's rows after gate's rows produces a
well-formed `2·intermDim × embDim` matrix. No requantization, no repacking. Assert this with a test
rather than trusting it.

### 2. Output buffer and views

Allocate `_ffnGateUp` at `2 × intermDim`. The matmul needs no change: `MatMul` derives `rows` from
`output.ElementCount`, so a `2·intermDim` output over the fused weight does the right thing.

For `SiLuMul`, add a backend method that registers a view handle over an existing allocation:

```csharp
// Registers a handle aliasing [byteOffset, byteOffset+size) of an existing tensor's allocation.
// The view must never be freed independently — it does not own the memory.
internal Tensor RegisterView(Tensor parent, long elementOffset, TensorShape shape, DType dtype);
```

Then `SiLuMul(gateView, upView)` works unmodified, since it only reads `gate.ElementCount`.

**Ownership is the trap here.** `_devPtrs` is also consulted by the free path and the pool
(`_pool.Rent`, `_exactHandles`). A view handle entering the free path would return a pointer the pool
does not own, at the wrong size. Track views in a separate set and make `Free` on a view either a
no-op or an assertion failure — decide which, and test it.

*Alternative if `RegisterView` proves invasive:* add an `upOffset` push-constant to `llm_silu_mul` and
bind the same buffer twice. NVRTC compiles kernel source at runtime, so unlike the Vulkan path there
is **no shader-toolchain blocker** — this is a live option, and possibly the smaller change.

### 3. Call sites

Replace the dense-FFN triple at `:1912-1915` (and `:2409`, `:2653`) with the fused form when
`_wGateUp[layer]` is non-null, keeping the existing three-call path as the fallback. Do the same in
`GpuMatMulBatched` for prefill — **this is the one that matters**.

### 4. Excluded

- **MoE.** `_wGateExps` has its own expert-batched layout; this touches the dense branch only.
- **`CudaForwardPass.cs:1511`**, a shape-dispatch path that indexes `_wGate[0]` directly. It must keep
  working — either leave the unfused weights uploaded alongside (costing VRAM) or update that path.
  Decide deliberately; do not discover it at runtime.

## How to measure it

The standing rule in this repo is that a performance claim needs a number that survives noise. On the
machine where this was investigated, decode variance across three identical runs was **±2.6%**, which
would have swamped an effect this size — one reason it was not attempted blind.

1. Establish a baseline with **≥5 runs**, fixed `--temp 0`, fixed `-n`, and report the spread, not the
   best run.
2. Measure **prefill and decode separately** — `-p` and `-n` in `llama-bench` are independent tests
   from fresh contexts. If comparing against llama.cpp, `-pg pp,tg` is the combined form; mixing them
   up produced a retracted claim earlier in this project.
3. Test with **`STINGRAY_CUDA_GRAPH=0` as well as the default**. Graphs already absorb launch
   overhead, so the flag separates "fusion helped the GEMM" from "fusion removed launches".
4. Use a model with a large `intermDim` — the effect scales with it. A 0.6B model will show nothing.

## Kill criteria — when to abandon this

Abandon it, and record that here, if any of these hold:

- Prefill improvement is **inside the run-to-run spread** on a model with a realistic `intermDim`.
- Decode shows no change with graphs enabled *and* the prefill gain is under ~2%. The complexity cost —
  a view/aliasing mechanism in the backend, a fallback path, a dtype guard, and an excluded MoE branch —
  is not worth a sub-noise win.
- Gate/up dtypes turn out to differ on the models you actually run, making the fused path rarely taken.

A negative result is a real result. Write it in this file rather than deleting the plan, so the next
person does not rediscover the same idea from the same CUTLASS reading.

## Correctness gate

Fusing must not change outputs. The fused and unfused paths read the same weight bytes into the same
GEMM in the same K order, so they should be **bit-identical**, not merely close. Assert exactly that:
run a fixed prompt greedily through both paths and compare logits bitwise. If they differ at all, the
concatenation or the view offsets are wrong — do not accept "close enough" here, because a subtly
transposed or misaligned second half produces a model that loads, runs, and emits plausible nonsense.
