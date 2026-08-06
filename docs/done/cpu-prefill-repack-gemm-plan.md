# CPU prefill, phase 2 — repacked-weight GEMM

**Status:** checkpoint 1 (§9), eligibility/budget report (§10), scalar GEMV port (§11, found to
be llama.cpp's non-SIMD fallback, ~8× slower than baseline, §12), and four rounds of AVX2
micro-optimization (§13-§17) are done. Combined: scalar→AVX2 took the kernel from 6.81ms to
4.00ms (real, ~41%), but the next three targeted fixes — redundant decode (§14), redundant
activation reload (§15), and eliminating the profiled-and-confirmed-real 32%-cost de-interleave
copy (§17) — each landed cleanly with correctness held and **each moved throughput ~0-3%,
not the expected win**. Currently 3.53ms/call, **still ~3.5× slower than the F32 baseline at
batch=1 — not wired into `ForwardPass`.** Three consecutive real, verified, near-zero-effect
fixes is itself the finding: this specific kernel shape (isolated batch=1 GEMV) appears to have
an irreducible per-call floor that piecemeal optimization isn't reaching. A follow-up attempt
at batch>1 (§18, `GemmQ4K16x16Q8`, externally proposed) confirmed this concretely: correct, and
faster than isolated single-token GEMV, but still 2-3× *slower* than the already-shipped
`MatMulBatched` on the same shape/batch — `GemvQ4K8x8Q8K`'s own per-unit cost, not
parallelization granularity, is the bottleneck. **This repacked-GEMV design, as implemented, is
not on a path to beating what's already shipped.** Follows `docs/cpu-prefill-plan.md`,
which landed the `_4In` int8 batched path (§12 there): gap to llama.cpp closed from ~6× to
~4.6×, gated off by default, quality-verified at both greedy-token (§13) and corpus-perplexity
(§14) level with a noise-level (−0.4%) delta. This document is phase 2, informed by reading
llama.cpp's actual CPU GEMM implementation (`ggml/src/ggml-cpu/repack.cpp`) rather than
guessing at what it does.
**Owner:** unassigned
**Last updated:** 2026-07-24

---

## 1. What llama.cpp actually does (read from source, not inferred)

`examples/cpp/llama.cpp/ggml/src/ggml-cpu/repack.cpp` implements genuine 2D-blocked GEMM:
multiple **weight rows** computed against multiple **activation columns** per call, via
functions like `ggml_gemm_q4_K_8x8_q8_K_generic` (8 rows × N columns) and a `_16x1` variant
(16 rows). This is a different axis of batching than our `_4In` work, which batches only the
activation side (4 tokens against 1 weight row).

### The mechanism: repack once at load, not per call

The weight reorganization is the whole trick. `make_block_q4_Kx8` (repack.cpp:2836) takes 8
separate `block_q4_K` structs — which are scattered in memory, row 0's 144 bytes then row 1's,
etc. — and interleaves them into one `block_q4_Kx8` struct (repack.h:43):

```c
struct block_q4_Kx8 {
    ggml_half d[8];      // super-block scale, one per row
    ggml_half dmin[8];   // super-block min-scale, one per row
    uint8_t scales[96];  // scales/mins for all 8 rows, re-packed 12B-per-subblock -> interleaved
    uint8_t qs[1024];    // 4-bit quants for all 8 rows, byte-interleaved
};
```

Critically, **this repack happens once, when the model loads**, not on the hot path. llama.cpp
implements it as a distinct "extra buffer type" (`ggml_backend_cpu_repack_buffer_type`,
ggml-cpu.cpp:65) — weight tensors get copied into this reorganized layout at load time, and the
scheduler automatically routes matmuls against them to the repacked GEMM kernels instead of the
plain per-row ones. The original (on-disk GGUF) layout is not touched at inference time at all.

### Why the GEMM kernel is faster once weights are repacked

With interleaved storage, one SIMD load pulls a column-slice across all 8 rows at once (they're
now physically adjacent), instead of 8 separate scattered loads — one per row, each needing its
own pointer arithmetic and cache-line fetch. Reading `ggml_gemv_q4_K_8x8_q8_K_generic`
(repack.cpp:958) and its sibling `_8x4` (repack.cpp:887): the inner loop accumulates, per
weight sub-block, `int32` products across 8 output columns (rows) in one pass, scaled by each
row's own fp16 `d`/`dmin` at the end — i.e. 8 dot products computed together with shared
activation reads, mirroring exactly what `_4In` does for activations but on the weight axis.

**This is the "genuine data-layout change" `docs/cpu-prefill-plan.md` predicted was needed
beyond loop reordering (§1, §3), now identified concretely rather than as a guess.**

---

## 2. Why this is different from — and larger than — the `_4In` work

| | `_4In` (landed) | Repacked GEMM (this plan) |
|---|---|---|
| Batches | 4 activation columns (tokens) | 8 weight rows × N activation columns |
| Weight storage | Untouched (still the on-disk GGUF layout) | Reorganized once at load into a new layout |
| One-time cost | None | Repack pass at model load (CPU time + a second copy of the weight data in memory) |
| Correctness contract | Byte-exact vs. per-token Q8 reference (proven, §12 there) | Almost certainly **not** byte-exact even vs. a Q8 reference — see §5 |
| Kernel surface | 1 new dispatch function reusing existing `_4In` kernels | New GEMM kernels per supported dtype, new repacked struct layouts, new load-time step |

This is a bigger, longer piece of work than `_4In` was. It is not a tuning knob; it changes
where weight bytes live in memory.

---

## 3. Proposed design for OpenTail.Stingray

### 3.1 Repacked layout

Mirror `block_q4_Kx8` as a plain byte-buffer layout in C# (not a literal struct — OpenTail's
existing dequant/dot kernels already work on raw `byte*` with hand-computed offsets, so the
repacked format follows the same convention): for every group of 8 weight rows, lay out
8× `d` (fp16) + 8× `dmin` (fp16) + interleaved `scales` + interleaved `qs`, byte-for-byte
matching `make_block_q4_Kx8`'s transform so the packing logic can be validated directly against
llama.cpp's own reference GGUF/logit outputs (the project already has `xcheck-llamacpp.ps1` and
reference-logit tooling for exactly this kind of cross-check).

**Row counts not divisible by 8** (padding): the last partial group is padded with zero rows
(computed but discarded), matching llama.cpp's approach — simpler than a separate narrow
kernel for the remainder, at the cost of a few wasted dot products per matrix, which is
negligible relative to the whole prefill.

### 3.2 Load-time repack step

A new method, analogous to the existing `_dequantWeightCache` mechanism already in
`ForwardPass.cs` (docs §1 references this — it's the existing "dequant once, reuse across
chunks" cache for the BLAS path): at model load, for every weight tensor whose dtype has a
repacked kernel, walk its rows in groups of 8 and build the repacked buffer once. Own this
memory explicitly (it's a full second copy of that tensor's bytes, not a view into the mmap'd
GGUF file) and free it on model teardown, mirroring how the dequant cache is disposed today.

**This must be opt-in and budgeted**, same reasoning as the existing dequant-cache budget flag
(`--dequant-cache-mb` / `prefillDequantCacheBytes` already in `RunCommand.cs`): repacking every
matrix in a large model doubles that model's *weight* memory footprint for the duration the
repacked copies are held, on top of the original mmap'd file staying resident. For SmolLM2
(1005 MiB) that's acceptable; for a 20GB MoE model it would not be, so this needs the same
kind of budget/opt-in gate the dequant cache already has, not an unconditional "always repack."

### 3.3 New GEMM kernels

Start with **one dtype (Q4_K) and one row-width (8)**, matching the model this whole
investigation has used (`SmolLM2-1.7B-Instruct-Q4_K_M`) and llama.cpp's own `8x8` variant
(not `16x1`, which needs AVX-512 to pay off and this CPU has none — confirmed in
`docs/cpu-prefill-plan.md` §2: Ryzen 7 5700G, no AVX-512). Reuses the existing
`QuantizeRowToQ8K`/`Q8KScratchBytes` activation-side infrastructure already built for `_4In` —
the activation quantization doesn't change, only how weight rows are read.

Q6_K, Q3_K, and other dtypes are explicitly **out of scope for the first landing** — get Q4_K
working and measured before generalizing, same incremental discipline as `_4In`'s rollout.

### 3.4 Dispatch and fallback

`MatMulBatched` gains a third path, checked before the existing `_4In` path: if a repacked
buffer exists for this weight tensor (resolved once, e.g. via a dictionary keyed by weight
pointer, populated at load) and the dtype/row-width match, use the repacked GEMM; otherwise
fall through to `_4In`, then the plain per-token loop, unchanged. No existing path's behavior
changes when this one isn't engaged.

### 3.5 Gating

This needs its **own** gate, separate from `STINGRAY_CPU_PREFILL_Q8` — repacking is a
load-time, memory-committing decision, not a per-call toggle like `_4In`'s. Proposed:
`STINGRAY_CPU_PREFILL_REPACK=1`, checked once at model load (not per `MatMulBatched` call),
default off. `_4In`'s gate stays independent so the two can be measured separately or together.

---

## 4. What I will actually do, in order

1. **Read `ggml_gemm_q4_K_8x8_q8_K_generic` and `make_block_q4_Kx8` in full** (only partially
   quoted above) and write out the exact byte-layout transform as a C# reference
   implementation, tested against llama.cpp's own dequantized output for a handful of real
   rows from `SmolLM2-1.7B-Instruct-Q4_K_M.gguf` (via `list-tensors`/existing GGUF tooling) —
   i.e. prove the *repack* step alone is correct (repacked-then-dequantized bytes match the
   original dequantized bytes) before writing any new dot kernel.
2. **Write the repacked-buffer builder** (`BuildQ4KRepacked8`, or similar) as an isolated,
   directly-testable function — input: original Q4_K byte buffer + row/col counts; output: the
   repacked byte buffer. Unit test: for every row, extracting that row back out of the repacked
   buffer and dequantizing it must equal dequantizing the original row directly (reuses the
   existing `Dequantize.ToFloat32` as the independent reference).
3. **Write the 8-row GEMM kernel** operating on the repacked buffer + Q8_K-quantized
   activations, porting the accumulation order from `ggml_gemm_q4_K_8x8_q8_K_generic` as
   directly as possible (same per-subblock int32 accumulate-then-scale order) rather than a
   from-scratch redesign, specifically so any numerical difference from the F32/Q8 references
   is attributable to something identifiable, not an accidental reordering.
4. **Build a correctness harness before wiring it into `MatMulBatched`** — same discipline as
   `MatMulBatchedEquivalenceTests`/`MatMulBatchedQ8EquivalenceTests`: compare the new kernel's
   output against the existing per-row `DotQ4K_Q8K`/`DotQ4K_Q8KS` reference. **Expect this to
   require a tolerance, not byte-exactness** (see §5) — decide and document the bound before
   writing the test, not after seeing what the numbers happen to be.
5. **Wire into `MatMulBatched` behind the new gate**, default off, falling back cleanly when
   the gate is off, the dtype/row-width isn't covered, or no repacked buffer was built for that
   tensor.
6. **Measure with `bench-vs-llamacpp.ps1 -Runs 3`** before/after, same protocol as the `_4In`
   work — including the load-time cost (does `[ForwardPass] Pre-faulted ... in 0.1s` become
   meaningfully slower with repacking folded in?). **If it's slower, revert and record why**,
   same rule as attempt 1.
7. **Update this document and `docs/cpu-prefill-plan.md`** with real measurements at each step,
   not just at the end.

---

## 5. Real risks (asking to be challenged on all of these)

- **Byte-exactness is unlikely even against the Q8 reference.** `_4In`'s biggest win was that
  batching didn't change the answer (§12 there: bit-identical to N independent per-token Q8
  calls). Reading `ggml_gemm_q4_K_8x8_q8_K_generic`, the accumulation groups 8 columns' `int32`
  products together per sub-block before scaling, which is a different summation structure than
  a scalar per-row loop — floating-point (and even fixed-point-then-scale) reassociation is not
  generally exact. This needs a **tolerance-based** correctness contract from the start, which
  is a genuinely different (and weaker) guarantee than what `_4In` achieved, and needs its own
  numerical-error budget decided up front, not discovered after the fact.
- **Memory cost.** A full second copy of every repacked tensor's bytes, held for the process
  lifetime (or until explicitly freed). Needs the same budget/opt-in treatment as the existing
  dequant cache, not an unconditional default.
- **Load-time cost.** The repack pass itself costs CPU time at model load, which the existing
  `bench-vs-llamacpp.ps1` doesn't currently measure (it only measures prefill/decode, not load
  time) — the harness may need a load-time metric added before this can be judged fairly.
- **Scope discipline.** llama.cpp supports repacked kernels for many dtypes and row-widths
  (4/8/16, `Q4_0`/`Q4_K`/`Q2_K`/etc.) built up over a long time. Landing all of that is not the
  goal here; one dtype, one row-width, measured, is.
- **Diminishing returns is possible.** Even a working repacked Q4_K/8-wide kernel may not close
  the remaining ~4.6× gap on its own — llama.cpp's advantage is the accumulation of many such
  optimizations (repacking across multiple dtypes/widths, plus whatever else `ggml`'s CPU
  backend does beyond just this one file) built over years. This phase should be judged on its
  own measured delta, not against an expectation of reaching parity.
- **This is real, open-ended engineering time**, not a bounded nudge-sized task like `_4In` was.
  Worth treating as its own tracked piece of work with checkpoints, not something to squeeze
  into hourly increments.

---

## 6. Explicit questions for review

1. Is starting with Q4_K/8-wide (not 4-wide, not 16-wide) the right first target, given this
   CPU has AVX2 but no AVX-512?
2. Is a tolerance-based correctness contract (rather than byte-exact) actually acceptable here,
   or is there a way to constrain the port (e.g. matching llama.cpp's exact accumulation order
   bit-for-bit, including intermediate rounding) that could still achieve byte-exactness against
   *some* well-defined reference, even if not the existing scalar Q8 dot?
3. Is per-tensor repacked-buffer ownership (a dictionary keyed by weight pointer, built once at
   load) the right integration point, or should this live inside `GgufModel`/`ForwardPass`
   instead of `SimdKernels`, given it's model-lifetime state rather than a pure compute kernel?
4. Any reason the row-padding-for-non-multiple-of-8 approach (compute and discard, matching
   llama.cpp) is unsound here specifically, e.g. interaction with existing GPU-offload row
   partitioning logic that assumes exact row counts?
5. Given the scale of this relative to `_4In`, is there a smaller intermediate step worth taking
   first — e.g. proving out just the repack-and-dequant-round-trip correctness (my step 1-2
   above) as its own checkpoint before committing to writing a new GEMM kernel at all?

---

## 7. External review (codex / gpt-5.6-terra) — before any implementation

> "The core idea is sound, and Q4_K × 8 rows is a sensible first target on AVX2. The plan
> correctly identifies that this is a layout-and-lifetime feature, not simply another
> `SimdKernels` dispatch tweak."

Two **high-severity architectural corrections** to §3.4's proposed integration point, both
accepted — this plan is revised accordingly before any code is written:

### 7.1 Reject: pointer-keyed dictionary inside `SimdKernels`

> "A global/static dictionary keyed by raw weight pointer inside `SimdKernels` would be the
> wrong boundary: it has no model lifetime, budget, disposal, or concurrency context, and stale
> entries could survive a model teardown and pointer reuse."

**Concrete failure scenario:** model A is disposed, model B is loaded and its allocator reuses
the same native address, and the stale pointer-keyed cache serves *model A's repacked weights*
to model B's inference — silently wrong output, not a crash. §3.2/§3.4 are revised: repacked
buffers move to an explicitly-owned cache on `ForwardPass` (or a model-owned CPU execution-cache
object), keyed by stable tensor identity (name + shape + dtype), not a raw pointer. A dedicated
GEMM entry point receives the repacked buffer explicitly as a parameter, rather than
`MatMulBatched` looking one up by pointer.

### 7.2 Reject: adding the repack path only inside `SimdKernels.MatMulBatched`

> "There is also an integration completeness problem: ... many batched prefill sites call
> `SimdKernels.MatMulBatched` directly. Adding a third path only inside the generic SIMD API
> can work only if a global raw-pointer registry is used ... Prefer consolidating CPU batched
> matmul through one `ForwardPass` wrapper that receives `TensorRef`."

This is the more important finding: §3.4 as written would only benefit whichever call sites
happen to route through the one function I was editing, while the model's other batched-prefill
projections keep calling `SimdKernels.MatMulBatched` directly and never see the repacked path at
all — silent, partial, and hard-to-notice under-coverage rather than an outright bug. Revised:
route CPU batched prefill through one `ForwardPass`-level wrapper (already partially true via
the existing `MatMulBatchedCached` pattern for the F32 dequant cache) that decides repacked vs.
`_4In` vs. plain per-token, rather than adding decision logic inside the low-level SIMD API.

### 7.3 Accept: eligibility/budget policy needs to be decided up front, not "repack everything"

> "'Every Q4_K tensor at model load' is too broad: this library has other CPU users of
> `MatMulBatched` (diffusion/text encoders) and GPU/hybrid paths where a CPU repack has no
> benefit. Budget allocation needs to be all-or-nothing for the selected prefill set ... not a
> lazy fill-and-stop policy, which will fail to retain weights across the full-model reuse
> distance."

§3.2 underspecified this. Revised: define the CPU-dense-prefill-eligible tensor set explicitly
(not "every Q4_K tensor anywhere") before implementation, and make the budget decision
all-or-nothing for that set — a partial repack that gets evicted mid-sweep costs memory and
load time while delivering nothing, which is worse than not repacking at all.

### 7.4 Accept: the new gate's relationship to the existing Q8 gate needs to be explicit

> "Specify the relationship between the new repack gate and `STINGRAY_CPU_PREFILL_Q8`,
> because the proposed GEMM still quantizes activations and therefore changes CPU prefill
> numerics ... a user enables what appears to be a layout/memory optimization but unexpectedly
> opts into Q8-prefill numerical behavior despite leaving the existing Q8 gate disabled."

Correct — §3.5 proposed the two gates as independent, but the repacked GEMM is *also* an int8
path (it consumes `Q8_K`-quantized activations, same as `_4In`), so it carries the same
divergence-from-decode risk `STINGRAY_CPU_PREFILL_Q8` exists to gate. Revised:
`STINGRAY_CPU_PREFILL_REPACK` either requires `STINGRAY_CPU_PREFILL_Q8=1` to also be set,
or is folded into that same gate as a mode (`STINGRAY_CPU_PREFILL_Q8=repack` vs `=1`) rather
than presented as an independent memory/layout-only toggle.

### 7.5 On the correctness contract (§5, §6 Q2)

> "Do not commit to an arbitrary numerical threshold before collecting a baseline. Define the
> *kind* of contract up front ... If the port mirrors the same per-output accumulation order as
> the Q8 reference, byte-exactness may still be achievable; treat that as an experiment rather
> than assuming it is impossible."

Fair correction to my own plan: §5 asserted byte-exactness was "almost certainly" unreachable.
Revised position: attempt the direct accumulation-order port first (plan step 3 already
proposed this) and *measure* whether it's byte-exact before assuming a tolerance is needed. If
it isn't, the threshold must come from a documented error analysis (adversarial scale/input
cases, worst-case bound with headroom) plus an end-to-end perplexity/greedy-token check — not a
number picked from one model's typical logits, which was the real risk in the original wording.

### 7.6 Accept: the recommended checkpoint

> "The best intermediate checkpoint is exactly steps 1–2, extended with an allocation/
> eligibility report for a real model. It validates the risky representation and quantifies
> memory/load cost before committing to the much larger SIMD kernel port. Add a stop criterion:
> only continue if the Q4_K tensors reachable from CPU batched prefill account for a material
> share of measured prefill time and the repack budget fits the intended deployment profile."

Adopted as the actual first deliverable (§8, replacing the flat step-by-step list in §4).

---

## 8. Revised first checkpoint (supersedes §4 steps 1-2 as the immediate next action)

Before writing a single GEMM kernel line:

1. Prove the repack-then-dequant round trip is correct (original §4 steps 1-2), against
   `SmolLM2-1.7B-Instruct-Q4_K_M.gguf`'s real Q4_K tensors.
2. Produce an **eligibility and budget report**: which tensors in this model are (a) Q4_K,
   (b) actually reached via CPU dense batched prefill (not diffusion/text-encoder/GPU/hybrid
   paths), (c) what repacking all of them would cost in memory and load time.
3. **Stop-or-continue gate**: only proceed to the GEMM kernel (§4 steps 3+) if those eligible
   tensors are a material share of measured prefill time (profile this, don't assume it) and
   the budget fits a normal deployment, not just this one 1005 MiB test model.
4. Only after that: design the `ForwardPass`-level dispatch wrapper (§7.2) and the
   identity-keyed owned cache (§7.1) — not a `SimdKernels`-internal pointer dictionary.

---

## 9. Checkpoint 1 — repack-then-dequant round trip, Q4_K/8-wide (done)

Implements step 1 above, nothing more. No GEMM kernel, no `ForwardPass` wiring, no gate. Two
files:

- `src/OpenTail.Stingray.Cpu/RepackedGemm.cs` — `RepackQ4K8Rows(src, blocksPerRow)`, a direct C#
  port of `make_block_q4_Kx8` (repack.cpp:2836, `interleave_block=8`, matching the `<block_q4_K,
  8, 8>` specialization used by the generic 8x8 GEMM per repack.cpp:3880-3881): takes 8 source
  rows' raw Q4_K bytes and produces the interleaved `block_q4_Kx8` byte layout (§3.1: 16B `d[8]`
  + 16B `dmin[8]` + 96B re-packed scales + 1024B interleaved `qs`, 1152 bytes per super-block
  group). Also `DequantizeRepackedQ4KRow(repacked, blocksPerRow, row, dst)` — dequantizes one
  row directly out of the repacked buffer (decoding the folded 6-bit scale/min packing in
  place, without reconstructing the original raw bytes) for the correctness check below; not
  used by anything else and not on any hot path.
- `tests/OpenTail.Stingray.Tests.ForwardPass/RepackedGemmQ4KRoundTripTests.cs` — for 8 rows of
  **random** Q4_K bytes (1, 2, and 8 super-blocks per row), repack then dequantize each row out
  of the repacked buffer and compare against `Dequantize.ToFloat32` on that row's original
  bytes directly. Random rather than realistic data deliberately: the repack is a pure byte
  permutation with no arithmetic, so it must be bit-exact for *any* input, not just plausible
  quantized weights — random bytes are a stronger adversarial check of the transform itself.

**Result: bit-exact on every element, first run, no tolerance needed.** This answers §6 Q2 and
§7.5 in the affirmative for the repack step specifically (not yet the GEMM kernel, which is the
part actually expected to need reassociation/tolerance per §5): the layout transform alone
introduces zero numerical difference, as expected for a pure permutation, now proven rather than
assumed. Full `Tests.ForwardPass` suite: 1047 total (4 new), 1031 passed / 16 failed — the same
16 pre-existing device-less Vulkan/GPU failures as every prior checkpoint in this project, zero
new regressions.

**What this does and doesn't establish:** confirms the byte-layout half of §3.1 is correct and
testable in isolation, exactly as the reviewed checkpoint (§7.6, §8) intended — a real, cheap,
low-risk unit before committing to the GEMM kernel. Does **not** yet answer whether repacking is
worth doing at all: no eligibility/budget report (§8.2) has been produced, no profiling has shown
Q4_K CPU-dense-prefill tensors are a material share of prefill time, and the stop-or-continue
gate (§8.3) has not been evaluated. The GEMM kernel itself (§4 step 3) — the part where
byte-exactness is genuinely uncertain per §5/§7.5 — has not been started; this checkpoint only
de-risks the data-layout half of the design.

**Next:** either (a) the eligibility/budget report (§8.2-8.3) to decide whether to continue at
all, or (b) if that's judged already-answered by the `_4In` work's own profiling, proceed
straight to the 8-row GEMM kernel (§4 step 3) with the same "port the accumulation order
directly, measure for byte-exactness before assuming a tolerance" discipline §7.5 called for.

---

## 10. Checkpoint 1b — eligibility and budget report, SmolLM2-1.7B-Instruct-Q4_K_M

Per §8.2-8.3, measured before writing any GEMM kernel.

### Eligibility (which tensors would actually route through repacked Q4_K/8)

`ForwardPass.cs`'s CPU dense batched-prefill call sites (`MatMulBatched` at the wq/wk/wv/wo and
ffn_gate/ffn_up/ffn_down lines, §2's dense path — the same set `_4In`/§10 there already
established is the only CPU dense path this model uses) touch 168 weight tensors per full
forward pass across 24 layers. Dumped via `list-tensors -m models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf`
and counted:

| | Count | Total bytes |
|---|---|---|
| Eligible tensors (attn q/k/v/o + ffn gate/up/down) | 168 | 921.6 MiB |
| — of which Q4_K (this phase's target dtype) | 144 | 724.8 MiB (**78.6%** of eligible bytes) |
| — of which Q6_K (out of scope this phase, per §3.3) | 24 | 196.8 MiB |

All 144 Q4_K tensors have row counts divisible by 8 (2048 or 8192 rows) — **zero padding-group
tensors** for this model, so §3.1's padding path is untested by this model specifically (would
need a model with a non-multiple-of-8 hidden dim to exercise it).

Since §2 already established CPU prefill is bandwidth-bound (flat-with-length on both engines,
scaling with weight bytes read, not compute), byte share is a direct proxy for time share here:
**Q4_K tensors account for ~79% of the weight traffic the dense CPU prefill path reads** —
a material majority, not a marginal slice.

### Budget — memory

Repacked `block_q4_Kx8` is **144 bytes per row-equivalent** (1152 bytes / 8 rows) — exactly the
same as the original Q4_K row size. Unlike the existing OpenBLAS F32-dequant cache (an 8×
blow-up), **repacking is size-neutral**: it's a straight second copy, not a blow-up multiplier.
For this model: **+724.8 MiB** resident to hold both the original mmap'd bytes and the repacked
copy simultaneously (the mmap page is not released — other non-repacked tensors, embeddings,
and the LM head still read from it), roughly **1.7× this model's ~1.06 GiB weight footprint**.
Acceptable for a 1.7B model on a normal desktop; **confirms §3.2/§5's memory-cost concern is
real and scales with model size** — a 20GB MoE model's Q4_K-eligible dense tensors would still
cost proportionally the same ~1.7×, which is why this must stay opt-in/budgeted (§3.2), not
default-on, independent of how phase 2 turns out for this specific model.

### Budget — load-time repack cost (measured, not estimated)

Built a throwaway reflection-based harness (session-local, not committed) that calls
`RepackedGemm.RepackQ4K8Rows` on byte buffers shaped like this model's actual eligible Q4_K
tensors (2048×2048 ×3/layer, 2048×8192 and 8192×2048 ffn, 24 layers — content is random bytes,
irrelevant to a pure byte-permutation's throughput). **Single-threaded: 810 MiB repacked in 403
ms (~2.0 GiB/s)** — scaling to the model's actual 724.8 MiB Q4_K set, **~360 ms**. For
comparison, this model's own weight pre-fault already costs ~0.2-0.3s (`[ForwardPass] Pre-faulted
1.06 GiB ... in 0.2-0.3s`) as an accepted normal load cost. The repack pass is embarrassingly
parallel (independent per row-group), so a threaded implementation should land well under
100ms — not measured here since the checkpoint's function is single-threaded by design (a
threaded repack is an implementation detail of the real load-time step in §3.2, not needed to
answer the budget question).

### Stop-or-continue decision

**Continue to the GEMM kernel (§4 step 3).** Both gate conditions from §8.3 are met, backed by
real numbers rather than assumption:

1. **Material share confirmed**: ~79% of dense-prefill weight bytes are Q4_K, on the actual
   model this whole investigation has used throughout.
2. **Budget fits**: ~360ms single-threaded repack cost (parallelizable, likely <100ms), ~1.7×
   memory footprint for this model — a normal desktop deployment, not disqualifying. Still
   requires the opt-in/budget gate for larger models, unchanged from §3.2's design.

Next real step: the 8-row `_8x8_q8_K` GEMM kernel itself (§4 step 3), the one place §5/§7.5
flagged as genuinely uncertain on byte-exactness — attempt the direct accumulation-order port
first and measure before assuming a tolerance is needed, same discipline that made this
checkpoint's repack-step verification unconditionally exact.

---

## 11. GEMV kernel — ported, measured, not yet wired in

Implements the **GEMV** variant first (one activation row against 8 repacked weight rows),
not the full multi-activation GEMM (§1's `ggml_gemm_q4_K_8x8_q8_K_generic`, which additionally
needs a `block_q8_Kx4`-interleaved *activation* format — new work on the activation side, not
just the weight side). GEMV needed no new activation format: llama.cpp's `block_q8_K` (one
scale + 16-way bsums per 256-element super-block) turned out to already exist in this codebase
as `SimdKernels.QuantizeRowToQ8K`/`Q8KScratchBytes` — currently used for Q6_K's dot kernel, not
Q4_K (Q4_K's existing `_4In` path uses a *different*, custom scratch format, `Q8_KS`, per-32
scales instead of per-256 — not interchangeable, confirmed while reading the code to avoid
routing through the wrong one). Reusing the existing Q8_K quantizer meant this slice needed
only the weight-side kernel.

### What was ported

`RepackedGemm.GemvQ4K8x8Q8K` in `src/OpenTail.Stingray.Cpu/RepackedGemm.cs` — a line-for-line port of
`ggml_gemv_q4_K_8x8_q8_K_generic` (repack.cpp:958): same `utmp` scale/min unpacking (the
`kmask1`/`kmask2`/`kmask3` bit-twiddling, operated on the *raw* packed 12-byte scale chunks per
sub-block — deliberately re-derived from the original bytes rather than reusing checkpoint 1's
`DecodeScaleMinPair` helper, so the accumulation order matches ggml's byte-for-byte rather than
being "equivalent by a different route"), same per-subblock int accumulate-then-scale structure,
same output layout (`sumf[j] - sum_minf[j]` per column). Produces 8 outputs (one per repacked
row) per call, consuming one `QuantizeRowToQ8K`-quantized activation row.

### Correctness — tolerance-based, measured rather than assumed (§7.5)

Per the external review, byte-exactness against a plain-float reference isn't the right bar
here (the kernel consumes int8-quantized activations; the reference doesn't) — so the test
measures the actual delta instead of picking a bound first. `RepackedGemmQ4KRoundTripTests.
GemvQ4K8x8Q8K_MatchesScalarReference_WithinQuantizationNoiseTolerance`: random Q4_K weight
bytes (finite d/dmin explicitly forced, since fully-random fp16 bit patterns hit NaN/Inf often
enough at this scale to poison the reference dot product) against random `[-1,1)` activations,
comparing `GemvQ4K8x8Q8K`'s 8 outputs to `SimdKernels.DotQ4K` (the existing scalar,
unquantized-activation reference) per row.

**Measured: 2.56% max relative error** across the 8 outputs — consistent with Q8_K's
per-256-element int8 quantization step (~1/127 relative ≈ 0.8%, compounding somewhat as it
propagates through 2048 summed terms), not a reassociation bug (the repack step itself, §9, is
proven bit-exact separately). Test asserts `< 5%` — real headroom above the measured figure,
not a loose bound chosen to pass. Full `Tests.ForwardPass`: 1048 total (5 new), 1032 passed / 16
failed — the same pre-existing device-less failures, zero new regressions.

### What's still open before this is useful

- **Not wired into `ForwardPass`/`MatMulBatched` at all.** No gate, no dispatch, nothing calls
  this kernel outside its own test. Per §7.1/§7.2's accepted review corrections, that wiring
  needs the `ForwardPass`-level wrapper + identity-keyed owned cache, not a `SimdKernels`-
  internal lookup — not done yet.
- **GEMV only, not GEMM.** This processes one activation row per call — it gets the weight-side
  reuse win (8 rows read together) but none of the *additional* reuse the full GEMM would add
  by also batching activations 4-at-a-time via `block_q8_Kx4`. Real prefill needs the GEMM
  variant (or this GEMV called once per token, which reuses repacked weights across the
  `MatMulBatched` call but not across tokens within it) to be worth the repack cost at all —
  worth measuring GEMV-called-per-token before investing in the GEMM's extra activation-format
  work, since it may already capture most of the win at much lower implementation cost.
- **No performance measurement yet.** Nothing here has been benchmarked against `_4In` or
  llama.cpp — this section is a correctness-only checkpoint, same split as §9 was for the
  repack step.

### Next

Wire `GemvQ4K8x8Q8K` into a real call path (even a standalone microbenchmark harness first,
before touching `ForwardPass`) and measure actual throughput before deciding whether the GEMM's
extra activation-interleave work is worth building — cheaper to learn that from GEMV numbers
now than to build the GEMM first and find out the weight-reuse win alone wasn't the bottleneck.

---

## 12. GEMV throughput — measured, and it's currently a regression (real finding, not yet fixed)

Built a throwaway microbenchmark (session-local, not committed) doing a full 2048×2048 Q4_K
matvec (SmolLM2's attn_q/k/o shape) three ways, single activation row (batch=1, the case GEMV
targets), 30 timed iterations after 5 warmup, repack cost excluded from the timing loop
(measured separately in §10):

| Method | ms/call | Effective GB/s |
|---|---|---|
| F32 `MatVec` (today's production baseline) | 0.86 | 2.74 |
| Q8_KS single-row dot (today's `_4In` per-row cost, batch=1) | 2.71 | 0.87 |
| **Repacked `GemvQ4K8x8Q8K` (new)** | **6.81** | **0.35** |

**The new kernel is ~8× slower than the F32 baseline and ~2.5× slower than the existing
Q8_KS per-row dot — a regression, not a win, as currently implemented.**

### Root cause (found by reading llama.cpp's own source, not guessed)

`ggml_gemv_q4_K_8x8_q8_K_generic` — the function ported in §11 — is llama.cpp's **scalar
fallback**, deliberately named `_generic` and compiled only when no SIMD ISA is available.
`examples/cpp/llama.cpp/ggml/src/ggml-cpu/arch/x86/repack.cpp:1464` has a *second*, unrelated-
looking function with the **same name minus `_generic`** (`ggml_gemv_q4_K_8x8_q8_K`) guarded by
`#if defined(__AVX2__)` — real AVX2 intrinsics (`_mm256_*` shuffle/permute/lookup-table
sign-extension of the packed nibbles, vectorized scale application), falling back to a call to
the `_generic` version only when AVX2 is unavailable. **§11 ported the fallback, not the fast
path** — this was not identified during the earlier read of `repack.cpp` (§1) because that
reading focused on the arch-independent file; the x86-specific override lives in a sibling file
that wasn't examined until this measurement forced a second look.

This is fully consistent with the rest of this investigation: `docs/cpu-prefill-plan.md` §2
established CPU prefill is bandwidth-bound and every existing production kernel in
`SimdKernels.cs` (`DotQ4K`, `DotQ4K_Q8KS`, the `_4In` family) is hand-vectorized with
`Avx2`/`Fma` intrinsics — a scalar kernel was never going to be competitive against them
regardless of its weight-reuse design, because weight-read reuse only pays off if the compute
per byte read is also fast. The **design** (8-row interleaved reads, proven bit-exact in §9,
correctness-verified within Q8_K noise in §11) is not invalidated by this result — only the
*generic* implementation of it is unfit to ship. This is exactly the outcome the plan's own
risk list (§5: "diminishing returns is possible... llama.cpp's advantage is the accumulation of
many such optimizations") flagged as possible, now confirmed concretely rather than assumed.

### Status and recommendation

**Do not wire `GemvQ4K8x8Q8K` into `ForwardPass`/`MatMulBatched` as-is** — it would make prefill
slower, not faster, for anyone who enables its gate. It stays as a correctness-proven building
block (§11's tests remain valid — the *numerics* are right) but is not deployment-ready.

To make this phase actually pay off, the real next step is porting the **AVX2 intrinsics
version** (`arch/x86/repack.cpp:1464`), not iterating further on the scalar one — a materially
bigger, riskier slice than everything in this phase so far (hand-translating `_mm256_*`
shuffle/permute/lookup-table logic to .NET's `Avx2`/`Vector256` intrinsics, which is a new kind
of work relative to the straightforward byte-permutation and scalar-port slices done so far).
Given the size and risk jump, this is a natural checkpoint to pause and get explicit direction
before committing to it, rather than plan-and-build it unprompted the way earlier slices in
this phase were sized to allow.

---

## 13. AVX2 kernel — implemented, real bug found and fixed, real speedup, still not a win

Per direction, ported an AVX2 fast path rather than staying on the scalar `_generic` kernel.
**Not** a literal instruction-for-instruction port of llama.cpp's `arch/x86/repack.cpp:1464`
(its cross-column `_mm256_shuffle_epi8`/`blend`/`permute` choreography is register-tetris-level
intricate and high-risk to hand-transliterate blind). Instead: adapted this codebase's own
already-production AVX2 idiom, `SimdKernels.DotQ4K_Q8KS_Avx2` (nibble unpack +
`Avx2.MultiplyAddAdjacent`), per output column — de-interleave one column's 128 `qs` bytes back
to original-row order (16 small copies out of the already-hot repacked group), decode its
scale/min via checkpoint 1's already-proven `DecodeScaleMinPair`, then run the same vectorized
32-element dot `DotQ4K_Q8KS_Avx2` uses, substituting Q8_K's single per-superblock scale + bsums
for Q8_KS's per-32 scale array. Lower-fidelity to llama.cpp's specific kernel, but reuses two
already-validated pieces instead of introducing a large block of unverified new shuffle logic.

### A real bug, found and fixed via cross-checking, not luck

First build: correctness test failed by **1717%** relative error — not noise, a real bug.
Root cause: `DecodeScaleMinPair`'s `baseOffset` parameter selects *which subblock* (0-7), and
for GEMV's `chunk` loop (0-3, each covering subblock pair `2*chunk`/`2*chunk+1`) the offset
needed the same `sb<4 ? sb*12 : (sb-4)*12+48` branch checkpoint 1 already established —
the first draft used `chunk*12`/`chunk*12+48` directly, correct only by coincidence for
`chunk=0`'s first value. Found by adding a direct scalar-vs-AVX2 cross-check (both kernels
should agree closely, since they're the same accumulation order just vectorized) rather than
only checking against the loosely-toleranced `DotQ4K` comparison, which couldn't have localized
the bug as precisely. That direct check is now a permanent test:
`GemvAvx2_MatchesGemvScalar_Closely` (8 columns, <0.1% relative error, i.e. FP-reassociation
noise only) alongside the existing `DotQ4K`-tolerance test (both green, full suite:
1049 total, 1032 passed, 17 pre-existing device-less Vulkan/GPU failures — confirmed by name,
none touch this code — zero real regressions).

### Measured: real speedup, still not a win over the F32 baseline

Same microbenchmark as §12, same shapes, method unchanged except the GEMV kernel now dispatches
to the AVX2 path (`Avx2.IsSupported && Fma.IsSupported`, true on this machine):

| Method | ms/call | GB/s |
|---|---|---|
| F32 `MatVec` (production baseline) | 0.96 | 2.45 |
| Q8_KS per-row dot (today's `_4In`, batch=1) | 2.89 | 0.82 |
| Repacked GEMV, scalar (§12) | 6.81 | 0.35 |
| **Repacked GEMV, AVX2 (this section)** | **4.00** | **0.59** |

**41% faster than the scalar port** (6.81ms → 4.00ms) — real, measured, from vectorizing the
inner dot alone. Still **~4.2× slower than the F32 baseline** and **~1.4× slower than the
existing Q8_KS per-row dot** at batch=1. Not yet a win.

### Where the remaining gap likely is (not yet measured directly)

- **The per-column de-interleave.** 128 bytes × 8 columns × `blocksPerRow` scalar byte copies,
  every call, with no vectorization — plausibly a meaningful fraction of the 4.00ms given it's
  pure overhead the F32/Q8_KS baselines don't pay (they read each row's `qs` in its original,
  already-contiguous layout).
- **Redundant per-column re-decoding.** `DecodeScaleMinPair` and the scale/min unpack run once
  per (column, chunk) — 8×4=32 times per super-block per call — instead of decoding all 8
  columns' scale/min together once per super-block the way llama.cpp's actual kernel and this
  repo's own scalar §11 port both do. This is very likely the largest single inefficiency:
  the whole *point* of the interleaved layout is reading 8 columns' worth of packed data
  together, and this implementation still processes columns one at a time end-to-end.
- **Single-token (batch=1) is `_4In`'s own weak point too** (§9 of `cpu-prefill-plan.md`:
  Q8_KS is *slower* than F32 at batch=1, only wins from batch≥4 where quantization overhead
  amortizes) — GEMV may have the same shape of problem: its real payoff is likely at higher
  `blocksPerRow`/batched-activation (the actual GEMM, §1, not GEMV) rather than single-token
  single-column-at-a-time processing.

### Status

Real, verified progress (scalar→AVX2: 41% faster, and a real bug caught by disciplined
cross-checking rather than shipped) but **still not deployment-ready** — this remains a
correctness-proven building block, not wired into `ForwardPass`. Given the identified
inefficiency (redundant per-column decode) is architectural, not a micro-tuning issue, the next
concrete step — if this phase continues — is restructuring the AVX2 kernel to decode all 8
columns' scale/min once per super-block (matching §11's scalar structure and llama.cpp's own
design) before spending more effort on this specific kernel shape.

---

## 14. Redundant-decode fix landed — correctness held, but it wasn't the bottleneck

Implemented §13's proposed fix: `GemvQ4K8x8Q8K_Avx2` restructured so the `utmp` scale/min
unpack (same bit-twiddling as §11's scalar kernel) runs **once per super-block**, producing all
8 columns' scale/min directly indexable (`utmp[sb*16+col]` = scale, `utmp[sb*16+8+col]` = min),
replacing the 32 redundant `DecodeScaleMinPair` calls per super-block from §13's first cut.

**Correctness held**: all 6 tests green, including the tight scalar-vs-AVX2 cross-check
(<0.1% — the restructuring didn't touch the math, only when/how often it runs). Full
`Tests.ForwardPass`: 1049 total, 1033 passed, 16 failed — back to the exact baseline
pre-existing device-less-Vulkan count (§9/§12's number), zero regressions.

**Measured: 3.87ms/call, vs 4.00ms before the fix — a 3% improvement, not the "likely
dominant cost" §13 predicted.** Real, honest miss on the hypothesis: eliminating 32 redundant
12-byte decodes per super-block barely moved the number. The actual bottleneck must be
elsewhere — most likely the per-column de-interleave (1024 individual 8-byte scalar copies per
call: `blocksPerRow`(8) × 8 columns × 16 chunks) or the `HSumI32_256` horizontal-sum calls (512
per call, each a multi-instruction extract/shuffle/add chain that AVX2's data-parallel width
doesn't help with) — neither has been measured in isolation yet, so this is diagnosis, not a
confirmed finding.

### External review (gemini 3.6) — one confirmed real finding, verified not to matter (again)

A code review from a different model flagged: `q8_0`/`q8_1` (the activation vectors) depend
only on `(l, chunk)`, not on the weight column, but the loop order (`col` outer, `chunk` inner)
reloaded them via `Vector256.LoadUnsafe` 8× redundantly — once per column instead of once per
chunk, shared. **This was a real, verifiable inefficiency** — confirmed by reading the code,
not taken on faith — and cheap to fix: hoisted the 4 chunks' `q8_0`/`q8_1` loads into a
`stackalloc Vector256<sbyte>[4]` computed once per super-block, before the column loop.

**Measured: 3.88ms — statistically the same as before the fix (3.87ms).** Same pattern as §14:
a plausible-sounding, verified-real inefficiency that turned out not to be where the time goes.
64-byte reloads that are almost certainly L1-resident (the whole repacked group is 1152 bytes,
comfortably L1-sized) are cheap regardless of redundancy — this predicts the earlier de-
interleave/HSum hypothesis (§14) is more likely correct, though still not directly measured.
Correctness held: same 6/6 tests green, `RepackedGemm*`-filtered suite 6/6, full
`Tests.ForwardPass` 1032/1049 (17 failed — re-verified this run includes an unrelated simulated
CUDA-teardown-exception test alongside the usual device-less Vulkan set; zero real
regressions).

The same review also repeated the block-tiled-GEMM (`M=64/128` token tiles) and VNNI (`vpdpbusd`)
recommendations from an earlier, separate analysis. The tiled-GEMM direction is independently
correct and matches this document's own conclusion (§14: batch=1 GEMV can't amortize per-call
overhead the way a real multi-token GEMM would) — plain AVX2 tiled GEMM would still win from
cache reuse alone. VNNI specifically does not apply on this machine (Ryzen 7 5700G / Zen 3 has
no AVX-VNNI or AVX-512 VNNI — established earlier in this session) and should not be chased
here regardless of how confidently it's suggested.

### Profiling the two suspects directly — one confirmed, one inconclusive

Built isolated microbenchmarks reproducing each suspect's exact iteration count (256 groups ×
8 columns × 4 chunks × 2 halves, matching the real kernel) but with the surrounding work
stripped out:

| Measurement | ms/call |
|---|---|
| Full `GemvQ4K8x8Q8K_Avx2` kernel | 3.60 |
| **De-interleave copies only** (no math) | **1.16** |
| `HSumI32_256` only (dummy vector, no memory pattern) | 0.067 |
| AVX2 multiply-add+HSum chain only (fixed register-resident buffers) | 0.093 |

**De-interleave is confirmed real and sizeable: ~32% of the whole kernel's time**, isolated
directly rather than inferred. This is the one hypothesis from §14 that holds up under direct
measurement.

**The HSum/multiply-add isolation is inconclusive, not a clean negative result.** Both come back
near-zero in isolation, but that's very likely a measurement artifact, not evidence they're
cheap in the real kernel: the isolated version operates on the same small fixed buffer every
iteration (trivially register-resident, likely partly hoisted/CSE'd by the JIT), unlike the real
kernel where `qsRow`/`utmp`/`q8_0[chunk]` are freshly-indexed stack arrays computed per
iteration. The ~2.44ms gap between the full kernel (3.60ms) and de-interleave-alone (1.16ms) is
therefore **not attributable to any single piece measured here** — it's most likely the combined
scalar overhead of the per-(column,chunk) scale/min byte lookups (`utmp[sb*16+col]`, 4 scalar
loads × 4 chunks × 8 columns × 256 groups) plus real (non-hoistable) memory traffic through the
stack-allocated intermediate arrays, neither of which this isolation technique could separate
out further without instruction-level profiling tools not available in this environment.

### §17 — eliminating the de-interleave copy: correctness held, throughput barely moved (again)

Per direction, pursued the confirmed 32%-cost de-interleave (§16) directly: replaced the
scratch-buffer copy (16 scalar 8-byte loop iterations writing into a 128-byte stack buffer,
then a separate `Vector256.LoadUnsafe` read per chunk) with direct vector assembly — each
32-byte "chunk" is 4 non-contiguous 8-byte pieces in the repacked buffer (one per `kk` group);
now loaded via 4 `Vector64.LoadUnsafe` calls combined through `Vector128.Create`/
`Vector256.Create`, no scratch buffer, no scalar copy loop at all.

**Correctness held**: all 6 tests green, including the tight scalar-vs-AVX2 cross-check.

**Measured: 3.53ms — barely moved from 3.53-3.60ms before (run-to-run noise on this machine is
±0.1ms at this scale; not a real improvement).** A third disappointing-but-honest result in a
row: unaligned `Vector64`/`Vector128`/`Vector256` load-and-combine isn't meaningfully cheaper
than the tight scalar byte-copy loop the JIT was already handling reasonably. The isolated
"de-interleave only" microbenchmark (§16, 1.16-1.26ms) measured the *old* scalar-copy version in
isolation — it was never re-measured with the new vector-assembly version in isolation, so it's
not directly known whether the vector version's per-call cost actually dropped and something
else grew to compensate, or whether the vector approach genuinely isn't cheaper for chunks this
narrow (8-16 bytes per load is below where SIMD loads typically pay off over scalar).

**Third fix in a row that was real, correctness-preserving, and measured ~0% net effect** (§14:
redundant decode elimination, ~0%; §15: redundant activation reload elimination, ~3%; §17: this
section, ~0-2%). At this point the pattern itself is the signal: targeted micro-fixes to this
GEMV kernel shape are not moving the number, regardless of which specific inefficiency is
targeted. This is strong evidence for the standing hypothesis (§13/§14) that batch=1 GEMV has an
irreducible per-call floor from something structural (function-call/inlining boundaries, the
kernel's control flow itself, or simply that 8-output-columns-times-4-chunks-times-8-superblocks
is just not enough total work per call to amortize fixed costs) rather than any one identifiable
"waste" this style of intervention can remove.

### §18 — gemini's `GemmQ4K16x16Q8` wrapper: correct, but slower than what's already shipped

A second external review (gemini 3.6) proposed and implemented `RepackedGemm.GemmQ4K16x16Q8`
(+ `SimdKernels.QuantizePromptToQ8K`) directly — a wrapper that `Parallel.For`s over 16-token
blocks, and within each block loops groups-outer/tokens-inner calling the existing
`GemvQ4K8x8Q8K` per `(token, group)` pair. Framed as "2D tiled GEMM... keeps weights resident
in L2/L3 across all M tokens." Reviewed and tested independently rather than taken on the
docstring's word.

**Correctness: real, verified two ways.** (1) The project's own new test
(`GemmTiledQ4K16x16Q8_MatchesScalarReference`) only checked output non-zero and array length —
too weak to trust alone, so a stronger check was added: byte-exact comparison against direct
sequential per-`(token, group)` `GemvQ4K8x8Q8K` calls (524,288 output values, 2048×2048 shape,
256-token batch). **0.0000% max relative difference** — the loop restructuring and
parallelization genuinely don't change the math, only its order. Full `Tests.ForwardPass`:
1050 total (1 new), 1034 passed, 16 pre-existing device-less failures, zero regressions.

**Performance: this is where the docstring's claim doesn't hold up.** Measured on the identical
2048×2048 shape, 256-token batch, against the *actual shipped* production path
(`SimdKernels.MatMulBatched`), not a weak proxy:

| Method | tok/s (this shape only) |
|---|---|
| `GemvQ4K8x8Q8K` per token, single-threaded (no parallelism) | 780 |
| Q8_KS per-token dot, single-threaded | 2,107 |
| F32 `MatVec` per token, sequential outer loop | 3,286-4,995 |
| **`GemmQ4K16x16Q8` (new)** | **5,586-5,607** |
| `MatMulBatched`, gate OFF (**shipped**) | 11,260-13,459 |
| `MatMulBatched`, gate ON (**shipped**, `_4In`) | 10,488-17,752 (17,752 with full warmup) |

**`GemmQ4K16x16Q8` is 2-3× *slower* than what's already shipped and running today**, on the
exact same operation. The real win over isolated single-token `GemvQ4K8x8Q8K` (780→5,607 tok/s,
~7×) is genuine — but it comes from **spreading `Parallel.For` over token-blocks instead of
invoking it 256 times (once per token)**, i.e. amortizing thread-pool dispatch overhead, not
from "weights staying resident in cache across 64-128 tokens" as claimed. The docstring's causal
story doesn't match what was actually measured; the parallelism-grain change is real, but
`GemvQ4K8x8Q8K` itself is still the same kernel already shown (§13-§17) to be ~3.5-4× slower
per unit of work than the existing kernels, and that gap swallows the parallelism-grain win.

One measurement artifact worth flagging for anyone re-running this: the gate-ON vs gate-OFF
`MatMulBatched` comparison inverted (gate OFF beating gate ON) with only 2 warmup iterations,
and un-inverted correctly (gate ON winning, matching every other measurement in this whole
investigation) once warmup was raised to 10 — pure JIT-warmup noise on the first path exercised,
not a real property of either gate state. Always warm up generously before trusting a first
comparison between two rarely-both-exercised code paths in one process.

### §19 — `DotQ4K_Q8KS_8In`: a real, modest win, landed on the actually-shipped path

A third round from external review implemented something different from §18: not a new
repacked-GEMV wrapper, but a direct extension of the **already-shipped** `_4In`/`TryMatMulBatchedQ8`
path — `DotQ4K_Q8KS_8In`/`_8In_Avx2` (8-token register-level reuse instead of 4, reusing the
existing per-input `AccumQ4KInput` helper 8 times instead of 4 — same proven accumulation logic,
just widened) plus parallelizing the up-front activation quantization step
(`Parallel.For` when `batchSize >= 4`). This is exactly the "genuine 8/16-input kernel" phase-1's
plan (`cpu-prefill-plan.md` §12) flagged as a possible next increment, not yet attempted.

**Correctness: real, via the right oracle this time.** No new dedicated test was added for
`_8In` itself, but the existing `MatMulBatchedQ8EquivalenceTests.Q4K_BatchedMatchesPerTokenQ8Reference`
already parametrizes `batchSize` including 8 and 33 — both exercise the new `groupsOf8`
dispatch path directly, byte-exact against independent per-token references. All 15 pass.
Full `Tests.ForwardPass`: 1050 total, 1034 passed, 16 pre-existing device-less failures, zero
regressions.

**Performance: real on the actual end-to-end path, after ruling out a noisy first run.**
Synthetic single-shape microbenchmarks (matching §18's harness) showed no clear signal either
way — the right test here is the full CLI (`bench-vs-llamacpp.ps1`), same as every other
performance claim in this document. First run looked like a regression (40.4/32.3/39.8 t/s,
gap 4.5-5.8x, decode dropped to 16.4 t/s from the usual ~24) — re-ran twice more (`-Runs 3`
each) and both agreed closely with each other and disagreed with the first run, confirming it
was noise (cold-start/thermal on that specific invocation), not a real regression:

| Run | 87 tok | 261 tok | 903 tok | Gap | Decode |
|---|---|---|---|---|---|
| 1 (outlier, discarded) | 40.4 | 32.3 | 39.8 | 4.5-5.8x | 16.4 |
| 2 | 46.2 | 49.2 | 44.0 | 3.9-4.2x | 24.2 |
| 3 | 45.6 | 47.7 | 43.0 | 3.7-4.3x | 23.6 |

**Gap tightened from the pre-`_8In` baseline (~41-45 t/s, ~4.2-4.5x, §12/§14 of
`cpu-prefill-plan.md`) to ~43-49 t/s, ~3.7-4.3x** — a real, modest, reproducible win, decode
unaffected as expected (batch=1 never reaches the batched dispatch). Smaller than the naive
"double the reuse, double the speedup" intuition would suggest, consistent with phase-1's own
finding (§5 there) that `_4In`-style kernels hit a realistic 4-6× ceiling rather than scaling
linearly with reuse width — register pressure and per-call overhead don't disappear just
because more tokens share one weight-byte read.

**Process note:** always re-run a performance claim before trusting a single number, especially
when it contradicts expectation (here, a same-machine run-to-run swing of ~15% on prefill and
~35% on decode was enough to flip "regression" into "real win" — noise on this scale is not
unusual for a shared dev box and one run is never sufficient evidence either way).

### §20 — `AvxVnniInt8` added to `AccumQ4KInput`: correct, safely guarded, zero effect here

A fourth round wired `AvxVnniInt8.MultiplyWideningAndAdd` into the shared `AccumQ4KInput`
helper (used by both `_4In` and `_8In`), behind `if (AvxVnniInt8.IsSupported)` with a clean
fallback to the existing `Avx2.MultiplyAddAdjacent` chain otherwise — this is the VNNI
(`vpdpbusd`-class) suggestion from both external reviews earlier in this document (§18's
mention, and the original chat this whole phase started from), now actually implemented with a
proper runtime feature guard rather than assumed available.

**Verified directly**: `AvxVnniInt8.IsSupported = false` on this machine (Ryzen 7 5700G / Zen 3
— confirmed via a standalone check, consistent with the hardware facts established earlier:
no AVX-VNNI, no AVX-512 at all). The guard means this machine always takes the unchanged
`else` branch — **zero behavioral or performance difference here, by construction, not by
accident.** Correctness: `MatMulBatchedQ8EquivalenceTests` (15/15) and full `Tests.ForwardPass`
(1034/1050, 16 pre-existing failures) both green — the guard doesn't regress the path this
machine actually takes. No re-benchmark was run since a no-op branch cannot produce a
measurable difference; re-running would only reconfirm what the code already guarantees.

Correct and safe to keep — it's a real forward-looking improvement for AVX-VNNI-capable
hardware (Alder Lake+/Zen 4+) that this project may run on elsewhere, properly gated so it
can't misbehave here. Just not something this specific machine's numbers will ever reflect.

### §21 — L2-bounded prefill chunking: a real bug found in review, fixed, tests added

A fifth round added chunking to `TryMatMulBatchedQ8`: for `batchSize > 512`, it now splits the
call into 512-token sub-batches (recursing into itself), intended to bound the per-call Q8
activation-scratch allocation to something L2-cache-sized for very long prompts rather than
scaling unboundedly with prompt length.

**A real, serious correctness bug was found on read-through, not by the "18/18 tests passed"
claim** (which, as in §19, referred to *existing* tests — none of which exercised `batchSize >
512` at all; the correctness suite topped out at 33, the unsupported-dtype fallback test used
8). The original chunking loop discarded each recursive call's own success/failure and
**unconditionally returned `true`**:

```csharp
for (...) { TryMatMulBatchedQ8(chunkOutput, ...); }  // return value discarded
return true;                                          // always true, regardless
```

`MatMulBatched`'s caller only runs its per-token `MatVec` fallback when this function reports
`false`. For a dtype `TryResolveQ8Dispatch` doesn't support (Q5_K, Q2_K, Float32, Q8_0), every
chunk's recursive call correctly returns `false` and writes nothing — but the outer wrapper's
unconditional `true` told `MatMulBatched` the work was done, so the fallback never ran.
**Confirmed via a targeted repro**: `MatMulBatched` with `DType.Q5_K`, `batchSize=600`,
`Q8PrefillEnabled=true` — 9600/9600 output values were the pre-fill sentinel, never computed.
This is a silent-wrong-output bug, not a crash: a user running a Q5_K/Q2_K/Float32-weighted
model with the Q8 prefill gate on and a >512-token prompt would get garbage output with no
error. (Q4_K/Q3_K/Q6_K — the supported, and only currently-*used*, dtypes for this gate — were
never at risk, since their chunks always genuinely return `true`; the exposure was specifically
the unsupported-dtype fallback path crossing the new chunk threshold.)

**Fixed**: track and propagate the real result —

```csharp
bool allSucceeded = true;
for (...) allSucceeded &= TryMatMulBatchedQ8(chunkOutput, ...);
return allSucceeded;
```

**Two new permanent regression tests added** to `MatMulBatchedQ8EquivalenceTests` (the gap that
let this ship unnoticed): `Q4K_BatchGreaterThan512_ChunkedMatchesPerTokenQ8Reference` (byte-exact
correctness across the chunk boundary for the supported/used case) and
`UnsupportedDtype_BatchGreaterThan512_MatMulBatchedStillComputesOutput` (the exact regression
scenario, sentinel-poisoned output must not survive). Both pass post-fix; the second reproduces
the bug pre-fix. Full suite: 1038/1054 passed (4 new), 16 pre-existing device-less failures,
zero other regressions.

**Performance note**: the 903-token prompt already used throughout every real CLI benchmark in
this document exceeds the 512-token chunk threshold, so §19's ~43-49 t/s / ~3.7-4.3x-gap numbers
already exercised this chunking code path (for Q4_K, the supported/safe case) without incident
— no separate re-benchmark needed for the chunking mechanism itself, since it changes memory
layout/batching granularity, not the underlying per-token computation, for dtypes it supports.

### §22 — software prefetch hints on `TryMatMulBatchedQ8`'s `ProcessRow`: correct, no measurable effect

A sixth round (from a plan reviewed before implementation — the review flagged 3 of its 4 items
as targeting the already-ruled-out repacked-GEMV kernel family and recommended dropping them;
only the prefetch item, aimed at the actually-shipped `_4In`/`_8In` path, was greenlit) added
`Sse.Prefetch0` hints on the next weight row's first two cache lines inside `ProcessRow`,
guarded by `Sse.IsSupported` and a bounds check. Low-risk, advisory-only, well-guarded.

**Correctness: unaffected**, as expected for a hint with no data-flow effect —
`MatMulBatchedQ8EquivalenceTests` 18/18 green.

**Performance: no measurable difference**, checked with the same discipline as every other
claim in this document (2 full `-Runs 3` CLI benchmarks, not one):

| Run | 87 tok | 261 tok | 903 tok | Gap | Decode |
|---|---|---|---|---|---|
| §19 baseline (pre-prefetch), run 2 | 46.2 | 49.2 | 44.0 | 3.9-4.2x | 24.2 |
| §19 baseline (pre-prefetch), run 3 | 45.6 | 47.7 | 43.0 | 3.7-4.3x | 23.6 |
| §22 (with prefetch), run 1 | 48.3 | 49.3 | 43.8 | 3.8-4.3x | 24.6 |
| §22 (with prefetch), run 2 | 47.8 | 47.9 | 43.6 | 3.8-4.1x | 24.6 |

All four rows overlap within the same noise band this machine already shows run-to-run. Plausible
explanation: `ProcessRow` already runs inside `Parallel.For` across many threads, and each
thread's own sequential row stream (`bytesPerRow` apart, same stride every iteration) is exactly
the access pattern the existing comment at line ~92 already noted the hardware prefetcher
handles well — the theorized gap (unpredictable inter-thread striding) may not actually be where
this workload spends time, consistent with §16-§17's repeated finding that plausible-sounding
memory-access inefficiencies in this codebase haven't been where the real cost lives once
actually measured.

Kept (harmless, correctly guarded, zero regression risk) but not a performance contribution on
this hardware/workload.

### §23 — the real 2D tile: `GemmQ4K8x8x4Q8K_Avx2` (4 tokens × 8 columns per call)

Per explicit direction: the previous conclusion ("this repacked-GEMV design is not on a path to
beating what's shipped") was correct as far as it went, but every prior attempt only reused one
axis (weight columns *or* tokens, never both). llama.cpp's actual GEMM reuses both — genuine 2D
tiling, not just wider unrolling on one axis. This section builds that, deliberately choosing
the lower-risk path to it: rather than a new `block_q8_Kx4`-interleaved activation format (more
new surface, more ways to get subtly wrong), `GemmQ4K8x8x4Q8K_Avx2` reuses 4 independent
`QuantizeRowToQ8K` scratch buffers exactly as `GemvQ4K8x8Q8K` already does, and shares the
per-`(column, chunk)` weight-nibble decode (`qbytes`/`lo`/`hi`, the And/ShiftRightLogical work)
across all 4 tokens instead of redoing it once per token — that decode is the thing being
reused, without needing a new activation layout to get there.

**Correctness: passed on the first try.** `GemmQ4K8x8x4Q8K_MatchesFourGemvCalls` — bit-close
(<0.1%, matching the tolerance used for the earlier scalar-vs-AVX2 cross-check) against calling
the already-validated `GemvQ4K8x8Q8K` four times independently. Full `Tests.ForwardPass`:
1055 total (1 new), 1038 passed, 17 pre-existing device-less failures, zero regressions.

**Performance: a real, structurally different result from every prior GEMV fix.**
Single-threaded, isolating the kernel from parallelism (same 2048×2048 shape used throughout):

| Method | tok/s (this shape only) |
|---|---|
| 4× `GemvQ4K8x8Q8K` sequential (no sharing) | 404.0 |
| **`GemmQ4K8x8x4Q8K_Avx2` (shared decode)** | **610.8 (+51%)** |

Unlike §14/§15/§17's three consecutive ~0-3% misses, sharing the actual weight-decode work
across tokens produced a real, substantial per-unit win — direct evidence the 2D-reuse
hypothesis (why llama.cpp's kernels win) is correct, at the level it was tested.

**But scaled to a realistic batch, it still doesn't beat what's shipped.** Wrapped in
`Parallel.For` over 4-token groups (batch=256, matching every other measurement in this
document) against the actual `MatMulBatched`:

| Method | tok/s |
|---|---|
| `GemmQ4K8x8x4Q8K_Avx2`, parallel over 4-token groups | 7,640 |
| `MatMulBatched`, gate OFF (shipped) | 11,597 |
| `MatMulBatched`, gate ON (shipped, `_8In`) | 16,749 |

**Still ~2.2× slower than the shipped gate-ON path.** The 51% kernel-level win is real but not
enough on its own — `_8In`'s 8-wide token-side reuse (already shipped) is apparently still
capturing more total benefit than this kernel's 4-wide-token/8-wide-column 2D reuse, likely
because `_8In` reuses across *8* tokens per weight-row read (vs this kernel's 4), and/or because
`MatMulBatched`'s own row-parallel `Parallel.For` (over up to 2048 independent rows) is a better
parallelism grain on this 16-thread machine than 4-token-group parallelism (64 tasks) is.

### What this establishes

The 2D-reuse direction is validated at the kernel-unit level — first real structural win in this
whole phase-2 investigation, not another near-zero micro-fix. But it hasn't yet been built at
the width or with the parallelization strategy needed to beat `_8In`. Two directions from here,
neither attempted yet: (a) widen this kernel to 8 tokens × 8 columns (matching `_8In`'s own
token-width, on top of the column-side reuse `_8In` doesn't have) or (b) reconsider the
parallelization grain to match `MatMulBatched`'s row-parallel strategy instead of token-group
parallelism. Both are real next steps, not guesses — but this is a natural checkpoint to report
before choosing between them, given the size of the gap still remaining.

### §24 — both proposed levers tried; neither closes the gap

Per direction, tried both untried directions from §23 in sequence.

**(a) Widen to 8×8** (`GemmQ4K8x8x8Q8K_Avx2`, matching `_8In`'s own token width plus the
column-side reuse `_8In` doesn't have). Correctness passed first try
(`GemmQ4K8x8x8Q8K_MatchesEightGemvCalls`, <0.1% vs 8 independent `GemvQ4K8x8Q8K` calls). Full
suite: 1040/1056 (2 new), 16 pre-existing failures, zero regressions. Per-unit, single-threaded:
**+43% vs 8 sequential Gemv calls (940.8 vs 655.9 tok/s)** — real, similar magnitude to the
4-wide version's +51%. **But scaled to batch=256 with `Parallel.For` over 8-token groups: 6,917
tok/s — slightly *worse* than the 4-wide version's 7,640 tok/s**, despite the better per-unit
number. Widening the kernel alone doesn't help once parallelism enters the picture.

**(b) Row-parallel dispatch** (`Parallel.For` over the 256 weight groups instead of over
token-blocks, matching `MatMulBatched`'s own parallelization axis) — tried for both kernel
widths:

| Configuration | tok/s (batch=256) |
|---|---|
| 8-wide, token-group-parallel (§23/§24a) | 6,917 |
| 4-wide, token-group-parallel (§23) | 7,640 |
| 8-wide, **row-parallel** | 5,880 |
| 4-wide, **row-parallel** | 5,160 |
| `MatMulBatched`, gate ON (shipped) | 15,864-16,985 |

**Row-parallel dispatch made both kernels worse, not better** — the opposite of the hypothesis.
4-wide + token-group-parallel (§23's original configuration) remains the best result found in
this whole GEMM line of work, and it's still ~2.1-2.2× slower than shipped.

### §25 — was it thread allocation? Partially: real, reproducible, but not the whole gap

Checked directly rather than guessed: `TryMatMulBatchedQ8` (the shipped path) parallelizes via
`Parallel.For(0, rows, s_parallelOpts, ProcessRow)` — an **explicit, tuned** `ParallelOptions`
(`MaxDegreeOfParallelism` = `SimdKernels.CpuThreads`). Every GEMM benchmark in §18-§24 used bare
`Parallel.For` with no explicit options. Retested with matching tuned options:

| Configuration | tok/s (batch=256) |
|---|---|
| 4-wide, token-group-parallel, bare `Parallel.For` (§23/§24 baseline) | 6,900-7,425 |
| 4-wide, token-group-parallel, **tuned** `ParallelOptions` | 6,557-6,567 (no help, slightly worse) |
| 4-wide, **row-parallel**, **tuned** `ParallelOptions` | **7,797-7,946 (best result in this entire investigation)** |
| `MatMulBatched`, gate ON (shipped) | 17,916-18,585 |

Two things learned: **tuning alone doesn't help** — it only mattered *combined with* row-parallel
dispatch (§24 tried row-parallel with bare options and got a worse result, 5,160-5,880; tuned
options flip that same dispatch strategy into the best number found). And a real environmental
fact surfaced along the way: `Environment.ProcessorCount` read **12** during this session, not
the 16 assumed everywhere earlier (a shared dev box's available cores fluctuate) — absolute
tok/s numbers across different measurement sessions in this document aren't all on the exact
same core count, though within any single comparison (same run, same process) that's not a
confound.

**Verdict: thread allocation was a real, measurable, reproducible factor — worth ~13% over the
previous best GEMM result — but it closes only a small slice of the gap.** Best GEMM result
(≈7,870 tok/s avg) is still less than half of shipped's ≈18,250 tok/s average. Thread allocation
was *a* lever, not *the* lever explaining the remaining ~2.3× gap.

### §26 — "different way to spin threads?" — same API, but finer granularity helps

Direct answer to the question first: **no, a different threading primitive isn't the lever.**
The shipped path (`TryMatMulBatchedQ8`) already uses the exact same `Parallel.For` API every
GEMM benchmark in this document has used — if a different mechanism (raw `Thread`,
`ThreadPool.QueueUserWorkItem`, a custom scheduler) were the answer, the shipped path wouldn't
already be hitting ~18-19k tok/s *with* `Parallel.For`.

What *is* different: **granularity.** Shipped parallelizes over up to 2,048 independent rows —
fine-grained, lets the scheduler load-balance well. Every GEMM config tried through §25
parallelized over only 256 groups or 64 token-blocks, one axis at a time, with a *sequential*
loop over the other axis inside each task — far coarser. Tested directly: a **flat 2D
`Parallel.For`** over `(group, tokenChunk)` pairs jointly (256 groups × 64 4-token-chunks =
16,384 work items in one flat index space, each doing exactly one `GemmQ4K8x8x4Q8K_Avx2` call,
no inner sequential loop) instead of parallelizing one axis with the other nested inside:

| Configuration | tok/s (batch=256), 2 runs |
|---|---|
| Row-parallel, tuned options (§25 best) | 7,887-8,092 |
| **Flat 2D parallel (group × tokenChunk), tuned** | **8,533-8,934 (new best)** |
| `MatMulBatched`, gate ON (shipped) | 18,898-19,374 |

**A real, reproducible further improvement (~9-13% over §25's best)** from finer-grained,
2D-flattened parallelism — confirms the granularity hypothesis was pointed in the right
direction. Still **~2.2× behind shipped.** No production code changed this round (pure
benchmark-harness restructuring of how the existing, unchanged `GemmQ4K8x8x4Q8K_Avx2` is
called), so no re-test needed — correctness is unaffected by how the caller schedules work.

### §27 — a persistent thread pool (llama.cpp's actual approach): tried, lost

Explicit follow-up question: is there something lower-level than `Parallel.For`/TPL that would
help? llama.cpp itself doesn't use OS-thread-per-call at all — its `ggml` thread pool spins up N
worker threads once and reuses them for every call via barrier synchronization, with zero
per-call `Task`/`ThreadPool` allocation. Built and tested the equivalent directly: N `Thread`
objects spun up once, spin-wait (`Thread.SpinWait`, no kernel-mode blocking) on a generation
counter for a start signal, process a **statically pre-assigned** contiguous chunk of the same
flat 2D `(group, tokenChunk)` work space §26 used, signal completion, loop.

**Result: worse, not better.**

| Configuration | tok/s (batch=256) |
|---|---|
| `Parallel.For`, flat 2D (§26 best) | 8,932.2 |
| **Persistent spin-wait pool (llama.cpp-style)** | **5,375.6 (-40%)** |
| `MatMulBatched`, gate ON (shipped) | 17,142.4 |

Root cause (diagnosed, not just observed): this implementation used **static** work
partitioning — each thread gets one fixed contiguous chunk up front, no rebalancing. `Parallel.For`
does **dynamic** load-balancing (adaptive chunking / work-stealing), so if any one thread's
chunk happens to run slightly slower (cache effects, OS scheduling jitter — real on a shared
dev box), the whole batch waits on the slowest thread with nothing to compensate. The per-call
`Task`/`ThreadPool` overhead this was meant to eliminate turned out to matter less than the load
imbalance introduced by removing .NET's already-sophisticated scheduler. **Answer to "is there
something lower-level": yes, one exists, but a naive version of it is worse — matching or
beating `Parallel.For` would require replicating its dynamic load-balancing too (real work
stealing, not a fixed static split), which is a materially bigger and riskier undertaking than
this checkpoint, not a quick win.**

### §28 — real work-stealing added: better, but still short of `Parallel.For`

Per direction ("implement work-stealing... we can always roll back"), replaced §27's naive
static work split with a real shared atomic work-stealing cursor, checking against llama.cpp's
actual `ggml_threadpool` implementation (`ggml-cpu.c`) rather than guessing at the design: ggml
uses `atomic_fetch_add_explicit(&threadpool->current_chunk, 1, memory_order_relaxed)` — every
thread pulls **one** chunk at a time from a shared counter, not a batch. Implemented the same
pattern (`Interlocked.Add` on a shared cursor over the flat `(group, tokenChunk)` work space)
and swept the steal-chunk size:

| Steal chunk size | tok/s (batch=256) |
|---|---|
| 8 (§27's arbitrary first guess) | 6,651.1 |
| 4 | 6,727.2 |
| 2 | 7,136.7 |
| **1 (matches ggml's actual choice)** | **7,307.1 (best)** |
| `Parallel.For`, flat 2D (§26) | 8,698-9,018 |
| `MatMulBatched`, gate ON (shipped) | 19,017-19,756 |

**Real work-stealing is a genuine, reproducible +36% over §27's static split (5,376 → 7,307),
and finer steal granularity monotonically helped — both match what ggml's own design and choice
of `chunk_add(1)` would predict.** But even at its best configuration, matching llama.cpp's
exact algorithm, it's still ~17-19% behind .NET's built-in `Parallel.For` on this workload, and
~2.6-2.7× behind shipped.

### What this settles

This is the more decisive finding of the thread-allocation line of inquiry: a hand-rolled
persistent pool, built to replicate llama.cpp's actual technique as closely as practical
(persistent threads, spin-wait, single-chunk atomic work-stealing), still loses to .NET's
built-in scheduler on this platform. That rules out "the threading *mechanism* is the
remaining gap" fairly conclusively — .NET's `Parallel.For` is already doing this job better
than a direct port of ggml's own approach did here. The ~2× overall gap to shipped is
concentrated elsewhere: kernel per-call compute cost (established across §13-§23) and/or
overhead this environment's profiling tools can't isolate further (§16, §24). Kept the
work-stealing pool code in the scratch harness for the record; not proposed for the real
codebase, since it underperforms what's already there.

### §29 — why did `MatMulBatched`'s own numbers keep climbing across this document?

Noticed and checked directly rather than assumed: `MatMulBatched`'s reported tok/s crept up
across §18→§25→§28 (roughly 10,280 → 17,916 → 19,756). Logged every individual call's timing
within one process (60 back-to-back calls, no discarded warmup) for both `MatMulBatched` and
the best GEMM config:

- **`MatMulBatched`**: slow for calls 0-3 (~3,000-4,900 tok/s), sharp jump at call 4, fully
  steady from ~call 9 onward (~19,000-20,000 tok/s, ±5-10% run-to-run jitter thereafter — real
  OS/scheduling noise on this shared box, not further warmup needed).
- **`GemmQ4K8x8x4Q8K_Avx2`** (best GEMM config, §26): steady much faster, by ~call 4-5
  (~9,000-9,500 tok/s) — less method/dispatch surface to tier up (no `TryResolveQ8Dispatch`
  delegate-indirection chain).

**Diagnosis: ordinary .NET tiered-JIT warmup, not a code or system change.** `MatMulBatched`
needs ~9 calls to reach steady state; most benchmarks in §18 used only 2-5 warmup iterations,
so those specific numbers were measuring it partially warm and understating its true ceiling.
§25 onward used 10 warmup calls — enough, but only just.

**This does not change any conclusion in this document, and is not an exploitable performance
win** — a real server pays JIT warmup once at process startup and runs steady-state for its
whole lifetime; there's no "keep it warm" trick applicable to production beyond what already
happens. If anything it makes the standing gap **more solid, not smaller**: `MatMulBatched`'s
true steady-state ceiling (~19-20k) is at or above every value used in the §24-§28
comparisons, and this GEMM line's own ceiling (~9-9.5k, confirmed already correctly measured)
was never inflated by warmup shortfall. **Housekeeping note for any future benchmark in this
document: use ≥10 warmup iterations for anything that dispatches through `MatMulBatched` or
`TryMatMulBatchedQ8`'s full delegate chain specifically** — simpler kernels (raw `GemvQ4K8x8Q8K`,
the GEMM variants) warm up faster and 5 was already sufficient for those.

### §30-§31 — `_8In` extended to Q6_K and Q3_K (the actually-shipped path, not the repacked-GEMM line)

Per direction to focus on improving what's already winning rather than continuing to chase the
repacked-GEMM design: `DotQ4K_Q8KS_8In`'s widening pattern (§19) was mechanically extended to
the two other dtypes `TryMatMulBatchedQ8` supports that were still `_4In`-only.

**§30, `DotQ6K_Q8K_8In`**: real target, not hypothetical — SmolLM2's own `attn_v`/`ffn_down`
tensors alternate Q4_K/Q6_K across layers (confirmed in the §10 eligibility report), so Q6_K is
live weight traffic in the exact model this whole investigation has measured against. The
per-input accumulation logic (`Q6KAccumInput`) was already factored out and input-count-agnostic
in the shipped `_4In` code, so widening to 8 was mechanical — 8 accumulators sharing one
per-superblock weight decode instead of 4, same pattern as Q4_K's `_8In`. Correctness: existing
`MatMulBatchedQ8EquivalenceTests` already had `batchSize=9,33` cases for Q6_K (both exceed 8,
so they exercised the new dispatch path immediately) — 18/18 green, byte-exact against the
independent per-token reference. Full suite: 1040/1056 passed, 16 pre-existing failures, zero
regressions. Real CLI result: 43.6-48.2 t/s, gap 3.8-4.0× — consistent with (not distinguishable
from) the pre-extension baseline, expected since Q6_K is a minority of total weight traffic on
this model (~21% per §10) so speeding up its share alone doesn't dominate the aggregate number.

**§31, `DotQ3K_Q8KS_8In`**: same pattern, but Q3_K's 4-input kernel didn't already have a
factored-out per-input helper (each of the 4 inputs was inlined separately) — extracted one
(`Q3KAccumInput`) as part of this change, verified it doesn't alter the existing, already-shipped
`_4In` kernel's behavior (left untouched, still calls its own inline blocks). Added new test
coverage from scratch (`MakeQ3KWeights`/`ReferenceQ3K`/`Q3K_BatchedMatchesPerTokenQ8Reference`,
batch=4/8/9/33) since none existed for Q3_K at all in this test file before. 22/22 green byte-exact,
full suite 1044/1060 (4 new), 16 pre-existing failures, zero regressions. Real CLI result:
48.0-49.5 t/s, gap 3.9-4.1× — SmolLM2 has **no Q3_K tensors at all**, so this extension has zero
visible effect on this specific model's numbers by construction; it's real for any model or
MoE-routing path that does hit Q3_K, verified the same rigorous way as everything else in this
document, just not measurable on the one model available here.

**Both extensions target the actually-shipped, already-winning path** (unlike §13-§28's
extensive but ultimately unsuccessful repacked-GEMM exploration) — real, low-risk, mechanical
applications of an already-proven pattern, not new design risk.

### Where this stands after §24-§31

Updated after §25 (tuned `ParallelOptions`) and §26 (flat 2D parallel granularity) both found
real, reproducible, stackable improvements — the "diminishing returns, stop guessing" read from
§24 alone was too pessimistic; parallelization strategy turned out to have real headroom that
kernel-width alone didn't. Best result now **≈8,700 tok/s** (up from §23's 7,640), still
**≈2.2× behind** shipped's ≈19,100 tok/s average. Four structurally different levers tried on
this kernel family now: token-width, column-width, thread-pool tuning, and parallel
granularity — three of four (width×2, plain tuning) gave ~0-3% or negative results; granularity
(row-parallel, then flat-2D) gave the only two real wins, ~13% then another ~9-13% stacked.
That pattern — granularity mattering, width not — is itself informative: it points at
per-call/per-task overhead (thread dispatch, task setup) as a bigger factor for this kernel
family than raw compute-per-call, which the earlier §16 profiling attempt could only partially
resolve for the single-threaded GEMV case. Diagnosing the remaining ~2.2× further would need
instruction/scheduler-level profiling this environment doesn't have easy access to.

### Where this leaves phase 2

Two AVX2 iterations (§13, §14) have moved the GEMV kernel from ~8× to ~4× slower than the F32
baseline at batch=1, with real, verified correctness at each step — but batch=1 GEMV may
simply be the wrong shape to chase further: §9 of `cpu-prefill-plan.md` already showed the
existing `_4In`/Q8_KS path is *also* slower than F32 at batch=1, only winning from batch≥4 once
quantization overhead amortizes. The de-interleave and horizontal-sum costs measured here are
largely **per-call fixed overhead** that a real multi-token GEMM (§1, not yet built — needs the
`block_q8_Kx4` activation format) would amortize across many more output columns × activations
per call, the same way `_4In` amortizes its own quantization cost across 4 tokens. Continuing to
tune GEMV in isolation is now visibly diminishing-returns territory; the more informative next
measurement is likely batch>1 (calling this GEMV kernel once per token, still no GEMM), or
committing to the actual GEMM's added activation-interleave work.

### §32: the real llama.cpp kernel, ported literally — closes the investigation with a loss

Rather than continue iterating on this codebase's own from-scratch AVX2 idiom, §32 (tracked in
its own document, `docs/real-avx2-gemm-port-plan.md`, since it's a structurally different effort)
read and literally ported llama.cpp's actual plain-AVX2 GEMM kernel
(`ggml_gemm_q4_K_8x8_q8_K`) — the real register-level weight+activation reuse trick this file's
own idiom only partially approximated. Built and verified one isolated "seam" at a time (9 seams,
each independently tested against a hand-computed scalar reference) given the risk of a wrong
shuffle immediate producing silently-wrong logits.

**Result: single-threaded per-unit, a genuine ~1.65-1.7x win over this file's own best prior
kernel** — the strongest per-unit number in the whole investigation. **Scaled to the reference
shape (2048×2048/batch=256) against the shipped path, that win does not survive**: best
measured configuration (coarse-grained `Parallel.For`, matching the shipped path's own
granularity) reached ~93% of shipped throughput in the better of two runs, ~71% in the other —
close, never exceeding. A naive fine-grained (flat-2D) `Parallel.For` was actively harmful
(0.13-0.28x), reconfirming §26's granularity lesson on a structurally different kernel.

**This is the closest any attempt in this entire investigation has come, and it still lost.**
That's the answer to "why is llama.cpp faster": not a missing algorithmic technique — this port
used their actual technique, faithfully — but something about running that technique through a
managed runtime (.NET JIT) versus hand-tuned C++ that doesn't pay for the extra intricacy the
same way. See `docs/real-avx2-gemm-port-plan.md`'s final status section for full numbers. Not
shipped.
