# Brief: independent investigation of OpenTail.Stingray CPU prefill performance

**For:** a second AI working independently of the session that produced today's results.
**Goal:** find remaining CPU prefill performance in OpenTail.Stingray, and challenge the conclusions
below. Disagreement backed by measurement is the most useful thing you can produce.

---

## 1. Where things stand

Box: Zen 3, 6c/12t, AVX2 only (**no AVX-512, no VNNI**), OpenBLAS **not installed**.
Model: `models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf`. Reference: `tools/llama.cpp/llama-completion.exe`.

Prefill gap to llama.cpp went **2.31x → ~1.14x** in one day:

| step | gap |
|---|---:|
| start | 2.31x |
| enable repack cache (`STINGRAY_Q4KX8_CACHE_MB=2048`) | ~1.85x |
| Path 2 (ported llama.cpp AVX2 GEMM) replaces Path 1 | 1.67x |
| route FFN gate+up through the repacked path (one `if`) | **1.14x** |

Decode is **1.26x** behind and was untouched — it runs an F32 `MatVec` path that never reaches any
of this.

Current phase breakdown (`STINGRAY_PROFILE_PREFILL=1`, 2431 tokens, all fixes active):

| phase | share |
|---|---:|
| FFN (batched GEMM) | 48.6% |
| **Attention (per-token, NOT batched)** | **30.3%** |
| QKV projection | 13.8% |
| Output projection | 4.1% |
| RoPE / RmsNorm / other | 3.1% |

**The current hypothesis is that attention holds most of the remaining gap** — implied ~1.6x slower
than llama.cpp's, which if true means closing it reaches parity. That hypothesis is one session old
and has not been independently checked. **Please attack it.**

Background documents: `docs/repack-gemm/port-log.md` (the full record, including every wrong turn),
`docs/repack-gemm/README.md` (source study of llama.cpp's GEMM), `docs/perf-loop-progress.md`
(64 prior iterations — read before re-treading anything).

---

## 2. How to turn the fast path on

```
STINGRAY_Q4KX8_CACHE_MB=2048   # REQUIRED. Without it the repacked path is unreachable.
STINGRAY_GEMM_PATH=2           # the ported GEMM; 1 or unset = incumbent
STINGRAY_PROFILE_PREFILL=1     # phase breakdown
```

`RepackedGemmPath2.EngagedCalls` returns how many times the ported GEMM actually ran. **Use it.**

---

## 3. Method — this matters more than the findings

Every significant result today came from one of these. Every significant *error* came from skipping one.

### 3.1 Instrument; do not infer

The first end-to-end A/B of the ported GEMM showed **zero** difference. That looked like a real
negative result. It was not: the code **never executed**, because the repacked path is disabled by
default. Four rounds of reasoning about chunk sizes and gates failed to find that; a two-line
`Console.Error.WriteLine` found it in one run.

**A null result is ambiguous between "no effect" and "never ran." Prove which before interpreting it.**

### 3.2 Isolated speedups do not predict end-to-end

Repeatedly, and in both directions:

- The repacked kernel is 2.6x isolated, +14% end-to-end (documented at `ForwardPass.cs:51`).
- Path 2 is 1.83x isolated over Path 1, 1.11x end-to-end.
- The vectorised activation quantiser was 1.19x on the matmul pass and **0%** end-to-end.
- Perf-loop iteration 24: a reproduced 2.4–2.6x isolated win became a ~12% end-to-end **loss**.

Always confirm with `scripts/bench-prefill-cli.ps1` or the CLI directly. Isolated numbers justify
*trying*, never *claiming*.

### 3.3 Re-examine what was already checked

**Three of today's biggest findings were things previously examined and written off.** Each was a
*correct* observation that quietly stopped being true when something else changed:

- The dual-Q8 gate for FFN gate+up was audited for correctness and left alone. When the repacked
  path arrived later, nobody re-asked whether dual-Q8 should still take precedence. It was starving
  ~55% of matmul FLOPs. Fixing the precedence was worth **1.52x** — more than the entire kernel port.
- `ForwardPass.cs:55-60` attributes two `PrefillPackedMulti_*` test failures to the repacked
  kernel's summation order. That diagnosis appears wrong: the real cause was the dual-Q8 gate's
  `N >= MinBatchForQ8Prefill` threshold making kernel choice batch-size-dependent. After the
  precedence fix, all three configurations pass 1191/1191.
- The repacked path being off by default is documented and deliberate — but its cost (1.25x) had
  not been re-measured since the surrounding code changed.

**"This was investigated" is not evidence it is still true.**

### 3.4 Read the doc comments on the code you are touching

Two errors today, and an entire wasted increment, came from not doing this. `ForwardPass.cs:48-62`
predicted and explained a "bug" that was hunted for an hour. The file was already open.

### 3.5 Grep the whole tree before declaring anything absent

Four times today something was called missing that already existed — including the CPU prefill
profiler itself (`PrefillProfileTimers`), which was declared nonexistent after grepping only the
env-var registry and the CUDA files.

### 3.6 llama.cpp is a design oracle, not just a benchmark

Sources are at `examples/cpp/llama.cpp`. Reading how it solves a problem repeatedly beat reasoning
from first principles. Two rules that came out of that:

- **Check which arm actually compiles.** The Q4_K GEMM is ~1450 lines, but half is AVX-512 and dead
  on this box. Only ~670 lines were live and relevant.
- **A/B llama.cpp against itself** to size a technique before porting it. `--no-repack` on
  `llama-completion` isolated the repack's worth at 1.51–1.66x, which is what justified the port.

---

## 4. Measurement traps in this repo (all cost real time today)

1. **`llama-bench` ignores `LLAMA_ARG_REPACK`.** It builds params from
   `llama_model_default_params()` and never calls `common_init_from_params`. An A/B through it
   measures identical code paths on both arms. Use `llama-completion`.
2. **`llama-cli` is interactive-only** (b8585). It rejects `-no-cnv` and blocks on stdin. Use
   `llama-completion` with `-st --simple-io`.
3. **`pwsh -File script.ps1 -Words 900,2400` concatenates** the array to the single integer
   `9002400`. Pass one value per invocation.
4. **`dotnet test --nologo` fails** on Microsoft.Testing.Platform with exit 5 / "Zero tests ran",
   which reads exactly like a discovery failure. Drop it. Filtering uses `--filter-class`, not
   `--filter`.
5. **Unregistered env vars print a stderr warning** that trips `$ErrorActionPreference = "Stop"` in
   the bench scripts. Register new ones in `KnownEnvironmentVariables.cs`.
6. **`DOTNET_TC_QuickJitForLoops=0`** is mandatory for any timing run; tiered JIT invalidates results.
7. **Interleave A/B arms within one pass** and take best-of. Interference here is one-sided. A
   comparison assembled from two separate runs was contaminated by a machine restart today and had
   to be withdrawn.
8. **Machine drift is real** — both engines measured ~4% slower across sessions. Trust *ratios*.

---

## 5. Suggested lines of attack

Ordered by expected value, but the ordering is a guess — challenge it.

1. **Attention is 30.3% of prefill and explicitly not batched.** llama.cpp computes prefill
   attention as batched GEMMs over all query positions. Ours walks tokens. Prior work
   (iterations 13/14/15/17/33/63) tuned tiling but did not batch it. Flash attention was tried and
   correctly rejected — it exists to work around GPU shared-memory limits and loses on CPU.
2. **Verify the claim that our matmuls now match llama.cpp's.** The whole "attention is the gap"
   conclusion rests on it, and it is an assumption, not a measurement. If matmuls are still behind,
   the attribution shifts.
3. **Anything else that takes precedence over the repacked path.** The dequant cache at
   `ForwardPass.cs:804` sits in front of it exactly as dual-Q8 did.
   `STINGRAY_PREFILL_DEQUANT_MB=0` measured +9% earlier — but that was taken when the repacked
   path was unreachable, so it needs re-taking.
4. **Per-call allocation.** Both paths `NativeMemory.Alloc`/`Free` activation scratch per matmul —
   roughly 120 alloc/free pairs of 1–10 MB per prefill chunk. Unmeasured. Measure before pooling.
5. **Decode (1.26x, untouched).** Different code path entirely (F32 `MatVec`). Nothing from today
   applies. For interactive use this matters more than prefill.

---

## 6. Ground rules

- **Do not commit.** Leave changes in the working tree.
- Verify builds before claiming anything: `dotnet build src/OpenTail.Stingray.Cpu -c Release`.
- Full suite before any completion claim: `dotnet test tests/OpenTail.Stingray.Tests.ForwardPass -c Release`
  (currently **1191/1191** in all three configurations).
- **One benchmark at a time.** Single machine — concurrent runs invalidate each other. If the other
  session is measuring, do not.
- Record negative results. Today's log deliberately keeps every wrong turn; they were more useful
  than the successes for calibrating what to trust.
