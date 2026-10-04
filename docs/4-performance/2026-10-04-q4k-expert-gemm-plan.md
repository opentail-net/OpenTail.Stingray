# Q4_K (and Q6_K) GEMM for CPU MoE expert prefill

Written 2026-10-04. **Status: plan only, nothing implemented.** Follows the CPU profile in
[2026-10-03-batched-moe-prefill-plan.md](../2-coverage/2026-10-03-batched-moe-prefill-plan.md) (section "CPU path profile and experiments").

## 1. Why this is worth planning

Every MoE prefill on this machine now runs on the CPU's batched expert path: pure CPU, the hybrid and the full-GPU
pass all hand a fresh prompt to it (CPU-prefill handoff, Phase 1/1b of that plan). Its speed is therefore the prefill speed
of every configuration, and today it is the slowest well-understood part of the stack.

Measured on OLMoE-1B-7B Q4_K_M, 740-token prompt, CPU only, Ryzen 5700G (8 cores, Zen 3: AVX2, no VNNI):

| Fact | Value |
|---|---|
| Prefill throughput | about 148-154 tok/s (for comparison, the dense Gemma models reach 0.95-0.97 of llama.cpp prefill speed on this box with the int8 batched path, see ADR-0003) |
| Share of prefill trunk time in the FFN | 79% (QKV 10%, attention 5%, output projection 3%) |
| Inside the routed-expert stage | gate matmul 37%, up matmul 35%, down matmul 27%; gather, activation, scatter each under 1% |
| Rows per expert per layer call | 93 on average (94,720 (token, slot) pairs over 1,014 expert runs) |
| Matmul time by dtype (worker-summed) | Q4_K 24.6 s, Q6_K 3.4 s |
| Achieved per worker | about 40 GFLOP/s on Q4_K, about 59 on Q6_K, against an fp32 peak of about 130 per Zen 3 core |
| Workers busy | 6.8 of 8 on average |

If the expert matmuls ran `s` times faster and nothing else changed, prefill would speed up by `1 / (0.21 + 0.79 / s)`:
1.22x at s = 1.3, 1.36x at s = 1.5, 1.66x at s = 2.0. Those are arithmetic, not predictions: whether `s` above 1 is reachable
is what the first step below decides.

## 2. What already exists (so we do not rebuild it)

| Piece | Where | Notes |
|---|---|---|
| Per-token int8-activation matvec and its batched siblings `MatVec2In` / `MatVec4In` | `SimdKernels.cs` | This is the **exact** path the expert loop uses today (`allowQ8: false`): bit-identical to token-by-token decode, asserted by `SimdKernelsQ8KSTests`. Its own comment measures it as about 87% dot/dequant compute, 13% weight streaming. |
| Batched int8 path `TryMatMulBatchedQ8` (`_4In` / `_8In`) | `SimdKernels.cs` | Quantises activations once per batch; used for MoE only with `STINGRAY_MOE_PREFILL_Q8=1`. Measured on OLMoE: +11%, identical text (not adopted). |
| Repacked Q4_Kx8 GEMM | `RepackedGemm.cs` (row-interleave transform ported from llama.cpp `make_block_q4_Kx8`), `RepackedGemmPath2.cs` (Q8_Kx4 activation quantiser plus 8-row GEMM), `ForwardPass.MatMulBatchedCached` / `GetRepackedQ4Kx8` | The dense default (ADR-0003): 2.2-2.9x over the per-token path on dense models (SmolLM2-1.7B 77.5 to 227 tok/s). Repacks each weight once and caches it. **MoE experts do not go through it.** |
| `Q6KPrefillGemm` | `ForwardPass.MatMulBatchedCached` | Dense Q6_K prefill GEMM. Also not used for experts. |
| `MicroGemmQ4K` | `MicroGemmKernel.cs` | Off by default (`STINGRAY_CPU_MICRO_GEMM`), at most 16 rows: never applies at 93 rows. |
| Shared expert loop | `MoeBatchedExperts.Run` | CSR-bucketed rows per expert, experts in parallel, per-worker buffers; the only caller of the expert matmuls in prefill. |

History that must not be repeated (all measured on this repo):
- Weight-stationary loop reordering (rows outermost, tokens tiled): **4x slower** (`SimdKernels.cs`, comment above `_4In`).
- Batch-1 repacked GEMV (`docs/done/cpu-prefill-repack-gemm-plan.md`): about 3.5x slower than the baseline; three rounds of micro-optimisation moved nothing. The plan there concludes the kernel shape, not the tuning, was the problem. The batch > 1 variant it tried (`GemmQ4K16x16Q8`) was still 2-3x slower than the shipped `MatMulBatched`.
- Routing MoE experts through the dense `MatMulBatchedCached` was never done, so there is no negative result for it.

## 3. Why experts are not on the fast dense path (and what that implies)

1. **Exactness.** `MoeBatchedExperts` defaults to the exact path since the Granite-4-H-Small investigation
   (`docs/done/13-granite4-h-small-moe-ppl-parity-plan.md`): int8 activation grouping in the batched expert path accounted for all
   of the batched-versus-per-token NLL difference. Note what that result is: **a reproducibility difference, not an accuracy
   loss.** Granite perplexity was 26.5483 (exact path), 26.4155 (int8 batched), 26.1080 (llama.cpp): the int8 batched number is
   the closer one. ADR-0003 accepted int8 batched prefill as the dense default on that basis. Applying it to MoE is a policy
   question for the project owner, not a technical one.
2. **Memory.** `GetRepackedQ4Kx8` keeps a repacked copy per weight. The Q4_Kx8 layout is the same size as Q4_K (8 rows x 144
   bytes = 1,152 bytes per group), so caching a repacked copy of every expert would duplicate the expert bytes (3.7 GB for OLMoE,
   far more for large MoE models). Not acceptable for the oversized-model goal.

Both points shape the options below.

## 4. Options

### Option A: reuse the existing repacked GEMM, repack each expert on the fly
Repack the used expert's three matrices into worker-local scratch (about 1.2 MB each for OLMoE; a copy costs far less than
the roughly 390 MFLOP matmul it feeds at 93 rows), run `TryMatMulBatchedQ4Kx8` (and `Q6KPrefillGemm` for Q6_K down), discard.
- Pros: kernels exist and are proven on dense models (2.6x there); no persistent memory cost; smallest amount of new code.
- Cons: **not bit-identical to the per-token path** (activation grouping differs, ADR-0003), so it needs the policy decision in
  section 3.1; the dense speedup may not carry over to 93-row, 1024x2048 shapes (to be measured, not assumed); repack cost
  and scratch traffic must be shown to be small.

### Option B: an exact tiled GEMM that keeps the per-token arithmetic
Bit-identical output is achievable because the arithmetic per output element decomposes cleanly: for each 256-wide block the
int8 dot product is an **integer** sum (exact in any order); only the float accumulation across blocks has an order, and it is
sequential per output. A kernel may therefore tile rows x tokens freely as long as each (row, token) accumulator is updated
block by block in the same order and with the same per-token activation quantisation (Q8_KS, which depends only on that token's
values, not on the batch). What it buys: each weight block's nibble unpacking, the dominant cost, is done once and reused for T
tokens held in registers (AVX2 has 16 ymm registers; the llama.cpp repacked kernels use 8 weight rows x several tokens).
- Pros: exact, so no policy question and the existing bitwise tests (`MatVec4In_BitwiseMatchesSingleMatVec` style) apply
  unchanged; reuses nothing risky.
- Cons: new kernel code on a path with several documented failures; speedup is unproven (the comment that the current kernel
  is compute-bound on dequant/dot suggests the unpack-sharing gain is real but its size is the open question).

### Recommendation
Do the measurement first (section 5, step 0) for **both**, because step 0 costs half a day and decides everything:
if A's ceiling is large, the question becomes the policy decision, and B becomes the answer only if the owner wants exactness;
if neither beats the current kernel by a clear margin on expert shapes, stop and record the result.

## 5. Plan

**Step 0: micro-benchmark on real expert shapes (about half a day, go/no-go).**
In a test project (not in the engine), take real OLMoE expert tensors (gate/up 1024 x 2048 Q4_K; down 2048 x 1024, half Q4_K and
half Q6_K) and realistic row counts taken from the profile (distribution with mean 93, include the extremes 1, 16, 93, 250).
Time, per call and as GFLOP/s per worker (single-threaded, then 8 workers each on a different expert, as the real loop runs):
1. today's exact path (`MatMulBatched`, `allowQ8: false`);
2. `STINGRAY_MOE_PREFILL_Q8` path (`TryMatMulBatchedQ8`);
3. repacked Q4_Kx8 GEMM with the repack done per call into scratch (Option A);
4. the same with pre-repacked weights (upper bound, shows the repack cost);
5. a prototype of Option B if the above leaves room.
Output: one table. **Go** if some variant is at least 1.4x faster than (1) at 93 rows with the repack cost included; otherwise
stop and write the negative result next to the earlier ones.

**Step 1: semantics decision (only if Option A wins).**
Measure NLL/perplexity and greedy tokens, exact path versus Option A, on OLMoE, Qwen3-Coder-30B-A3B and the Granite model that
raised the issue, using the corpus-perplexity harness ADR-0003 used (`docs/done/cpu-prefill-quality-gate.md`). Present the numbers;
the owner decides whether MoE follows ADR-0003 (default on, exact path as `STINGRAY_MOE_PREFILL_Q8=0`) or stays exact.

**Step 2: implement the winner.**
- Option A: a worker-scratch repack helper next to `MoeBatchedExperts`, dispatch by row count (below some N, keep the current
  path) and by dtype (Q4_K gate/up and Q4_K down; Q6_K down through `Q6KPrefillGemm` or left on the current path in a first
  version, it is 12% of matmul time on OLMoE).
- Option B: the tiled kernel behind the same dispatch, AVX2 first; ISA-dispatched so an AVX-512/VNNI machine can keep a better
  existing path.
- Either way: shapes the kernel cannot take (columns not a multiple of 256, rows not a multiple of the tile) fall back to the
  current path; scratch is per worker and sized from `maxCnt` like the existing buffers.

**Step 3: tests.**
- Option B: bitwise equality against per-token `MatVec` for every row count 1..64 and a spread up to 512, odd row counts, all
  three expert shapes; cross-checked by the existing batched-versus-per-token expert test.
- Option A: tolerance-based equality against the exact path plus the NLL gate from step 1; greedy-token equality on the three
  models.
- Both: the end-to-end parity tests that already cover MoE (CPU batched versus sequential, hybrid handoff, full-GPU handoff),
  thread-safety under the real parallel loop, no growth of private memory after repeated prefills.

**Step 4: end-to-end measurement.**
Repeat the Phase 0 matrix of the MoE plan (OLMoE; 41 / 168 / 740 / 2,597 tokens; CPU, hybrid with handoff, full GPU with handoff;
3 runs each) and the `[MoeExperts]` breakdown (`STINGRAY_PROFILE_PREFILL=1`). Record before and after in this document. Run the
Qwen3-Coder-30B case once as well: its experts have a different shape (768 x 2048, 128 experts; check their dtypes with
`list-tensors` first, a Q4_K_M file mixes Q4_K and Q6_K), and about 47 rows per expert at 740 tokens, the regime where tiling matters most.

**Effort (one engineer, focused):** step 0 half a day; step 1 one day including the harness runs; step 2 two to four days
(Option A closer to two, Option B closer to four); step 3 one day; step 4 half a day. Roughly a week if the step 0 gate passes,
half a day if it does not.

## 6. Risks and non-goals

- **The gain may not exist.** The history in section 2 is three negative results in a row for this kind of work. Step 0 exists
  to find that out cheaply.
- **One machine.** Zen 3, 8 cores, no VNNI. The kernel choice must stay ISA-dispatched, and numbers from here say nothing about
  AVX-512 or Apple silicon.
- **Load imbalance is not the issue.** Longest-expert-first scheduling was tried and made prefill 10% slower (the kernels
  parallelise internally); do not retry it as part of this work.
- **Not in scope:** decode (a different shape; the earlier claim that it is "memory-bound, already at its streaming limit" is UNSUBSTANTIATED as of 2026-10-04: no decode measurement supports it, and the 2026-10-04 MoE speed league has our decode at 0.21-0.59x of llama.cpp, see PerformanceLeague.md "MoE lot" and the decode investigation below), GPU kernels (see the MoE plan's
  Phase 3), other quantisation formats (Q5_K, IQ*), changing the routing or reduction order (slot-order reduction is a parity
  contract and stays).

## 7. Decisions needed from the project owner

1. Is an approximate-but-reproducible MoE prefill (Option A semantics, ADR-0003 applied to experts) acceptable, or must MoE prefill
   stay bit-identical to per-token decode (Option B)? Step 1 produces the evidence; this can wait until step 0 passes.
2. Should step 0 go ahead now? (It touches no engine code.)
