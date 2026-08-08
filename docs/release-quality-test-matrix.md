# Release-quality test matrix

Every published package needs a machine-readable test transcript and a completed row in this
matrix. Passing a test on an unsupported machine is not evidence for a hardware/backend row.

| Gate | Required evidence | Runs where | Release rule |
|---|---|---|---|
| Build and managed regression | Release build; Core, CLI, Pipeline, Server, TurboQuant, Sessions and Vision results | hosted CPU CI | Required for every package |
| CPU inference | dense greedy prefill/decode, batch admission, prefix reuse and numerical parity | AVX2 release runner | Required when CPU engine changes |
| CUDA dense | load, greedy/sample decode, batched decode, long-context KV dtype paths, capability JSON | designated NVIDIA runner | Required when CUDA/placement/KV changes |
| Vulkan | load, greedy decode, model-family smoke and capability JSON | designated Vulkan runner | Required when Vulkan/placement changes |
| Hybrid MoE | CPU-expert and GPU-cache paths, representative prefill/decode, planner output retained | designated CUDA MoE runner | Required when hybrid/MoE changes |
| Speculation | no-spec versus greedy MTP equivalence and explicit rejected-case checks | runner with an MTP GGUF | Required when speculative decoding changes |
| Session restart continuation | actual model, persisted state, process exit, new process, resumed turn, token-by-token full replay comparison | dedicated CPU dense runner | Required before sessions are advertised as supported |
| HTTP server | OpenAI, Anthropic, `/capabilities`, `/status`, queue limit and configuration assertions | hosted CPU CI; selected backend runners | Required when server/protocol changes |
| Packaging | pack, CLI startup/version, `inspect --json`, package contents and notices | release job | Required for every package |

## Release receipt

The publish workflow writes TRX files for the managed suites and uploads them as the
`release-quality-results` artifact. A reviewer links that artifact and records hardware rows
actually exercised in the release notes. A skipped model/hardware test remains skipped; it must
never be described as a pass. The release receipt also retains
`scripts/check-test-model-coverage.ps1` output, which names which fixtures were absent.

Model- and GPU-gated tests now call `Assert.SkipUnless` and are reported as **skipped**. They
previously `return`ed early, so an absent fixture produced a PASS and suite totals silently
overstated what had executed — the coverage transcript was the only way to tell. Read a suite's
skip count as the honest measure of what a fixture-less runner did not exercise.

## Restart-continuation acceptance test

The acceptance test must use a real GGUF, not a fake forward pass. It must:

1. Create a session and materialise at least two turns.
2. Persist its manifest and KV payload.
3. Stop the owning process without retaining an in-memory engine/cache reference.
4. Start a new process, load the same model/runtime ABI, and restore the session.
5. Generate one more greedy turn.
6. Compare every continuation token against a fresh full replay of the exact logged token IDs.

`HotSessionGreedyReplayTests.ColdSession_RealModel_CrossProcessRestore_MatchesFullGreedyReplay`
uses two child processes and fresh runtime objects: two turns are persisted before process exit,
then a third is generated after restore and every generated segment is compared to full replay.
That clears the reference
lane's proof and the CPU-dense named-session product surface: it is now lifecycle- and
capability-gated, with bounded completed-operation replay after restart. It does not prove every
backend/cache family; those remain separate rows rather than being inferred from the CPU result.

## History and release notes

Use [CHANGELOG.md](../CHANGELOG.md) as the human release index. Pull requests should use a concise,
imperative subject with a component prefix, for example `engine: preserve BF16 KV pages on restore`.
Avoid generic messages such as `ok`; add a changelog/release-note bullet rather than rewriting
shared history.

## Evidence status — 2026-08-07 (single developer machine)

Recorded so the gaps are legible rather than assumed. **This machine is a Ryzen 7 5700G: Zen 3,
AVX2 only (no AVX-512, no VNNI), integrated Radeon graphics, no NVIDIA GPU.** Per the rule at the
top of this document, a pass here is not evidence for a row whose runner this machine is not.

| Gate | Evidence today | Where it came from |
|---|---|---|
| Build and managed regression | **Yes** — Core 421/421, CLI 367/367, Server 246/246, Sessions 79/79, Release, 0 warnings | this session |
| CPU inference | **Yes** — dense greedy prefill/decode, numerical parity, corpus perplexity | `done/cpu-prefill-quality-gate.md`, `cpu-benchmark-llamacpp-baseline.md` |
| CUDA dense | **No — blocked.** No NVIDIA driver on this machine; nothing was run and nothing is claimed | — |
| Vulkan | **Partial.** Functional load + greedy decode, and CPU/Vulkan logit parity measured. **On an integrated APU, not a discrete card** — every number is APU-specific and does not predict discrete-GPU behaviour | `vulkan-backend-evidence.md` |
| Hybrid MoE | **Partial.** MoE model runs on CPU with prefill numerics in the dense band. The row's actual subject — CPU-expert and GPU-cache offload paths — is **untested**, needing constrained VRAM this machine cannot provide | `moe-backend-evidence.md` |
| Speculation | **Measured and negative.** Draft-model speculation is a 37% regression on CPU because the verify pass cannot amortise; no MTP GGUF present for the MTP equivalence check | `done/cpu-speculative-decoding-findings.md` |
| Session restart continuation | **CPU-dense GGUF lane proven.** Two real turns persist across child-process exit; a fresh runtime restores them, adds a third greedy turn, and exactly replays every generated segment (1/1 focused receipt, 2026-08-07). Other backend/cache families remain separate conformance work | `HotSessionGreedyReplayTests` |
| HTTP server | **Yes** — 246/246 including OpenAI, Anthropic, capabilities, status, queue limits | this session |
| Packaging | **Not run this session** | — |

### Defects found and closed while producing this evidence

- Generation was not bounded by the active context: decoding past the ceiling wrote past
  `_ctxLen`-sized native scratch, reachable from a client's `max_tokens` on the server.
- int8 prefill collapsed on single-repeated-token prompts (cosine 0.40-0.48, -0.124 for a repeated
  space), affecting ordinary words too. Fixed by routing such prompts to exact F32.
- `STINGRAY_VABL`, a registered switch whose own comment said it produced wrong output, was
  declared but referenced nowhere; `doctor` reported it to operators as valid.
- The server's queue-overload warning advised raising a variable nothing read.
- `STINGRAY_PER_LAYER_HD_PREFILL=1` entered a path that raised `AccessViolationException` rather
  than the "wrong output" its comment promised — the batched route indexes KV with the model-wide
  head dim on layers carrying a smaller one. It now fails fast. The switch existed to make the
  outstanding per-layer-head-dim work measurable, which it could never do.

### Known open, with evidence rather than suspicion

- **`committed_revision` is not a usable concurrency token.** The API advertises a value it then
  rejects: a live session reports `committed_revision: 6` after one turn and answers the next
  request carrying 6 with `409 "Expected revision 6, but current revision is 1"`. Read-then-send is
  the only pattern optimistic concurrency admits. **This is a live defect on a shipped HTTP
  surface** and should gate advertising sessions as supported. Both single-sided fixes were
  implemented and measured and both break something else; the correct fix is an on-disk
  persisted-revision decision. Full characterisation and reproduction in
  `done/session-revision-contract-defect.md`.
- Gemma 4 per-layer-head-dim models get **no batched prefill**: 3.8 t/s prefill against 3.7 t/s
  decode, where every other model measured runs prefill at 1.9-6.6x decode. Against the nearest size
  class (Qwen3-8B) it decodes 1.7x slower — explainable by parameter count — but prefills 9.7x
  slower, which is not, leaving roughly a **5.7x penalty beyond what size accounts for**. Confirmed
  in code at the `perLayerHdUnsupported` gate; see `cpu-performance-baseline.md`.
- Flash-64 head dims 128/256 remain opt-in: +14% prefill for +0.52% perplexity, which is worse than
  the exact path and outside the envelope of anything else shipped.
