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
`scripts/check-test-model-coverage.ps1` output: several older model-gated tests return early
when their local fixture is absent, so a managed-suite pass count alone is not model coverage.

## Restart-continuation acceptance test

The acceptance test must use a real GGUF, not a fake forward pass. It must:

1. Create a session and materialise at least two turns.
2. Persist its manifest and KV payload.
3. Stop the owning process without retaining an in-memory engine/cache reference.
4. Start a new process, load the same model/runtime ABI, and restore the session.
5. Generate one more greedy turn.
6. Compare every continuation token against a fresh full replay of the exact logged token IDs.

`HotSessionGreedyReplayTests.ColdSession_RealModel_CrossProcessRestore_MatchesFullGreedyReplay`
passed on the local CPU-dense SmolLM2 GGUF on 2026-08-07. It uses two child processes and fresh
runtime objects, then compares each generated segment to full replay. That clears the reference
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
| CPU inference | **Yes** — dense greedy prefill/decode, numerical parity, corpus perplexity | `cpu-prefill-quality-gate.md`, `cpu-benchmark-llamacpp-baseline.md` |
| CUDA dense | **No — blocked.** No NVIDIA driver on this machine; nothing was run and nothing is claimed | — |
| Vulkan | **Partial.** Functional load + greedy decode, and CPU/Vulkan logit parity measured. **On an integrated APU, not a discrete card** — every number is APU-specific and does not predict discrete-GPU behaviour | `vulkan-backend-evidence.md` |
| Hybrid MoE | **Partial.** MoE model runs on CPU with prefill numerics in the dense band. The row's actual subject — CPU-expert and GPU-cache offload paths — is **untested**, needing constrained VRAM this machine cannot provide | `moe-backend-evidence.md` |
| Speculation | **Measured and negative.** Draft-model speculation is a 37% regression on CPU because the verify pass cannot amortise; no MTP GGUF present for the MTP equivalence check | `cpu-speculative-decoding-findings.md` |
| Session restart continuation | **Partial.** Six of seven conformance dimensions covered, and rollback closed this session after a production seam was added. Full restart-with-real-model replay not run | `sessions-release-gate-matrix.md` |
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

### Known open, with evidence rather than suspicion

- Gemma 4 per-layer-head-dim models get **no batched prefill**: 3.5 t/s prefill against 4.0 t/s
  decode, where a dense model of similar shape runs prefill at ~4x decode. Roughly a 4x prefill
  penalty, confirmed in code at the `perLayerHdUnsupported` gate.
- Flash-64 head dims 128/256 remain opt-in: +14% prefill for +0.52% perplexity, which is worse than
  the exact path and outside the envelope of anything else shipped.
