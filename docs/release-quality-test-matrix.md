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
lane's proof, not every backend/cache family or a product-surface commitment; `inspect` continues
to report restart continuation as unsupported until the named-session lifecycle is exposed and
capability-gated.

## History and release notes

Use [CHANGELOG.md](../CHANGELOG.md) as the human release index. Pull requests should use a concise,
imperative subject with a component prefix, for example `engine: preserve BF16 KV pages on restore`.
Avoid generic messages such as `ok`; add a changelog/release-note bullet rather than rewriting
shared history.
