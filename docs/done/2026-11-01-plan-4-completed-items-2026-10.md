# Completed items split from 2026-11-01-plan-4.md (2026-11-01)

Finished items moved verbatim out of [../2026-11-01-plan-4.md](../2026-11-01-plan-4.md). Open items stay there. Each block is preceded by the heading it sat under. Relative links inside the blocks were rebased for this folder.

> from: ### A. Numerics and engine
- [x] **A1** *(done 2026-11-01, measured harmless: Q8_KS is the more precise format, 0.66-0.77 of the Q8_K error; PPL 605.7 int8 vs 546.8 F32 vs llama.cpp 617.1 ± 75.5 on a degenerate LM, so the kernel error is the evidence; ADR-0003 addendum; decode kept on Q8_K)* Q3_K prefill and decode quantise activations differently. *Checked 2026-11-01:* `SimdKernels.TryResolveQ8Dispatch` maps `Q3_K` to `QuantizeRowToQ8KS`
  (eight scales per 256) for the batched path, while the single-token decode dot is `DotQ3K_Q8K` (one scale per 256); ggml uses one scale for both. Done when the
  Q3_K batched path uses the same activation format as decode (or the difference is measured to be harmless), judged by perplexity on Granite-Vision-3.2-2B Q3_K_S
  (llama.cpp reference 617.1 from perf-sweep 10.2) and prefill speed, with an ADR note. Source box: prefill-numerics follow-ups.

> from: ### A. Numerics and engine
- [x] **A2** *(checked 2026-11-01, no code change: `MoeBatchedExperts` gathers each expert bucket's rows into one buffer and calls `SimdKernels.MatMulBatched` for gate, up and down with `allowQ8: Q8PrefillEnabled`, which is `STINGRAY_MOE_PREFILL_Q8=1` and off by default. By default no activation is quantised at all (exact F32 batched path, parity with per-token). Only with the experiment on does each bucket quantise its gathered rows separately for gate and again for up, so a token is quantised about `2 x numActive + numActive` times instead of once. That cost applies to an opt-in path nobody runs, so the change is not made; reopen it only if A3 turns the default on.)* MoE: quantise the token activations once before expert dispatch instead of per expert bucket. *Not yet checked:* read the batched MoE path first and write
  down what it does today; the box was written from memory. Done when the current behaviour is recorded and, if it quantises per bucket, the change is measured on
  Granite-4-H-Small (speed and PPL).

> from: ### A. Numerics and engine
- [x] **A3** *(decided 2026-11-01: keep `STINGRAY_MOE_PREFILL_Q8` off. A1 changed no code and A2 changed no code, so the inputs to the earlier Granite-4-H-Small Q2_K comparison, where the Q3_K expert-down projection on Q8 moved PPL and token NLLs, are unchanged and no re-run was made; evidence in `docs/1-correctness/13-granite4-h-small-moe-ppl-parity-plan.md`.)* Decide `STINGRAY_MOE_PREFILL_Q8` (default off) from a re-run of the Granite-4-H-Small PPL/NLL comparison after A1 and A2. Recorded decision, not just a number.

> from: ### B. Product surface
- [x] **B1** `stingray pull` keeps its own HTTP clients. *Checked 2026-11-01:* `PullCommand.cs` built a plain `HttpClient` for both the repo listing and the downloads, while `Core/Net/HubClient.cs` and `ExternalHttpClient` (HTTPS to Hugging Face hosts only, every redirect hop checked, `HF_TOKEN` sent to the Hub and never to the CDN, response size capped) exist.
  - [x] The listing now goes through `ExternalHttpClient` (`PullCommand.ListGgufFiles`, internal). Three tests with an injected handler: sizes, hashes and the resolved commit are read; the token reaches the Hub and a pinned revision uses the revision route; a redirect to a non-Hugging-Face host is refused before any request is made. Two mutations fail them (2026-11-01).
  - [x] The downloads keep their `HttpClient` signature (resume, progress and the hash check sit on it) but the client `ModelDownloader.CreateClient` returns now runs on `HuggingFaceOnlyHandler`: external access re-checked per request, HTTPS to Hugging Face hosts only, redirects followed by hand with every hop checked, `Authorization` kept only for the Hub. `stingray pull` and `stingray setup` both use it. Six tests with an injected inner handler; dropping the token stripping or the hop check each fails one (2026-11-01). A real catalog download through it (chat bundle, temporary model home) passed in `scripts/qualify-tasks.ps1`.

> from: ### B. Product surface
- [x] **B2** *(done 2026-11-01: `StatusMatrixCitationTests` fails when STATUS.md cites a test class or a `*.golden.json` that does not exist; its first run found two vision rows citing `KimiYoutuVisionEmbedderParityTests`, folded into `LlamaMtmdVisionParityTests` earlier, now corrected; cited hashes are still not checked)* Docs-as-tests beyond links. *Checked 2026-11-01:* `DocumentationLinkTests` covers relative links, backticked repo paths in the entry-point docs and the

  development project map; nothing checks that a cited hash or evidence file exists, or that a STATUS row still names a test that exists. Done when a test fails when a
  STATUS row cites a test class or golden file that is gone.

> from: ### B. Product surface
- [x] **B4** *(done 2026-11-01: A2 lifecycle and lock protocol was already implemented (re-check after build under the lock, children disposed outside `Model._lock`) but its tests and the `ModelParams` refusal test returned early when no checkpoint is on disk, which is always the case in CI, so none of it ran; `ModelLifecycleTests` now covers it with a synthetic GGUF (context then model, model with live children, independent siblings, executor over a disposed context, 25 rounds each of create-versus-dispose and child-versus-parent dispose with a 30 s bound; removing the post-build disposed re-check fails the race test). A3 found one dead option: `ContextParams.TurboQuantHeadDim` was copied and never read; `CreateContext` now refuses it with `NotSupportedException` (mutation-checked). A4 was already done and its tests are hermetic: custom `IInferenceParams` and `IModelContext`, turn rollback on exception and cancellation.)* Library surface audits from the public-surface plan: A2 (ownership, disposal and lock-order protocol), A3 (misleading parameter options, the library
  counterpart of the server parameter audit) and A4 (interface conformance and incomplete-turn semantics). *Not yet checked:* each starts by reading the public types;
  each ends with findings and tests, like the server audits did (they found a 500 and a fake image endpoint).

> from: ### C. Server
- [x] **C1** `/v1/sessions/*` parameter audit, the last endpoint group the 2026-11-01 audit did not cover. **Done 2026-11-01:** the turn digest ignored `allowed_choices` (a retry with other choices came back as a replay) and `temperature` and `top_p` were unchecked; both fixed, see F7 in the audit. *Checked:* `/tokenize`, `/detokenize`, `/props`, `/status` and
  `/api/embed` were read the same day: missing payloads are 400 (tested), out-of-range token ids decode to empty (`GgufTokenizer.DecodeBytes`), the embedding alias shares the
  validated parser. No defect found there.

