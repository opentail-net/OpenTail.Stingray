# Doc audit handoff (2026-09-27, session paused by user request)

The user asked for every doc in `docs/` to be checked against real code/tests/`PerformanceLeague.md`/
`STATUS.md` evidence: fully-completed plans moved to `docs/done/` with a status banner citing the
evidence, partially-completed ones split or annotated in place. This ran as a self-paced `/loop`
across several iterations; the user is now closing the session, so this is the handoff.

## Done this session (committed: `0e43969`, `1bc98a8`, `59ef972`)

Moved to `docs/done/` with an evidence-cited status banner, cross-references fixed in
`docs/README.md`/`docs/00-current-work.md`:

- `done/01-gguf-model-coverage-plan.md` — cross-checked against `ModelCompatibility.cs`'s live allowlist
- `done/055-ltx-video-implementation-plan.md`, `done/077-ltx-video-gpu-residency-plan.md` — both end
  unresolved in their own text (2026-09-14) but `STATUS.md` shows LTX-Video closed 2026-09-18
- `done/067-sdxl-unet-gpu-residency-plan.md`, `done/068-sdxl-cpu-side-overhead-plan.md` — both say
  "complete" in their own final sections
- `done/072-wan21-gpu-kernel-tuning-plan.md` — Phase 1 objective met (073 continues, stays open)
- `done/074-wan21-umt5-gpu-residency-plan.md`, `done/075-zimage-turbo-gpu-residency-plan.md`,
  `done/076-sd35-medium-gpu-residency-plan.md` — confirmed via real `*GpuWeights` classes in source
  plus matching `PerformanceLeague.md` timings
- `done/081-master-gpu-perf-and-accuracy-plan.md` — Priority-0 Wan correctness bug confirmed resolved
  within the doc itself
- `done/2026-09-24-audio-recheck-plan.md` — all 4 phases checked done
- `done/vl-migration-plan-2026-08-20.md` — "migration fully closed", confirmed dead code removed
- `done/qwentts-cosyvoice3-handoff.md` — superseded, both targets green in `STATUS.md`
- `done/cpu-performance-baseline.md` — a completed measurement snapshot, not an open plan

Annotated in place (partial, not moved): `103-front-door-design.md` (step 1 done, steps 2-4 not
started). Also fixed the stale CosyVoice-3 pointer in `docs/00-current-work.md` and in
`docs/done/cosyvoice3-voice-identity-chatgpt-prompt.md`.

Note: `087-flux2-implementation-plan.md`, `089-qwen-image-text-conditioning-plan.md`,
`091-flux2-vulkan-gpu-residency-scoping.md` were already archived to `docs/done/` by a separate,
concurrent docs-only agent working in the same repo this session (pre-existing commit `e484cec`,
not part of the above list).

## Confirmed correct as-is (checked, left alone — do not re-check these)

- `00-current-work.md`, `docs/README.md`, `STATUS.md`, `MODELS.md`, `cli-option-inventory.md`,
  `env-var-inventory.md` — living index/reference docs, not "plans," correctly never moved
- `02-qwen35moe-plan.md` — tiny, already self-describes done (items 1-2) vs open (3-4) inline
- `051-hotsession-capability-wiring-plan.md` — already correctly split (done half in `docs/done/`)
- `bugstofix.md` — already correctly split (resolved entries repeatedly moved to `docs/done/`)
- `069-flux-vulkan-gemm-perf-handoff.md`, `073-wan21-kernel-fusion-and-qkv-plan.md` — both have an
  explicitly stated target that was **not** met (069: 99.8s vs achieved 374.7s; 073: <450ms vs
  achieved 770.1ms) — genuinely still open despite large measured speedups
- `050-ggml-op-coverage-gap-plan.md` — verified zero of its 5 op kernels exist in
  `src/OpenTail.Stingray.Cpu` — genuinely 0% done, correctly left active
- `docs/reference/`, `docs/research/` — architecture docs/ADRs/marketing/generated inventories,
  not plans with an end state; out of scope for this task

Also spot-checked and confirmed still genuinely open (no action taken, no need to re-check):
`03-gemma4-e4b-vision-plan.md`, `032-multi-model-inference-runtime-plan.md`,
`04-quality-of-life-improvements-plan.md`, `052`/`053` (Vulkan TTS backend audits),
`070-gemma4-batched-prefill-plan.md`, `082`, `084`, `086`, `088`, `092`, `101`, `102`.

## Not yet checked — remaining work for the next session

These are the docs this audit had not yet reached when the session paused. None have been verified
either way — don't assume either "done" or "still open" without checking:

- `090-flux2-cpu-perf-handoff.md`, `093-flux2-gpu-performance-optimization-plan.md`,
  `094-diffusion-performance-plan.md` — **in progress when paused**: `Flux2DiT.cs` still has a
  `GetWeight` method (line 112) that 090 flags as the anti-pattern to fix (naive per-call weight
  fetch instead of the packed-panel approach other models use), but `Flux2GpuWeights.cs` *does*
  have `TryGetRaw`-based fast-path loading (line 236) — this needs a closer read to determine
  whether the CPU path (090's actual target) got the fix or only the GPU path did, before
  concluding either way. `PerformanceLeague.md`'s FLUX.2-dev row (258.8s CPU / 236.7s GPU at
  512²/2-step) is much better than 090's "killed after 30+ min at 20-step" baseline, but the step
  counts aren't directly comparable — verify the actual per-step cost before calling this done.
- `perf-sweep-plan.md` — large (14 phases, 32 unchecked boxes vs 28 checked) — spot-checked only
  the checkbox count, not verified individually; likely still genuinely open but each phase should
  be checked, not assumed
- `tts-performance-baseline-and-plan.md` — Turn 1 done, Turn 2 (Batched CFG & ODE step reduction)
  never appears in the doc; unclear whether it happened elsewhere and was never written back here
  — check `PerformanceLeague.md`'s QwenTTS/CosyVoice3 rows and `docs/audio-review-new-progress.md`
- `PerformanceLeague-expansion-plan.md` — has some deferred items noted inline; not individually
  re-verified this session
- `058-deepseek-full-lineage-implementation-plan.md` — last known status: Phase 0/1 alpha code
  written, never run against real weights — worth checking if that changed
- `064-acestep-implementation-plan.md` — V1 working end-to-end but explicitly not
  "production-ready" (numeric golden-parity, missing `silence_latent` buffer, audible quality
  assessment all still open per the doc) — not re-verified this session
- `066-minimax-music3-future-plan.md` — all 8 components golden-verified per the doc, but a real
  VAE-decode performance gap (29-33s vs reference 13.1s) was still open and "not yet attempted";
  not re-verified this session
- `docs/profiles/` (JSON config files), `docs/audio-samples/README.md`,
  `docs/diffusion-samples/README.md` — not opened this session at all

## Process notes for whoever continues this

- Don't trust a doc's own "DONE"/"complete" claim without cross-checking `PerformanceLeague.md`,
  `STATUS.md`, or the actual source file it claims changed — several docs in the "done" list above
  had stale self-assessments that needed correcting before archiving (e.g. `081`'s own "broader
  checklist" was stale relative to its own body text).
- When moving a file, grep the whole `docs/` tree (and root `README.md`) for its old filename
  before finishing — several docs are cited by bare filename in other active docs' prose, and the
  navigational ones (`docs/README.md`, `docs/00-current-work.md`) need the link fixed, though
  historical prose citations elsewhere are lower priority (left as-is this session, not broken
  links, just stale paths in text).
- A separate, concurrent docs-only agent was also active in this repo during this session
  (confirmed via unexpected file mtimes / already-committed archive moves). Check
  `git log --oneline` for new commits before resuming, to avoid redoing already-committed work.
