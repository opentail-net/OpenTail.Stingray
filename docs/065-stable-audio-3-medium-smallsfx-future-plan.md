# Stable Audio 3 Medium / Small SFX — future plan (kept, not started)

Status: **saved for later, 2026-09-03** — user explicitly asked to keep this plan for "when we are
free from current stuff" (ACE-Step 1.5 Turbo port is the active thread). Not scoped against real
checkpoints yet; the plan below is the user's own detailed proposal, recorded verbatim in
structure so it can be picked up directly once prioritized, adaptable as needed once real
archaeology happens (per this project's own rule: check real config/tensor inventory before
assuming anything, including this plan's own architectural guesses).

Also on the same want-list, per the user (2026-09-03): **MiniMax-Music3** — a genuinely new
architecture family, not yet investigated at all, no relationship to the already-shipped Stable
Audio 3 runtime assumed.

## Why Small SFX first, then Medium

Official Stable Audio 3 family split:
- **Small SFX**: ~433M diffusion model + SAME-S autoencoder, CPU-oriented, up to ~120s.
- **Medium**: ~1.4B diffusion model + SAME-L autoencoder, GPU-oriented, up to ~380s.

Both share the same broad pipeline (T5Gemma text conditioning -> flow-matching DiT -> SAME
decoder -> 44.1kHz stereo), matching this project's already-working Stable Audio 3 (small/music)
runtime. The strategic bet: Small SFX mostly tests "how configurable is the existing runtime,"
and Medium mostly adds "a bigger DiT config + SAME-L" on top of what Small SFX proves out — not
two independent ports.

**Real caveat this project's own conventions demand before trusting that bet**: this project's
existing Stable Audio 3 VAE (`AcousticVae`) and ACE-Step's `AutoencoderOobleck` looked similar on
paper too (both "the VAE for a Stability-adjacent audio diffusion model") and turned out to be
genuinely different architectures (see docs/064-acestep-implementation-plan.md) — so "Small
SFX/Medium share the current runtime's SAME/DiT shape" is a hypothesis to verify against real
checkpoint tensor names first (this plan's own Phase 0), not something to build on without
checking.

## Phase 0 — architecture comparison first (before any code)

Build a hard comparison matrix (`StableAudio3/docs/SA3_MODEL_MATRIX.md` or similar) between the
currently-working SA3 runtime, small-sfx, and medium: DiT config (hidden size, layers, heads, head
dim, latent channels), text encoder, duration conditioning, flow scheduler, SAME variant, max
duration, editing support. Exit criterion: able to say precisely "Small SFX needs X code changes;
Medium needs Y additional changes" before porting starts.

## Recommended sequencing (user's own plan, condensed)

1. **Sprint 1 — archaeology**: real config/tensor inventories for both checkpoints, compared
   against the existing runtime, `SA3_MODEL_MATRIX.md` produced.
2. **Sprint 2 — Small SFX**: config support, weight mapping, DiT forward pass, golden tensor
   tests (input projection -> block 0 -> middle -> final -> velocity/noise prediction, NOT
   jumping straight to full 8-step generation), scheduler, reuse SAME-S if the existing runtime
   already has it, end-to-end SFX generation. Milestone: "Stable Audio 3 Small SFX runs natively."
3. **Sprint 3 — consolidate**: refactor around what Small SFX proved out BEFORE starting Medium —
   config-driven DiT/loader, a generic `ISameAutoencoder` interface (`SameSmallAutoencoder`/
   `SameLargeAutoencoder` implementations), shared generation pipeline. Explicitly flagged by the
   user as an important, non-skippable step — don't fork the runtime into parallel bespoke ports.
4. **Sprint 4 — Medium DiT**: config, weight mapping, forward-pass golden tests, full sampling —
   latent generation only, no audio yet is fine at this stage.
5. **Sprint 5 — SAME-L decoder** (decoder before encoder, same reasoning as ACE-Step's Oobleck —
   text-to-audio only needs decode, not the full encode/decode/editing suite): architecture
   mapping, attention (SAME-L uses sliding-window attention, a real Medium-specific concern worth
   its own `IWindowedAttention` abstraction, correctness-first before SIMD/blocking optimization),
   decoder blocks, latent->PCM golden tests, full WAV. Milestone: "Stable Audio 3 Medium generates
   music natively."
6. **Sprint 6 — long-form + optimization**: test 5s/30s/60s/120s/240s/380s durations (peak RAM,
   allocation rate, GC pressure, DiT/VAE/total time per length), chunked SAME-L decoding
   (`DecodeChunked` with overlap-and-stitch, compared numerically and audibly against full
   decode) — the official repo notes chunked decoding meaningfully reduces Medium's peak memory.

## Public API shape (user's proposal)

A unified `StableAudio3Variant` enum (`SmallMusic`, `SmallSfx`, `Medium`) behind one
`IStableAudio3Engine`/`StableAudio3Request` surface, so `runtime.Load(variant).Generate(request)`
works the same way regardless of which checkpoint is loaded — editing modes (audio-to-audio,
inpainting, continuation) explicitly deferred to a later phase, after text-to-audio works for both
sizes.

## Golden test ladder (same discipline for both, per the user's plan)

Config parser -> tensor inventory -> tokenizer -> T5Gemma embeddings -> duration embedding -> one
DiT block -> full DiT -> one scheduler step -> full-step latent generation -> SAME decode -> WAV,
repeated for Medium with explicit "reuse validation" checkpoints (reuse T5Gemma? reuse duration
conditioner? reuse scheduler unchanged?) rather than re-deriving each piece.
