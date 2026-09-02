# MiniMax-Music3 — future plan (kept, not started)

Status: **saved for later, 2026-09-03** — on the radar per user request, not scoped against real
checkpoints yet. ACE-Step 1.5 Turbo is the active thread; this is recorded so the plan doesn't
need to be re-derived when picked up. Sequencing note from the user: MiniMax-Music3 after
ACE-Step, before spending time on niche audio models.

## Why this is architecturally distinct from every other audio model on this project's list

Not another MusicGen/AudioGen codec-LM, not another ACE-Step-style single DiT+VAE — a genuine
**hybrid autoregressive + flow-matching system** with FOUR real, separately-sized components:

```
Global LLM (8B, Qwen3-8B-initialized) -- long-range song structure, predicts the first
    semantic RVQ codebook (16,384 entries) at 25 frames/sec
Local LLM (0.6B) -- predicts the remaining 7 acoustic RVQ codebooks (1,024 entries each)
    per frame, conditioned on the Global model's per-frame output
    -> Global + Local HIDDEN STATES (not just the discrete tokens) get fused
Flow Matching (2.4B) -- synthesizes from the fused hidden-state representation
Flow-VAE (123M, adapted from MiniMax Speech, retrained for music) -- latent -> stereo PCM
```

The real, load-bearing detail: inference does NOT simply decode the discrete RVQ tokens through a
codec — the Global/Local models' HIDDEN STATES are fused and fed into the flow-matching
synthesizer. This is why MiniMax-Music3 needs its own hidden-state-fusion primitive
(`MiniMaxMusic3HiddenStateFusion`) that neither MusicGen/AudioGen nor ACE-Step need.

## Real scale warning (verify before committing to a residency strategy)

The official repository is cited at ~57.4GB total (8B + 0.6B + 2.4B + 123M plus condition
encoder/tokenizer/vocoder assets) — roughly 11B+ parameters across the pipeline before KV cache
and activations. This is a different class of problem from MusicGen/AudioGen/ACE-Step (all
sub-5B): a naive "load everything into RAM at once" strategy is unlikely to work well on a
consumer machine. The user's own plan proposes a staged residency pipeline (Global+Local resident
-> generate -> evict -> load Flow+VAE -> synthesize), explicitly tying into this project's
existing `ModelRuntimeManager`/admission/eviction infrastructure rather than building a new one.
**Real numbers (checkpoint size, actual per-component parameter counts, whether staged eviction is
actually necessary on this machine) need confirming against the real checkpoint before assuming
the 57.4GB figure or the staged-residency necessity — same "verify before building" discipline as
every other model doc in this series.**

## V1 scope (user's own framing)

Lyrics + music description -> 8B Global LLM -> 0.6B Local LLM -> hidden-state fusion -> 2.4B
flow-matching -> Flow-VAE -> stereo audio. Target: **5 seconds first**, not 5 minutes — a naive
autoregressive Global model over 7,500 frames (5 minutes @ 25Hz) would be brutal on CPU; prove the
mechanics at 5s, then 10s/30s/60s, with long-form (up to the real 5-minute target) as a later
phase requiring flow-stage chunking (the real MiniMax Music3 flow stage reportedly operates on
overlapping chunks for long songs, per current Diffusers documentation — needs confirming against
real source, not assumed). No prompt-rewriting/structured-caption support, no editing/continuation
in V1.

**Real sample rate note**: the user flagged a real discrepancy between the "official" 32kHz stereo
claim and a Diffusers implementation description citing a 44.1kHz DAC-style decoder — explicitly
says to treat the real official checkpoint/repo behavior as authoritative over secondary
implementation descriptions once archaeology happens, not to hard-code either number now.

## What should genuinely reuse existing Stingray infrastructure

- **Global LM**: real, load-bearing clue -- initialized from Qwen3-8B. Should build on this
  project's existing Qwen3/GQA/RoPE/RMSNorm/SwiGLU kernel support (the same infrastructure
  ACE-Step's Qwen3 text encoder and this engine's general LLM inference already use), not a
  bespoke reimplementation — mirrors how ACE-Step's DiT reused Qwen3-shaped attention/FFN
  primitives. Verify real config differences from stock Qwen3-8B before assuming a clean drop-in.
- **Flow matching**: a generic `IFlowMatchingScheduler` shared across ACE-Step/Stable-Audio-3/
  MiniMax-Music3, per the user's explicit recommendation — build this once there are 2+ real,
  verified callers (this project's own DRY-timing rule: after MusicGen+AudioGen needed the same T5/
  EnCodec math, THOSE got extracted; the same discipline applies here once ACE-Step's real flow
  scheduler exists to compare against).
- **VAE decoder shape**: `IAudioLatentDecoder`-style interface shared across models where the
  actual decode math permits — but given ACE-Step's Oobleck VAE and this project's existing
  Stable Audio 3 VAE already turned out to be genuinely different architectures despite looking
  similar on paper, do NOT assume Flow-VAE shares real code with either without checking real
  tensor names first.

## Golden test ladder (user's own, condensed)

Global (prompt/lyrics -> tokens -> hidden state -> semantic logits) -> Local (global frame ->
7 acoustic tokens + local hidden state) -> Fusion (global+local hidden -> fused representation) ->
Flow (fused representation + known noise + known timestep -> flow output) -> VAE (known latent ->
PCM) -> end-to-end (lyrics+caption+seed -> complete audio). Debug one stage at a time, not all five
simultaneously — same discipline as every other model doc in this series.

## Suggested project layout (user's proposal, for when picked up)

```
OpenTail.Stingray.Models.MiniMaxMusic3/   (or under Diffusion/Audio per this project's existing
    MiniMaxMusic3Model.cs                  per-domain convention, TBD at archaeology time)
    MiniMaxMusic3Config.cs
    MiniMaxMusic3GenerationParams.cs
    MiniMaxMusic3Pipeline.cs
    Global/    -- GlobalMusicModel, GlobalConfig, MusicTokenHead, GlobalKvCache
    Local/     -- LocalMusicModel, LocalConfig, AcousticTokenDecoder, LocalKvCache
    Conditioning/ -- Music3PromptEncoder, LyricsProcessor, Music3Condition
    Fusion/    -- HiddenStateFusion, FusionProjection
    Flow/      -- Music3FlowTransformer/Block/Attention/Scheduler/Sampler
    Vae/       -- Music3FlowVaeDecoder/Encoder
```

## Structured caption support (later phase, optional)

The official model reportedly supports/recommends a structured caption format (Global Metadata:
genre/BPM/key/scale/emotional progression/production; Vocal Details: gender/timbre/performance/
harmony; Arrangement: instruments/groove/bass/percussion/section evolution) plus an optional
music-caption-rewriter that expands concise prompts into this structure. User's recommendation:
make this optional, not a hard dependency, and it could eventually inform a real UI (structured
fields instead of one prompt string) — a product-layer decision for well after V1 works.

## Implementation sequence (user's own phase list)

Phase A archaeology (config, tensor inventory, checkpoint structure, module map) -> Phase B Global
(Qwen3 reuse, semantic embedding/output head, KV cache, 25Hz generation, hidden-state capture) ->
Phase C Local (local transformer, 7-codebook generation, hidden-state capture) -> Phase D Fusion
(projection, fusion, golden test) -> Phase E Flow (2.4B transformer, scheduler, chunking, one-step
then full-flow golden tests) -> Phase F Flow-VAE (decoder, latent->PCM golden test) -> Phase G
end-to-end (5s, 10s, WAV, regression corpus) -> Phase H long-form (30s/60s/180s/300s,
overlap/stitching) -> Phase I optimization (quantization, KV, attention, memory residency, CPU
threading) -- correctness at FP32/BF16 before any quantization, matching this project's standing
rule.
