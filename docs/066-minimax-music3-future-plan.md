# MiniMax-Music3 — future plan

Status: **Phase A archaeology done, 2026-09-03** — ACE-Step 1.5 Turbo and Stable Audio 3 Small
SFX/Medium are now complete (all V1-working); this is next per the user's own sequencing note
(after ACE-Step, before niche audio models). Real checkpoint located and inspected
(`MiniMaxAI/MiniMax-Music3` on Hugging Face) -- several of this doc's own original figures (cited
from documentation, not yet checkpoint-confirmed at the time) turned out to be significantly off.
See "Phase A archaeology, real findings" below before trusting anything in the older sections past
this point.

## Phase A archaeology, real findings, 2026-09-03 -- corrects several documented-only figures above

Real total repo size: **57.35GB** (confirms the ~57.4GB figure this doc originally cited from
documentation -- that part held up). But the real, load-bearing correction: **the actual inference
pipeline (`modular_model_index.json`'s real `MiniMaxMusic3ModularPipeline` component graph) only
uses SIX of the repo's components, totaling ~28.5GB** -- roughly HALF the repo:

```
condition_encoder/  (diffusers, MiniMaxMusic3ConditionEncoder)        ~100MB
language_model/     (transformers, Qwen3ForCausalLM -- REAL, STOCK    ~17.2GB (4 safetensors shards)
                      Qwen3 config: hidden=4096, layers=36, heads=32,
                      kv_heads=8, intermediate=12288 -- an ~8B-class
                      dense Qwen3, loadable via this project's
                      EXISTING Qwen3/GGUF support, not a bespoke port)
rvq_depth_decoder/  (diffusers, MiniMaxMusic3RVQDepthDecoder --        ~1.29GB
                      hidden=4096, 4 layers, 16 heads, 8 codebooks,
                      audio_vocab_size=1024, max_position=16)
transformer/        (diffusers, MiniMaxMusic3Transformer1DModel --     ~9.7GB (2 shards)
                      the flow-matching DiT: condition_dim=2048,
                      36 layers, 32 heads, head_dim=64, in_channels=128,
                      rotary_dim=32)
vocoder/            (diffusers, MiniMaxMusic3Vocoder -- latent_dim=128,~217MB
                      decoder_hidden=1536, upsampling_ratios=[8,8,4,2]
                      =512 hop @ 44100Hz)
scheduler/          (diffusers, FlowMatchEulerDiscreteScheduler,       tiny (JSON only)
                      shift=1.0, no dynamic shifting)
```

**The other ~29GB in the repo is NOT referenced by the real pipeline's component graph at all**:
`qwen_7B/qwen_7B/` (~19GB across 48 shards -- a real, DIFFERENT, MiniMax-native `AbabForCausalLM`
architecture, `model_type: mixtral`, real custom fields `layernorm_full_attention_alpha/beta`,
`layernorm_linear_attention_alpha/beta` suggesting a hybrid linear+full attention scheme, distinct
audio-token decoder head fields `audio_num_codebooks=8`/`decoder_num_layers=4` -- despite the
directory's confusing "qwen_7B" name, this is NOT the same as `language_model/`'s real, stock
`Qwen3ForCausalLM`), `flowmatching_vae.pth` (9.83GB, single `.pth` file, real purpose
unconfirmed -- given its size (~80x the `vocoder/`'s 217MB) and that the real inference pipeline
never references it, the leading hypothesis is a training-time encode-side asset, not an
inference-time decode component; NOT confirmed), and `dav.pth` (491MB, real purpose also
unconfirmed, not in the pipeline graph either).

**This retroactively corrects this doc's own original "FOUR real, separately-sized components"
framing** (Global 8B + Local 0.6B + Flow 2.4B + Flow-VAE 123M, cited from documentation before any
real checkpoint was inspected): the real modular pipeline graph shows FIVE real weighted
components (condition_encoder, language_model, rvq_depth_decoder, transformer, vocoder), and there
is no directly-visible standalone "0.6B Local LLM" matching that description among them --
`rvq_depth_decoder` (a real, modest 4-layer/4096-hidden model) is the closest real candidate for
"predicts the remaining RVQ codebooks conditioned on the Global model's output," but whether it
also does real hidden-state fusion (this doc's own "real, load-bearing detail" about the Global/
Local fusion mechanism) is NOT yet confirmed from config alone -- needs real source/tensor-name
archaeology, not assumed from the old framing. The `vocoder/`'s real 217MB size (~108M params in
bf16) is much closer to the originally-cited "123M Flow-VAE" than `flowmatching_vae.pth`'s 9.83GB
is -- the real `vocoder/` component is likely what that original figure meant, not the giant `.pth`
file.

**Real, concrete next steps** (before any code): (1) download the small components first
(`condition_encoder/` 100MB + `vocoder/` 217MB + `rvq_depth_decoder/` 1.29GB ≈ 1.6GB total, safe
and cheap) and inspect their real tensor names/shapes the same way every other model in this
series was archaeology'd; (2) find and read the real `diffusers` source for
`MiniMaxMusic3ConditionEncoder`/`MiniMaxMusic3Transformer1DModel`/`MiniMaxMusic3RVQDepthDecoder`/
`MiniMaxMusic3Vocoder` (these class names suggest they may already be merged into a real `diffusers`
branch/PR, matching how ACE-Step's and Stable Audio 3's real classes were found in `diffusers`/
`stable-audio-tools` source this session -- check before assuming no reference exists); (3) confirm
whether `rvq_depth_decoder` really does the Global/Local hidden-state fusion this doc's older
sections describe, or whether that mechanism lives somewhere else entirely (real source read, not
assumed); (4) the ~17.2GB `language_model/`+~9.7GB `transformer/` are the two big real downloads
needed for a working V1 (~27GB combined) -- this environment's disk is currently tight (11GB free
as of this archaeology pass; ACE-Step/Stable-Audio-3-Medium checkpoints already consumed most of
this session's downloaded-checkpoint budget), so these need either freed disk space or a staged
download-use-delete cycle, not assumed to just fit.

**The sections below this point are the ORIGINAL plan, written before this real archaeology pass
-- read them as the user's own framing/sequencing preferences, not as confirmed architecture facts
where they conflict with the real findings above.**

## Real diffusers source ALREADY EXISTS for all five components -- major find

The installed `diffusers==0.40.0` in this environment already ships real, complete classes for
every real pipeline component: `MiniMaxMusic3Transformer1DModel`, `MiniMaxMusic3ConditionEncoder`,
`MiniMaxMusic3RVQDepthDecoder`, `MiniMaxMusic3Vocoder`, plus a full real modular pipeline
(`diffusers/modular_pipelines/minimax_music3/`) with real block classes and their own docstrings
describing the exact real data flow. This gives MiniMax-Music3 the SAME real-source-oracle
advantage ACE-Step and Stable Audio 3 Medium had this session (golden-parity checks, exact real
formulas, no guessing) -- a much stronger starting position than this doc's original framing
assumed, and found essentially for free (no large download needed, the classes were already
installed alongside everything else fetched this session).

**Real pipeline flow, read directly from `MiniMaxMusic3Blocks`'/`MiniMaxMusic3SemanticGenerationStep`'/
`MiniMaxMusic3CoreDenoiseStep`'s own docstrings** (`diffusers/modular_pipelines/minimax_music3/
modular_blocks_minimax_music3.py`):

1. **Tokenize + autoregressive generation** (`tokenizer` + `language_model` (Qwen3, real stock
   architecture) + `rvq_depth_decoder`): assembles a special-token prompt from the caption + lyrics,
   runs the real Qwen3 autoregressively, producing `frame_hiddens` of shape `[1, frames,
   num_codebooks * hidden_size]` (`num_codebooks=8`, `hidden_size=4096` -> 32768-wide per frame).
   **This directly answers the original doc's "hidden-state fusion" question**: there is no separate
   fusion module at all -- `rvq_depth_decoder` (the real "Local" component, 4 layers/8 codebooks)
   produces the additional per-codebook hidden states, and they are simply CONCATENATED into one
   wide per-frame vector. The mysterious `MiniMaxMusic3HiddenStateFusion` primitive this doc
   originally proposed does not need to exist as a separate class.
2. **Chunked flow-matching denoise** (`condition_encoder` + `transformer` + `scheduler` +
   `guider`/CFG): splits `frame_hiddens` into **200-frame windows** and flow-matches each window's
   Flow-VAE latent from noise, **blending each window into the previous one over their overlap**
   (real, concrete long-form mechanism -- resolves this doc's earlier "reportedly operates on
   overlapping chunks... needs confirming" uncertainty into a confirmed fact).
3. **Vocoder decode** (`vocoder`, real "DAC-style"): latents -> stereo waveform. **Real sample rate
   confirmed as 44.1kHz** (`vocoder/config.json`'s own `sampling_rate: 44100`, and the real
   pipeline docstring says so explicitly too) -- resolves this doc's earlier flagged 32kHz-vs-44.1kHz
   discrepancy definitively in favor of 44.1kHz.

Also downloaded the three small real component weight files this session
(`condition_encoder/diffusion_pytorch_model.safetensors` ~100MB,
`vocoder/diffusion_pytorch_model.safetensors` ~217MB,
`rvq_depth_decoder/diffusion_pytorch_model.safetensors` ~1.29GB) -- real tensor-name/shape
archaeology against them (matching this session's established technique) is the concrete next
step, not yet done.

**Revised, real next steps**: (1) real tensor-name archaeology on the three small downloaded
components against their real diffusers class `__init__`/`load_state_dict` shapes (same technique
used for every ACE-Step/Stable-Audio-3 component this session); (2) read the real
`MiniMaxMusic3TokenizeStep`/`MiniMaxMusic3AutoregressiveStep` source (`encoders.py` in the same
modular_pipelines directory) for the exact real special-token prompt format and per-codebook
generation loop; (3) the two big real downloads (`language_model/` ~17.2GB,
`transformer/` ~9.7GB) remain the real disk-space blocker -- everything else can be prototyped and
even partially golden-verified (condition_encoder, rvq_depth_decoder, vocoder in isolation) without
them.

## Vocoder ported, golden-verified, 2026-09-03 -- first real MiniMax-Music3 component landed

`src/OpenTail.Stingray.Diffusion/MiniMaxMusic3/MiniMaxMusic3Vocoder.cs`: real, transcribed directly
from the installed `diffusers` source (`minimax_music3_vocoder.py`). A single-parameter-Snake
DAC-style decoder -- `alpha` used DIRECTLY (no `exp()`, real `torch.ones` init), unlike ACE-Step's
Oobleck two-parameter LOG-SCALE Snake; confirmed from real source, not assumed from either prior
Snake variant this project has ported. Real "folded stereo" detail: the decoder network is MONO
(`latent_channels/2 = 64` input channels); stereo comes from reshaping the real `[128, T]` latent
into two `[64, T]` channel-streams (decoded independently, batch-folded in the real reference) --
NOT a stereo-aware architecture. Real channel progression `1536 -> 768 -> 384 -> 192 -> 96` across
4 blocks (strides `8,8,4,2`, hop=512), confirmed against the real checkpoint's own tensor shapes.

`MiniMaxMusic3VocoderGoldenParityTests`: loaded the real `diffusers.MiniMaxMusic3Vocoder` with the
same real weights (`load_state_dict(strict=False)` reported zero missing/unexpected keys,
confirming tensor-name understanding exactly) and diffed against a fixed-seed synthetic latent --
**passes on the FIRST run**, well under 0.1% relative error. This is the first real, verified
MiniMax-Music3 component -- structurally close to `Audio.Parler.DacDecoder`'s shape but kept
self-contained (CLAUDE.md rule 7 -- DRY once a second real caller of the SAME exact formula
exists, not speculatively here since Snake's parameterization differs from Parler's).

## RVQ depth decoder ported, golden-verified, 2026-09-03

`src/OpenTail.Stingray.Diffusion/MiniMaxMusic3/MiniMaxMusic3RvqDepthDecoder.cs`: the real "Local"
component. Real, simple architecture confirmed from source (`minimax_music3_rvq_depth_decoder.py`):
unlike every other transformer this project has ported this session, it uses LEARNED absolute
positional embeddings (`pos_embedding`, `max_position_embeddings=16`) added at the input rather than
RoPE -- standard causal full self-attention otherwise (no GQA, `heads=16, head_dim=256`), standard
pre-norm RMSNorm + SwiGLU blocks. Real per-step flow: the projected global hidden state followed by
up to 7 embedded residual-codebook codes (`audio_embeddings`, real row offset
`codebookIdx*audioVocabSize + code`), each step's hidden state feeding that step's own real output
head (`audio_heads`, 7 heads for the 7 residual codebooks c1..c7).

Loaded the real `diffusers.MiniMaxMusic3RVQDepthDecoder` with the same real weights (zero missing/
unexpected keys) and diffed against a fixed-seed synthetic input -- `MiniMaxMusic3RvqDepthDecoder
GoldenParityTests` passes on the FIRST run, well under 0.1% relative error.

## Condition encoder ported, golden-verified, 2026-09-03 -- all three small components now done

`src/OpenTail.Stingray.Diffusion/MiniMaxMusic3/MiniMaxMusic3ConditionEncoder.cs`: real, tiny (only
3 learned parameters: an 8-way softmax layer-mixing weight, a scalar scale, and a
`Conv1d(k=3,pad=1)` projection). Real per-frame flow: the 8 concatenated condition layers (1 global
+ 7 residual-codebook hidden states, matching `MiniMaxMusic3RvqDepthDecoder`'s own real per-step
output) get mixed with LEARNED softmax weights (ELMo-style), scaled, projected `4096 -> 2048`, then
nearest-neighbor-resampled from the language model's real 25Hz frame rate to the Flow-VAE's real
`44100/512 ≈ 86.13Hz` latent frame rate (ratio confirmed `≈3.4453125`, exact real formula from
source, not assumed).

Loaded the real `diffusers.MiniMaxMusic3ConditionEncoder` with the same real weights (zero missing/
unexpected keys) and diffed against a fixed-seed synthetic input -- passes on the FIRST run, well
under 0.1% relative error.

**All three small real inference-time components are now landed and golden-verified**
(`vocoder`, `rvq_depth_decoder`, `condition_encoder`) -- ~1.6GB of real weights, three real golden-
parity tests, all passing on their first run.

## Real autoregressive generation loop, fully specified from source, 2026-09-03

Read `diffusers/modular_pipelines/minimax_music3/encoders.py` in full (362 lines,
`MiniMaxMusic3TokenizeStep` + `MiniMaxMusic3AutoregressiveStep`) -- this is a REAL, extremely
precise specification (exact special-token ids, exact prompt string, exact CFG top-k masking) that
would be near-impossible to reconstruct correctly without the real source; worth recording in full
here rather than re-deriving later.

**Real prompt template** (`_clean_caption`/`_normalize_lyrics` do real, non-trivial text
normalization first -- markdown stripping, structural-tag lowercasing, `[verse]`-style tags forced
onto their own line):
```
<|im_start|><|caption_start|>{cleaned_caption}<|caption_end|><|lyrics_start|>{normalized_lyrics}<|lyrics_end|><|im_end|><|audio_start|>
```
Tokenized once via the real `Qwen2Tokenizer`; the REAL unconditional (CFG-null) branch is not a
separate empty prompt but the SAME token sequence with every token except the first and the two
trailing structure tokens replaced by a single real `_AUDIO_CFG_TOKEN_ID` (`151654`) -- both
branches run through the language model together as a real batch-of-2.

**Real special token ids** (checkpoint-specific, not derivable from the tokenizer's own vocab):
`_AUDIO_END_TOKEN_ID=151670`, `_AUDIO_CFG_TOKEN_ID=151654`, `_AUDIO_CODE_OFFSET=151675`,
`_SEMANTIC_VOCAB_SIZE=16384`. `_MAX_PROMPT_TOKENS=5000`, `_MAX_AUDIO_FRAMES=9000` (six minutes at
the real 25Hz frame rate).

**Real per-frame generation loop** (`MiniMaxMusic3AutoregressiveStep.__call__`):
1. Real `language_model.model.embed_tokens(text_ids)` -> real Qwen3 forward with `use_cache=True`
   -> `last_hidden`, real KV cache kept across the whole loop.
2. Real semantic-code sampling: `lm_head(last_hidden)` -> mask everything outside the real audio-
   code vocab range and the END token -> real CFG (`unconditional + (conditional-unconditional) *
   1.5`, `_AR_CFG_SCALE=1.5`) restricted to the conditional branch's real top-50
   (`_AR_CFG_TOP_K=50`) candidates before re-masking (avoids NaN from guiding two `-inf` logits) ->
   real top-50 (`_AR_SAMPLING_TOP_K=50`) multinomial sample. Sampling the real END token id stops
   generation.
3. Real depth-code generation (`_generate_depth_codes`, uses `MiniMaxMusic3RvqDepthDecoder` as a
   real autoregressive per-frame mini-transformer, NOT a single forward call): starts the real
   depth sequence with `[projection(last_hidden), projection(embed(semantic_code))]`, then for each
   of the 7 residual codebooks: run the depth decoder forward over the sequence so far, take the
   real LAST step's hidden state, apply that codebook's real output head, apply the SAME real CFG
   formula (`cfg_scale=1.5`), sample top-50, append the sampled code's real embedding
   (`audio_embeddings` row `code + (index-1)*audio_vocab_size`) projected into the sequence for the
   next depth step.
4. Real per-frame hidden state = `cat([last_hidden(global, 1 layer), depth_hidden(7 residual-
   codebook hiddens concatenated)])` -- confirms the exact real 8-layer concatenation this doc's
   earlier "hidden-state fusion" finding already established, now with the EXACT real assembly
   order (global first, then c1..c7 in codebook order).
5. Real feedback embedding for the next frame (`_embed_audio_frame`): sums the semantic code's real
   language-model token embedding with the SUM of all 7 residual codes' real `audio_embeddings`
   rows, then scales by `num_codebooks**-0.5` (`8**-0.5`) -- real, easy-to-miss normalization
   constant.
6. Real "off-by-one" detail: frame index 0 only advances the language model's state past the real
   `<|audio_start|>` token and does NOT emit a frame -- only `frame_index > 0` iterations get
   appended to `frame_hiddens`.

This is now a COMPLETE real specification for Phase B (Global) + Phase C (Local/depth) generation
-- the two big real downloads (`language_model/`, `transformer/`) are the only remaining blocker
before this loop can actually be exercised end-to-end; the logic itself needs no further real
archaeology to implement.

**Not yet done**: read `denoise.py`/`decoders.py`/`before_denoise.py` for the real chunked flow-
matching denoise loop's exact mechanics (the 200-frame-window/overlap-blend scheme is already
confirmed at a high level from the block docstrings, but not yet the real per-chunk math); the two
big downloads remain blocked on this session's current disk space.

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
