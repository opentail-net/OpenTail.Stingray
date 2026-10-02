# MiniMax-H3 port plan (video + native 32 kHz stereo audio)

**Status:** not started. **Policy:** port now, prove later; kept off the CLI's model list and the
diffusion docs until verified (CLAUDE.md rule 14; [ported-families-todo](ported-families-todo.md)).

## Architecture

Source: TensorSharp `docs/models/minimax-h3.md`, `Models/MiniMaxH3/` (20 files, including a 100%
pure-C# backend), upstream HF code (`MiniMaxAI/MiniMax-H3`).

One diffusion transformer denoises a **packed video+audio latent** in a single token sequence:
up to 15 s at 24 fps, plus a 32 kHz stereo soundtrack written as a sidecar `.wav`. Seven whole
networks:
- a text encoder (a Qwen3-VL-32B derivative, its own GGUF, no tokenizer embedded);
- a vision encoder (for image / reference conditioning);
- the DiT;
- encode + decode for the video VAE (3D, tiled; 5 latent frames at a time past 22 frames);
- encode + decode for the audio VAE.

Two denoiser checkpoints with different conditioning:
- `fl2va` (text + first/last keyframes; also t2v and i2v);
- `ref2va` (text + reference images, clips and soundtracks).

**Files** (about 35.5 GB per set):

| File | Size |
|---|---|
| denoiser GGUF (Q4_K) | ~10.6 GiB |
| `qwen3vl_32b_minimax_h3` text encoder | 17 GiB Q4_K_M, or 12.2 GiB Q2_K |
| video VAE (fp16 safetensors) | 5.2 GB |
| audio VAE (fp32) | 0.6 GB |

**Memory:** this fits the 64 GB host **for sequential loading of weights.** Networks load and
release in turn, so peak weight residency is about the largest single component (the ~17 GiB text
encoder), not the ~35.5 GB sum. Generation tensors, latent buffers, decoded video frames and audio
add RAM on top, so don't treat the summed checkpoint size, or the largest file alone, as the runtime
peak. Measure it.

**External-file dependencies (integration work, not detail):**
- **The text-encoder GGUF carries no tokenizer.** The tokenizer is an external `vocab.json` +
  `merges.txt` (TensorSharp documents this; a config cannot auto-fetch it).
- The VAEs are safetensors, not GGUF.
- The denoiser choice (`fl2va` vs `ref2va`) decides which conditioning modes work.

## Reuse in Stingray

- The diffusion stack: Wan / HunyuanVideo / LTX DiT blocks, 3D VAE decoders (`HunyuanVaeDecoder3D`),
  flow-matching schedulers, tiled VAE decode, CPU and Vulkan paths.
- Audio: VAE / vocoder pieces from the audio engines; WAV writing.
- The text encoder: Qwen3-VL text tower (`UnifiedVisionPipeline` / LLM forward for hidden states).

## New work (phases)

- [ ] **0. Read** `MiniMaxH3Pipeline.cs`, `MiniMaxH3DiT.cs` / `DirectDiT.cs`, `MiniMaxH3Layout.cs`
  (latent packing), `MiniMaxH3Scheduler.cs`, both VAEs, and the encoders. Write the spec here:
  token layout, conditioning injection per mode, scheduler, guidance.
- [ ] **1. Text encoder:** hidden-state extraction from the Qwen3-VL-32B GGUF (external tokenizer).
- [ ] **2. DiT** over the packed video+audio latent (CPU first; Vulkan later).
- [ ] **3. VAEs:** video (3D, tiled) and audio decode first, then encode for i2v / fl2v / ref.
- [ ] **4. Pipeline + scheduler:** t2v first, then i2v, fl2v, ref.
- [ ] **5. Output:** an MP4 (existing H.264 writer, if the diffusion stack has one) plus a sidecar
  WAV.
- [ ] **6. Gate:** an experimental flag; not listed in `stingray image` / video help until verified.

**Promotion is pipeline-specific,** like the other diffusion pipelines: per-network fixture checks,
an end-to-end deterministic clip, exposure in the CLI's video/diffusion registry, and a STATUS row.
It doesn't go through `ModelCompatibility` / `admit-arch`.

## Deferred (not in the initial port)

`ref2va` reference clips and soundtracks, i2v / fl2v until t2v is verified, GPU (Vulkan) paths,
long clips past the FP16 attention ceiling, and the HTTP API.

## Verification (levels as in [ported-families-todo](ported-families-todo.md))

1. **Independent implementation (level 3):** fixtures from the upstream HF/PyTorch reference
   (`MiniMaxAI/MiniMax-H3`): text embedding, one DiT step's velocity, VAE decode of a fixed latent,
   on the same seed and inputs. TensorSharp's own verification uses upstream/PyTorch fixtures, not
   self-comparison.
2. **Second reading:** TensorSharp's pure-C# backend on the same files (useful for localising
   differences; not independent of TensorSharp-derived code).
3. **End to end:** a short t2v clip at small size, judged visually with an audio sanity check. Only
   the per-network fixture checks count as verification.

**Effort:** port + per-network checks multi-day (about 3-5). The largest item; a diffusion project.
The closeout performance + DRY pass and promotion are separate.
