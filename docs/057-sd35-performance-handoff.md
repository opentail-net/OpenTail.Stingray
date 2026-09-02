# 057 — SD3.5-medium performance handoff

**Status: paused 2026-09-02, ready to hand off.** This doc is written as a self-contained brief
for whoever (human or AI) picks this back up — you should not need to re-read the whole session
history to continue.

## The task

Make `Sd3Pipeline`/`MMDiTModel` (`src/OpenTail.Stingray.Diffusion/SD3/`) fast enough to iterate on
practically — the current CPU path is too slow to use as a normal dev loop. This is now a
**performance** problem, not a correctness one: every bug found by structural comparison against
the real reference has been found and fixed this pass (see below), and the pipeline runs the full
24-block trunk with zero crashes. What's genuinely unverified is whether the OUTPUT is numerically
correct, and that can't be checked yet because a single real run doesn't finish in a practical
amount of time.

## What's already fixed (five real, verified bugs)

All five are already applied in `src/OpenTail.Stingray.Diffusion/SD3/MMDiTModel.cs` and
`Sd3Pipeline.cs`. Do not re-investigate these — they're confirmed against the real vendored
`examples/diffusers/src/diffusers/models/{transformers/transformer_sd3.py,attention.py,
normalization.py}` source, not guessed, and each is covered by an inline doc comment at its fix
site:

1. **CLI could only load one gated checkpoint layout.** `Sd3Pipeline.Load` only matches the
   StabilityAI single-file `..._incl_clips[_t5xxlfp8].safetensors` export, which is gated on
   HuggingFace with no ungated mirror. Added `Sd3Pipeline.LoadSeparate` (independent
   clip-l/clip-g/transformer/vae files) and a `--clip-g` CLI flag; `RunSd3` branches to it when
   `--clip-l`/`--clip-g`/`--vae` are all given.
2. **Fused-QKV misassumption.** `MMDiTModel` read three separate `qkv.0/1/2` weight matrices per
   block; the real checkpoint stores ONE fused `qkv.weight` `[dim, 3*dim]`. This alone made every
   real checkpoint unloadable — exactly why the pipeline had never been run once before this pass.
3. **Missing QK-RMSNorm.** Real `attn.ln_q.weight`/`attn.ln_k.weight` tensors exist and were never
   read or applied.
4. **Missing SiLU before every AdaLN modulation linear.** Real `emb = self.linear(self.silu(emb))`
   — `MMDiTModel` applied no SiLU at any of its three modulation call sites.
5. **SD3.5-medium's real "dual-attention" (MMDiT-X) extension was entirely unimplemented.** The
   first 13 of this checkpoint's 24 blocks each declare a full SECOND attention module
   (`x_block.attn2.*`) with a 9-chunk (not 6-chunk) modulation; real `JointTransformerBlock(
   use_dual_attention=True)` runs this as an extra image-only self-attention pass, gated and added
   as a second residual before the MLP. Also implemented `context_pre_only` handling for the last
   block (its `context_block` uses a different 2-chunk modulation with no gate/MLP at all).

See `docs/00-current-work.md`'s "SD3/3.5 run for the first time ever" section for the full
per-bug narrative if you want more context on any of these — but you should not need to
re-derive or re-verify them.

## The remaining problem: CPU is too slow to verify correctness

A 512×512, 15-step CPU run (Q8_0 GGUF DiT + fp16 HF diffusers text encoders/VAE) did not finish
inside a 15-minute wait (no crash, no error — it was still denoising when stopped). A follow-up
4-step run was still in flight when this pass was time-boxed and stopped, so **no completed timing
number exists yet for this checkpoint at this resolution**. Do not assume any specific per-step
time from this doc — measure fresh.

What's known architecturally that affects the perf profile specifically for SD3.5-medium (as
opposed to SD3.5-large, which doesn't have dual-attention at all, or SD3-medium/large before it,
which have neither dual-attention nor the same block count):
- 24 joint blocks, `HiddenSize=1536`, `NumHeads=24`, `HeadDim=64`.
- The first 13 of 24 blocks run a full SECOND self-attention pass (`attn2`) end-to-end (fused QKV
  projection, QK-norm, attention, output projection) — roughly **1.5-2x the per-block compute**
  for those 13 blocks compared to a plain (non-dual-attention) block.
- At 512×512 with `patch_size=2`, that's a `(512/8/2)^2 = 32^2 = 1024`-token image stream (VAE
  8x-downsamples first) attending jointly with whatever the T5/CLIP text-token count is — the
  attention itself is `O(n^2)` in that combined token count, on top of the doubled-compute blocks.

## Where to look first (suggested priority order, not mandatory)

1. **Measure before optimizing.** Get one real, complete timing number first — a full run at a
   SMALL resolution (try 128×128 or 256×256 first; SD3.5's real minimum resolution constraints
   haven't been checked, but VAE downsampling + patch_size=2 means width/height need to be
   divisible by at least 16) and low step count (4), so you have a real baseline to compare
   against before touching anything. This project's own `CLAUDE.md` rule 7 applies directly here:
   "measure, don't assume... only keep a change if it's measurably better."
2. **Check whether GPU offload (`-g -1` / Vulkan / CUDA) already works and is faster**, before
   assuming CPU-only optimization is the right lever. `MMDiTModel`'s `Lin` helper already has a
   `_backend`-gated GPU dispatch path (`GetGpuWeight`/`_backend.Upload`/`_backend.Allocate`) —
   check whether it actually gets exercised correctly for this pipeline (it was never run once
   before this pass, so this GPU path is equally unverified) before assuming it's a free win.
3. **`DiffusionOps.Linear`'s CPU path** (the function whose `AccessViolationException` led to
   finding bug #5 above) is a plain `Parallel.For`-over-`outDim` scalar dot-product loop
   (`TensorPrimitives.Dot`-vectorized per output row) — reasonable but not obviously
   SIMD/cache-optimal for a 1536-wide hidden size at this many blocks/steps. Profile before
   assuming this is the hot path, though — the joint attention (`JointMultiHeadAttention`,
   `O(n^2)` over the combined ~1024+ token sequence) is a real candidate too, especially since it
   now runs TWICE (once for `attn`, once for `attn2`) on the first 13 blocks.
4. **Once a real baseline number exists**, decide whether this is a "make the existing kernels
   faster" problem (parallelization, vectorization, avoiding redundant allocations — `MMDiTModel`
   allocates a fresh `float[]` for every intermediate at every block, e.g. `SplitQkv`/`ModulateNorm`
   both `.Clone()`/allocate rather than reusing scratch buffers) or a "this resolution/step count is
   just not viable on CPU, GPU is the real answer" problem. Don't assume before measuring which one
   it is.

## How to reproduce

Checkpoints are NOT vendored (deleted after every pass per this project's convention — see
`CLAUDE.md`). Re-download (all ungated, no HF auth needed):

```
DiT (real StabilityAI joint_blocks/x_embedder naming, NOT the diffusers-native
     transformer_blocks re-export — see docs/00-current-work.md for why the diffusers-native
     variant doesn't work with MMDiTModel as written):
    city96/stable-diffusion-3.5-medium-gguf → sd3.5_medium-Q8_0.gguf (or a smaller quant, e.g.
    Q4_K_M, for faster loading/iteration — MMDiTModel reads whichever dtype is in the GGUF)

Text encoders + VAE (standard HF diffusers multi-file layout):
    ckpt/stable-diffusion-3.5-medium (or adamo1139/stable-diffusion-3.5-medium-ungated, same repo
    contents, both ungated mirrors of the gated stabilityai/stable-diffusion-3.5-medium):
      text_encoder/model.fp16.safetensors    (CLIP-L)
      text_encoder_2/model.fp16.safetensors  (OpenCLIP-bigG)
      vae/diffusion_pytorch_model.safetensors
```

```bash
dotnet run --project src/OpenTail.Stingray.Cli -c Release -- image \
  -m sd3.5_medium-Q8_0.gguf \
  --clip-l text_encoder/model.fp16.safetensors \
  --clip-g text_encoder_2/model.fp16.safetensors \
  --vae vae/diffusion_pytorch_model.safetensors \
  --clip-tokenizer models/clip_tokenizer.json \
  -p "a red apple on a wooden table" \
  --steps 4 --seed 42 --verbose -W 256 -H 256 -g 0 \
  --output sd35-out.png
```

Start smaller (256×256 or even 128×128, 4 steps) than the 512×512/15-step repro that didn't
finish in 15 minutes above — get a real, complete number first, then scale up deliberately.

## House rules for whoever picks this up (from this project's `CLAUDE.md`)

- **No subagents** — do all work directly in the main session for this project.
- **Measure, don't assume.** A plausible-sounding optimization that isn't actually measurably
  faster (interleaved control/candidate samples, not a single run) gets reverted, even if the
  reasoning behind it seemed sound. Write the measured numbers down (in this doc or
  `docs/00-current-work.md`), not just "should be faster."
- **Check the real reference before "fixing" code that looks wrong** — every bug fixed this pass
  was confirmed against the real vendored `examples/diffusers` source before writing code, not
  guessed. If you find yourself wanting to change math (not just performance), apply the same
  discipline.
- **Delete transient checkpoints after use** — do not commit large model files.
- **Once correctness is actually verified (a real, recognizable image), that's the point to loop
  back to `docs/00-current-work.md` and update SD3/3.5's status** — this handoff is scoped purely
  to "make it fast enough to check," not "confirm it's right." Those are two different, sequential
  jobs; don't skip straight to declaring victory on the second without actually looking at output.
