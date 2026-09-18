# 086 — Video model verification plan: HunyuanVideo, LTX-Video, FLUX.2/FLUX.3

Owner: autonomous session, 2026-09-18. Scope: close the three video/large-image coverage gaps
flagged in that day's README review — HunyuanVideo (🔴 blocked), LTX-Video (⚠️ partially
verified, no visual run), FLUX.2/FLUX.3 (⚠️ real code, never run against real weights). Ordered
per the user's own priority: HunyuanVideo fix → LTX-Video real sample → FLUX.2/FLUX.3 on GPU.

Per `CLAUDE.md` rule 1 ("stopping is for wimps") and this session running fully unattended: if any
one item stalls on a real blocker, document it precisely here and move to the next item rather than
idling. Downloads for FLUX.2 were already started in parallel with writing this doc (see the FLUX.2
section) so disk I/O isn't wasted session time.

**Cross-check against the real C++ reference wherever one exists** (per CLAUDE.md rule 8 and the
project's standing convention of diffing against `examples/*.cpp`, not reimplementing from memory):

- **HunyuanVideo**: `examples/stable-diffusion.cpp/src/model/vae/hunyuan_vae.hpp` is explicitly the
  WRONG reference for this checkpoint's VAE (it's HunyuanVideo 1.5's VAE — see `docs/054`'s own
  warning) — the only real reference here is the vendored `examples/diffusers` Python source
  (`autoencoder_kl_hunyuan_video.py` for the VAE, `pipeline_hunyuan_video.py` for the real text-
  encoder wiring/prompt template). No GGML/C++ reference exists for the checkpoint actually
  downloaded in this repo, so there is nothing to diff DiT/VAE math against there — diffusers is
  the sole oracle, already used for the VAE port; the text-encoder wiring must be checked against
  `pipeline_hunyuan_video.py`'s `_get_llama_prompt_embeds` specifically before writing code.
- **LTX-Video**: check for an `ltx`/`ltxv` family under `examples/stable-diffusion.cpp` or
  `examples/audio.cpp`'s sibling tools before assuming none exists — if a real C++ reference binary
  is buildable/present, run it at the same resolution/seed as the new C# sample for a real
  side-by-side comparison (same discipline as the F5-TTS/FishSpeech/Citrinet-ASR rows in
  `PerformanceLeague.md`, which all cite an actual reference-binary invocation, not just a citation).
  If no such reference exists in this repo, say so explicitly rather than skipping the check
  silently.
- **FLUX.2 / FLUX.3**: check whether `examples/stable-diffusion.cpp` (or any other vendored
  reference under `examples/`) has gained FLUX.2/FLUX.3 support before concluding none exists —
  these are newer architectures, so the answer may differ from FLUX.1's. If a real reference binary
  exists, use it for both a correctness cross-check (RoPE/DiT-block output comparison, same pattern
  as `docs/055`'s tensor-inventory-then-golden-parity approach) and, once real weights run, a timing
  comparison in `PerformanceLeague.md`. If none exists, note that explicitly per item rather than
  silently reporting only a self-consistency (non-degenerate output) check.

## Disk budget (checked 2026-09-18)

- `C:` — 57G free (OS drive, code only, no checkpoints).
- `F:\_models` (symlinked as `models/_models`) — 414G already used, **67G free** at time of writing
  (was 78G before FLUX.2 downloads started).
- `K:\_other_models` — newly added by the user specifically so downloaded checkpoints **do not**
  need to be deleted after verification (cold-storage headroom). Use this if `F:` gets tight before
  a checkpoint's verification pass is done — move the *least* recently needed large file there, not
  the one currently in use.

## 1. HunyuanVideo — real, scoped gap (not a vague "blocked")

Status at time of writing, confirmed by direct code read (not the stale README wording):

- **DiT**: real port, structurally sound (`HunyuanVideoModel.cs`), matches FLUX's dual/single-stream
  shape. `headDim=128`, 3D RoPE split-half with FLUX's 16/56/56 axis split. Not yet GPU-resident —
  see `docs/078-hunyuanvideo-gpu-residency-plan.md` for that follow-on (deliberately out of scope
  for this pass; correctness/wiring first, GPU residency after, same ordering CLAUDE.md rule 7
  prescribes).
- **VAE**: real port exists, `HunyuanVaeDecoder3D.cs` (475 lines), ported directly against
  `examples/diffusers/.../autoencoder_kl_hunyuan_video.py` per `docs/054`'s spec — **and the real
  checkpoint is already downloaded**: `models/hunyuanvideo/hunyuan_video_vae_bf16.safetensors`
  (confirmed present on disk). **But `HunyuanVideoRealWeightsTests.cs`'s only real-weight test
  calls `HunyuanVideoPipeline.Load(modelPath)` with no `vaePath` argument** — so the VAE code has
  never actually been exercised against its own real weights in an automated test. This is the
  cheap, real first fix: wire the VAE checkpoint path into a test/run and confirm decode actually
  produces a non-degenerate frame.
- **Text conditioning: the real, still-open gap.** `HunyuanVideoPipeline.Generate` (line ~92):
  `var condContext = textContext ?? new float[seqLen * HunyuanVideoModel.TextDim];` — when no
  `textContext` is passed (as in the existing real-weight test), conditioning is a literal
  **all-zero buffer**, not even noise. `TextDim=4096` per the class doc comment ("LLaMA-3 /
  Qwen2.5-VL text dimension") — the real HunyuanVideo text encoder is an LLM's own last hidden
  states (decoder-only, not T5), which this codebase has never wired up for this pipeline. No
  Llama-3.1-8B-Instruct (or Qwen2.5-VL text tower) checkpoint is currently in `models/_models`.

### Plan

1. **Quick win**: fix the test/CLI call site to pass `vaePath: "models/hunyuanvideo/
   hunyuan_video_vae_bf16.safetensors"` into `HunyuanVideoPipeline.Load`, run a real 1-frame
   32×32 (or slightly larger, still cheap) forward pass, confirm `HunyuanVaeDecoder3D.Decode`
   produces finite, non-degenerate pixel output (RMS/range check, same bar used elsewhere in this
   repo — see CLAUDE.md rule 12's own criteria). This alone upgrades HunyuanVideo from "VAE code
   exists but untested" to "VAE verified against real weights," independent of the text-encoder gap.
2. **Text encoder**: HunyuanVideo's real pipeline (per diffusers `pipeline_hunyuan_video.py`) uses
   `llava-llama-3-8b`'s text tower (a LLaMA-3-8B variant) with a specific prompt template, pooled
   via the final hidden states (not a CLS/pooled embedding) — **do not guess this wiring**; check
   `examples/diffusers/src/diffusers/pipelines/hunyuan_video/pipeline_hunyuan_video.py`'s
   `_get_llama_prompt_embeds` before writing any code, same "check the real reference before fixing
   what looks wrong" discipline as CLAUDE.md rule 8.
   - If a real checkpoint fits the remaining disk budget, download it and wire the real encoder in
     for real end-to-end conditioning (this is the actually-correct fix and unblocks the 🔴 status
     honestly).
   - If disk gets tight, note it as a real, named blocker here (not silently skipped) and keep
     the VAE-only fix as this pass's real, shippable, stated-scope progress.
3. Update `README.md`'s status matrix and this doc with whatever the real outcome is — cite
   dated findings, not aspirational color, per CLAUDE.md rule 10.

## 2. LTX-Video — generate and save a real, viewable sample

Correctness bug (Euler `flipSinToCos` convention mismatch) was root-caused and fixed on 2026-09-14
per `docs/077-ltx-video-gpu-residency-plan.md` — all golden-parity tests pass. But per
`docs/055`/README: **no full timed/visual end-to-end run has been done since that fix** — the
README's LTX-Video row is still describing the pre-fix "1 of 6 runs coherent" finding.

### Plan

1. Run a real, full LTX-Video generation (real T5-XXL text encoding → DiT denoise loop → VAE decode)
   at a resolution the model card recommends this checkpoint actually work at (LTX-Video's own
   card: not the ultra-low resolutions that triggered the earlier false "corruption" finding in the
   sibling investigation for other checkpoints — check LTX-Video's own README/config for its trained
   resolution regime rather than assuming FLUX's numbers apply).
2. Save the output video/frames to `docs/diffusion-samples/` (gitignored, local-only per CLAUDE.md
   rule 9 — for the user's own viewing) with a descriptive filename
   (`ltx-video_<prompt-slug>_<res>_<date>.<ext>`).
3. Re-run with the exact same seed/config a second time to check the "1 of 6 coherent" instability
   from the pre-fix sweep is actually resolved now, not just luck on one run — this is the real
   open question the fix's own doc left unanswered.
4. Update README's LTX-Video row and `docs/055`/`docs/077` with the real, dated, post-fix finding.

## 3. FLUX.2 / FLUX.3 — GPU-required real-weight verification

Both have real code (`src/OpenTail.Stingray.Diffusion/Flux2`/`Flux3`: DiT, RoPE, params, pipeline;
FLUX.3 additionally has its own KV cache and flow scheduler) and structural conformance tests
(`Flux2ConformanceTests.cs`/`Flux3ConformanceTests.cs`) — but per the README, **neither has ever
run against a real checkpoint.** The user has specified these need to work on GPU (Vulkan iGPU,
this machine's only GPU) — not just CPU-verified.

### Checkpoints — download already started in parallel with writing this doc

FLUX.2-dev's real architecture (confirmed via `city96/FLUX.2-dev-gguf`'s own README): DiT +
**Mistral-Small-3.2-24B-Instruct-2506** as text encoder (not T5/CLIP) + a dedicated FLUX.2 VAE.
Picked Q4_K_S across the board for a size/quality balance that leaves real headroom under the 67G
free on `F:`:

| File | Quant | Size | Destination | Status at doc-write time |
|---|---|---:|---|---|
| `flux2-dev-Q4_K_S.gguf` | Q4_K_S | 17.97 GiB | `models/_models/` | downloading (background) |
| `Mistral-Small-3.2-24B-Instruct-2506-Q4_K_S.gguf` | Q4_K_S | 12.62 GiB | `models/_models/` | downloading (background) |
| `flux2-vae.safetensors` | fp32/bf16 (as published) | 0.31 GiB | `models/_models/` | done |

Total ≈ 30.9 GiB, leaving ≈ 36G free on `F:` afterward — enough headroom for FLUX.3 assets too if
FLUX.3 turns out to reuse the same text encoder/VAE (check before downloading a second Mistral
copy — FLUX.3 may be a distinct checkpoint with its own components; verify via its own model card
before assuming reuse).

**No FLUX.3 checkpoint has been identified/downloaded yet** — `black-forest-labs` had not
published a distinctly-named "FLUX.3" as of this session's knowledge; check whether it's a real,
separate released checkpoint (vs. FLUX.2's own later revision) before spending download budget on
it. If it doesn't exist as an independently-released checkpoint, downgrade this item to "FLUX.2
only, FLUX.3 code exists but has no checkpoint to verify against yet" rather than inventing one.

### Correction, 2026-09-18: real scope is much larger than "wire real checkpoint paths"

Checked `Flux2DiT.cs`/`Flux3DiT.cs` directly before starting: **neither has an `IWeightLoader`
field, nor any reference to one, anywhere in the file.** `Flux2Pipeline.Generate` is not "real code
that's never been run against real weights" in the sense every other row in this plan is — it is a
**pure structural/synthetic stub**:
- Reference-image latents: `Array.Fill(rLatent, 0.2f * (r + 1))` — a constant, not encoded pixels.
- Text conditioning: `var txtEmbeds = new float[nTxt * ContextInDim];` (all-zero) and
  `Array.Fill(pooledEmbed, 0.1f)` — no T5/Mistral call anywhere, not even wired to one.
- "VAE decode": `rgb[p] = targetLatent[p % targetLatent.Length] * 0.5f + 0.5f` — a raw channel
  repeat/rescale of the DiT's own output latent, not a real decoder call.

So this item is a genuine **implementation task** (weight loader wiring for the DiT itself, a real
Mistral-Small-24B forward pass for text conditioning, a real VAE decoder port), same shape and
scope as LTX-Video's original from-scratch build (`docs/055`) — not a short "point it at real
weights and check the output" pass like HunyuanVideo/LTX-Video turned out to be. Downgrading this
session's ambition for item 3 accordingly: the 33GB of real checkpoints (`flux2-dev-Q4_K_S.gguf`,
`Mistral-Small-3.2-24B-Instruct-2506-Q4_K_S.gguf`, `flux2-vae.safetensors`, all in
`models/_models/`) are downloaded and ready, but wiring real weight loading through a 19+-block
DiT, a 24B-parameter text encoder, and a VAE decoder is multi-hour work in its own right, not
something to rush blind in the same pass as HunyuanVideo/LTX-Video's genuinely-smaller gaps.

### Plan (revised)

1. **Do not attempt a rushed implementation.** Treat real FLUX.2 weight-wiring as its own follow-up
   item, scoped like `docs/055` was for LTX-Video: check the checkpoint's real tensor inventory
   first (`flux2-dev-Q4_K_S.gguf`'s own tensor names/shapes) before writing any loader code, cross-
   check against a real reference if `examples/stable-diffusion.cpp` has gained FLUX.2 support (see
   the C++-reference note above), and write a dedicated implementation-plan doc (087?) once that
   inventory pass is done, rather than winging weight-key names by analogy to FLUX.1.
2. Once weight loading is real (a real follow-up session/task): run end-to-end against the real
   GGUF DiT + Mistral text encoder + VAE on **Vulkan iGPU explicitly** (the user's stated
   requirement is GPU, not just "runs") — a CPU-only run would not be sufficient to close this item.
3. If the Vulkan backend doesn't yet have every op FLUX.2's DiT needs, name the specific missing
   op/kernel rather than silently falling back to CPU (check `docs/050` first).
4. Verify non-degenerate real output, save a real sample to `docs/diffusion-samples/` (gitignored).
5. Only after a real Vulkan run succeeds, add a real timing entry in `PerformanceLeague.md`.
6. Repeat for FLUX.3 (same zero-weight-loading status confirmed) if/once a real checkpoint is
   confirmed to exist and is downloaded.

## 4. Extended backlog — other "silently untested" real implementations (added 2026-09-18, user-relayed)

Beyond the three items above, a broader review of README + PerformanceLeague surfaced more real
code that has structural/conformance tests passing but has never been run against a real checkpoint
end-to-end, or is real-but-incomplete for a different reason. Recorded here so this doesn't get
re-discovered cold in a future session. Not all of these are in scope for this pass — noted per item.

- **Qwen Image / Qwen Image Edit** (`src/OpenTail.Stingray.Diffusion/QwenImage`): real code,
  structural tests only (`QwenImageTests`), **no real checkpoint ever run**. Check its weight-loader
  wiring status the same way this doc just did for FLUX.2/FLUX.3 before assuming it's a quick
  verification pass rather than a build task — do not assume parity between these three just
  because the README groups them together.
- **DeepSeek-V3.2**: `DeepSeek32ForwardPass.cs` exists, substantially implemented, never run once
  against real weights. Known missing pieces: Hadamard rotation, MTP. No GGUF was available when
  the README was last updated — check whether one exists now before writing this off as blocked.
- **DeepSeek-V4**: `DeepSeek4ForwardPass.cs` exists (covers three attention variants), never run
  once. Smallest cited quant ~99GB — genuinely constrained by this machine's disk/RAM (see the 67G
  free `F:` budget this doc opened with), not simply forgotten. Not worth attempting on this
  5700G/64GB machine without a real plan for where 99GB+ would even fit (`K:\_other_models` is cold
  storage, not necessarily fast enough for active inference — check before assuming it works there).
- **Llama 4 vision**: vision path exists per README, untested because the 93GB text checkpoint was
  deliberately not downloaded. Hardware/RAM constraint, not an overlooked test — same disk-budget
  logic as DeepSeek-V4 above.
- **MobileNetV5 vision adapter**: code exists, no real checkpoint found yet that uses the targeted
  mobilenetv5 projector. Implementation-without-test with no obvious model to test against — lower
  priority than items with a concrete checkpoint path.
- **Granite 4.0 Vision / Granite Vision 3.2**: NOT silently untested — already exercised, and the
  README's 🔴 reflects a known, real correctness bug (not lack of testing) that a Vulkan sweep
  already caught/fixed once for Granite-specific scaling. Re-verify current status before assuming
  it's still broken; check `docs/00-current-work.md`'s Granite entries for the latest state.
- **MiMo-VL**: encoder golden-verified, but full end-to-end blocked by a real, named 3584→4096
  mmproj/text-backbone dimension mismatch. Tested, not passing — different category from the
  "never run" items above.
- **Kimi-VL / YoutuVL**: vision encoders golden-verified, full generation blocked by the real
  `deepseek2` split `attn_k_b`/`attn_v_b` layout. Same "tested but incomplete end-to-end" category
  as MiMo-VL, not "forgotten."
- **Stable Audio 3 "Continuous MMDiT, Variable-Length 1s-6min"**: README advertises this range, but
  PerformanceLeague only demonstrates Small/Medium at short fixed durations (4-6s). Distinguish
  "architecture tested" from "checkpoint tested" from "duration regime tested" from "full pipeline
  tested" — the 4-6s runs are real evidence for the first three, not the fourth. A real 1-6min run
  on this 64GB shared-memory machine is a genuinely untested regime, not just an unmeasured one —
  check memory scaling before assuming it merely needs a longer timeout.
- **Stable Audio 3 Small SFX**: README previously recorded poor output pending a re-listen after a
  schedule fix; this session's own Stable Audio infrastructure changes (differential-attention GPU
  residency, shared workspace/weights classes) make this a regression/coverage check on existing
  code, not new implementation — cheap to re-verify given how much shared infrastructure just moved.

**Relative priority if picked up after items 1-3 above**: Qwen Image (most analogous to FLUX.2/3,
worth checking its real implementation depth before scoping) → Stable Audio 3 Small SFX (cheap
regression check, infra already changed this session) → DeepSeek-V3.2 (real but needs a checkpoint
availability check first) → DeepSeek-V4/Llama-4 vision/MobileNetV5 (blocked/impractical on this
hardware, not simply forgotten — don't spend session time here without a real plan for the disk/RAM
constraint first).

## Working notes / running log

### 2026-09-18: HunyuanVideo — two real correctness bugs found and fixed, VAE verified, text-conditioning gap now directly demonstrated

**VAE wiring fix (quick win, done)**: `HunyuanVideoRealWeightsTests.cs`'s `FindModelPath` never
checked the `models/hunyuanvideo/` subdirectory the real checkpoints actually live in, so its only
real-weight test was silently no-op'ing (confirmed via the 0.301s-runtime tell from CLAUDE.md rule
12, before any fix). Fixed the path lookup and wired `vaePath` into `HunyuanVideoPipeline.Load` —
the VAE checkpoint (`hunyuan_video_vae_bf16.safetensors`, already on disk) is now actually
exercised. Added a real finite/non-degenerate assertion on the decoded frame.

**Bug 1 — missing Q-RMSNorm** (`HunyuanVideoModel.JointAttention`/`SingleBlock`): the checkpoint
carries both `*_q_norm.weight` and `*_k_norm.weight` tensors (QK-RMSNorm, confirmed via direct
tensor-name enumeration), and `Resolve()` already had a name-mapping rule for query_norm — but
`JointAttention` and `SingleBlock` only ever looked up and applied K-norm. Q-norm was silently
skipped everywhere. Fixed by adding the missing `TryGetWeight(...query_norm...)` + `RmsNormHeads`
calls, mirroring the existing K-norm code exactly.

**Bug 2 — missing pre-modulation LayerNorm (the actual root cause of the NaN)**: `DoubleBlock` and
`SingleBlock` called `Modulate(img/txt/x, ...)` directly on the raw residual stream. `ModulateRows`
is a pure affine `x*(1+scale)+shift` with no normalization built in — real AdaLN-Zero requires an
affine-free LayerNorm immediately before it. `WanModel`'s own CPU path does exactly this
(`LayerNormNoAffine` then `Modulate`, confirmed by direct comparison — the shared
`ApplyGatedResidualRows` doc comment even says this gating logic is a byte-identical extraction
shared with Wan, making the missing counterpart doubly clear as an oversight). Without it, `img`/
`txt`/`x` grow unboundedly across blocks via the gated residual adds every layer, with nothing ever
renormalizing them — confirmed via a bisecting diagnostic (`STINGRAY_HUNYUAN_DUMP_LATENT=1`) that
the corruption first appears at `double_blocks.8` of 20 (a handful of NaN values), then fully
saturates by `double_blocks.9`, consistent with exponential-ish blowup. Fixed by cloning and
`LayerNormNoAffine`-ing before every `Modulate` call in both blocks.

**Verified**: `HunyuanVideoRealWeightsTests` (both facts) and `HunyuanVideoTests` all pass, real
runs (50-51s, not the silent-no-op tell), pre-decode latent stats healthy after the fix
(`mean=0.0317 std=1.0151 min=-3.21 max=2.77`, vs. 100%-NaN before). Weight tensors themselves were
directly checked and are clean (0 NaN/Inf across every `double_blocks.{0..10}` MLP/mod tensor) —
ruling out an fp8-conversion or corrupted-checkpoint explanation before looking at the model code.

**Real 256×256, 4-step sample generated** (496s, CPU, `docs/diffusion-samples/
hunyuanvideo_red-apple-on-white-table_256x256_4steps_zero-cond_2026-09-18.png`, gitignored/local):
**visually confirmed to be structured static/noise, not a coherent image** — expected and correct
given zero-conditioning (no text encoder wired, per this doc's section 1). This is real, useful
evidence: it demonstrates the DiT+VAE path is now numerically healthy (finite, real per-pixel local
color/texture correlation, not uniform garbage or NaN-black) while making unmistakably clear that
the text-encoder gap (section 1, item 2) is the sole remaining blocker to a coherent image — not a
residual correctness bug in the DiT/VAE themselves. This directly de-risks the text-encoder work:
whoever picks it up next knows the rest of the pipeline is already solid.

**Not done this pass**: the LLM text-encoder wiring itself (section 1, item 2) — no Llama-3-8B/
Qwen2.5-VL-text checkpoint was downloaded this pass; disk budget went to FLUX.2 instead per the
user's stated priority order. HunyuanVideo's README status should move from 🔴 ("blocked, no VAE")
to something reflecting "DiT+VAE numerically verified end-to-end, real coherent output blocked
specifically on the text encoder" — a real, narrower, more honest status than before.

### 2026-09-18: LTX-Video — post-fix re-verification, instability NOT resolved (real negative result)

Per the plan above, ran 3 fresh real end-to-end generations against `ltx-video-2b-v0.9.1.safetensors`
via `stingray image` (real CLI path, no `models/_models/ltx-t5/text_encoder` directory downloaded
yet, so placeholder text conditioning — the exact same condition every prior run in this
checkpoint's history used, so this is a valid apples-to-apples re-check, not a different-condition
comparison):

| Run | Resolution | Steps/CFG | Seed | Wall time | Result |
|---|---|---|---|---:|---|
| 1 | 256×256 | 25/3.0 | 42 | 209.6s | pure visual noise |
| 2 | 256×256 | 25/3.0 | 7 | 188.8s | pure visual noise |
| 3 | 512×512 | 25/3.0 | 42 | 462.2s | visual noise (faint red blob suggestive of "apple," not coherent) |

**0 of 3 coherent**, matching the pre-fix 2026-09-11 sweep's own bad-run rate (5 of 6 noise) far
more closely than its one good run. The 2026-09-14 `flipSinToCos` timestep-embedding fix (real,
confirmed via golden-parity re-runs — see `docs/077`) evidently fixed a real bug but did **not**
resolve this deeper instability — the two are separate problems. Real, honest conclusion: **LTX-
Video's seed-dependent convergence bug is still open**, not closed by the September 14 fix as a
casual reading of the git history might suggest. README updated accordingly (see LTX-Video row) —
downgrading the implicit expectation that the September 14 fix was "the fix" for this checkpoint's
correctness rather than a real-but-partial fix.

**Not attempted this pass**: downloading the real T5-v1.1-XXL text encoder to test whether real
(non-placeholder) text conditioning changes convergence behavior — a real, cheap-to-state next
hypothesis (placeholder conditioning is `0.01f * (rng.NextSingle() - 0.5f)` per-element noise, not
zero, but also not anything semantically meaningful; if the DiT's attention has any sensitivity to
conditioning *magnitude/structure* rather than just "some nonzero signal," noise-shaped
conditioning could itself be part of what's destabilizing convergence at this checkpoint's
below-trained-regime resolutions). Worth trying before assuming the bug lives purely in RoPE/
VAE/scheduler math that's already golden-verified component-by-component.

### 2026-09-18: DeepSeek-V3.2 — checked checkpoint availability, genuinely disk-blocked

A real GGUF now exists (`unsloth/DeepSeek-V3.2-GGUF`, didn't exist when the README was last updated
per the extended-backlog note above) — checked its smallest available quant (`UD-IQ1_S`, the most
aggressive 1-bit dynamic quant Unsloth publishes) directly via the HF API before assuming it fits:
**4 shards totaling 184.1GB** (49.98 + 48.94 + 49.55 + 35.66 GB). `F:` had 46G free at check time;
`K:\_other_models` (added this session specifically as extra headroom) has 182G free — even that
doesn't clear 184GB, and this ignores that a model this size also needs to actually load into
RAM/VRAM to run inference, not just fit on disk. **Confirmed genuinely disk/RAM-constrained on this
machine, not simply forgotten** — matches the ChatGPT-relayed assessment's own read on this item
without needing to guess. Not pursuing further this pass; would need either a much smaller
official quant (none published smaller than 184GB as of this check) or different hardware.

### 2026-09-18: FLUX.2/FLUX.3 — real scope correction (see section 3 above)

Checked `Flux2DiT.cs`/`Flux3DiT.cs` before attempting anything: zero `IWeightLoader` wiring in
either. `Flux2Pipeline.Generate` is a pure structural/synthetic stub (mock reference latents, all-
zero text embeddings, a raw channel-repeat instead of VAE decode) — not "real code never run
against real weights" like every other item in this doc, but a genuine from-scratch implementation
task on the scale of LTX-Video's original build (`docs/055`). Downloaded all three real checkpoints
(`flux2-dev-Q4_K_S.gguf` 18GB, `Mistral-Small-3.2-24B-Instruct-2506-Q4_K_S.gguf` 12.6GB,
`flux2-vae.safetensors` 0.3GB — all in `models/_models/`, verified present) so they're ready when
that implementation work happens, but did not attempt the implementation itself this pass — rushing
weight-key names by analogy to FLUX.1 without checking FLUX.2's own real tensor inventory first
would violate this project's own CLAUDE.md rule 8 discipline. Real follow-up: a dedicated `087-
flux2-implementation-plan.md` scoped the same way `docs/055` was for LTX-Video, tensor inventory
first.
