# CosyVoice3 LLM forward-pass bug — request for advice

## Context

I'm porting CosyVoice3's TTS pipeline from a real C++ reference implementation
(`cosyvoice.cpp`, a llama.cpp-style GGUF/GGML port) to a native C# engine
(OpenTail.Stingray). The pipeline is: text → LLM generates speech tokens →
flow encoder → DiT (conditional flow matching) → HiFT vocoder → waveform.

I already found and fixed two real bugs this session (a HiFT vocoder padding
bug, and a missing LLM zero-shot-conditioning gap — the LLM wasn't being told
about the reference speaker's prompt text/tokens at all). Since fixing those,
**every stage of the pipeline except one is now numerically proven correct**
against the C++ reference, given identical real inputs (I built real
comparison harnesses that dump intermediate tensors from both sides and
compute cosine similarity):

- Frontend extraction (speaker x-vector, reference-audio speech tokenizer,
  mel extractor): cosine similarity 1.000000, and the speech tokenizer
  produces an exact 65/65 token match against the reference.
- Flow encoder output tensors (`mu`, `spks`, `conds`): cosine similarity
  1.000000 on all three, given identical token/embedding inputs.
- The DiT (flow-matching ODE solve) + HiFT vocoder, run end-to-end: when fed
  the reference's own real tokens/embedding/prompt-mel, produces
  correctly-transcribed, clean speech (verified with Whisper ASR).

**The one broken piece: the LLM's own forward pass** (a plain Qwen2
architecture backbone — 896 hidden dim, 14 attention heads, 2 KV heads, 4864
FFN dim, 24 layers, head_dim 64, RoPE — reading real GGUF weights). Given the
IDENTICAL real composed input token sequence (sos token + instruction-prefix
text tokens + an "endofprompt" special token + the reference audio's own real
transcript tokens + the new synthesis text's tokens + a task token + the
reference audio's own real prompt speech tokens — all real values, dumped
from an actual working run of the C++ reference), my C# engine's first-step
logits do NOT match the reference's own dumped logits for that same sequence:

- **Cosine similarity: 0.30** (should be ~1.0)
- **Argmax is a completely different token** (mine: token 2387; reference's:
  token 4011 — not a near-miss along a shared ranking)
- My top logit is about 4x larger in magnitude than the reference's (17.9 vs
  4.3) — but since the argmax itself differs, I don't think this is just a
  scale/temperature difference; the underlying computation seems to diverge
  in kind, not just magnitude.

The full real sequence is roughly 140 token positions long (65 real prompt
speech tokens + ~10 text/instruction tokens + ~1 task token + a handful of
special tokens). I have NOT yet tested whether a much SHORTER sequence (say,
under 20 positions, skipping the reference-audio conditioning) matches
correctly or diverges the same way — that's the next thing I plan to check.

## The audible symptom (real, from actually listening to the output)

Before my fixes, the output was a high-pitched "drone"/"drill" tone with no
recognizable words at all. After fixing the HiFT vocoder bug and the LLM
conditioning gap above, **real words are now audible** — genuine progress.
But the remaining distortion is described (by the person listening) as
**"like lowest-quality MP3 noise sprinkled around the clip, like a stream of
digits"** — i.e. real intelligible speech content is present, interspersed
with short harsh/bitcrushed-sounding bursts, rather than one uniform
tone/noise or total gibberish throughout. My working theory: the LLM
occasionally samples a badly-wrong speech token (because its logits are
wrong, per the 0.30 cosine above), and that one wrong token, decoded through
an otherwise now-fully-correct flow-encoder/DiT/vocoder chain, produces one
short burst of harsh noise, while the surrounding correctly-or-near-correctly
sampled tokens still produce recognizable speech around it.

## What I've already ruled out for this specific bug (checked directly, not assumed)

- **CORRECTION, found right after I first wrote this**: I originally believed
  a sibling model in the same codebase (CosyVoice2, which uses the exact same
  Qwen2 architecture shape and the same shared engine) was already verified
  working, making a shared-engine bug unlikely. That belief turned out to be
  wrong — the "verification" I was relying on only checked that logits were
  finite and non-degenerate, never against a real oracle. When I actually
  generated real audio from CosyVoice2's own pipeline and ran it through
  Whisper ASR, it transcribed as **"[Music]"** — no recognizable words at
  all, which is actually WORSE/more degenerate than CosyVoice3's output
  (which has real, if distorted, audible words). Given that difference in
  severity, I'm treating "are these the same underlying bug, or two
  independent ones" as an open question, NOT a conclusion — it's tempting to
  assume one shared root cause in the common engine code, but the different
  symptom severity is real evidence against that being a safe assumption,
  and I don't want to bias your reasoning toward it.
- **Not a metadata/hyperparameter mapping bug**: I checked that my C# tensor
  adapter correctly translates this GGUF checkpoint's own (non-standard)
  metadata key names into the values the generic hyperparameter parser
  expects — head counts, RoPE theta, RMS norm epsilon, and a vocab-size
  override for a combined text+speech embedding table all look correct.
- **Not the token-composition sequence itself**: I transcribed the exact
  real token sequence directly from the reference's own C++ source (not
  guessed), and separately confirmed via the model's own GGUF metadata that
  the special token ids (sos/task/stop tokens) are being read correctly.
- **Not the embedding-table composition trick**: I build one combined
  [text-vocab rows ; speech-vocab rows] embedding table so plain integer
  token ids can be used directly with my engine's ordinary API, instead of
  needing raw-embedding injection like the C++ reference does (its two
  embedding tables are genuinely separate weight tensors in that
  implementation). This exact pattern is already used successfully elsewhere
  in my codebase for a different, already-working model.

## What I'm planning to try next, in order

1. Narrow by sequence length: test a MUCH shorter real sequence (no
   reference-audio conditioning, ~15-20 positions) and see if it matches the
   reference (cosine ~1.0) or still diverges. If short sequences match but
   long ones don't, the bug is something that scales with sequence length or
   KV-cache size — attention masking, RoPE position ids beyond some
   threshold, or KV-cache read/write consistency across many positions. If
   even the short sequence diverges, the bug is more fundamental.
2. Dump post-RoPE Q/K vectors directly (not just final logits) from both
   sides, to narrow "somewhere in the transformer" down to a specific
   operation: Q after RoPE? K after RoPE? raw attention scores? the weighted
   V-sum? the output projection? the MLP? the final norm?
3. Verify KV-cache write/read consistency: is the cached K/V for an early
   position (written during a multi-token prefill) byte-identical to what a
   fresh, uncached computation of that same position alone would produce?
4. Check the causal attention mask construction specifically for a long,
   multi-token prefill with no prior cache — the reference builds its own
   mask with a specific indexing scheme; I want to compare that carefully
   against how my own engine builds its equivalent mask for the same
   scenario, since a subtle off-by-one in "how many prior positions are
   visible from position N" would produce exactly this symptom: fine for
   short/simple cases, increasingly wrong as the sequence gets longer and
   more context-dependent.

## What I'd like your advice on

- Given the evidence above (every other pipeline stage numerically proven
  correct; only the LLM's forward pass diverges, and only/mostly for a long,
  multi-position real sequence with a KV cache spanning ~140 positions),
  which of my four planned next steps would you try first, and why?
- Is there a known class of bug — in causal-mask construction, RoPE position
  indexing, or KV-cache indexing specifically for multi-token PREFILL as
  opposed to single-token incremental DECODE steps — that commonly produces
  exactly this "correct enough for short/simple inputs, degrades as context
  grows" symptom in a hand-written (non-HuggingFace) transformer
  implementation?
- Given the audible symptom (real words present, but with short
  bitcrushed/harsh noise bursts sprinkled in, rather than one uniform
  tone or total gibberish), does that pattern suggest anything specific to
  you about WHERE in a causal LM's forward pass a bug would need to live to
  produce "mostly right, occasionally badly wrong" token choices, as opposed
  to a bug that would corrupt every single token equally?
- Any other classic hand-rolled-transformer porting bugs (RoPE convention
  mismatches, attention-mask off-by-ones, KV-cache aliasing, GQA
  head-repeat-convention mismatches, etc.) that specifically tend to show up
  only once a real multi-position KV cache is exercised, rather than in a
  trivial single-token test?

I have full read/write access to both the real C++ reference source and my
own C# port, and can add targeted debug dumps to either side to test whatever
hypothesis you suggest.
