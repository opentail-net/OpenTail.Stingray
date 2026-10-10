# Where help would change something

Open items that are waiting on something a contributor could supply. This is a pointer page; the
detail and the acceptance evidence for each item are in the linked documents, which are the source
of truth. Last reviewed 2026-10-10. How to take part: [CONTRIBUTING.md](../CONTRIBUTING.md).

## You have an NVIDIA GPU

The development machine has none, so these are unproven rather than known-bad. Source:
[external-hardware queue](9-external-hardware/90-external-hardware-work.md).

- **CUDA release evidence:** dense load/decode/sample, long-context KV dtype, placement output,
  CUDA-hybrid MoE, MTP/speculation receipts.
- **Gemma 4 12B:** re-run the CPU / CUDA / hybrid acceptance sequence.
- **CUDA RoPE frequency factors:** applied for Gemma 4 only, so Llama-3.1-style models run unscaled
  RoPE on CUDA (wrong only at long context). Needs a reproduction on real hardware.
- **CUDA partial offload** for the newer architectures (currently falls back to CPU with a note).
- **CUDA graph default:** dense and hybrid paths interpret `STINGRAY_CUDA_GRAPH` differently.
  Measure both before anyone changes the default.

## You have a discrete GPU of any make

Results from an integrated GPU do not generalise to a discrete one, so these defaults are waiting
for measurements: speculative decoding with a draft model on Vulkan, the UMT5 text-encoder GPU
path against CPU, and gpt-oss on Vulkan. See items 5-7 in the
[external-hardware queue](9-external-hardware/90-external-hardware-work.md).

## You have an ARM64 machine

No ARM64 baseline exists. The first receipt should report ISA features (NEON, dot-product, i8mm),
the model and dtype, correctness parity, and a measured baseline, before any ARM-specific kernel is
written.

## You have a different CPU, or more RAM

Every row in [RUNNING.md](RUNNING.md) was measured on one Ryzen 7 5700G. A reproduction on another
CPU, or on a machine with enough memory to run the larger rows comfortably, is a genuine second
data point; so is a number that disagrees. Two specific discrepancies are open:

- **gpt-oss-20b decode speed:** [STATUS](STATUS.md) quotes about 11 tok/s and [RUNNING](RUNNING.md)
  records 5.5 tok/s. Nobody has reconciled them.
- **gpt-oss on Vulkan:** the code lists a Vulkan path, the matrix says there is none, and there is
  no receipt either way. See the [gpt-oss card](models/gpt-oss.md).

## You have a large disk or a lot of RAM

Some families are written and waiting for a real checkpoint to be compared against an independent
reference. See the "Ported, not verified" table in the
[takeaways plan](2-coverage/2026-10-02-tensorsharp-takeaways-plan.md); each row says what is
missing.

## You have a GGUF that does not load

Run `stingray scout -m <file> --format json` and open a **New architecture** issue with the output.
See [coverage tooling](reference/061-coverage-tooling.md).

## You like reading documentation

Cards in [docs/models](models/README.md) cite measured results and Hugging Face file names. File
names drift as uploaders reshuffle repositories. The Qwen3.8-27B, Qwen3.6-35B-A3B and Ornith cards
already note quant files that could not be confirmed in the current repo listings; confirming or
correcting those is a small, real contribution.
