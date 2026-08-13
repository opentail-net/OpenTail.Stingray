Plan: DeepSeek-V2-Lite (deepseek2) Support
Objective

Add full CPU inference support for the downloaded:

DeepSeek-V2-Lite-Chat.Q2_K.gguf

using Stingray's existing model-loading, graph, tensor, KV-cache, batching and sampling infrastructure.

The target architecture is:

architecture: deepseek2
parameters:   ~15.7B total
active:       ~2.4B/token
layers:       27
hidden:       2048
vocab:        102400
MoE layers:   26
experts:      64 routed + 2 shared
top-k:        6
MLA KV latent: 512
attention heads: 16
head dimension: 128
context:      163840
YaRN factor:  40

The implementation must be metadata/tensor-driven, not hard-coded around this particular GGUF file.

Phase 0 — Repository and GGUF audit

Before writing inference code, inspect the existing implementations for:

ModelGraph
ForwardPass
IBatchedForwardPass
existing MoE implementations
existing attention implementations
existing KV-cache interfaces
RoPE/YaRN implementation
quantised matrix multiplication
model compatibility/architecture dispatch
model metadata extraction
GGUF tensor-name mapping
model validation
existing model-specific graph builders

Stingray already has a dedicated ModelGraph abstraction, so deepseek2 should become another graph family rather than bypassing it.

Also inspect the existing MoE work before introducing a new dispatcher; the repository already contains MoE expert-offloading research.

Deliverable

Produce an implementation inventory:

Existing component                  Reuse?
------------------------------------------------
Dense attention                     ...
GQA attention                       ...
RoPE                                ...
YaRN                                ...
MoE router                          ...
MoE expert execution                ...
Shared experts                      ...
KV cache                            ...
Quantized matmul                    ...
Batched forward                     ...
ModelGraph                          ...

Do not implement anything in Phase 0.

Phase 1 — Architecture admission

Add:

deepseek2

to the supported architecture registry in the existing compatibility mechanism.

Do not create a separate DeepSeek model type if the existing architecture abstraction can accommodate it.

The architecture should resolve to something like:

DeepSeek2ModelGraph

or an equivalent existing graph-builder abstraction.

Required metadata

The loader must obtain these from GGUF metadata wherever available:

block_count
embedding_length
attention.head_count
attention.head_count_kv
attention.key_length
attention.value_length
expert_count
expert_used_count
expert_feed_forward_length
expert_shared_count
attention.layer_norm_rms_epsilon
rope.dimension_count
rope.freq_base
rope.scaling.type
rope.scaling.factor

Do not assume:

27 layers
2048 hidden
64 experts
top 6

merely because this particular model has those values.

Phase 2 — Tensor inventory and validation

Before implementing execution, build a DeepSeek-specific tensor inventory from the actual GGUF.

Expected families include equivalents of:

blk.{n}.attn_norm
blk.{n}.attn_kv_a_mqa
blk.{n}.attn_kv_a_norm
blk.{n}.attn_kv_b
blk.{n}.attn_q_a
blk.{n}.attn_q_a_norm
blk.{n}.attn_q_b
blk.{n}.attn_k_b
blk.{n}.attn_v_b
blk.{n}.attn_output
blk.{n}.ffn_gate_inp
blk.{n}.ffn_shexp.*
blk.{n}.ffn_exps.*
blk.{n}.ffn_norm

Do not assume these exact names.

The coding agent must dump the actual tensor names, dimensions, types and offsets from:

DeepSeek-V2-Lite-Chat.Q2_K.gguf

and map them explicitly.

Acceptance criterion

A diagnostic command should be able to say:

Architecture: deepseek2
Layers:       27
Hidden:       2048
Experts:      64
Active:       6
Shared:       2
MLA KV:       512
Heads:        16
Head Dim:     128

All required tensors present: YES

before inference is attempted.

Phase 3 — MLA

This is the most important part of the implementation.

Do not model MLA as ordinary GQA with a funny KV cache.

DeepSeek-V2's MLA fundamentally changes what is stored in the KV cache.

The implementation needs to distinguish:

Q path

hidden
  ↓
q_a
  ↓
q_a_norm
  ↓
q_b
  ↓
query heads

and:

KV path

hidden
  ↓
kv_a
  ├── latent KV
  └── RoPE K component

The latent KV representation is what should be retained in the cache.

3.1 MLA prefill

Implement:

MlaPrefill(...)

producing the compressed KV state for each token.

Conceptually:

x
 │
 ├── Q projection
 │
 └── KV-A projection
       │
       ├── compressed latent KV
       │
       └── K-pe RoPE component

Apply the appropriate latent RMSNorm to the compressed KV component.

Phase 4 — MLA KV cache

This is where I would not reuse the existing standard K/V cache blindly.

Create an abstraction along the lines of:

IMlaKvCache

or extend the existing KV-cache abstraction so that DeepSeek can provide a different representation.

The cache should contain the compressed latent representation, not fully expanded K/V heads.

For example:

Standard model:

KV cache
 ├── K[heads]
 └── V[heads]

DeepSeek MLA:

MLA cache
 ├── compressed KV latent
 └── K-pe positional component

The exact storage type must follow the actual tensor layout in the GGUF and existing Stingray quantisation/cache conventions.

Important

Do not immediately expand:

512 → 16 × 128

and store that expanded result per token.

That would throw away one of the main memory advantages of MLA.

Phase 5 — MLA decode attention

Implement the decode path separately from prefill where necessary.

Conceptually:

Current hidden
      │
      ▼
   Q projection
      │
      ▼
 query heads
      │
      │
      ├───────────────┐
      │               │
      ▼               ▼
cached latent KV    K-pe
      │               │
      └──────┬────────┘
             ▼
       MLA attention
             │
             ▼
       output projection

The implementation must avoid unnecessary materialisation of full historical K/V.

Required tests

At minimum:

one-token decode
two-token decode
long decode
prefill + decode
batch decode
cache reuse
session continuation
Phase 6 — RoPE + YaRN

DeepSeek-V2-Lite reports:

context: 163840
YaRN factor: 40

Do not simply set the global context length to 163840.

Inspect Stingray's existing RoPE implementation and determine whether its current scaling abstraction can represent the DeepSeek metadata.

Implement a DeepSeek-specific RoPE configuration only if the generic abstraction cannot express it.

Acceptance tests

Compare position encoding behaviour at:

position 0
position 1
position 128
position 2048
position 8192
position 32768
position 65536

against a reference implementation.

Phase 7 — DeepSeekMoE

Implement the 26 MoE layers using the existing MoE execution infrastructure wherever possible.

Architecture:

                 hidden
                   │
             ┌─────┴─────┐
             │           │
             ▼           ▼
       2 shared       router
        experts          │
             │        top-6
             │       / | | \
             │      /  | |  \
             │     ▼   ▼ ▼   ▼
             │   routed experts
             │        │
             └────┬───┘
                  ▼
             fused output

The router must:

calculate expert logits;
select top 6;
calculate routing weights using the model's specified routing semantics;
execute only selected routed experts;
execute the shared experts;
combine the results correctly.
Phase 8 — Shared expert implementation

Do not treat:

ffn_shexp

as another routed expert.

They are always active.

The implementation should expose this distinction explicitly:

SharedExperts(...)
RoutedExperts(...)
CombineExperts(...)

This will also make later DeepSeek-family models easier to support.

Phase 9 — Expert dispatch optimisation

The first implementation should prioritise correctness.

Start with:

for each token
    route top-6
    execute selected experts
    accumulate

Then optimise.

For batched inference, group tokens by expert:

Token 0 → E3 E7 E12 E18 E29 E44
Token 1 → E3 E4 E7 E21 E31 E44
Token 2 → E1 E7 E12 E18 E29 E63

             ↓

E1  → tokens [...]
E3  → tokens [...]
E4  → tokens [...]
E7  → tokens [...]
...

This is important for CPU performance.

The existing continuous batching engine currently explicitly says it does not support MoE models, so DeepSeek support must either add MoE-aware batching or deliberately keep DeepSeek on the single-sequence path initially.

Do not silently enable continuous batching for DeepSeek until routing correctness and expert batching are validated.

Phase 10 — Q2_K compatibility audit

Your actual test model is:

DeepSeek-V2-Lite-Chat.Q2_K.gguf

Therefore the implementation must verify every DeepSeek tensor's quantisation type.

Do not assume that because existing Q2_K models work, every DeepSeek tensor can use the same path.

Create a report:

Tensor family             Quantisation       Supported?
-------------------------------------------------------
attention projections     Q2_K               YES/NO
router                    F32                YES/NO
shared experts            Q2_K               YES/NO
routed experts            Q2_K               YES/NO
norms                     F32                YES/NO
output                     Q2_K               YES/NO

If the router/norm tensors are F32/F16 while weights are Q2_K, preserve that.

Phase 11 — Graph integration

Build the model as a normal Stingray graph.

Something conceptually like:

DeepSeek2ModelGraph
    │
    ├── Embedding
    │
    ├── Layer 0
    │     ├── RMSNorm
    │     ├── MLA
    │     ├── residual
    │     └── dense FFN
    │
    ├── Layers 1–26
    │     ├── RMSNorm
    │     ├── MLA
    │     ├── residual
    │     └── DeepSeekMoE
    │
    ├── Final RMSNorm
    │
    └── LM Head

Note the first layer is dense according to your inspected model description. Do not accidentally run the first layer through the MoE path.

Phase 12 — KV/session integration

This deserves its own phase because of the work you've already done.

DeepSeek must work with:

session continuation
retained KV state
context accounting
prefix reuse where compatible
context compaction where compatible
forked sessions where supported

The session runtime already has explicit cache lifecycle work, so MLA must integrate with that lifecycle rather than creating a DeepSeek-only persistence mechanism.

Critical test
Prompt A
   ↓
generate N tokens
   ↓
save/retain session
   ↓
append prompt B
   ↓
generate

must produce the same result as an uninterrupted reference execution, subject only to expected sampling nondeterminism.

Phase 13 — Continuous batching

Do this after single-sequence inference works.

Then extend:

IBatchedForwardPass

to support:

DeepSeek2 + MLA + MoE

The existing ContinuousBatchingEngine already has admission/KV-budget logic and packed prefill/decode infrastructure.

The new DeepSeek path should exploit that infrastructure rather than bypass it.

The MoE scheduler should ideally become:

batch tokens
      ↓
MLA batch
      ↓
router
      ↓
expert dispatch
      ↓
expert grouped GEMMs
      ↓
recombine token outputs
      ↓
MLA output
Phase 14 — Correctness validation

Create a deterministic DeepSeek test suite.

A. Shape tests

Every tensor operation should validate expected dimensions.

B. Router tests

For known hidden states:

top-6 expert IDs
routing weights

must match reference output.

C. MLA tests

Compare:

Q
compressed KV
K-pe
attention scores
attention output

against a trusted implementation.

D. Layer tests

Compare individual layers before comparing the entire model.

E. Full model

Test:

1 token
8 tokens
32 tokens
128 tokens
512 tokens

prefill.

Then:

1-token decode
10-token decode
100-token decode
Phase 15 — Reference comparison

Do not use generated text alone as the correctness criterion.

Build a small reference harness that can obtain:

logits
hidden states
router selections
MLA intermediate values

from a known-good DeepSeek implementation.

Then compare Stingray.

Target:

embedding
    ↓
layer 0
    ↓
layer 1
    ↓
...
    ↓
final logits

with tolerances appropriate to Q2_K.

Phase 16 — Performance

Only after correctness.

Measure separately:

Prompt processing tok/s
Decode tok/s
Memory consumption
KV bytes/token
Expert dispatch overhead
Router overhead
MLA overhead

Particularly:

MLA memory

Compare:

standard expanded KV
vs
DeepSeek compressed KV

This is one of the major reasons we're adding DeepSeek.

MoE

Measure:

total GEMM work
active expert GEMM work
expert dispatch overhead
expert weight cache hit rate
Phase 17 — Optimisation opportunities

Once correct:

1. Expert grouping

Batch tokens destined for the same expert.

2. Expert weight residency

Use the existing MoE expert-offloading work rather than inventing a second mechanism.

3. MLA fused operations

Investigate fusing:

latent projection
normalisation
attention preparation

where the current tensor abstraction permits it.

4. Avoid temporary allocations

MLA and MoE can generate substantial intermediate buffers.

Use pooled/reused buffers.

5. SIMD

Only optimise kernels after profiling demonstrates the hotspot.

Phase 18 — Continuous batching re-enable

Remove the initial DeepSeek/MoE restriction only once:

single sequence        PASS
multiple sequences     PASS
different routing      PASS
different lengths      PASS
KV isolation           PASS
session continuation   PASS

Then test:

batch = 2
batch = 4
batch = 8

and compare throughput against sequential execution.

Phase 19 — Model coverage registration

Add the model to the GGUF coverage inventory.

Document:

Architecture: deepseek2
Model: DeepSeek-V2-Lite
Quantisation: Q2_K
Parameters: 15.7B
Active: 2.4B
MLA: yes
MoE: yes
Shared experts: 2
Routed experts: 64
Top-k: 6
Definition of Done

DeepSeek-V2-Lite support is complete when:

 deepseek2 is recognised by model compatibility.
 GGUF metadata is parsed correctly.
 All required DeepSeek tensors are discovered.
 Q2_K tensors load correctly.
 Dense first layer executes correctly.
 MLA prefill works.
 MLA decode works.
 Compressed MLA KV cache works.
 RoPE/YaRN behaviour is correct.
 2 shared experts execute correctly.
 top-6 routed experts execute correctly.
 routing weights match reference.
 MoE output matches reference.
 full single-sequence generation works.
 session continuation works.
 KV accounting works.
 no standard-KV expansion is used unnecessarily.
 deterministic reference tests pass within quantisation tolerance.
 memory usage is measured.
 tok/s is benchmarked.
 continuous batching remains disabled until its MoE path is explicitly validated.
 continuous batching is subsequently enabled only if the MoE batched path passes its isolation/correctness tests.
 documentation/model coverage is updated.
One important correction to your original three-step plan

I wouldn't start by writing MlaAttention.cs and DeepSeekMoeGraph.cs immediately.

The right order is:

GGUF tensor audit
       ↓
existing abstraction audit
       ↓
architecture registration
       ↓
tensor mapping
       ↓
MLA cache representation
       ↓
MLA correctness
       ↓
MoE correctness
       ↓
full graph
       ↓
session/KV integration
       ↓
batching
       ↓
optimisation

That matters because MLA is not simply another attention node, and the existing engine's continuous-batching layer explicitly assumes non-MoE execution today.

I'd call this Plan 012 — DeepSeek-V2-Lite / MLA + DeepSeekMoE Support. It's a substantially more meaningful inference-engine feature than just adding another model family: MLA gives Stingray a new KV-cache architecture, while DeepSeekMoE exercises expert routing and sparse execution at a size you can actually test with your 6.13 GB Q2_K model.