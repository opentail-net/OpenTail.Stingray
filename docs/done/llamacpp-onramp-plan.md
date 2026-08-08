# llama.cpp on-ramp plan

> **Goal.** Someone who knows llama.cpp — or an AI assistant that has read a lot of llama.cpp
> documentation — should be able to run OpenTail.Stingray on the first try, using the spellings they
> already have in their fingers.
>
> **Explicit non-goal.** Matching llama.cpp is *not* a long-term objective. This is a migration ramp,
> not a compatibility guarantee, and it is deliberately bounded by the tiers below. When a llama.cpp
> spelling and a better OpenTail design conflict, OpenTail wins and the llama.cpp spelling stays an
> alias.

Last updated 2026-08-07.

## Why this is worth doing

The first command a new user runs is usually copied from somewhere else. If it fails on flag syntax,
they conclude the project is immature — not that the parser differs. That impression is expensive and
the fix is cheap. The same applies with more force to coding assistants, which will confidently emit
llama.cpp flags because that is what their training data is full of.

The benefit is concentrated almost entirely in **the first five minutes**. That is why the tiers below
are ordered by "does this unblock a first run", not by how close it gets us to llama.cpp.

## Two premises that were wrong, corrected before planning

**1. Spectre.Console is gone.** The CLI has its own parser and terminal layer:
`Cli/CommandLine/` (`CommandApp`, `Command`, `OptionBinder`, `OptionModel`, `HelpRenderer`) and
`Cli/Terminal/` (`AnsiConsole`, `Markup`, `Table`, `Status`). No Spectre package reference remains in
`OpenTail.Stingray.Cli.csproj` or `Directory.Packages.props`.

**2. Single-dash multi-character flags already work.** This was assumed to be the blocker; it is not.
`OptionBinder.TryBind` builds an exact-match `Dictionary<string, OptionModel>` over alias strings with
`StringComparer.Ordinal`. There is no dash counting, no short-option clustering, no `-abc` expansion:

```csharp
var byAlias = new Dictionary<string, OptionModel>(StringComparer.Ordinal);
foreach (var opt in options)
    foreach (var alias in opt.Aliases)
        byAlias[alias] = opt;
```

`OptionModel` splits the attribute template on `|` and stores each alias verbatim. `RunCommand`
already proves it with `--ngl|--n-gpu-layers|--gpu-layers|-g`.

**Consequence: Tier 0 needs no parser work at all.** Adding `-ngl`, `-fa`, `-ctk`, `-ts` is editing
template strings. No argv normalizer, no framework change. An earlier version of this analysis
recommended writing a normalizer shim — that recommendation is void, and anyone who finds it
elsewhere should ignore it.

## The rule that governs every tier

**Never accept a flag and ignore it.** An accepted-but-unhonoured `--mlock` or `-ts` produces "it ran
but the numbers are wrong" reports that cannot be reproduced, and it is worse than not supporting the
flag at all. This is the same *refuse rather than run* principle already governing the SafeTensors
capability profile in `08-safetensors-support-plan.md`, and a compatibility surface is exactly where the
temptation to accept-and-ignore is strongest.

Every llama.cpp flag lands in one of three states, and there is no fourth:

1. **Aliased** — maps onto an existing option with the same meaning.
2. **Implemented** — new plumbing to a capability that genuinely exists.
3. **Refused** — recognised by name, rejected with a message that says what OpenTail does instead.

State 3 is a feature. `Error: -ts/--tensor-split is not supported; OpenTail places layers with
--auto or an explicit -g <N>.` is a good user experience. Silence is not.

## Tier 0 — aliases only, no new behaviour

Zero new capability. Each entry is an extra token in an existing `[CommandOption]` template.

| llama.cpp | Maps to | Note |
|---|---|---|
| `-ngl` | `--ngl` (also `--n-gpu-layers`, `--gpu-layers`, `-g`) | Implemented and binding-tested. |
| `-ctx`, `--n-ctx` | `-c\|--ctx-size` | Implemented and binding-tested. |
| `-npredict` | `-n\|--n-predict` | Implemented and binding-tested. |
| `--n-predict -1` | `-n` | **Refused.** OpenTail does not support llama.cpp's “until EOS” sentinel; use an explicit non-negative bound. |
| `-ctk` / `--cache-type-k` | `--kv-type` | Implemented; conflicting K/V values are explicitly rejected. |
| `-ctv` / `--cache-type-v` | `--kv-type` | Implemented with the same agreement check. |
| `--repeat_penalty` (underscore) | `--repeat-penalty` | Implemented and binding-tested. |
| `-md` | `--model-draft` | Implemented and binding-tested. |
| `--draft` | `--spec-lookahead` / `--draft-tokens` | Implemented as draft length and binding-tested. |

**Acceptance:** a test enumerates every advertised alias, binds a command line containing it, and
asserts the bound property. Aliases are cheap to add and cheap to break silently — a typo in a
template string produces "unexpected argument", which no existing test would catch.

**Before writing any of these down as truth, re-derive the llama.cpp column from the release you
intend to target.** llama.cpp renames flags between releases; this table reflects a general
recollection and must be checked against `common/arg.cpp` in the actual version, not trusted.

## Tier 1 — cheap implementations where the capability already exists

| llama.cpp | Work | Notes |
|---|---|---|
| `-t` / `--threads` | Complete | Sets the SIMD kernel worker count; binding and behavior are tested. |
| `--repeat-last-n` | Complete | Plumbed to the decode history window; behavior-tested for the default, disabled (`0`), bounded, and full-context (`-1`) cases. |
| `--presence-penalty`, `--frequency-penalty` | Small–medium | Only if `Sampler` supports them; if not, this is Tier 3. Check before promising. |
| `--logit-bias` | Complete | Parses repeatable llama.cpp `TOKEN_ID(+/-BIAS)` entries into `SamplingParams.LogitBias`; parser behavior is tested. |
| `--chat-template` | Complete | Raw Jinja override bypasses the embedded template; named shortcuts are refused rather than approximated. |
| `-e` / `--escape` | Complete | Processes `\n`, `\t`, `\r` and `\\` in `-p`; behavior-tested. |
| `--no-warmup` | Complete (inert) | There is no separate warmup phase. The flag is accepted with a warning, not silently claimed to do work. |

**Acceptance per item:** the flag changes observable behaviour, and a test proves it — not merely that
it binds. A bound-but-inert flag is the failure mode this whole document exists to prevent.

## Tier 2 — llama-server endpoint on-ramp

Current routes: `/v1/chat/completions`, `/v1/messages`, `/v1/models`, `/v1/responses`, `/health`,
`/metrics`, `/status`, `/capabilities`.

Take only the cheap three first:

| Endpoint | Work | Why it's cheap |
|---|---|---|
| `/tokenize` | Complete | Tokenizer relay shape mapping; wire-contract tested. |
| `/detokenize` | Complete | Tokenizer relay shape mapping; wire-contract tested. |
| `/props` | Complete | Safe model/template facts; wire-contract tested. |

Deliberately deferred, each needing its own decision: `/completion` (a re-skin of the generate path in
llama.cpp's field and streaming shapes — a few days), `/embedding` (needs pooled embeddings that may
not exist at all), `/infill` (needs FIM token handling), `/slots` (only meaningful against
`ContinuousBatchingEngine`).

## Tier 3 — real features wearing a flag's clothing

Do **not** treat these as parity work. Each is a feature with its own design cost, and each should be
justified on its own merits rather than because llama.cpp has it:

- `--grammar` / `--grammar-file` (GBNF). OpenTail has JSON-schema constraints and tool grammars via
  `ITokenConstraint`, but not GBNF. A real parser and compiler.
- The interactive family: `-i`, `-cnv`, `-r`/`--reverse-prompt`, `--in-prefix`, `--in-suffix`,
  `--keep`. Interactive semantics, not flags.
- `--lora`, `--lora-scaled`, `--control-vector`.
- `--rope-scaling`, `--rope-freq-base`, `--rope-freq-scale`, `--yarn-*` overrides.
- `-b` / `-ub` (batch / micro-batch sizing) — only meaningful if the engine exposes equivalent knobs.

## Refuse explicitly — these are decisions, not omissions

Recognise the name, fail with a message naming the OpenTail alternative:

| Flag | Refusal message should say |
|---|---|
| `-ts` / `--tensor-split` | Placement is chosen by `--auto` or an explicit `-g <N>`. |
| `-sm` / `--split-mode` | Same. |
| `-mg` / `--main-gpu` | Use `--device`. |
| `--mlock`, `--no-mmap` | Not implemented; say so rather than no-op. |
| `--numa` | Not implemented. |
| `-fa` / `--flash-attn` | State whether attention is already fused, so the flag is meaningless rather than ignored. |

## Next decision

The bounded on-ramp is complete: Tier 0 aliases, its explicit refusals, Tier 1 features, and the
three cheap llama-server endpoints all have binding, behavior, or wire-contract tests. Do not add
more flags for parity's sake. Any Tier 3 item needs an independent product proposal and acceptance
evidence before it re-enters active work.

## How this stays bounded

- The capability surface is **published, not inferred**: a `--help` section (and ideally a doc table)
  lists exactly which llama.cpp spellings are accepted. Anything absent is unsupported.
- **No flag is added without one of the three states above.** If it cannot be aliased or implemented
  now, it gets a refusal message now.
- **Parity is never a release gate.** No work is scheduled because llama.cpp added a flag; it is
  scheduled because a user or an assistant tried it here and hit a wall.
- Revisit this document when a llama.cpp release renames things. Do not chase it continuously — the
  point is a first-run experience, and first-run flags change slowly.
