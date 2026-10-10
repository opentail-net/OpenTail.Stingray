# Contributing to OpenTail.Stingray

OpenTail.Stingray is MIT-licensed and built in the open. If you want to help, there is room for
you, and you do not have to write C# to do it. If you do not want to help, that is fine too; the
project works for you either way.

This page explains the ways in, what each one needs from you, and what happens to your work
afterwards.

## Start from what you have

| You have… | You can… | Skill needed |
|---|---|---|
| **Any computer and a model you like** | Run it and tell us what happened, good or bad ([model report](#1-run-a-model-and-report-what-happened)) | None beyond a terminal |
| **Hardware we do not have** (an NVIDIA GPU, a discrete AMD/Intel GPU, an ARM64 machine) | Produce a measured receipt that nobody else can ([hardware receipt](#2-measure-on-hardware-we-do-not-have)) | A terminal and patience |
| **An eye for detail** | Fix a wrong number, dead link or stale command in the docs ([docs](#3-fix-the-docs)) | None |
| **A GGUF whose architecture is not supported** | Triage it, or take it all the way to admission ([new architecture](#4-bring-a-new-architecture)) | Terminal; C# for the full port |
| **A lot of RAM or disk** | Prove a family that is already ported but not yet checked on a real checkpoint ([prove a port](#5-prove-a-port-we-could-not-check)) | Terminal |
| **C# / SIMD / GPU-kernel experience** | Fix bugs, add kernels, make things faster ([code](#6-change-the-code)) | C#, and the habit of measuring |

Concrete, current open items live in **[docs/WANTED.md](docs/WANTED.md)**. It is a list of things
that are blocked on something a contributor could supply, kept next to the evidence for each.

## 1. Run a model and report what happened

The most useful report is a plain one: the command you ran, the model file, your hardware, and
what you saw. Open a **Model report** issue (there is a form). A failure is as valuable as a
success: this project's status matrix only has value if it records where things break.

```bash
stingray doctor -m <model.gguf> --bundle support.zip   # redacted diagnostics; look inside before attaching
```

If your result matches the documented one, that is useful (it is a second machine). If it differs,
say how.

## 2. Measure on hardware we do not have

The development machine has an AMD integrated GPU and no NVIDIA device. Several items can only be
closed by someone with the right hardware, and a number from one machine is only evidence for that
machine. The queue is in
[docs/9-external-hardware/90-external-hardware-work.md](docs/9-external-hardware/90-external-hardware-work.md).

A good receipt records: commit SHA, exact model file and quantization, backend / device / driver,
the command, warm-up, repeated samples, the numerical or token-parity result, and whether other
workloads were running. Use the **Hardware receipt** issue form; it asks for exactly these.

## 3. Fix the docs

The documentation is checked against measured results, so a wrong number matters. A pull request
that corrects a command, a link, a file name or a figure is welcome as it stands; no issue needed.
Architecture cards are in [docs/models/](docs/models/README.md); the verification matrix is
[docs/STATUS.md](docs/STATUS.md); measured commands are in [docs/RUNNING.md](docs/RUNNING.md).

## 4. Bring a new architecture

Found a GGUF that Stingray refuses? There is a ladder, and you can stop at any rung:

1. `stingray scout -m <gguf> --format json` reads the header only and tells you what is in the
   file and what looks blocking. Attach the output to an issue (**New architecture** form). This
   alone helps.
2. `stingray admit-arch -m <gguf>` runs the model under a diagnostic bypass and, given a reference
   token sequence from an independent implementation (llama.cpp), compares token for token.
   Many architectures need no new forward-pass code; the blocker is often the tokenizer.
3. Port the missing pieces. Start with
   [docs/reference/adding-an-architecture.md](docs/reference/adding-an-architecture.md).

The full tool guide is [docs/reference/061-coverage-tooling.md](docs/reference/061-coverage-tooling.md).
An architecture is **admitted** only when a real checkpoint matches an independent reference. A
family ported before that is kept internal until it is proven (see the next section).

## 5. Prove a port we could not check

Some families are written but not yet admitted, because the maintainers could not run a real
checkpoint (too large for the development machine, for example). The list, with what each one is
missing, is the "Ported, not verified" table in
[docs/2-coverage/2026-10-02-tensorsharp-takeaways-plan.md](docs/2-coverage/2026-10-02-tensorsharp-takeaways-plan.md).
If you can run one of these checkpoints and compare it against an independent reference, that is
the single most direct way to move a family from "in the repo" to "supported".

## 6. Change the code

Read [DEVELOPMENT.md](DEVELOPMENT.md) for the layout, build and test tiers. The points that most
often trip people up:

- Warnings are errors, and the build is NativeAOT/trim-checked: no reflection-heavy patterns.
- Do not pass `--nologo` to `dotnet test`.
- Heavy tests skip unless `STINGRAY_RUN_HEAVY_TESTS=1`. A test named `*RealWeights*` that finishes
  in a fraction of a second probably did not run against weights (it returns early when the
  checkpoint is absent). Check the timing before you cite a green run.
- Performance changes are kept only if they are measurably faster on real weights over several
  runs. A result from an integrated GPU says nothing about a discrete one.
- Do not add Python reference scripts. The independent oracle is the vendored llama.cpp binaries.
- Put scratch output in a temp directory, not the repository root.

[CLAUDE.md](CLAUDE.md) holds the longer version of these rules. It is written for AI coding agents
but is also the most complete list of the project's conventions. AI-assisted contributions are
fine; the same evidence standard applies to them.

## The one standard

> A capability may be advertised only when its implementation status and verification evidence
> support the exact claim being made.

That is why a pull request that adds a model to the docs needs a result behind it, and why "it
loaded" is not "it works". A small, honest claim is always welcome; an unevidenced large one will
be asked for evidence.

## What happens after you submit

- Issues and pull requests are read by the maintainers, who are a small team. Replies can take a
  few days. There is no obligation in either direction.
- A receipt or report that passes the evidence standard can be cited in
  [docs/STATUS.md](docs/STATUS.md) or [docs/RUNNING.md](docs/RUNNING.md) with its date and, if you
  agree, your handle. Contributions are also credited in [CHANGELOG.md](CHANGELOG.md).
- A report that does not reproduce, or a result we cannot use, will be answered with the reason.
- You keep authorship of what you wrote. By contributing you agree it is licensed under the
  project's [MIT license](LICENSE).

## Ground rules

- Do not attach or upload model weights. Link to the Hugging Face repo and name the exact file.
- Check that code you copy is compatible with MIT, and say where it came from. Model files have
  their own licenses; see each model's page.
- Check bundles and logs for anything private before posting them.
- Be kind. People here are at different levels and most are volunteering their time.

## Not sure where to start?

Open an issue describing what you have (hardware, a model you care about, an hour or a month) and
we will point at the nearest open item.
