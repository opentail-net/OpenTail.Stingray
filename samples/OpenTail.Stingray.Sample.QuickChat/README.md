# OpenTail.Stingray.Sample.QuickChat

> **Audience:** Package Consumer
> **Task:** One prompt, one streamed answer, via the one-call facade
> **API Contract:** `Stingray.CreateAsync`, `StingraySession.ChatAsync`

---

## 1. What This Sample Demonstrates

The simplest possible chat program against the public API: `Stingray.CreateAsync` resolves a
catalog model by id (auto-downloading and SHA-256-verifying it if it isn't installed yet), loads
it, and returns a ready `StingraySession` — no explicit `Model.Load`, `CreateContext`,
`InteractiveExecutor` or `ChatSession` wiring. This is the facade [`OpenTail.Stingray.Sample.Chat`](../OpenTail.Stingray.Sample.Chat/)
builds from those lower-level types directly; reach for that sample (or the types it uses) once
you need multi-turn interactive chat, an explicit model path/backend, or anything this facade
doesn't expose.

---

## 2. Prerequisites & Asset Setup

None required — if the default chat model (Qwen2.5 0.5B Instruct, 469 MB) isn't already installed,
`Stingray.CreateAsync` downloads and verifies it on first run, exactly as `stingray setup chat`
does. To pre-install it instead:

```bash
stingray setup chat
```

---

## 3. How to Run

### Default prompt
```bash
dotnet run --project samples/OpenTail.Stingray.Sample.QuickChat -c Release
```

### Your own prompt
```bash
dotnet run --project samples/OpenTail.Stingray.Sample.QuickChat -c Release -- \
    "Why do developers write unit tests?"
```

---

## 4. Expected Output

```text
loading qwen2.5-0.5b ... qwen2 (ctx 32768)
Developers write unit tests to ensure individual components or functions work as intended, helping to identify and fix bugs before the application is deployed.
```

(Vulkan/GPU diagnostic lines between those two go to stderr — see §5.) The wording differs from
run to run (sampling is random), and the context size line reflects whatever the model's own
default is — `StingrayOptions.ContextSize` defaults to `0`, meaning "use the model's default,"
unlike `OpenTail.Stingray.Sample.Chat`'s explicit `2048`.

## 5. Known Limitations

- One prompt, one answer, then exit — no multi-turn loop (the underlying `StingraySession.History`
  would carry one across repeated calls, but this sample doesn't loop).
- Backend diagnostic lines (Vulkan device limits, "Pre-faulted ... weights") print to stderr; they
  are not errors.
- Backend selection (CPU/Vulkan/CUDA) is automatic, same as the rest of the library — there's no
  option on `StingrayOptions` to pin one yet.

## 6. Verification

- **Run against real weights, 2026-11-06:** `dotnet run --project samples/OpenTail.Stingray.Sample.QuickChat -c Release`
  with Qwen2.5 0.5B Instruct already installed by `stingray setup chat`. Auto-selected the Vulkan
  GPU backend and produced the answer shown above. One run.
- **Package contract:** builds against project references in-repo; pass
  `-p:StingrayPackageVersion=<version>` to build against the packed `OpenTail.Stingray` NuGet
  package instead, as `scripts/verify-package-consumers.ps1` does for the other package-consumer
  samples (not yet run for this sample — add it to that script's list when it's updated next).
