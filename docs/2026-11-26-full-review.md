# OpenTail.Stingray: full review (2026-11-26)

- **Revision reviewed:** `main` at `2277fb71`. Two commits (`6a88abf3`, `2277fb71`) are local only; `origin/main` is at `3cfcc60b`.
- **Size:** 3,520 tracked files. About 348k lines of C# in `src/` (13 projects, the largest being Engine at 86k) and about 208k in `tests/` (23 projects).
- **Prepared by:** Claude (Haiku 5.5), review only. No code or docs were changed apart from this file.

## 1. Verdict

The engine code I read is careful. The GGUF parser bounds every length against the file, the torch reader will not run arbitrary pickle callables, and the download path checks where files land. The project is **not releasable today**:

- **`main` is red.** The latest CI run (`3cfcc60b`, 2026-11-26 00:56 UTC) fails in all three CI jobs. Two of the causes are one-line fixes. A third, a timing flake, has now hit the trigger the backlog set for treating it as a real defect.
- **1.0.10 is not tagged.** The release workflow runs the Core suite, so the registry failure (M2) would also fail the release.
- **Repository hygiene has drifted from the project's own rules.** 29 scratch test files and 8 tuning dumps are committed, and the OpenBLAS directive is not reflected in most references.

## 2. Scope and method

**Covered:** build configuration and CI workflows; the last 8 CI runs and the failing job logs; a secret scan; the server's network surface; download and extraction code; native library loading; the GGUF and torch readers; the env-var registry checked against source; the project dependency graph; the largest files; tracked binaries and blob sizes; a markdown link check; the CLAUDE.md rules; README claims against the docs.

**Not covered (limits):**
- I did not build or run any test suite on this machine. Test results come from CI logs (run `45732348723` and the earlier red runs in section 9). M2 I also reproduced by running the same source scan by grep.
- I did not read the 348k lines of source. The CUDA and Vulkan kernels, the audio pipelines, the diffusion models and the session code were sampled by search, not read in full.
- No runtime or security testing. The rebinding exposure in M5 is inferred from configuration and must be confirmed by test.
- The evidence rows in `docs/STATUS.md` were not audited one by one.

## 3. Must fix before tagging 1.0.10

**M1. The AOT publish fails: NETSDK1129.** The CLI project now targets `net10.0;net11.0` (`src/OpenTail.Stingray.Cli/OpenTail.Stingray.Cli.csproj:5`). A `dotnet publish -r <rid>` with no `-f` refuses to choose a framework.
- Where: `.github/workflows/ci.yml:122` (Linux) and `:183` (Windows); `scripts/check-aot-telemetry.ps1:23`.
- Evidence: run `45732348723`, job "NativeAOT telemetry self-test": `error NETSDK1129: The 'Publish' target is not supported without specifying a target framework`. The multi-target came in with `341a1cfc` (release candidate 1.0.10, 2026-11-25 23:58). The last green run, `9ef49a8f` (08:50 the same day), predates it. The Windows job never reached its AOT or package-smoke steps, because the suite step failed first.
- Fix: add `-f net10.0` to each publish, or let the owner decide whether AOT ships net11 as well. `release.yml` uses `pack`, which this error does not affect.

**M2. `ListMatchesSource` fails: an env var is read but not registered.** `STINGRAY_CPU_PREFILL_FAST_GEMM` is read at `src/OpenTail.Stingray.Cpu/SimdKernels.cs:471`. It is not in `src/OpenTail.Stingray.Core/KnownEnvironmentVariables.cs`. Commit `3cfcc60b` added it.
- Evidence: the Core test fails on Linux and Windows (`KnownEnvironmentVariablesTests.ListMatchesSource`). A grep of `STINGRAY_` in `src/*.cs` against the registry finds this as the only real unregistered read.
- Consequence beyond CI: `Server.Host/Program.cs:5-12` prints "is set but is not read by this build — it will have no effect" for any `STINGRAY_*` name missing from the registry (`KnownEnvironmentVariables.cs:348-357`). A user who sets `STINGRAY_CPU_PREFILL_FAST_GEMM=1` gets a false warning, although the opt-in works.
- Fix: register the name, classified in `EnvironmentVariableClasses.cs` (probably `experimental`). This has happened before: `e5051079 Register STINGRAY_FFMPEG in the environment-variable registry (CI fix)`. Run the Core suite before every push.

**M3. Admission-queue stall, at its documented trigger.** `ModelRuntimeManagerTests.AcquireAsync_WokenWaiter_ReservesLoadAtomically_FreshCallerCannotStealTheSlot` took 30 s 036 ms and failed on Linux (`tests/OpenTail.Stingray.Tests.Server.Fast/ModelRuntimeManagerTests.cs:1504`).
- The backlog says (`docs/00-current-work.md:179`): "If it recurs with a stack that stays at the same wait, treat it as a real admission-queue defect and look at the wake path first." Earlier occurrences: G14 (2026-11-03, Windows, `docs/done/2026-11-01-plan-6-completed-items-2026-11.md:35`) and A17 (2026-11-24, `00-current-work.md:179`).
- The timeout was widened to 30 s instead of finding the cause. A permanent lost wake-up would still fail, but only after 30 s.
- What I read: `ModelRuntimeManager.cs` wake path (about lines 497-560). It looks right to me; I did not prove a lost wake-up. The test's `Load` helper is synchronous (`ModelRuntimeManagerTests.cs:38`), so once `older` is woken its load finishes at once. The 30 s therefore sits either on the wake, which never fired, or on a continuation that was delayed (thread-pool starvation would be more likely with the 2-CPU setting the test uses). Neither is ruled out.
- Next step: loop the test under `DOTNET_PROCESSOR_COUNT=2` with a dump of `_admissionQueue`, `_pendingLoads` and the thread-pool state at the stall. Do not widen the timeout again.

**M4. Server DNS-rebinding exposure (security, cheap to fix, verify first).** The demo host binds `127.0.0.1:8080` with no authentication, by design (`src/OpenTail.Stingray.Server/README.md:36-41`). But `src/OpenTail.Stingray.Server.Host/appsettings.json` sets `"AllowedHosts": "*"`, which accepts any `Host` header. That is the usual condition for a web page that rebinds its domain to 127.0.0.1 to read responses from a local unauthenticated API. This is inferred from configuration, not tested.
- Fix: set `AllowedHosts` to `localhost;127.0.0.1;[::1]`. Add a test that a request with `Host: evil.example` is refused. Say in the Server README that the host is for loopback use only.
- Verify first: send a request with a foreign `Host` header to a running host and record the status code.

## 4. Should fix

**S1. Scratch test files are committed.** 29 files `tests/**/Zz*Tmp.cs` are tracked. Their headers say "Scratch (untracked)" (for example `tests/OpenTail.Stingray.Tests.Diffusion/ZzAceStepCmpTmp.cs`). They compile into the test projects. Seven of them contain the silent-pass pattern `if (... is null) return;`, so they pass when their model is absent. Several hard-code `C:\Git-Public\OpenTail.Stingray\models\`. This breaks CLAUDE.md rule 9, which says investigation artifacts belong in the scratchpad.
- Fix: after owner approval, remove them from the index (a deletion of files the owner created, so it needs approval).

**S2. Tuning dumps are committed.** Eight files `docs/diffusion-samples/zz_scratch_csharp_t5_*.bin` (about 1.2 MB each, added in `6b656cdc`, 2026-10-13). Also an untracked `docs/diffusion-samples/zz_flux2_prof.png`. CLAUDE.md rule 9 forbids "tuning-iteration dumps" in the samples. The archived notes cite them (`docs/done/096-t5-bisection-howto.md:66-67`).
- Fix: after owner approval, remove them; add a line in the archived notes saying they were removed.

**S3. `.mcp.json` is tracked.** It holds per-machine absolute paths (`C:/git/opentail/...`) and passes `--allow-writes` to an MCP server that writes a database at `C:/git/opentail/opentail.db`. This is developer-local configuration in a public repository. Fix: untrack and ignore it (owner decision).

**S4. The OpenBLAS directive is not reflected in the references.** CLAUDE.md requires every reference to say OpenBLAS is never to be used, and forbids running `scripts/setup-openblas.ps1`. In the non-doc lines I sampled:
- `src/OpenTail.Stingray.Cpu/BlasInterop.cs:9` tells readers to run the setup script.
- `src/OpenTail.Stingray.Cli/RunCommand.Settings.cs:176` advertises an OpenBLAS SGEMM path as a CLI option.
- `benchmarks/OpenTail.Stingray.Bench/InferenceBenchmark.cs:1417` compares against OpenBLAS.
- `benchmarks/OpenTail.Stingray.Bench/ContinuousBatchingHarness.cs:58` prints OpenBLAS status.

The gate itself is correct: `BlasInterop.cs:52` returns false unless `STINGRAY_USE_OPENBLAS=1`. Across tracked files there are 312 OpenBLAS lines in 76 files. A separate owner instruction says every reference is to be removed, while CLAUDE.md says keep the gate and state never-use. These differ; see section 7.

**S5. A public number has no date or command.** The README's "Hardware and speed" table says "~60-70 tokens/s (Qwen2.5 0.5B)" (`README.md:299`). The repo's own measurements for this model are 28.2 t/s (decode, 2026-09-28, `docs/4-performance/PerformanceLeague-LOG.md:648`) and 75.48 t/s at 8 threads (2026-11-24, `docs/4-performance/2026-11-08-performance-plan.md:56`). Public text must match its evidence: date the figure, cite the command, or reconcile it.

**S6. A config sketch looks like the shipped config.** `docs/reference/OpenTail.Stingray-Design.md`, section 9.6 (around line 1245), shows `"Server": {"Host": "0.0.0.0", "Port": 8080}` and a `Models` array. The shipped host reads `Kestrel:Endpoints:Http:Url` and `OpenTail.Stingray:*` (`appsettings.json`). The section has no "sketch" label, and copying it gives a LAN bind with no authentication.

**S7. CI cannot catch silent passes.** CI runs only the CPU-safe suites, and the runner has no `models/`, so the real-model suites are skipped there. Eleven test files contain the silent-return pattern. Seven of them are the scratch files in S1, and four (`ScoutSignatureTests`, `CudaDSparkE2ETests`, `Sd3TimestepEmbedParityTests`, `SafeTensorsRamProbeTests`) are not. About 171 skip sites exist. CLAUDE.md rule 12 covers this, but nothing enforces it. Fix: have CI report real-weight tests as skipped rather than passed, or assert a minimum duration for them.

**S8. Repository size and tracked artifacts.** Tracked samples are about 23 MB in 44 files, against CLAUDE.md's "keep the whole set to a few MB". Twenty untracked WAVs (3.2 MB) sit in `docs/audio-samples`. `tests/fixtures` holds 28 files, about 20.5 MB in total, mostly `.f32` tensors. The generated `src/OpenTail.Stingray.Vulkan/Shaders.Precompiled.g.cs` is 10.6 MB and tracked. Owner decision on each.

**S9. Silent `catch { }` blocks.** At least 45 single-line empty catches are in `src/`. Most are cleanup or `finally` guards (for example `ControlNetModel.cs:579`, `ZImagePipeline.cs:273`, `CudaBackend.cs:7350`), which is acceptable. The ones that could hide a real failure from the caller need a log line or a comment saying why. I did not check `ImageCommand.cs:186` and `:216`.

**S10. Git state.** Six commits on `origin/main` came after the last green run (`9ef49a8f`), and CI ran only on the last one (`3cfcc60b`). Two commits are unpushed. A second worktree (branch `backlog/2026-11-08-other`) and `stash@{0}` ("WIP on main") remain. The global rule is no worktrees or branches unless asked. I did not touch any of them. Push only after M1 and M2 are fixed and the Core suite passes locally.

## 5. Notes

- **N1.** `global.json` pins SDK `10.0.301` with `rollForward: latestMajor`. CI selected the .NET 11.0.100 SDK (the failure log shows `/usr/share/dotnet/sdk/11.0.100/`), while the README asks for the .NET 10 SDK. Check that this is intended.
- **N2.** `Directory.Packages.props` sets `ManagePackageVersionsCentrally=false` with a comment about a "standalone subtree" that does not match this repo. It is harmless and confusing.
- **N3.** `src/OpenTail.Stingray.Cli/Tui/**` and the `#if STINGRAY_TUI` paths are excluded by default (`EnableTuiChat=false`). They are never compiled or tested in CI. The csproj documents this.
- **N4.** Twelve `scripts/*_ref.py` files are tracked. `docs/reference/061-coverage-tooling.md` says no new ones should be added. Owner decision on removing the old ones.
- **N5.** There are 82 broken relative links, all in the archive under `docs/done/`. Active docs have none.
- **N6.** Seven files in `docs/done/native-port-plans/` (`s*.txt`) are pasted chat replies, not documents. Owner decision.
- **N7.** Across tracked code, 96 files hard-code `C:\Git-Public`, `C:\git` or `C:\Users\` paths (151 lines in `.cs` files alone). Across all tracked files, 36 lines name a `C:\Users\` profile path. Mostly real-weight tests and `scripts/bench-390-*.ps1`. This makes them non-portable. A model-root setting would fix it.
- **N8.** `src/OpenTail.Stingray.Cpu/native/f16c_shim.dll` is a tracked prebuilt binary. Its source (`f16c_shim.c`) and `native/build.bat` are in the repo. Record a build hash to confirm reproducibility.
- **N9.** Thirteen files in `src/` exceed 2,000 lines. The largest are `Shaders.cs` (15,022, shader strings), `SimdKernels.cs` (13,204), `CudaTextKernels.cs` (10,544), `CudaBackend.cs` (8,493) and `CudaHybridGdnForwardPass.cs` (7,889). A maintenance risk, not a defect.
- **N10.** The dependency graph has no cycles. Core is a leaf and Server is the top. One coupling to check: `OpenTail.Stingray.Diffusion` references `OpenTail.Stingray.Audio`, so a diffusion build pulls in the audio stack.

## 6. What is solid (checked in code)

- **GGUF parser** (`src/OpenTail.Stingray.Core/GgufModel.cs`): string lengths are checked against the remaining bytes before reading (`EnsureAvailable`, about line 668). Array counts are refused when they exceed the remaining bytes, before anything is sized from them (`RequirePlausibleCount`, about line 710). Tensor rank is refused before allocation (about lines 446-450). Tensor extents use `checked` arithmetic and are checked against the file size (about lines 509-518).
- **Torch reader** (`src/OpenTail.Stingray.Core/TorchCheckpointReader.cs:252-262`): an allowlist of two callables, `OrderedDict` and `_rebuild_tensor_v2`. Every other global throws. No code runs from a `.pt` file.
- **Pull path** (`src/OpenTail.Stingray.Cli/PullCommand.cs:116-124`): rooted names and `..` escapes are refused before writing.
- **Hub downloads** (`src/OpenTail.Stingray.Core/Net/HuggingFaceOnlyHandler.cs:35-45`): every redirect hop is host-checked, and the bearer token is sent only to the Hub, never to the CDN.
- **OpenBLAS gate** (`src/OpenTail.Stingray.Cpu/BlasInterop.cs:52`): off unless `STINGRAY_USE_OPENBLAS=1`.
- **Build settings** (`Directory.Build.props`): `TreatWarningsAsErrors`, nullable, trim and AOT analyzers, pinned SourceLink.
- **Release guard** (`.github/workflows/release.yml`): refuses a tag that does not match `<Version>`, and documents the trap that would silently skip a duplicate version.
- **CI floors** (`.github/workflows/ci.yml`, `run_suite`): every suite must discover at least a minimum number of tests, and every suite runs even when an earlier one fails.
- **Secret scan:** no credentials in tracked files. The one `hf_` literal is a redaction fixture (`tests/OpenTail.Stingray.Tests.Cli/DoctorBundleTests.cs:59`).
- **Deployment boundary:** the Server README says plainly that the host has no authentication and must not be bound to a LAN without one (`src/OpenTail.Stingray.Server/README.md:36-41`).
- **Code markers:** 9 TODO/FIXME markers and 2 warning suppressions in all of `src/`.
- **Active docs:** zero broken relative links.

## 7. Owner decisions needed

1. **OpenBLAS:** keep the gate and state "never used" on the remaining references (CLAUDE.md), or remove the references (the separate owner instruction)?
2. **AOT and net11:** fix M1 with net10 only, or ship AOT for both targets?
3. **Removal from git:** approve removing the 29 `Zz*Tmp.cs` files (S1), the 8 `zz_scratch` dumps (S2), `.mcp.json` (S3), and the 12 `*_ref.py` scripts (N4)?
4. **Git cleanup:** the backlog worktree and `stash@{0}` (S10).
5. **Samples:** add or ignore the 20 untracked WAVs (S8), and trim the tracked set to a few MB?

## 8. Suggested order

1. Fix M1 and M2 in one commit. Run the Core suite locally, then push. This should turn `main` green.
2. Verify and fix M4 (`AllowedHosts`).
3. Investigate M3 with the repro loop. Do not widen the timeout.
4. Tag `stingray-v1.0.10` only after CI is green on the tagged commit.
5. Then the owner decisions in section 7, followed by the cleanup in S1 to S3.

## 9. Verification log

- `gh run list --limit 8`: the eight most recent runs and their head commits.
- `gh run view 45732348723 --log-failed`: the failing tests (`KnownEnvironmentVariablesTests.ListMatchesSource` on both platforms, `ModelRuntimeManagerTests.AcquireAsync_WokenWaiter...` at 30 s 036 ms), the `NETSDK1129` error, and "suites failed: Core" on Windows.
- `gh run view 45570543687` and `45575886689`: earlier red runs (Core and Sessions.Fast on 2026-11-25).
- `git grep` for `STINGRAY_` in `src/*.cs`, compared with the registry: `STINGRAY_CPU_PREFILL_FAST_GEMM` is the only real unregistered read (`SimdKernels.cs:471`).
- Project references parsed from every `.csproj` in `src/`.
- Link check over tracked `*.md`: 82 broken, all under `docs/done/`.
- `git cat-file --batch-check` over the tracked blobs: about 98 MB in total; the largest files are listed in S8 and N8.
- Line numbers are from `2277fb71` and may drift.
