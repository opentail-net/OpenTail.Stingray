# Recommended configurations

These are conservative starting points, not benchmark claims. Run `inspect` first against the
exact GGUF on the deployment machine, then run `plan` with the selected profile. The report opens
the model index but does not load weights or run inference, so it is safe to use in deployment checks.

```powershell
dotnet run --project src/OpenTail.Stingray.Cli -c Release -- `
  inspect -m C:\models\model.gguf --json
dotnet run --project src/OpenTail.Stingray.Cli -c Release -- `
  plan -m C:\models\model.gguf --profile docs\profiles\cpu.json --explain
```

`inspect --json` is the live capability report. It states the model architecture, tensor dtype
census, detected CPU/CUDA/Vulkan availability, MTP/speculation eligibility, batching eligibility,
and the status of restart-continuation sessions. `plan --explain` adds the resolved configuration
and placement decision. Do not substitute a model-name heuristic for either report.

## CPU-only

Use this for a portable or shared machine. It avoids GPU probing and keeps the baseline unambiguous.

```powershell
dotnet run --project src/OpenTail.Stingray.Cli -c Release -- `
  -m C:\models\model.gguf -p "Hello" --backend cpu -g 0 --temp 0
```

For a server, start with one batch and size CPU threads for the machine's other workload. Increase
`MaxBatchSize` only after measuring aggregate latency and memory use.

```powershell
$env:STINGRAY_MODEL = 'C:\models\model.gguf'
$env:STINGRAY_BACKEND = 'cpu'
$env:STINGRAY_N_GPU_LAYERS = '0'
$env:STINGRAY_CPU_THREADS = '8'       # example: choose deliberately; 0 is automatic
$env:STINGRAY_MAX_BATCH = '1'
dotnet run --project src/OpenTail.Stingray.Server.Host -c Release
```

`STINGRAY_KV_STORE=auto` is a CPU dense-path optimisation for long-lived contexts. It is not a
universal format switch: keep it unset unless a workload test establishes that the selected model
uses the supported dense CPU path.

## CUDA dense model

Use full offload only when `plan` says the model and requested context fit detected VRAM.

```powershell
dotnet run --project src/OpenTail.Stingray.Cli -c Release -- `
  -m C:\models\model.gguf -p "Hello" --backend cuda -g -1 --temp 0
dotnet run --project src/OpenTail.Stingray.Cli -c Release -- `
  plan -m C:\models\model.gguf --profile docs\profiles\cuda-dense.json --explain
```

For long context on a supported CUDA dense path, request the KV dtype explicitly and record it in
the profile. `bf16` roughly halves KV memory; `q8_0` trades more precision for further capacity.
The plan rejects combinations that do not apply.

## Vulkan

Treat Vulkan as a distinct backend, not as a CUDA spelling. Probe it and force it deliberately:

```powershell
dotnet run --project src/OpenTail.Stingray.Cli -c Release -- `
  plan -m C:\models\model.gguf --backend vulkan -g -1 --explain
```

Keep `kv_type` at `fp32`; CUDA KV dtypes do not apply. With MoE or TurboQuant, retain the
planner's decision trace with the deployment record because codec and placement paths are
model- and backend-specific.

## Hybrid MoE

Hybrid MoE deployment is capacity planning first. Start with automatic placement and let the
planner report the exact CPU/GPU layer split; do not copy a layer count from a different GPU.

```powershell
dotnet run --project src/OpenTail.Stingray.Cli -c Release -- `
  plan -m C:\models\model.gguf --profile docs\profiles\hybrid-moe.json --explain
```

For CUDA hybrid MoE, `STINGRAY_CPU_MOE=1` is an intentional alternative to the GPU expert
cache, not a universal speed flag. Evaluate both on the representative prompt mix. If the deployment
uses GPU MoE prefill, retain the default `STINGRAY_MOE_GPU_PREFILL=1` unless a measured
compatibility or latency constraint says otherwise.

## Local API server

Copy `src/OpenTail.Stingray.Server.Host/appsettings.json` to the ignored
`appsettings.Local.json` for durable operator-owned settings; use environment variables only for
machine-local paths or secrets. For a multi-user server, begin with bounded admission:

```powershell
$env:STINGRAY_MODEL = 'C:\models\model.gguf'
$env:STINGRAY_BACKEND = 'cuda'
$env:STINGRAY_N_GPU_LAYERS = '-1'
$env:STINGRAY_MAX_BATCH = '8'
$env:STINGRAY_MAX_QUEUE = '16'
$env:STINGRAY_PREFILL_CHUNK = '256'
dotnet run --project src/OpenTail.Stingray.Server.Host -c Release
```

Verify the running process—not just its intended configuration—with `GET /capabilities`, `GET
/status`, and `GET /metrics`. Continuous batching makes tool-argument grammar ineligible; the
capability endpoint says so directly. The server host deliberately does not use NativeAOT/trim
because the CUDA/NVRTC loading path is not trim-safe.

## Saved planning profiles

The example profiles under [docs/profiles](profiles) contain only static planning knobs: no paths,
prompts, credentials, or benchmark-specific tuning. A profile is an input, not proof; pair it with
the model-specific `inspect`/ `plan` output from the actual machine.
