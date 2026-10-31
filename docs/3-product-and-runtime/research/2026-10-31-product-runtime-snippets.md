# Code excerpts behind the 3-product-and-runtime research (2026-10-31)

Companion to the research sections appended to the plans in [docs/3-product-and-runtime](../). Excerpts are copied from the working tree on 2026-10-31. Nothing here has been changed or built.

## 1. Forward-pass context isolation (plan 010)

`src/OpenTail.Stingray.Core/IForwardPass.cs`, line 14:

```csharp
IForwardPass CreateContext() => this;
```

`src/OpenTail.Stingray.Core/IForwardPass.cs`, line 139 (the default that GDN passes inherit):

```csharp
bool SupportsPartialRewind => false;
```

Search results (2026-10-31): no `CreateContext()` override under `src/OpenTail.Stingray.Engine`, and `ThrowIfForwardPassNotIsolated` is not present under `src/`. The `DeepSeek2GpuForwardPass.cs:139` override (`SupportsPartialRewind => true`) is the only true override found.

## 2. Multi-model admission defaults (plan 032)

`src/OpenTail.Stingray.Server/ModelRuntimeManager.cs`, lines 146 and 150:

```csharp
public TimeSpan? AdmissionWaitTimeout { get; set; }
...
public int MaxQueuedAdmissions { get; set; } = 16;
```

## 3. Hugging Face listing query (scout plan, Phase 4 and `scout -r`)

`src/OpenTail.Stingray.Cli/Scout/Backlog.cs`, line 72:

```csharp
string urlStr = $"https://huggingface.co/api/models?filter=gguf&pipeline_tag=text-generation&sort=downloads&direction=-1&limit={effectiveLimit}&expand[]=gguf&expand[]=downloads&expand[]=gated";
```

`src/OpenTail.Stingray.Core/GgufModel.cs`, line 287 (version gate):

```csharp
throw new InvalidDataException($"Unsupported GGUF version: {version}");
```

## 4. Catalogue licence for the speech voice (known-good plan, P5 and the licence rule)

`src/OpenTail.Stingray.Core/Catalog/ModelCatalog.cs`, lines 162-164 (`piper-lessac`):

```csharp
Licence: "MIT (code); the voice was trained on the Lessac Blizzard 2013 data, whose own licence applies: "
    + "https://www.cstr.ed.ac.uk/projects/blizzard/2013/lessac_blizzard2013/license.html",
LicenceNeedsConsent: true,
```

The linked CSTR licence (fetched 2026-10-31) says commercial use is not permitted, redistribution is not permitted, the terms are for research use, and audiobook use is excluded. The consent flag is set, but the summary text leads with "MIT".

## 5. Telemetry instrument names (OpenTelemetry plan, Phases 1 to 4)

`src/OpenTail.Stingray.Engine/EngineTelemetry.cs`, lines 72-76 and 191-193 (excerpt):

```csharp
"opentail.engine.prefill.duration", "s", "Prompt processing time (new prompt tokens only).");
"opentail.engine.prefill.tokens", "{token}", "New prompt tokens processed (reused prefix tokens excluded).");
...
"opentail.engine.placement.gpu_layer_share", "1", "Share of the model's layers placed on the GPU at load (0 = all CPU). Selecting a GPU does not mean the whole model runs on it.");
"opentail.engine.backend.fallbacks", "{load}", "Loads that ran on a different backend than the one explicitly requested; reason is a closed category.");
```

## 6. OpenTelemetry pins in the host

`src/OpenTail.Stingray.Server.Host/OpenTail.Stingray.Server.Host.csproj`, lines 36-38:

```xml
<PackageReference Include="OpenTelemetry.Extensions.Hosting" Version="1.19.1" />
<PackageReference Include="OpenTelemetry.Exporter.OpenTelemetryProtocol" Version="1.19.1" />
<PackageReference Include="OpenTelemetry.Exporter.Console" Version="1.19.1" />
```

NuGet page fetched 2026-10-31: `OpenTelemetry.Extensions.Hosting` latest stable is 1.20.1 (updated 10/20/2026).

## 7. VRAM budget (multi-model plan, Phase 3 slice 7)

`src/OpenTail.Stingray.Server/ResourceBudget.cs`, line 111 (comment only; no call to the extension):

```csharp
/// has no portable "free VRAM right now" query without the <c>VK_EXT_memory_budget</c> extension,
```

The Vulkan extension itself returns `heapBudget` and `heapUsage` per memory heap through `VkPhysicalDeviceMemoryBudgetPropertiesEXT` (Khronos spec, fetched 2026-10-31 via search). The values are estimates and depend on the OS and load.
