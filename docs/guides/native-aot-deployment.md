# Technical Guide: NativeAOT & Zero-Dependency Deployment

> **Topic:** Building self-contained, single-file native executables and Docker images with .NET 10 NativeAOT  
> **Audience:** DevOps engineers and application developers shipping standalone binaries  
> **Key Characteristics:** No .NET runtime installation required; measured cold start and size are in [Measured numbers](#measured-numbers)

---

## 1. Why NativeAOT for AI Runtimes?

Traditional .NET deployments require either a pre-installed .NET runtime or a bulky self-contained runtime bundle containing JIT compilers.

With **NativeAOT**, the .NET 10 compiler ahead-of-time compiles your managed C# code directly into machine code for the target OS and architecture:
- **Fast Startup:** No JIT warm-up for the CLI itself; model loading time is separate (see [Measured numbers](#measured-numbers)).
- **Single Executable:** Ships as a single binary file (`.exe` on Windows, ELF binary on Linux).
- **Reduced Memory:** Eliminates JIT compilation metadata and overhead from the process working set.

---

## 2. Project Configuration for NativeAOT

To compile an application consuming `OpenTail.Stingray` with NativeAOT, configure your `.csproj`:

```xml
<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net10.0</TargetFramework>
    <Nullable>enable</Nullable>
    <ImplicitUsings>enable</ImplicitUsings>
    
    <!-- NativeAOT Flags -->
    <PublishAot>true</PublishAot>
    <InvariantGlobalization>true</InvariantGlobalization>
    <StripSymbols>true</StripSymbols>
  </PropertyGroup>

  <ItemGroup>
    <PackageReference Include="OpenTail.Stingray" Version="1.0.8" />
  </ItemGroup>

</Project>
```

---

## 3. Trimming & Reflection Rules

Because NativeAOT trims unused assemblies and removes reflection metadata, adhere to these development rules:

1. **Source-Generated JSON:** When serializing requests, responses, or configurations, use `System.Text.Json` source generation contexts (`[JsonSerializable]`). Do not use unannotated `JsonSerializer.Deserialize<T>()`.
2. **Reflection-Free API Facade:** The primary Stingray API (`Model`, `ModelContext`, `InteractiveExecutor`, `ChatSession`) is built strictly without dynamic code generation or unbound reflection.
3. **P/Invoke Isolation:** Stingray uses no native C++ P/Invoke libraries; all vector math is generated directly by the AOT compiler via hardware intrinsics.

---

## 4. Publishing Commands

### Publishing for Windows (x64)
```bash
dotnet publish -c Release -r win-x64 --self-contained
```

### Publishing for Linux (x64)
```bash
dotnet publish -c Release -r linux-x64 --self-contained
```

The resulting executable in `bin/Release/net10.0/<rid>/publish/` can be copied to any compatible machine without requiring .NET 10 or external dependencies installed.

---

## 5. Minimal Container Deployment

Because NativeAOT binaries require only the standard C library (or musl), you can deploy using minimal container images:

```dockerfile
FROM mcr.microsoft.com/dotnet/nightly/aot-chiseled:10.0 AS runtime
WORKDIR /app
COPY bin/Release/net10.0/linux-x64/publish/MyStingrayApp .
ENTRYPOINT ["./MyStingrayApp"]
```

The image size depends on the base image and on which native libraries you ship; it has not been measured here, so check your own build.

## Measured numbers

Measured 2026-10-11 on Windows 11, AMD Ryzen 7 5700G, by publishing `src/OpenTail.Stingray.Cli` with `dotnet publish -c Release -r win-x64` (NativeAOT) and timing `stingray --help` (process start to exit, 9 runs each, nothing else running):

| Build | `--help` wall time (median, min) | Peak working set (median) |
|---|---|---|
| NativeAOT `stingray.exe` | 62 ms (61 ms) | 13 MiB |
| Same CLI, JIT (`bin/Release/net10.0/stingray.exe`, needs the .NET 10 runtime) | 111 ms (109 ms) | 25 MiB |

The NativeAOT executable is 27.4 MiB; the whole publish folder is about 132 MiB, most of it `.pdb`/`.xml` debug and documentation files you do not need to ship (not stripped for this measurement). **This is the startup of the command line, not of inference:** loading a model (mapping the file, reading the tokenizer, preparing the backend) and producing the first token take far longer and depend on the model and the machine. The 62 ms figure should not be read as a time to first token. Other platforms and the server host were not measured.
