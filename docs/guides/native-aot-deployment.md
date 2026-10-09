# Technical Guide: NativeAOT & Zero-Dependency Deployment

> **Topic:** Building self-contained, single-file native executables and Docker images with .NET 10 NativeAOT  
> **Audience:** DevOps engineers and application developers shipping standalone binaries  
> **Key Characteristics:** Zero .NET runtime installation required, sub-50ms startup time, tiny disk footprint

---

## 1. Why NativeAOT for AI Runtimes?

Traditional .NET deployments require either a pre-installed .NET runtime or a bulky self-contained runtime bundle containing JIT compilers.

With **NativeAOT**, the .NET 10 compiler ahead-of-time compiles your managed C# code directly into machine code for the target OS and architecture:
- **Instant Startup:** Zero JIT compilation warm-up; model initialization begins immediately.
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
    <PackageReference Include="OpenTail.Stingray" Version="1.0.7" />
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

This produces production container images under **40 MB** before models are mounted.
