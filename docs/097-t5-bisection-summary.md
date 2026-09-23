# T5 Encoder Bisection Implementation Summary

**Date**: 2026-09-21  
**Status**: Diagnostic infrastructure complete and ready to use

## What Was Done

Implemented a comprehensive T5 encoder bisection diagnostic system to isolate where the T5 encoder diverges from the C++ reference (current cosine similarity: 0.169, should be ~0.99+).

## Changes Made

### 1. C# Implementation (OpenTail.Stingray)

**Modified**: `src/OpenTail.Stingray.Diffusion/TextEncoders/T5Encoder.cs`

Added diagnostic dumps at 4 key checkpoints:
- **E0**: Token embedding output (before any blocks)
- **A0**: After block 0's self-attention + residual (before FFN)
- **F0**: After block 0's FFN + residual (= input to block 1)
- **final**: After all 24 blocks + final RMSNorm

**Key additions**:
- `DiagnosticDump(checkpoint, data)` helper method
- Environment variable support: `STINGRAY_T5_DUMP_PREFIX`
- Call counter to distinguish cond vs uncond passes
- Console logging for verification

**New test file**: `tests/OpenTail.Stingray.Tests.Diffusion/ZZ_ScratchT5BisectionDumpTest.cs`

Two test methods:
- `Scratch_DumpT5Checkpoints` - generates C# dumps for comparison
- `Scratch_CompareT5CheckpointsAgainstReference` - compares against C++ dumps with cosine similarity stats

### 2. C++ Implementation (stable-diffusion.cpp)

**Modified**: `examples/stable-diffusion.cpp/src/model/te/t5.hpp`

Added diagnostic dumps at the same 4 checkpoints to match C# implementation.

**Key additions**:
- `diagnostic_dump_t5_tensor(checkpoint, tensor, backend)` helper function
- Environment variable support: `SD_DUMP_T5_PREFIX`
- Downloads tensor from backend before dumping
- Same binary format as C# (raw float32)

**Modified methods**:
- `T5Block::forward` - added `block_idx` parameter and A0 dump
- `T5Stack::forward` - added E0, F0, and final dumps

### 3. Documentation

**New guide**: `docs/096-t5-bisection-howto.md`

Complete step-by-step instructions for:
- Generating C# dumps
- Building and running C++ reference with dumps
- Comparing dumps with statistical analysis
- Interpreting results to locate the bug
- Next actions based on which checkpoint diverges

**Updated**: `docs/095-sd35-t5-encoder-bisection-plan.md`

Added note that diagnostic infrastructure is complete and ready to use.

## Build Verification

✅ C# code builds successfully:
```bash
cd c:\Git-Public\OpenTail.Stingray
dotnet build src/OpenTail.Stingray.Diffusion/OpenTail.Stingray.Diffusion.csproj
```

Build completed in 32.8s with no errors.

## How to Use

See `docs/096-t5-bisection-howto.md` for complete instructions.

**Quick start**:

1. **Generate C# dumps**:
```bash
cd tests/OpenTail.Stingray.Tests.Diffusion
dotnet test --filter "FullyQualifiedName~ZZ_ScratchT5BisectionDumpTest.Scratch_DumpT5Checkpoints"
```

2. **Build C++ reference**:
```bash
cd examples/stable-diffusion.cpp
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

3. **Generate C++ dumps**:
```bash
$env:SD_DUMP_T5_PREFIX="..\..\..\..\docs\diffusion-samples\zz_scratch_cpp_t5"
.\sd.exe --model ... --t5xxl ... --prompt "a red apple on a wooden table" ...
```

4. **Compare**:
```bash
dotnet test --filter "FullyQualifiedName~ZZ_ScratchT5BisectionDumpTest.Scratch_CompareT5CheckpointsAgainstReference"
```

## Expected Outcome

The comparison will show cosine similarity for each checkpoint:
- If **E0** matches (~0.99+) but **A0** diverges → Self-attention bug
- If **A0** matches but **F0** diverges → FFN bug
- If **F0** matches but **final** diverges → Later-layer accumulation bug
- If **E0** diverges → Embedding lookup bug (unlikely, but possible)

This pinpoints the exact subsystem containing the bug, eliminating the need to guess.

## Important Notes

1. **Temporary diagnostic code** - All dump infrastructure is for debugging only and should be removed after the bug is fixed
2. **Prompt must match exactly** - Use `"a red apple on a wooden table"` (no punctuation) for both C# and C++ runs
3. **Binary format** - Raw float32, little-endian, no header (platform-native endianness)
4. **Expected tensor shapes** - All checkpoints: `[77, 4096]` = 315,392 floats = 1,261,568 bytes

## Files Changed

**Modified** (keep after bug is fixed):
- `docs/095-sd35-t5-encoder-bisection-plan.md` (updated status)

**Modified** (temporary, remove after fix):
- `src/OpenTail.Stingray.Diffusion/TextEncoders/T5Encoder.cs` (DiagnosticDump calls)
- `examples/stable-diffusion.cpp/src/model/te/t5.hpp` (diagnostic_dump_t5_tensor calls)

**New** (temporary, delete after fix):
- `tests/OpenTail.Stingray.Tests.Diffusion/ZZ_ScratchT5BisectionDumpTest.cs`
- `docs/096-t5-bisection-howto.md` (reference during investigation)
- `docs/097-t5-bisection-summary.md` (this file)

**Generated** (temporary, delete after fix):
- All `docs/diffusion-samples/zz_scratch_*.bin` files

## Next Steps

1. Run the bisection following `docs/096-t5-bisection-howto.md`
2. Identify which checkpoint first diverges
3. Read the corresponding C++ and C# code for that subsystem
4. Perform line-by-line comparison
5. Fix the bug
6. Verify the fix by re-running the full SD3.5 comparison test
7. Clean up all diagnostic infrastructure
8. Commit only the actual fix

## Success Criteria

- T5 context tensor cosine similarity improves from 0.169 to ~0.99+
- SD3.5 composition bug is resolved (apple appears full-frame and centered)
- Both CPU and GPU backends produce identical results to the C++ reference
