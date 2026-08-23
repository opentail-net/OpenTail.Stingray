@echo off
REM Rebuilds f16c_shim.dll from f16c_shim.c. Requires MSVC Build Tools (adjust the vcvarsall.bat
REM path below if your install location differs -- find it via:
REM   "%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * ^
REM     -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
REM See docs/audio-review-progress.md's F16C investigation entry for why this exists: real
REM hardware F16C (VCVTPH2PS) conversion of Whisper's F16 weights straight into the FMA loop,
REM matching ggml's own ggml_vec_dot_f16 -- not reachable from managed .NET code (Half is not a
REM legal Vector128/256<T> element type, and every managed attempt measured 9-15x SLOWER than F32).
cd /d %~dp0
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
if errorlevel 1 exit /b 1
cl.exe /LD /O2 /arch:AVX2 /Fe:f16c_shim.dll f16c_shim.c
