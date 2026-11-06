global using System;
global using System.Runtime.InteropServices;

// Required for LibraryImport to pass MlxOptionalFloat (a bool-containing struct) by value, same as
// TensorSharp's own MlxNative.cs.
[assembly: System.Runtime.CompilerServices.DisableRuntimeMarshalling]
