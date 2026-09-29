# DFW native DLL deployment

The 64-bit Windows library is deployed at:

```text
src/flow4d/native/flow4d_dfwavelet.dll
```

The current file was built from the vendored sources in
`native/dfwavelet/` and copied from
`native/dfwavelet/build/Release/flow4d_dfwavelet.dll`. Its architecture must
match the Python process architecture. The default Python backend checks only
the fixed package location above; callers may instead pass an explicit path to
`DfwNativeBackend`.

If the DLL is missing, cannot be loaded, or lacks the expected C ABI symbol,
the backend raises `DfwBackendError`. It does not skip DFW, return the input, or
fall back to an approximate wavelet implementation.

Current validation boundary: the C++ source, C ABI, Python backend,
multi-phase orchestration, Release build, and package-location deployment are
present. This migration round did not load or execute the DLL. MATLAB-reference,
real-case, ABI loading, dependency, and MATLAB-free Windows validation have not
yet been completed. Do not describe this backend as numerically equivalent or
independently runnable until `docs/dfw_external_validation.md` is completed.
