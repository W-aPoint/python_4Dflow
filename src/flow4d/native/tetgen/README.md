# TetGen executable deployment

The default executable location is:

```text
src/flow4d/native/tetgen/tetgen.exe
```

An explicit path passed to `TetGenTetrahedralizationBackend` takes precedence.
The backend does not search `PATH` or the filesystem. A missing, inaccessible,
or non-file path raises `TetGenBackendError`.

Use a Windows executable whose architecture is compatible with the final
runtime. The active MATLAB path used the iso2mesh TetGen command semantics
`-A -q1.414a1`; the Python backend retains those options and does not retry
with weaker quality or volume constraints.

No TetGen binary has been copied or bundled by this migration. The TetGen
source license found in the MATLAB iso2mesh checkout describes an AGPLv3 /
commercial dual-license scheme, while the existing local executable reports
itself as an iso2mesh TetGen 1.4.2 build. Do not infer redistribution rights
from the executable's presence. Before packaging an executable, identify the
exact TetGen version and license, then satisfy all corresponding source-code,
notice, distribution, or commercial-license obligations.

Current validation boundary: the Python format adapter, executable interface,
and pipeline parameters have been migrated, but TetGen has not been executed
in this round. MATLAB-reference, real-case, MATLAB-free Windows, and license
validation remain external requirements.
