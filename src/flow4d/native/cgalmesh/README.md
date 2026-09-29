# CGALMesh runtime

This directory documents the external executable used to replace the active
MATLAB/iso2mesh surface path:

```text
run_iso2mesh.m
  -> v2s(..., 'cgalmesh')
  -> v2m / vol2mesh
  -> cgalv2m
  -> cgalmesh executable
```

No CGALMesh executable is included here. The Python caller must provide an
explicit absolute path:

```python
from flow4d.mesh import (
    CgalMeshSurfaceBackend,
    TetGenTetrahedralizationBackend,
    run_mesh_pipeline,
)

surface_backend = CgalMeshSurfaceBackend(
    r"D:\path\confirmed-cgalmesh-build\cgalmesh.exe"
)
tetra_backend = TetGenTetrahedralizationBackend(
    executable_path=r"D:\path\confirmed-tetgen-build\tetgen.exe"
)

result = run_mesh_pipeline(
    mask,
    velocity,
    voxel_size_mm,
    surface_backend,
    tetrahedralization_backend=tetra_backend,
    radius_bound=1.2,
)
```

## Command-line parameters

For one backend call, Python supplies these six numeric arguments after the
input and output paths:

```text
30.000000 <radius_bound> 0.500000 3.000000 1000.000000 1648335518
```

They correspond to `ang`, `ssize`, `approx`, `reratio`, `maxvol`, and the
iso2mesh seed `hex2dec('623F9A9E')`. `run_mesh_pipeline()` already supplies
`2 * radbound` as `radius_bound`; the backend does not multiply it again.

The shared interface also supplies `iso_value=0.5`. The original
`cgalmesh` branch casts the input volume to `uint8` and does not pass this
isovalue to the executable. The Python backend preserves that behavior; it
does not reinterpret the value as a marching-cubes threshold.

## Coordinate semantics

The returned nodes remain in the cropped and twice-refined local index space.
The backend applies the explicit `node = node + 0.5` operation from
`cgalv2m.m`. It does not restore crop offsets, a DICOM origin, direction
cosines, or patient coordinates. The existing pipeline performs the later
axis mapping and `0.5 * voxel_MR` scaling used by `run_iso2mesh.m`.

## Distribution boundary

Being able to run a local executable does not establish permission to copy or
redistribute that executable. Before building a standalone application,
record the exact binary origin, version, hash, CGAL dependencies, and all
applicable licenses. Development-machine execution is not evidence that the
binary may be included in a distributable package.

See `docs/cgalmesh_external_validation.md` before treating this backend as
numerically equivalent to the MATLAB path.
