# COLITE vs COLMAP 4.0 MVS A/B

Date: 2026-07-14

The comparison uses Release CPU builds and the deterministic synthetic fixture
created by `scripts/create_synthetic_mvs_fixture.py`. Both implementations
receive the same oriented points, visibility file, mesh, calibrated images,
and sparse model. Timings include process startup and are intended as smoke
measurements, not statistically stable benchmarks.

| Stage | COLITE output | COLMAP 4.0 output | COLITE time | COLMAP time |
|---|---:|---:|---:|---:|
| Poisson | 3990 vertices, 7976 faces | 3990 vertices, 7976 faces | 3.927 s | 8.738 s |
| Delaunay | 258 vertices, 784 faces | 257 vertices, 692 faces | 0.307 s | 0.701 s |
| Simplifier | 130 vertices, 256 faces | 130 vertices, 256 faces | 0.162 s | 0.724 s |
| Texturer | 512 x 219 atlas | 512 x 217 atlas | 0.296 s | 1.073 s |

Poisson produces identical vertex and face counts because COLITE now embeds
the same PoissonRecon generation used by the COLMAP 4.0 reference. File sizes
differ because COLITE deliberately writes ASCII PLY while COLMAP writes binary
PLY.

Delaunay is not expected to produce identical connectivity: COLITE uses its
own ray-casting and graph-cut implementation. On this fixture it produces 92
more faces (13.3%) and one more referenced vertex. A real-scene comparison
should add sampled Chamfer, normal consistency, and topology metrics.

Both simplifiers hit the exact 50% face target and produce the same compact
vertex/face counts. Their individual collapse order and boundary policies can
still yield different geometry.

The initial Texturer A/B exposed a 4096 x 4096 COLITE atlas versus COLMAP's
512 x 217 atlas. COLITE now crops unused atlas rows; with a common 512 starting
width the result is 512 x 219, only two rows taller than COLMAP. Image bytes
are also close (29724 vs 29396). COLITE writes OBJ/MTL/PNG while COLMAP writes
textured PLY/PNG.

The machine-readable final report is generated at:

```text
build-wsl-release/compare-mvs-final/comparison.json
```

Reproduce:

```bash
python3 scripts/create_synthetic_mvs_fixture.py \
  --output build-wsl-release/compare-mvs-fixture

python3 scripts/compare_colmap4_mvs.py \
  --colite build-wsl-release/src/colite/exe/colite \
  --colmap build-wsl-colmap4-reference/install/bin/colmap \
  --workspace build-wsl-release/compare-mvs-fixture \
  --mesh build-wsl-release/compare-mvs-fixture/mesh.ply \
  --output build-wsl-release/compare-mvs-final
```
