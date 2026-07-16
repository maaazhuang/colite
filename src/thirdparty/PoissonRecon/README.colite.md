# PoissonRecon vendoring note

This directory contains the PoissonRecon source bundled with COLMAP 3.12.6.
COLITE builds these sources directly as the `colite_poisson_recon` static
library; no system PoissonRecon package, download, or separate installation is
required.

Upstream project: PoissonRecon by Michael Kazhdan and Matthew Bolitho.

License: MIT. See `LICENSE` in this directory.

The integration wrapper lives in `src/colite/mvs/meshing.cpp`. COLMAP is the
behavioral reference for the command-line arguments passed to `PoissonRecon`
and `SurfaceTrimmer`.
