# Third-party dependencies

`PoissonRecon` is a Colite source dependency managed directly by this Git
repository. It is a source snapshot rather than a Git submodule.

| Directory | Upstream snapshot | License | Use |
|---|---|---|---|
| `PoissonRecon` | Source bundled from the COLMAP reference integration | MIT; see `PoissonRecon/LICENSE` | Compiled into Colite as `colite_poisson_recon` |

The production VLFeat-style SIFT backend contains an optimized adaptation of
VLFeat routines, but the full VLFeat source tree is not vendored. Its BSD
notice is retained in the source file and in the repository-level
`THIRD_PARTY_NOTICES.md`. A local `references/vlfeat-0.9.21` tree is used only
for optional reference-consistency tests.

`vcglib/` is an optional local dependency used only to build the isolated
OpenMVS reference environment. Colite does not include or link against it, so
the directory is excluded by `.gitignore` and must not be staged in this
repository. The currently prepared local copy came from vcglib commit
`5cae2abc2f9056785b0537dcbe156c40da2aea20` and is licensed under GPL-3.0.

When updating `PoissonRecon`, record the upstream version, retain its license
file, remove nested `.git` metadata, and run the relevant build and tests before
committing.
