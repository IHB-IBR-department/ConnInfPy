# Study data

This directory holds study inputs and manifests, including ABIDE and Open-Close.
Local participant data remain subject to their existing acquisition and ignore rules.

Atlas resources have separate owners:

| Resource | Location |
|---|---|
| Offline atlas label images, lookup/hierarchy tables and BrainMap profiles | Installed `conninfpy/atlas_data/`, accessed through `conninfpy.atlas.resources` |
| Optional Brainnetome probabilistic/tractography originals | `~/conninfpy_data/brainnetome/`, or a configured data directory |
| Original Brainnetome hierarchy-building tables | The external directory's `sources/` subfolder |
| ENIGMA Schaefer structural-connectivity simulation fixtures | [examples/simulations/sc_prior/data/](../examples/simulations/sc_prior/data/) |
| Derived Brainnetome arrays | `~/.cache/conninfpy/bna246/`, or the configured cache |

Set `CONNINFPY_DATA` to a data root; optional maps are resolved in its `brainnetome/`
subfolder. `CONNINFPY_BNA_DIR` remains a direct override; explicit `data_dir=` takes
precedence. The retired `datasets/atlases/` location is not searched implicitly.
See the [resource documentation](../docs/source/architecture.rst).
