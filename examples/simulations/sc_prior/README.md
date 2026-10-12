# Structural-prior simulation inputs

`data/` contains the existing ENIGMA Toolbox HCP-derived group SC matrices for
Schaefer-100/200/400, their ordered labels and extraction sidecars. These are
simulation fixtures, separate from the optional Brainnetome tractography maps.

The matrices, labels and original JSON sidecars were relocated byte-for-byte.
Original sidecars retain the paths recorded at extraction. `data/locations.json`
provides current paths and SHA256 hashes without rewriting that provenance.

`extract_enigma_sc.py` regenerates fixtures with the optional ENIGMA Toolbox
installation and writes future sidecars at the current location. `phase0_feasibility.py`
reads the Schaefer-100 fixture and checks matrix properties and the existing
simulation setup. Neither script is invoked by ordinary atlas loading.

From the repository root:

```bash
python -m examples.simulations.sc_prior.phase0_feasibility
```

Optional Brainnetome PM/SC originals live in the configured external directory;
see [atlas resource ownership](../../../../docs/source/architecture.rst).
