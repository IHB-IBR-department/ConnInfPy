Package organization
====================

``conninfpy.analyze`` is the main entry point. Its orchestration stays in
``analysis.py``; result containers stay in ``results.py``. Implementations are
grouped by responsibility:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Package
     - Responsibility
   * - ``stats``
     - Pairwise and GLM inference, design/estimability, missing-data paths,
       permutation execution, correction and accelerated tails.
   * - ``enhance``
     - Pure enhancement scores and the public ``apply_*`` wrappers.
       ``tfnbs.py``, ``ni_tfnbs.py`` and ``fbc_tfnbs.py`` hold the three TFNBS
       variants; ``nbs.py`` holds NBS/cNBS and the legacy BCT wrapper.
   * - ``atlas``
     - ``metadata.py`` holds ``AtlasInfo``; ``brainnetome.py`` holds
       ``BrainnetomeReference``; ``resources.py`` locates and reads reference data.
   * - ``interpret``
     - ROI/edge annotation, descriptive anatomical/network summaries, evidence
       packets and interpretation reports.
   * - ``harmonize``
     - ``combat.py`` implements ComBat; ``diagnostics.py`` contains design
       checks; ``_arrays.py`` converts between matrices and edge features.
   * - ``sim``
     - Synthetic datasets and topology scenarios.
   * - ``plot``
     - Matrix, network and connectogram rendering; ``summary.py`` composes panels.
   * - ``loaders``
     - Reusable input adapters, manifest parsing and input validation.

Shared array/scale and random-state helpers live in ``_shared``. Result types,
tail containers, numerical defaults, input preparation and exports keep their
existing locations. Inference consumes enhancement scores; interpretation and
plotting consume results. Study-specific protocols belong outside the library.

Atlas data and interpretation
-----------------------------

Small versioned resources remain in ``conninfpy/atlas_data/``. Atlas
constructors, manifest ``BUILTIN:`` tokens and Brainnetome annotation use
``atlas.resources`` for access. Materialized package-resource paths remain
valid for the process lifetime, including lazy NIfTI image reads.

Optional originals use ``atlas.resources.brainnetome_data_dir()``. Resolution
is explicit ``data_dir``, then the direct legacy ``CONNINFPY_BNA_DIR`` override,
then ``CONNINFPY_DATA/brainnetome`` (default ``~/conninfpy_data/brainnetome``).
The caller's working directory is not searched. PM/SC maps occupy this directory;
hierarchy-build tables occupy its ``sources/`` subdirectory. The bundled
``brainnetome_external.json`` records the reference originals' locally computed
SHA256 values, not provider-published checksums. Different source versions
are rejected when used, rather than silently changing an interpretation.
``brainnetome_resource_path(..., required=True)`` explains manual acquisition
when a file is missing. No implicit atlas download occurs.

Derived arrays use ``cache_dir`` / ``CONNINFPY_CACHE`` separately, defaulting to
``~/.cache/conninfpy/bna246``. Ordinary BrainMap profile loading uses bundled
snapshots; the explicit profile-refresh API retains its download/cache behavior.
ENIGMA Schaefer SC fixtures belong to ``examples/simulations/sc_prior/data/``.
``datasets/`` contains study inputs and manifests, not package atlas copies.

``scripts/build_bna_hierarchy.py`` reads configured external sources by default
and supports ``--source-dir`` and ``--output`` for explicit rebuilds.

``interpret`` annotates the ROIs/edges of the supplied result. Anatomical
summaries do not change its parcellation, p-values or inference unit.

Compatibility
-------------

Existing top-level functions and public module paths remain importable.
The former implementation modules, such as ``pairwise_stats``, ``glm_stats``,
``tfnbs_score``, ``brainnetome`` and ``synth_datasets``, are small compatibility
imports sharing their canonical module objects. Old saved class paths resolve
through these aliases and the domain-package facades. New code should use the
domain packages; edit the canonical implementations rather than the aliases.

Private helpers extracted into shared modules should be patched at their
canonical call site in development tests.

Process and thread execution
----------------------------

``stats._workers`` owns the shared permutation pool for pairwise, GLM,
available-case and legacy NBS inference. By default ``use_mp=True`` uses
forkserver where available, otherwise spawn. The default pool size is the
usable logical CPU count (including affinity/container quota and
``LOKY_MAX_CPU_COUNT``), capped at the number of permutations. One worker
or one permutation runs directly; calls from child processes run serially.
``n_processes`` must be a positive integer when multiprocessing is requested.

Each worker loads its task and associated data once. The work queue then
contains only permutation seeds, with the same seed order and randomization
as serial execution. Data are copied per process, so reduce ``n_processes``
when memory is limited. Progress reporting uses the same chunking as silent
execution and can advance in bursts as chunks finish.

Workers limit loaded BLAS/OpenMP libraries to one thread through
``threadpoolctl``. TFNBS Numba kernels use serial ``njit``; the process pool
supplies their parallelism. Parent thread settings are preserved. Pool startup
is protected when analyses are launched from concurrent background threads;
this does not impose a global CPU budget across independent analyses.

For small graphs or inexpensive statistics, process startup can cost more
than the permutations. Use ``use_mp=False`` for such runs, or explicitly set
``n_processes`` to control CPU and memory use. For CLI runs these settings
belong in the YAML ``inference`` section; ``--no-mp`` also selects serial
execution. Statistical defaults (permutation count, thresholds and enhancement
parameters) are independent of these execution settings.

Package tasks can run from an unguarded script. Custom functions/classes
defined in a caller's script still require Python's usual
``if __name__ == "__main__":`` guard for spawn/forkserver execution.
