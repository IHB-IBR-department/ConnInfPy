conninfpy
=========

Start with :func:`conninfpy.analyze`; reach advanced functions through
their module.

Main entry point and result types:

.. toctree::
   :maxdepth: 1

   conninfpy.analysis
   conninfpy.results

Atlases, anatomical interpretation and figures:

.. toctree::
   :maxdepth: 1

   conninfpy.atlas
   conninfpy.interpret
   conninfpy.plot

Advanced inference (custom designs, F-contrasts, null distributions):

.. toctree::
   :maxdepth: 1

   conninfpy.stats
   conninfpy.influence

Network enhancement:

.. toctree::
   :maxdepth: 1

   conninfpy.enhance

Multi-site harmonization:

.. toctree::
   :maxdepth: 1

   conninfpy.harmonize

Synthetic data and benchmark scenarios:

.. toctree::
   :maxdepth: 1

   conninfpy.sim

Utilities and constants:

.. toctree::
   :maxdepth: 1

   conninfpy.defaults
   conninfpy.preprocessing

Shared array helpers
--------------------

:func:`~conninfpy._shared.arrays.fisher_r_to_z` (also exported as
:func:`conninfpy.fisher_r_to_z`) and the connected-component labelling used by
the enhancement operators.

.. automodule:: conninfpy._shared.arrays
   :members:
   :show-inheritance:
