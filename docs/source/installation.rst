Installation
============

ConnInfPy supports Python **3.12, 3.13 and 3.14** (enforced by the package
metadata, so ``pip`` refuses other versions instead of failing later inside
``numba``).

From a clone (until the first PyPI release)
-------------------------------------------

.. code-block:: bash

    git clone https://github.com/IHB-IBR-department/ConnInfPy.git
    cd ConnInfPy
    conda create -n conninfpy python=3.13 -y
    conda activate conninfpy
    python -m pip install -e .

After the release this becomes ``python -m pip install conninfpy``.

Optional extras
---------------

=================  ==============================================================
Extra              Adds
=================  ==============================================================
``[app]``          Streamlit interface (``streamlit``, ``nilearn``, ``nibabel``, ...)
``[decode]``       NiMARE meta-analytic decoding of significant edges
``[full]``         ``scikit-learn`` (needed by ``conninfpy.sim.generate_fc_matrices``), ``networkx``
``[notebooks]``    Jupyter, for the tutorials in ``examples/notebooks/``
``[dev]``          ``pytest``, Sphinx and the other tools used to test and build the docs
=================  ==============================================================

For example ``python -m pip install -e ".[app,dev]"``. Contributor setup is
described in ``CONTRIBUTE.md``.

Requirements
------------

Core: ``numpy``, ``scipy``, ``statsmodels``, ``pandas``, ``matplotlib``,
``numba``, ``threadpoolctl``, ``pyyaml``.

.. note::
   **Without numba.** If ``numba`` / ``llvmlite`` cannot be installed on your
   platform, install without dependencies and add the rest by hand:

   .. code-block:: bash

       python -m pip install -e . --no-deps
       python -m pip install numpy scipy statsmodels pandas matplotlib pyyaml threadpoolctl

   ConnInfPy detects the missing ``numba`` and falls back to the SciPy backend.
