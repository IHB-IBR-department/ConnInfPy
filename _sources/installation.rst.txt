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

From a built artifact
----------------------

``python -m build`` produces ``dist/conninfpy-2.0-py3-none-any.whl`` and
``dist/conninfpy-2.0.tar.gz``. Either installs with its dependencies and the
bundled atlas data, and the checkout is no longer needed afterwards:

.. code-block:: bash

    python -m build
    python -m pip install dist/conninfpy-2.0-py3-none-any.whl   # or the .tar.gz

Optional extras
---------------

=================  ==============================================================
Extra              Adds
=================  ==============================================================
``[app]``          Streamlit interface (``streamlit``, ``nilearn``, ``nibabel``, ...)
``[full]``         ``networkx``
``[notebooks]``    Jupyter, for the tutorials in ``examples/notebooks/``
``[dev]``          ``pytest``, Sphinx and the other tools used to test and build the docs
=================  ==============================================================

For example ``python -m pip install -e ".[app,dev]"``. Contributor setup is
described in ``CONTRIBUTE.md``.

Requirements
------------

Core: ``numpy``, ``scipy``, ``statsmodels``, ``pandas``, ``matplotlib``,
``numba``, ``scikit-learn``, ``joblib``, ``threadpoolctl``, ``pyyaml``,
``nibabel>=3.2``.

.. note::
   **Without numba.** If ``numba`` / ``llvmlite`` cannot be installed on your
   platform, install without dependencies and add the rest by hand:

   .. code-block:: bash

       python -m pip install -e . --no-deps
       python -m pip install numpy scipy statsmodels pandas matplotlib scikit-learn joblib threadpoolctl pyyaml "nibabel>=3.2"

   Which code runs instead depends on the method and scoring path: exact TFNBS
   (``n=None``) executes its pure-Python graph and block sweeps, exact
   NI-TFNBS / FBC-TFNBS use vectorized NumPy, and the legacy threshold grid
   (an explicit ``n``) plus NBS use their SciPy implementations. The active
   fallback backend is logged once per call, e.g. ``Numba is not installed;
   TFNBS (exact) is running on the 'python' fallback backend.``
   The permutations, the correction and the
   significance level are not affected by this choice; the runtime of the
   uncompiled paths is not benchmarked here. ``conninfpy info`` reports whether
   ``numba`` was found.
   ``nibabel`` is not optional — the atlas and the interpretation step read
   their label images with it, so annotation fails without it even though the
   statistics still run.
