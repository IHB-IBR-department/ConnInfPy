For developers information.


## Installation

```bash
# Until the PyPI release is out, install from a source checkout instead:
git clone https://github.com/IHB-IBR-department/ConnInfPy.git
cd ConnInfPy

# Create the conda env (Python 3.13)
conda create -n conninfpy python=3.13 -y
conda activate conninfpy
python -m pip install -r requirements/dev.txt
```

To work on the decoding functionality, also install the `decode` extra:

```bash
python -m pip install -e .[decode]
```

---

## Running the tests

The test suite runs with the development dependencies installed via
`requirements/dev.txt` (pytest is included). The recommended runner is pytest —
it shows a compact progress bar and collects a few function-style tests that
`unittest` misses:

```bash
python -m pytest tests              # full suite
python -m pytest tests -q --disable-warnings   # one-line output
python -m pytest tests/test_glm_stats.py -k fstat   # single file / pattern
python -m pytest --doctest-modules conninfpy   # docstring examples
```

> **Note:** `tests/test_interpret.py::test_load_dotenv_manually` expects a
> local `.env` file with an `OPENROUTER_API_KEY` (gitignored). On a machine
> without it, that one test fails under pytest — everything else should be
> green. (unittest also skips a test needing
> `datasets/eeg_dataframe_nansfilled.csv`, which is likewise local-only.)

If pytest is unavailable (e.g. a bare environment), the suite also runs with
the standard library:

```bash
python -m unittest discover -s tests -t .
python -m unittest tests.test_glm_stats.TestFStatCompute.test_fstat_single_row_equals_tstat_squared
```

## Code style

**KISS first**: prefer the boring, obvious solution; extract a helper when
the same three lines appear three times; delete code instead of commenting
around it. SOLID where it's free (single responsibility per module, no
hidden surprises), but never at the cost of simplicity.

**Docstrings: NumPy style, everywhere** (`conninfpy` and `apps`). Keep them
short and factual — parameters, returns, raises, and an `Examples` doctest
when the function is nontrivial:

```python
def fisher_r_to_z(r):
    """Fisher r-to-z transform of a correlation matrix.

    Parameters
    ----------
    r : ndarray of shape (..., N, N)
        Correlation coefficients in [-1, 1].

    Returns
    -------
    ndarray
        Fisher-z values; |z| capped at 5 where r = ±1.

    Examples
    --------
    >>> fisher_r_to_z(np.array([[0.0, 0.5], [0.5, 0.0]]))[0, 1].round(3)
    0.549
    """
```

Comments are for the *why*, not the *what* — if code needs a comment to
say what it does, rewrite the code. Doctests run in CI-reachable form
(`pytest --doctest-modules conninfpy`) and must stay green; every doctest
is self-contained (sets up its own data).

## Type checking

The repo ships a soft-strict mypy config in `pyproject.toml`:

```bash
python -m mypy conninfpy
```

There is a known annotation backlog (~108 errors, mostly missing type
hints); don't introduce new ones in code you touch.

## Building the docs

```bash
cd docs
sphinx-build source _build
# Docs auto-build on push to main via GitHub Actions → gh-pages.
```

## Releasing to PyPI

The release flow is fully automated by `.github/workflows/publish.yml`
(PyPI Trusted Publishing — no API tokens or secrets in CI). A tag push
builds sdist + wheel, runs the test suite against it, twine-checks the
artefacts, and publishes:

- tag `v2.0.1-rc1` (anything with `-rcN`) → **TestPyPI**, automatic
- tag `v2.0.1` (bare `vX.Y.Z`) → **real PyPI**, requires manual approval
  of the `pypi` environment in GitHub repo → Settings → Environments

To cut a release:

```bash
# 1. Bump the version in pyproject.toml ([project] version = "...")
#    AND in CITATION.cff (version + date-released — otherwise GitHub's
#    "Cite this repository" button serves stale info)
# 2. Make sure the working tree is clean and tests pass:
python -m pytest tests --disable-warnings
# 3. Tag and push:
git tag v2.0.1
git push origin v2.0.1
# 4. Watch the "Publish to PyPI" Actions run; approve the `pypi`
#    environment when it pauses for review.
```

For a dry run, tag `-rcN` first, check the package at
`https://test.pypi.org/project/conninfpy/`, then tag the bare version.

> **One-time setup (already done if publishing works):** the PyPI and
> TestPyPI projects must each have a Trusted Publisher configured for this
> repo's `publish-pypi` / `publish-testpypi` workflows.
