conninfpy.defaults
==================

TFNBS-family scoring defaults to E=0.4, H=3.0 and exact integration of
its piecewise support function (``n=None``). A positive integer ``n``
selects the historical right-endpoint grid from ``start_thres + dh`` to
``max(statistic)``, where ``dh = (max - start_thres) / n``. Active edges
satisfy ``weight >= threshold`` after the initial strict ``weight > start``
filter. Inputs are rounded to ten decimal places before integration.

Record E, H, n, the start threshold, statistic scale and software version to
reproduce an earlier run. Explicit ``n`` retains the old grid; exact scores
and p-values need not equal finite-grid results. ``bh_fdr_perm`` uses raw
edge statistics and does not use TFNBS integration settings. T and beta tails
are enhanced separately; an F statistic is a single omnibus map.

For F with TFNBS and no start supplied, the start is F's 0.90 quantile with
the numerator and residual degrees of freedom. With available-case data and
different tested residual degrees of freedom, provide one common start
explicitly. An explicitly supplied F start below any applicable quantile emits
a warning. Beta thresholds remain in coefficient units.

.. automodule:: conninfpy.defaults
   :members:
   :undoc-members:
   :show-inheritance:
