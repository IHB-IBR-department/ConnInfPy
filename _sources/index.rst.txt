.. ConnInfPy documentation master file

Welcome to ConnInfPy's documentation!
=====================================

ConnInfPy (Connectivity Inference in Python) is a Python package for statistical inference on brain connectivity
networks (fMRI). It provides permutation-based tests for group comparisons (two-sample, paired and
repeated-measures designs, with between-subject predictors of within-subject change) and continuous
predictors with confound regression (GLM with Freedman–Lane), together with a family of enhancement methods: classical NBS,
TFNBS (Threshold-Free Network-Based Statistics), cNBS, network-informed TFNBS, and functional-block clustering
TFNBS. The TFNBS family eliminates the need for an arbitrary statistical threshold by integrating cluster statistics
across a range of thresholds via TFCE-style enhancement. Detected connections can then be described
anatomically with the Brainnetome atlas (BNA-246): the regions each ROI's tissue covers, the hemispheric
relation, circuit class and distance of each connection, and BrainMap literature context of its endpoints.
This description adds no test and leaves the inference unchanged.


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   installation
   usage
   architecture
   approach
   workflows
   conninfpy
   references



To **cite this toolbox**, use the metadata in ``CITATION.cff`` at the root of
the repository (`github.com/IHB-IBR-department/ConnInfPy
<https://github.com/IHB-IBR-department/ConnInfPy>`_; ConnInfPy 2.0, MIT
licence). No DOI has been registered for the package yet and the software
paper is in preparation; this section will be updated once a DOI is assigned.
