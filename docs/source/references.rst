References
==========

The methods implemented in ConnInfPy draw on the following works. Grouped by
the pipeline stage they underpin (see :doc:`approach`).

Network-based statistics and enhancement
----------------------------------------

- Zalesky, A., Fornito, A., & Bullmore, E. T. (2010). Network-based statistic:
  identifying differences in brain networks. *NeuroImage*, 53(4), 1197–1207.
- Smith, S. M., & Nichols, T. E. (2009). Threshold-free cluster enhancement:
  addressing problems of smoothing, threshold dependence and localisation in
  cluster inference. *NeuroImage*, 44(1), 83–98.
- Baggio, H. C., Abos, A., Segura, B., et al. (2018). Statistical inference in
  brain graphs using threshold-free network-based statistics. *Human Brain
  Mapping*, 39(6), 2289–2302.
- Noble, S., & Scheinost, D. (2020). The constrained network-based statistic: a
  new level of inference for neuroimaging. *MICCAI 2020*, LNCS 12267.
- Hao, Z., Wang, P., Xia, X., Pan, Y., & Dou, W. (2024). Threshold-free
  network-oriented statistics in neuroscience. *bioRxiv*, 2024-01.
- Vinokur, L., Smith, R. E., Dhollander, T., Vaughan, D., Jackson, G. D., &
  Connelly, A. (2023). Parameter Sensitivity of Network-Based Statistical
  Inference. *Research Square* (preprint).
  doi:10.21203/rs.3.rs-3081615/v1. (Reports up to 75-fold variation in
  detected edge counts across (E, H).)
- Rubinov, M., & Sporns, O. (2010). Complex network measures of brain
  connectivity: uses and interpretations. *NeuroImage*, 52(3), 1059–1069.

Permutation inference and the GLM
---------------------------------

- Freedman, D., & Lane, D. (1983). A nonstochastic interpretation of reported
  significance levels. *Journal of Business & Economic Statistics*, 1(4),
  292–298.
- Anderson, M. J., & Legendre, P. (1999). An empirical comparison of
  permutation methods for tests of partial regression coefficients in a linear
  model. *Journal of Statistical Computation and Simulation*, 62(3), 271–303.
- Winkler, A. M., Ridgway, G. R., Webster, M. A., Smith, S. M., & Nichols, T. E.
  (2014). Permutation inference for the general linear model. *NeuroImage*, 92,
  381–397.
- Winkler, A. M., Ridgway, G. R., Douaud, G., Nichols, T. E., & Smith, S. M.
  (2016). Faster permutation inference in brain imaging. *NeuroImage*, 141,
  502–516. (GPD tail acceleration.)
- Phipson, B., & Smyth, G. K. (2010). Permutation P-values should never be
  zero: calculating exact P-values when permutations are randomly drawn.
  *Statistical Applications in Genetics and Molecular Biology*, 9(1).

Multi-site harmonization
------------------------

- Johnson, W. E., Li, C., & Rabinovic, A. (2007). Adjusting batch effects in
  microarray expression data using empirical Bayes methods. *Biostatistics*,
  8(1), 118–127. (ComBat.)
- Fortin, J.-P., et al. (2018). Harmonization of cortical thickness
  measurements across scanners and sites. *NeuroImage*, 167, 104–120.
- Nygaard, V., Rødland, E. A., & Hovig, E. (2016). Methods that remove batch
  effects while retaining group differences may lead to exaggerated confidence
  in downstream analyses. *Biostatistics*, 17(1), 29–39. (The label-leak
  motivation for excluding the tested variable in the ComBat arms.)

Atlases and anatomical interpretation
-------------------------------------

- Fan, L., Li, H., Zhuo, J., et al. (2016). The Human Brainnetome Atlas: a new
  brain atlas based on connectional architecture. *Cerebral Cortex*, 26(8),
  3508–3526. doi:10.1093/cercor/bhw157. (BNA-246: label images, hierarchy,
  Yeo crosswalk, probabilistic and tractography maps, BrainMap profiles.)
- Laird, A. R., Lancaster, J. L., & Fox, P. T. (2005). BrainMap: the social
  evolution of a human brain mapping database. *Neuroinformatics*, 3(1),
  65–78. doi:10.1385/NI:3:1:065.
- Laird, A. R., Eickhoff, S. B., Fox, P. M., et al. (2011). The BrainMap
  strategy for standardization, sharing, and meta-analysis of neuroimaging
  data. *BMC Research Notes*, 4, 349. doi:10.1186/1756-0500-4-349.
  (Behavioural-domain and paradigm-class taxonomy of the profiles.)
- Yeo, B. T. T., Krienen, F. M., Sepulcre, J., et al. (2011). The organization
  of the human cerebral cortex estimated by intrinsic functional connectivity.
  *Journal of Neurophysiology*, 106(3), 1125–1165. (Yeo-7 / Yeo-17 networks.)
- Schaefer, A., Kong, R., Gordon, E. M., et al. (2018). Local-global
  parcellation of the human cerebral cortex from intrinsic functional
  connectivity MRI. *Cerebral Cortex*, 28(9), 3095–3114. (Bundled
  Schaefer-100/200/400 with Yeo-7 labels.)
- Lancaster, J. L., Tordesillas-Gutiérrez, D., Martinez, M., et al. (2007).
  Bias between MNI and Talairach coordinates analyzed using the ICBM-152 brain
  template. *Human Brain Mapping*, 28(11), 1194–1205. (Talairach to MNI
  conversion before annotation.)
- Power, J. D., Barnes, K. A., Snyder, A. Z., Schlaggar, B. L., & Petersen,
  S. E. (2012). Spurious but systematic correlations in functional
  connectivity MRI networks arise from subject motion. *NeuroImage*, 59(3),
  2142–2154. (The distance check of the annotation summaries.)
