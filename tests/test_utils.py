"""Tests for conninfpy.utils.

Covers:
- fisher_r_to_z: transform of typical correlations, the ±5 cap at r = ±1,
  and the ValueError for inputs outside [-1, 1].
- get_components: component sizes, partition, and isolated-node handling
  on a fixed 6-node graph (3 + 2 + 1 nodes).
- binarize: nonzero→1 / zero→0 conversion and the copy/in-place modes.
- create_prior_weights: intra-network boosting, symmetry, diagonal,
  and target_network_id selection.
"""
from unittest import TestCase
import warnings

import numpy as np

from conninfpy.utils import (
    binarize,
    create_prior_weights,
    fisher_r_to_z,
    get_components,
)


class TestFisherRToZ(TestCase):
    """fisher_r_to_z across its input space.

    r inside (-1, 1)  → finite z, no warnings.
    r exactly ±1      → one UserWarning, value capped at ±5.
    r outside range   → ValueError, no math runs.
    """

    def test_typical_values_no_warning(self):
        """Check: interior and near-boundary r transform silently.

        Input:    [[0, 0.5], [-0.5, 0.9999]]
                  (0.9999 is outside the np.isclose cap zone).
        Expected: no warning of any kind; all z finite.
        Failure:  warning spam on clean data, or a too-wide cap zone
                  silently replacing 0.9999's honest z = 4.95 with 5.0.
        """
        r = np.array([[0.0, 0.5], [-0.5, 0.9999]])
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning fails the test
            z = fisher_r_to_z(r)
        self.assertTrue(np.all(np.isfinite(z)))

    def test_perfect_correlation_warns_and_caps(self):
        """Check: r = ±1 warns once and caps at exactly ±5.

        Input:    [[0, 1], [-1, 0]] — both boundary signs present.
        Expected: UserWarning matching "±1"; z[0,1] == 5.0 and
                  z[1,0] == -5.0 (exact: the cap is an assignment,
                  not a computation).
        Failure:  silent capping, wrong cap value, or a changed warning
                  message that downstream code may be matching on.
        """
        r = np.array([[0.0, 1.0], [-1.0, 0.0]])
        with self.assertWarnsRegex(UserWarning, r"±1"):
            z = fisher_r_to_z(r)
        self.assertEqual(z[0, 1], 5.0)
        self.assertEqual(z[1, 0], -5.0)

    def test_out_of_range_raises(self):
        """Check: r outside [-1, 1] fails fast.

        Input:    [1.5] — unambiguously illegal.
        Expected: ValueError, raised before any math.
        Failure:  the guard is gone; arctanh(1.5) would produce nan
                  with only a numpy warning, surfacing far downstream.
        """
        with self.assertRaises(ValueError):
            fisher_r_to_z(np.array([1.5]))


class TestGetComponents(TestCase):
    """get_components on a fixed 6-node graph.

    Layout:  0--1--2   3--4   5(isolated)
    Answer by hand: three components, sizes {3, 2, 1}.
    Guaranteed: sizes and partition. NOT guaranteed: numbering order.
    """

    def setUp(self):
        adj = np.zeros((6, 6))
        for i, j in [(0, 1), (1, 2), (3, 4)]:
            adj[i, j] = adj[j, i] = 1.0
        self.adj = adj

    def test_sizes(self):
        """Check: component sizes are {1, 2, 3}.

        Input:    the 6-node graph from setUp.
        Expected: sorted(sizes) == [1, 2, 3].
        Failure:  a component got split or merged. Compared sorted
                  because numbering order is not guaranteed.
        """
        _, sizes = get_components(self.adj)
        self.assertEqual(sorted(sizes), [1, 2, 3])

    def test_partition(self):
        """Check: connected nodes share a label, separated nodes differ.

        Input:    the 6-node graph from setUp.
        Expected: comps[0]==comps[1]==comps[2]; comps[3]==comps[4];
                  the two groups differ; node 5 differs from both.
        Failure:  any split or merge error — each breaks at least one
                  of these five relations.
        """
        comps, _ = get_components(self.adj)
        self.assertEqual(comps[0], comps[1])
        self.assertEqual(comps[1], comps[2])
        self.assertEqual(comps[3], comps[4])
        self.assertNotEqual(comps[0], comps[3])
        self.assertNotEqual(comps[0], comps[5])

    def test_isolated_node_is_own_component(self):
        """Check: an edgeless node is a size-1 component, not dropped.

        Input:    the 6-node graph; node 5 has no edges.
        Expected: 1 appears in the sizes.
        Failure:  isolated nodes were skipped — downstream NBS-family
                  code counts components from this output.
        """
        _, sizes = get_components(self.adj)
        self.assertIn(1, sizes)


class TestBinarize(TestCase):
    """binarize: nonzero→1, zero→0, and the copy=True/False behavior.

    copy=True  → input array untouched.
    copy=False → input array modified in place.
    """

    def test_binarize(self):
        """Check: arbitrary magnitudes collapse to a 0/1 mask.

        Input:    [[0, 2.5, 0], [1.1, 0, 0.3]] — mixed magnitudes,
                  all positive (sign is irrelevant: what's thresholded
                  is the presence of an edge).
        Expected: [[0, 1, 0], [1, 0, 1]], same shape.
        Failure:  wrong threshold semantics — e.g. sign-sensitive or
                  magnitude-preserving output.
        """
        w = np.array([[0.0, 2.5, 0.0],
                      [1.1, 0.0, 0.3]])
        self.assertTrue(np.array_equal(
            binarize(w),
            np.array([[0, 1, 0],
                      [1, 0, 1]]),
        ))

    def test_copy_vs_in_place(self):
        """Check: copy=True preserves the input; copy=False mutates it.

        Input:    [[0, 2.5]], binarized twice — first with copy=True,
                  then with copy=False.
        Expected: after copy=True the input still holds 2.5; after
                  copy=False it holds 1.0.
        Failure:  the modes are swapped — anyone reusing an input after
                  binarize(copy=False) would get silent corruption.
        """
        w = np.array([[0.0, 2.5]])
        binarize(w, copy=True)
        self.assertEqual(w[0, 1], 2.5)   # original untouched
        binarize(w, copy=False)
        self.assertEqual(w[0, 1], 1.0)   # modified in place


class TestCreatePriorWeights(TestCase):
    """create_prior_weights: (N, N) weights from network labels.

    Same-label pairs      → boost_factor.
    Different-label pairs → 1.0 (background).
    Diagonal              → 1.0 always (self-connection, not an edge).
    target_network_id     → boost only pairs inside that one network.
    """

    def test_basic_boosting(self):
        """Check: all same-label pairs boosted, symmetric, diagonal kept.

        Input:    labels [1, 1, 2, 2, 3], boost_factor=3.0.
        Expected: w[0,1]==w[1,0]==3.0 (same label); w[2,3]==3.0;
                  w[0,2]==w[0,4]==1.0 (different labels);
                  diagonal all 1.0.
        Failure:  asymmetry, diagonal boosting, or cross-network leaks.
        """
        labels = np.array([1, 1, 2, 2, 3])
        w = create_prior_weights(labels, boost_factor=3.0)

        self.assertEqual(w.shape, (5, 5))
        self.assertTrue(np.allclose(np.diag(w), 1.0))
        self.assertEqual(w[0, 1], 3.0)
        self.assertEqual(w[1, 0], 3.0)
        self.assertEqual(w[2, 3], 3.0)
        self.assertEqual(w[0, 2], 1.0)
        self.assertEqual(w[0, 4], 1.0)

    def test_target_network_only(self):
        """Check: target_network_id boosts only that network's pairs.

        Input:    labels [1, 1, 2, 2, 3], target_network_id=2,
                  boost_factor=4.0.
        Expected: w[2,3]==4.0 (both nodes in network 2);
                  w[0,1]==1.0 (same label, wrong network);
                  w[0,2]==1.0 (cross-network).
        Failure:  the target selector was ignored and every
                  intra-network pair got boosted.
        """
        labels = np.array([1, 1, 2, 2, 3])
        w = create_prior_weights(labels, target_network_id=2, boost_factor=4.0)

        self.assertEqual(w[2, 3], 4.0)
        self.assertEqual(w[0, 1], 1.0)
        self.assertEqual(w[0, 2], 1.0)
