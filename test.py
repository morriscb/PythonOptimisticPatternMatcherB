
from __future__ import division, print_function, absolute_import

from copy import copy
import unittest

import numpy as np

from _optimistic_pattern_matcher_b import OptimisticPatternMatcherB

__deg_to_rad__ = np.pi/180


class TestPythonOptimisticPatternMatcherB(unittest.TestCase):

    """Unittest suite for the python implimentation of
    Optimistic Pattern Matcher B.
    """

    def setUp(self):
        np.random.seed(12345)

        n_points = 1000
        self.reference_catalog = np.random.rand(n_points, 3)*4096
        self.reference_catalog[:, 2] = (
            np.random.power(1.2, size=n_points)*4 + 20)
        self.source_catalog = copy(self.reference_catalog)

    def testInit(self):
        self.pyOPMb = OptimisticPatternMatcherB(
            reference_catalog=self.reference_catalog, max_shift=1000.0,
            max_rotation=10.0, dist_tol=75, ang_tol=0.75, max_match_dist=3,
            min_matches=30, max_n_patterns=250)

    def testConstructAndMatchPattern(self):
        self.pyOPMb = OptimisticPatternMatcherB(
            reference_catalog=self.reference_catalog, max_shift=1000.0,
            max_rotation=10.0, dist_tol=75, ang_tol=0.75, max_match_dist=3,
            min_matches=30, max_n_patterns=250)

        pattern_list, cos_theta, sin_theta = self.pyOPMb._construct_and_match_pattern(
            self.source_catalog[:5, :2], 5)
        self.assertGreater(len(pattern_list), 0)
        self.assertIsNotNone(cos_theta)
        self.assertIsNotNone(sin_theta)

        pattern_list, cos_theta, sin_theta = self.pyOPMb._construct_and_match_pattern(
            self.source_catalog[:8, :2], 5)
        self.assertGreater(len(pattern_list), 0)
        self.assertIsNotNone(cos_theta)
        self.assertIsNotNone(sin_theta)

    def testMatchPerfect(self):
        self.pyOPMb = OptimisticPatternMatcherB(
            reference_catalog=self.reference_catalog, max_shift=1000.0,
            max_rotation=10.0, dist_tol=75, ang_tol=0.75, max_match_dist=3,
            min_matches=30, max_n_patterns=250)

        matches, distances = self.pyOPMb.match(self.source_catalog, 8, 5)
        self.assertEqual(len(matches), len(self.reference_catalog))
        self.assertTrue(np.all(distances < 1e-8))

    def testMatchMoreSources(self):
        self.pyOPMb = OptimisticPatternMatcherB(
            reference_catalog=self.reference_catalog[:500], max_shift=1000.0,
            max_rotation=10.0, dist_tol=75, ang_tol=0.75, max_match_dist=3,
            min_matches=30, max_n_patterns=250)

        matches, distances = self.pyOPMb.match(self.source_catalog, 8, 5)
        self.assertEqual(len(matches), len(self.reference_catalog[:500]))
        self.assertTrue(np.all(distances < 1e-8))

    def testMatchMoreReferences(self):
        self.pyOPMb = OptimisticPatternMatcherB(
            reference_catalog=self.reference_catalog, max_shift=1000.0,
            max_rotation=10.0, dist_tol=75, ang_tol=0.75, max_match_dist=3,
            min_matches=30, max_n_patterns=250)

        matches, distances = self.pyOPMb.match(self.source_catalog[:500], 8, 5)
        self.assertEqual(len(matches), len(self.reference_catalog[:500]))
        self.assertTrue(np.all(distances < 1e-8))

    def testMatchShiftRotated(self):
        self.pyOPMb = OptimisticPatternMatcherB(
            reference_catalog=self.reference_catalog, max_shift=1000.0,
            max_rotation=10.0, dist_tol=75, ang_tol=0.75, max_match_dist=3,
            min_matches=30, max_n_patterns=250)

        # Shift the source catalog by a given amount.
        shift_xy = np.random.uniform(-500, 500, size=2)
        # Pick a random rotation center.
        center = np.random.uniform(1, 4096, size=2)
        # Pick a random rotation.
        rot = np.random.uniform(-6, 6)
        print("\tShift %.2f %.2f" % (shift_xy[0], shift_xy[1]))
        print("\tCenter %.2f %.2f, Rotation %.4f" %
              (center[0], center[1], rot))

        tmp_source = np.empty_like(self.source_catalog)
        tmp_source[:, 2] = self.source_catalog[:, 2]
        tmp_source[:, 0] = (
            (self.source_catalog[:, 0] -
             center[0])*np.cos(rot*__deg_to_rad__) -
            (self.source_catalog[:, 1] -
             center[1])*np.sin(rot*__deg_to_rad__) + shift_xy[0])
        tmp_source[:, 1] = (
            (self.source_catalog[:, 0] -
             center[0])*np.sin(rot*__deg_to_rad__) +
            (self.source_catalog[:, 1] -
             center[1])*np.cos(rot*__deg_to_rad__) + shift_xy[1])
        tmp_source[:, 0] += center[0]
        tmp_source[:, 1] += center[1]

        matches, distances = self.pyOPMb.match(tmp_source, 8, 5)
        self.assertEqual(len(matches), len(self.reference_catalog))
        self.assertTrue(np.all(distances < 1e-8))

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
