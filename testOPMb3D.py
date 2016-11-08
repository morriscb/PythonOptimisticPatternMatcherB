
from __future__ import division, print_function, absolute_import

from copy import copy
import unittest

import numpy as np

from optimistic_pattern_matcher_b_3D import OptimisticPatternMatcherB

__deg_to_rad__ = np.pi/180


class TestPythonOptimisticPatternMatcherB(unittest.TestCase):

    """Unittest suite for the python implimentation of
    Optimistic Pattern Matcher B.
    """

    def setUp(self):
        np.random.seed(12345)

        n_points = 1000
        self.reference_catalog = np.empty((n_points, 4))
        cos_theta_array = np.random.uniform(
            np.cos(np.pi/2 + 0.5*__deg_to_rad__),
            np.cos(np.pi/2 - 0.5*__deg_to_rad__), size=n_points)
        sin_theta_array = np.sqrt(1 - cos_theta_array**2)
        phi_array = np.random.uniform(-0.5, 0.5, size=n_points)*__deg_to_rad__
        self.reference_catalog[:, 0] = sin_theta_array*np.cos(phi_array)
        self.reference_catalog[:, 1] = sin_theta_array*np.sin(phi_array)
        self.reference_catalog[:, 2] = cos_theta_array
        self.reference_catalog[:, 3] = (
            np.random.power(1.2, size=n_points)*4 + 20)
        self.source_catalog = copy(self.reference_catalog)

    def testInit(self):
        self.pyOPMb = OptimisticPatternMatcherB(
            reference_catalog=self.reference_catalog, max_rotation_theta=0.5,
            max_rotation_phi=45.0, dist_tol=15/3600., max_dist_cand=1000,
            ang_tol=0.5, max_match_dist=15/3600, min_matches=30,
            max_n_patterns=50)

    def testConstructAndMatchPattern(self):
        self.pyOPMb = OptimisticPatternMatcherB(
            reference_catalog=self.reference_catalog, max_rotation_theta=0.5,
            max_rotation_phi=45.0,  dist_tol=15/3600., max_dist_cand=1000,
            ang_tol=0.5, max_match_dist=15/3600, min_matches=30,
            max_n_patterns=50)

        pattern_list = self.pyOPMb._construct_and_match_pattern(
            self.source_catalog[:6, :3], 6)
        self.assertGreater(len(pattern_list), 0)
        self.assertEqual(pattern_list[0], 0)
        self.assertEqual(pattern_list[1], 1)
        self.assertEqual(pattern_list[2], 2)
        self.assertEqual(pattern_list[3], 3)
        self.assertEqual(pattern_list[4], 4)
        self.assertEqual(pattern_list[5], 5)

        pattern_list = self.pyOPMb._construct_and_match_pattern(
            self.source_catalog[:9, :3], 6)
        self.assertGreater(len(pattern_list), 0)
        self.assertEqual(pattern_list[0], 0)
        self.assertEqual(pattern_list[1], 1)
        self.assertEqual(pattern_list[2], 2)
        self.assertEqual(pattern_list[3], 3)
        self.assertEqual(pattern_list[4], 4)
        self.assertEqual(pattern_list[5], 5)

        pattern_list = self.pyOPMb._construct_and_match_pattern(
            self.source_catalog[[2, 4, 8, 16, 32, 64], :3], 6)
        self.assertGreater(len(pattern_list), 0)
        self.assertEqual(pattern_list[0], 2)
        self.assertEqual(pattern_list[1], 4)
        self.assertEqual(pattern_list[2], 8)
        self.assertEqual(pattern_list[3], 16)
        self.assertEqual(pattern_list[4], 32)
        self.assertEqual(pattern_list[5], 64)

    def testMatchPerfect(self):
        self.pyOPMb = OptimisticPatternMatcherB(
            reference_catalog=self.reference_catalog, max_rotation_theta=0.5,
            max_rotation_phi=45.0,  dist_tol=15/3600., max_dist_cand=1000,
            ang_tol=0.5, max_match_dist=15/3600, min_matches=30,
            max_n_patterns=50)

        matches, distances = self.pyOPMb.match(self.source_catalog, 9, 6)
        self.assertEqual(len(matches), len(self.reference_catalog))
        self.assertTrue(np.all(distances < 0.01/3600.0*__deg_to_rad__))

    def testMatchMoreSources(self):
        self.pyOPMb = OptimisticPatternMatcherB(
            reference_catalog=self.reference_catalog[:500],
            max_rotation_theta=0.5, max_rotation_phi=45.0,  dist_tol=15/3600.,
            max_dist_cand=1000, ang_tol=0.5, max_match_dist=15/3600,
            min_matches=30, max_n_patterns=50)

        matches, distances = self.pyOPMb.match(self.source_catalog, 9, 6)
        self.assertEqual(len(matches), len(self.reference_catalog[:500]))
        self.assertTrue(np.all(distances < 0.01/3600.0*__deg_to_rad__))

    def testMatchMoreReferences(self):
        self.pyOPMb = OptimisticPatternMatcherB(
            reference_catalog=self.reference_catalog, max_rotation_theta=0.5,
            max_rotation_phi=45.0,  dist_tol=15/3600., max_dist_cand=1000,
            ang_tol=0.5, max_match_dist=15/3600, min_matches=30,
            max_n_patterns=50)

        matches, distances = self.pyOPMb.match(self.source_catalog[:500], 9, 6)
        self.assertEqual(len(matches), len(self.reference_catalog[:500]))
        self.assertTrue(np.all(distances < 0.01/3600.0*__deg_to_rad__))

    def testPhiRotation(self):
        self.pyOPMb = OptimisticPatternMatcherB(
            reference_catalog=self.reference_catalog, max_rotation_theta=0.5,
            max_rotation_phi=45.0,  dist_tol=15/3600., max_dist_cand=1000,
            ang_tol=0.5, max_match_dist=15/3600, min_matches=30,
            max_n_patterns=50)
        phi = 10.0*__deg_to_rad__
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        phi_rotation = np.array(
            [[1., 0., 0.],
             [0., cos_phi, -sin_phi],
             [0., sin_phi,  cos_phi]])

        self.source_catalog[:, :3] = np.dot(
            phi_rotation, self.source_catalog[:, :3].transpose()).transpose()
        matches, distances = self.pyOPMb.match(self.source_catalog, 9, 6)
        self.assertEqual(len(matches), len(self.reference_catalog))
        self.assertTrue(np.all(distances < 0.01/3600.0*__deg_to_rad__))

if __name__ == '__main__':
    unittest.main()
