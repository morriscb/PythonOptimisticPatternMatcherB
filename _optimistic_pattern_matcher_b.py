
from __future__ import division, print_function, absolute_import

import numpy as np
from scipy.spatial import cKDTree

__deg_to_rad__ = np.pi/180.0


class OptimisticPatternMatcherB(object):
    """Class implimenting Optimistic Pattern Matcher B from Tubak 2007. The
    class loads and stores the reference object in a convienent data structure
    for matching any set of source obejcts that are assumed to contain each
    other.
    ----------------------------------------------------------------------------
    Attributes:
        reference_catalog: Input array of x, y, mag of reference objects.
        dist_tol: float epislon distance to consider a pair of objects the same.
        max_angle: float max rotation to consider a possible match
        max_shift: float max distance to shift the image for a match
    """
    
    def __init__(self, reference_catalog, dist_tol, max_angle, max_shift):
        self._reference_catalog = reference_catalog
        self._n_reference = len(self._reference_catalog)
        self._dist_tol = dist_tol
        self._max_angle = max_angle
        self._max_shift = max_shift

        self._build_distances_and_angles()
    
    def _build_distances_and_angles(self):
        """Internal function for constructing for searchable distances and
        angles between pairs of objects in the reference catalog. 
        """
        self._id_array = np.empty(
            (self._n_reference*(self._n_reference - 1)/2, 2), dtype=np.int_)
        self._dist_array = np.empty(
            self._n_reference*(self._n_reference - 1)/2, dtype=np.float_)
        self._theta_array = np.empty(
            self._n_reference*(self._n_reference - 1)/2, dtype=np.float_)
        self._quadrant_array = np.empty(
            (self._n_reference*(self._n_reference - 1)/2, 2), dtype=np.bool_)

        start_idx = 0
        for ref_idx, ref_obj in enumerate(self._reference_catalog):
            end_idx =  self._n_reference - ref_idx
            self._id_array[start_idx: start_idx + end_idx, 0] = ref_idx
            self._id_array[start_idx: start_idx + end_idx, 1] = np.arange(
                ref_idx + 1, self._n_reference, dtype=np.int_)
            tmp_dx = self._reference_catalog[ref_idx + 1:, 0] - ref_obj[0]
            tmp_dy = self._reference_catalog[ref_idx + 1:, 1] - ref_obj[1]
            dist_sq = (tmp_dx*tmp_dx + tmp_dy*tmp_dy)
            self._dist_array[start_idx: start_idx + end_idx] = dist_sq
            self._theta_array[start_idx: start_idx + end_idx] = np.arccos(
                2*tmp_dy*tmp_dy/dist_sq - 1)/2
            self._quadrant_array[start_idx: start_idx + end_idx, 0] = (
                tmp_dx <= 0)
            self._quadrant_array[start_idx: start_idx + end_idx, 1] = (
                tmp_dy <= 0)
            start_idx += end_idx

        self._sorted_args = self._dist_array.argsort()
        self._id_array = self._id_array[self._sorted_args]
        self._dist_array = self._dist_array[self._sorted_args]
        self._theta_array = self._cos2theta_array[self._sorted_args]
        self._quadrant_array = self._quadrant_array[self._sorted_args]

        return None
    
    def _construct_and_match_pattern(self, dist_candidates):
        """Given the current best distance match we 
        """
        pattern_found = False
        matched_references = None
        
        tmp_dx = dist_candidates[1:, 0] - dist_candidates[0, 0]
        tmp_dy = dist_candidates[1:, 1] - dist_candidates[0, 1]
        dist_sq = (tmp_dx*tmp_dx + tmp_dy*tmp_dy)
        cos2theta = 2*tmp_dy*tmp_dy/dist_sq - 1
        _quadrant = np.empty((len(tmp_dx), 2), dtype = np.bool_)
        _quadrant[:,0] = (tmp_dx <= 0)
        _quadrant[:,1] = (tmp_dy <= 0)

        min_idx = np.searchsorted(self._dist_array, dist_sq[0] - self._dist_tol)
        max_idx = np.searchsorted(self._dist_array, dist_sq[0] + self._dist_tol,
                                  side='right')
        
        if min_idx == max_idx:
            return pattern_found, None
        
        for dist_idx in xrange(min_idx, max_idx):
            self._quadrant_array
            
        return pattern_found, matched_references
        
    def _compute_shift(self):
        pass
    
    def _compute_matches(self):
        pass
    
    def _test_valid_shift(self, shift):
        pass
    
    def match(self, source_catalog, n_match):
        sorted_catalog = source_catalog[source_catalog[:, 2].argsort()]
        n_source = len(sorted_catalog)
        
        for pattern_idx in xrange(n_source - n_match):
            pattern = sorted_catalog[pattern_idx: pattern_idx + n_match]
            if self._construct_and_match_pattern(pattern):
                shift = self._compute_shift(pattern)
                if self._test_valid_shift(shift):
                    matches = self._compute_matches(source_catalog, shift)
                    if matches > self._min_matches:
                        break
        return matches

