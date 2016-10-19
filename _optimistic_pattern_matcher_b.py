
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
        self._max_angle = max_angle*__deg_to_rad_
        self._max_cos = np.cos(self._max_angle)
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

        start_idx = 0
        for ref_idx, ref_obj in enumerate(self._reference_catalog):
            end_idx =  self._n_reference - ref_idx
            self._id_array[start_idx: start_idx + end_idx, 0] = ref_idx
            self._id_array[start_idx: start_idx + end_idx, 1] = np.arange(
                ref_idx + 1, self._n_reference, dtype=np.int_)
            tmp_dx = self._reference_catalog[ref_idx + 1:, 0] - ref_obj[0]
            tmp_dy = self._reference_catalog[ref_idx + 1:, 1] - ref_obj[1]
            dist_sq = (tmp_dx*tmp_dx + tmp_dy*tmp_dy)
            tmp_theta = np.arccos(2*tmp_dy*tmp_dy/dist_sq - 1)/2
            self._dist_array[start_idx: start_idx + end_idx] = dist_sq
            self._theta_array[start_idx: start_idx + end_idx] = np.where(
                tmp_dx > 0,
                np.where(tmp_dy > 0, tmp_theta, np.pi - tmp_theta),
                np.where(tmp_dy > 0, 2*np.pi - tmp_theta, np.pi + tmp_theta))
            start_idx += end_idx

        self._sorted_args = self._dist_array.argsort()
        self._id_array = self._id_array[self._sorted_args]
        self._dist_array = self._dist_array[self._sorted_args]
        self._theta_array = self._theta_array[self._sorted_args]

        return None
    
    def _construct_and_match_pattern(self, dist_candidates):
        """Given the current best distance match we 
        """
        pattern_found = False
        
        tmp_dx = dist_candidates[1:, 0] - dist_candidates[0, 0]
        tmp_dy = dist_candidates[1:, 1] - dist_candidates[0, 1]
        dist_sq = (tmp_dx*tmp_dx + tmp_dy*tmp_dy)

        start_idx = np.searchsorted(self._dist_array,
                                    dist_sq[0] - self._dist_tol)
        end_idx = np.searchsorted(self._dist_array, dist_sq[0] + self._dist_tol,
                                  side='right')
        
        if start_idx == end_idx:
            return pattern_found, matched_references
        if start_idx < 0:
            start_idx = 0
        if end_idx > id_array.shape[0]:
            end_idx = self._dist_array.shape[0]
        
        theta = np.arccos(2*tmp_dy*tmp_dy/dist_sq - 1)/2
        theta = np.where(
            tmp_dx > 0,
            np.where(tmp_dy > 0, theta, np.pi - theta),
            np.where(tmp_dy > 0, 2*np.pi - theta, np.pi + theta))
        
        for dist_idx in xrange(start_idx, end_idx):
            
            matched_references = []
            
            delta_theta = theta[0] - self._theta_array[dist_idx]
            cos_delta_theta = np.cos(delta_theta)
            flipped = False
            passed = False
            rot = delta_theta
            reference_id = -1
            if np.fabs(cos_delta_theta) < self._max_cos:
               passed = True
               if cos_delta_theta < 0:
                   flipped = True
                   delta_theta += np.pi

            if not passed:
                continue
            
            if not flipped:
                matched_references.append(self._id_array[dist_idx, 0])
            else:
                matched_references.append(self._id_array[dist_idx, 1])
            
            id_mask = np.logical_or(
                self._id_array[:, 0] == matched_references[0],
                self._id_array[:, 1] == matched_references)
            tmp_ref_dist_arary = self._dist_array[id_mask]
            tmp_ref_theta_array = self._theta_array[id_mask]
            tmp_ref_id_array = self.__id_array[id_mask]
            
            for cand_idx in xrange(1, len(dist_sq)):
                match = self._pattern_spoke_test(
                    dist_sq[cand_idx], theta[cand_idx], matched_references[0],
                    delta_theta, tmp_ref_dist_arary, tmp_ref_theta_array,
                    tmp_ref_id_array)
                if match is None:
                    break
                matched_references.append(match)
        
        if len(matched_references) == len(dist_candidates):   
            return matched_references
        else:
            return []
    
    def _pattern_spoke_test(self, cand_dist, cand_theta, ref_center_id,
                            delta_theta_center, ref_dist_array, ref_theta_array,
                            ref_id_array):
        
        start_idx = np.searchsorted(ref_dist_array,
                                    cand_dist - self._dist_tol)
        end_idx = np.searchsorted(ref_dist_array, cand_dist + self._dist_tol,
                                  side='right')
        if start_idx == end_idx:
            return pattern_found, matched_references
        if start_idx < 0:
            start_idx = 0
        if end_idx > id_array.shape[0]:
            end_idx = self._dist_array.shape[0]
        for dist_idx in xrange(start_idx, end_idx):
            cos_delta_theta = np.cos(cand_theta - ref_theta_array[dist_idx] -
                                     delta_theta_center)
            if np.fabs(cos_delta_theta) < self._max_cos:
                
            
        
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

