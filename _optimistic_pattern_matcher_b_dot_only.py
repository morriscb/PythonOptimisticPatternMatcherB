
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
    
    def __init__(self, reference_catalog, max_rotation, max_shift,
                 dist_tol, ang_tol):
        self._reference_catalog = reference_catalog
        self._n_reference = len(self._reference_catalog)
        self._max_cos = np.cos(max_rotation*__deg_to_rad__)
        self._max_shift = max_shift
        self._dist_tol = dist_tol
        self._ang_tol = ang_tol

        self._build_distances_and_angles()
    
    def _build_distances_and_angles(self):
        """Internal function for constructing for searchable distances and
        angles between pairs of objects in the reference catalog. 
        """
        self._id_array = np.empty(
            (self._n_reference*(self._n_reference - 1)/2, 2), dtype=np.int_)
        self._dist_array = np.empty(
            self._n_reference*(self._n_reference - 1)/2, dtype=np.float_)
        self._dx_array = np.empty(
            self._n_reference*(self._n_reference - 1)/2, dtype=np.float_)
        self._dy_array = np.empty(
            self._n_reference*(self._n_reference - 1)/2, dtype=np.float_)

        start_idx = 0
        for ref_idx, ref_obj in enumerate(self._reference_catalog):
            end_idx =  self._n_reference - ref_idx
            self._id_array[start_idx: start_idx + end_idx, 0] = ref_idx
            self._id_array[start_idx: start_idx + end_idx, 1] = np.arange(
                ref_idx + 1, self._n_reference, dtype=np.int_)
            self._dx_array[start_idx: start_idx + end_idx] = (
                self._reference_catalog[ref_idx + 1:, 0] - ref_obj[0])
            self._dy_array[start_idx: start_idx + end_idx] = (
                self._reference_catalog[ref_idx + 1:, 1] - ref_obj[1])
            self._dist_array[start_idx: start_idx + end_idx] = (tmp_dx*tmp_dx +
                                                                tmp_dy*tmp_dy)
            start_idx += end_idx

        self._sorted_args = self._dist_array.argsort()
        self._id_array = self._id_array[self._sorted_args]
        self._dx_array = self._dx_array[self._sorted_args]
        self._dy_array = self._dy_arra[self._sorted_args]

        return None
    
    def _construct_and_match_pattern(self, dist_candidates):
        """Given the current best distance match we 
        """
        pattern_found = False
        matched_references = None
        
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
        
        for dist_idx in xrange(start_idx, end_idx):
            matched_references = []
            dot = (tmp_dx[0]*self._dx_array[dist_idx] +
                   tmp_dy[0]*self._dy_array[dist_idx])
            cos_theta_scr = dot/dist_sq
            cos_theta_ref = dot/self._dist_array[dist_idx]
            if (np.fabs(cos_theta_scr) < self._max_cos or
                np.fabs(cos_theta_ref) < self._max_cos):
                continue
            
            # Posible check for max distance in these two.
            if dot > 0:
                matched_references.append(self._id_array[dist_idx, 0])
                delta_dx = tmp_dx[0] - self._dx_array[dist_idx]
                delta_dy = tmp_dy[0] - self._dy_array[dist_idx]
            else:
                matched_references.append(self._id_array[dist_idx, 1])
                delta_dx = tmp_dx[0] + self._dx_array[dist_idx]
                delta_dy = tmp_dy[0] + self._dy_array[dist_idx]

            x,y = self._reference_catalog[matched_references[0],:2]
            if (np.fabs(dist_candidates[0, 0] - x) >= self._max_shift or
                np.fabs(dist_candidates[0, 1] - y) >= self._max_shift):
                continue
    
            id_mask = np.logical_or(
                self._id_array[:, 0] == matched_references[0],
                self._id_array[:, 1] == matched_references[0])
            
            tmp_ref_dist_arary = self._dist_array[id_mask]
            tmp_ref_dx_array = self._dx_array[id_mask]
            tmp_ref_dy_array = self._dy_array[id_mask]
            tmp_ref_id_array = self.__id_array[id_mask]
            
            delta_dx = tmp_dx[0] - self._dx_array[dist_idx]
            delta_dy = tmp_dy[0] - self._dy_array[dist_idx]
            
            for cand_idx in xrange(1, len(dist_sq)):
                match = self._pattern_spoke_test(
                    dist_sq[cand_idx], tmp_dx[cand_idx], tmp_dy[cand_idx],
                    matched_references[0], cos_theta_ref, delta_dx, delta_dy,
                    tmp_ref_dist_arary, tmp_ref_dx_array, tmp_ref_dy_array,
                    tmp_ref_id_array)
                if match is None:
                    break
                matched_references.append(match)
            if len(matched_references) == len(dist_candidates):
                break
        
        if len(matched_references) == len(dist_candidates):   
            return matched_references
        else:
            return []
    
    def _pattern_spoke_test(self, cand_dist, cand_dx, cand_dy,
                            ref_center_id, cos_theta_ref, delta_dx, delta_dy,
                            ref_dist_array, ref_dx_array, ref_dy_array,
                            ref_id_array):
        start_idx = np.searchsorted(ref_dist_array,
                                    cand_dist - self._dist_tol)
        end_idx = np.searchsorted(ref_dist_array, cand_dist + self._dist_tol,
                                  side='right')
        for dist_idx in xrange(start_idx, end_idx):
            dot = (cand_dx*self._dx_array[dist_idx] +
                   cand_dy*self._dy_array[dist_idx])
            if (np.sign(dot) < 0 and not
                ref_id_array[dist_idx,1] == ref_center_id):
                continue
            
        
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

