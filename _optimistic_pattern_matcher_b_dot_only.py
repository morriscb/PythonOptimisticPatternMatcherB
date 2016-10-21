
from __future__ import division, print_function, absolute_import

from copy import copy

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
    
    def __init__(self, reference_catalog, max_shift, max_rotation,
                 dist_tol, ang_tol, max_match_dist, min_matches,
                 max_n_patterns):
        self._reference_catalog = copy(reference_catalog)
        self._n_reference = len(self._reference_catalog)
        self._max_cos_sq = np.cos(max_rotation*__deg_to_rad__)
        self._max_shift = max_shift
        self._dist_tol = dist_tol*dist_tol
        self._ang_tol = ang_tol*__deg_to_rad__
        self._max_match_dist = max_match_dist
        self._min_matches = min_matches
        self._max_n_patterns = max_n_patterns
        
        self._ref_kdtree = cKDTree(self._reference_catalog[:,:2])

        self._build_distances_and_angles()
    
    def _build_distances_and_angles(self):
        """Internal function for constructing for searchable distances and
        angles between pairs of objects in the reference catalog. 
        """
        self._id_array = np.empty(
            (int(self._n_reference*(self._n_reference - 1)/2), 2), dtype=np.int_)
        self._dist_array = np.empty(
            int(self._n_reference*(self._n_reference - 1)/2), dtype=np.float_)
        self._dx_array = np.empty(
            int(self._n_reference*(self._n_reference - 1)/2), dtype=np.float_)
        self._dy_array = np.empty(
            int(self._n_reference*(self._n_reference - 1)/2), dtype=np.float_)

        start_idx = 0
        for ref_idx, ref_obj in enumerate(self._reference_catalog):
            end_idx =  self._n_reference - 1 - ref_idx
            self._id_array[start_idx: start_idx + end_idx, 0] = ref_idx
            self._id_array[start_idx: start_idx + end_idx, 1] = np.arange(
                ref_idx + 1, self._n_reference, dtype=np.int_)
            self._dx_array[start_idx: start_idx + end_idx] = (
                self._reference_catalog[ref_idx + 1:, 0] - ref_obj[0])
            self._dy_array[start_idx: start_idx + end_idx] = (
                self._reference_catalog[ref_idx + 1:, 1] - ref_obj[1])
            self._dist_array[start_idx: start_idx + end_idx] = (
                self._dx_array[start_idx: start_idx + end_idx]**2 +
                self._dy_array[start_idx: start_idx + end_idx]**2)
            start_idx += end_idx

        self._sorted_args = self._dist_array.argsort()
        self._dist_array = self._dist_array[self._sorted_args]
        self._id_array = self._id_array[self._sorted_args]
        self._dx_array = self._dx_array[self._sorted_args]
        self._dy_array = self._dy_array[self._sorted_args]

        return None
    
    def _construct_and_match_pattern(self, source_candidates):
        """Given a list of source canidates we check the pinwheel pattern they
        create against the reference catalog by checking distances an angles.
        We keep the calculations as simple and fast as posible by keeping most
        distances squared and using cosines and sines.
        """
        matched_references = []
        # Create our vector and distances for the source object pinwheel.
        # TODO:
        #     Could possibly move calculating the distances to the other parts
        #     of the pinwheel until they are absolutely needed.
        tmp_dx = source_candidates[1:, 0] - source_candidates[0, 0]
        tmp_dy = source_candidates[1:, 1] - source_candidates[0, 1]
        
        dist_sq = (tmp_dx*tmp_dx + tmp_dy*tmp_dy)
        # We first test if the distance of the first (AB) spoke of our source
        # pinwheel can be found in the array of reference catalog pairs.
        start_idx = np.searchsorted(self._dist_array,
                                    dist_sq[0] - self._dist_tol)
        end_idx = np.searchsorted(self._dist_array, dist_sq[0] + self._dist_tol,
                                  side='right')
        # If we couldn't find any candidate references distances we exit. We
        # also test if the edges to make sure we are not running over the array
        # size.
        if start_idx == end_idx:
            return ([], None, None)
        if start_idx < 0:
            start_idx = 0
        if end_idx > self._dist_array.shape[0]:
            end_idx = self._dist_array.shape[0]
        # Now that we have candiates reference distances for the first spoke of
        # the pinwheel we loop over them and attempt to construct the rest of
        # the pinwheel.
        hold_cos_theta = None
        hold_sin_theta = None
        for dist_idx in xrange(start_idx, end_idx):
            # Reset the matched references to an empty list because we haven't
            # found any sure matches yet.
            matched_references = []
            # Compute the dot product and cosine^2 of the source vector and
            # reference vector. 
            dot = (tmp_dx[0]*self._dx_array[dist_idx] +
                   tmp_dy[0]*self._dy_array[dist_idx])
            cos_theta_sq = dot*dot/(dist_sq[0]*self._dist_array[dist_idx])
            # Test if the angle between the two vectors are co-aligned within
            # the user specified max rotation. This is the major rejection point
            # for candidates.
            if cos_theta_sq < self._max_cos_sq:
                continue
            # Check the positivity of the dot product. If it is negative we
            # just need to flip the direction of the reference vector. We have
            # still found a valid match candidate we just need to make sure we
            # pick the correct reference id from the reference pair.
            if dot > 0:
                matched_references.append(self._id_array[dist_idx, 0])
                matched_references.append(self._id_array[dist_idx, 1])
            else:
                matched_references.append(self._id_array[dist_idx, 1])
                matched_references.append(self._id_array[dist_idx, 0])
            # One quick test we can do now that we have a source and a reference
            # candidate for that source is check if the absolute distances
            # in x and y between the two are within our max allowed shift. If
            # they are not we can discard the this reference pair.
            x,y = self._reference_catalog[matched_references[0],:2]
            if (np.fabs(source_candidates[0, 0] - x) >= self._max_shift or
                np.fabs(source_candidates[0, 1] - y) >= self._max_shift):
                continue
            # Now that we have a firm reference candidate for the first spoke of
            # our  pinwheel we can start computing quantities we will need for
            # finding the remaining spokes. Our convention is that we rotate the
            # source vector into the reference vector.
            cross = np.sign(dot)*(tmp_dx[0]*self._dy_array[dist_idx] -
                                  tmp_dy[0]*self._dx_array[dist_idx])
            # We want to avoid square roots and arch(cos)sines as much as
            # possible but here we can at least take one square root now that
            # we're at least protected by several exit clauses above. We will
            # only need this once per pinwheel candidate.
            sin_theta = np.sign(cross)*np.sqrt(
                cross*cross/(dist_sq[0]*self._dist_array[dist_idx]))
            # We want to check if the other spokes of our pinwheel rotate in the
            # correct direction, however if our candidate rotation is already
            # close to zero within tolerance we can avoid checking for this. 
            cross_approx_zero = False
            if -self._ang_tol < sin_theta < self._ang_tol:
                cross_approx_zero = True
            # Since we already have the first two reference candidate ids we can
            # narrow our search to only those pairs that contain our pinwheel
            # reference and exclude the reference we have already used to match
            # the first spoke.
            id_mask = np.logical_or(
                np.logical_and(self._id_array[:, 0] == matched_references[0],
                               self._id_array[:, 1] != matched_references[1]),
                np.logical_and(self._id_array[:, 1] == matched_references[0],
                               self._id_array[:, 0] != matched_references[1]))
            tmp_ref_dist_arary = self._dist_array[id_mask]
            tmp_ref_dx_array = self._dx_array[id_mask]
            tmp_ref_dy_array = self._dy_array[id_mask]
            tmp_ref_id_array = self._id_array[id_mask]
            # Now we can start our loop to look for the remaining candidate
            # spokes of our pinwheel. 
            for cand_idx in xrange(1, len(dist_sq)):
                match = self._pattern_spoke_test(
                    dist_sq[cand_idx], tmp_dx[cand_idx], tmp_dy[cand_idx],
                    matched_references[0], cross, sin_theta, cross_approx_zero,
                    tmp_ref_dist_arary, tmp_ref_dx_array, tmp_ref_dy_array,
                    tmp_ref_id_array)
                # If we don't find a mach for this spoke we can exit early.
                if match is None:
                    break
                matched_references.append(match)
            # If if we've found a match for each spoke we can exit early and
            # then return the matches. We can also send off the rotations we
            # have already computed.
            if len(matched_references) == len(source_candidates):
                hold_cos_theta = np.sqrt(cos_theta_sq)
                hold_sin_theta = sin_theta
                break
        # Return the matches. If found.
        if len(matched_references) == len(source_candidates):   
            return (matched_references, hold_cos_theta, hold_sin_theta)
        return ([], None, None)
    
    def _pattern_spoke_test(self, cand_dist, cand_dx, cand_dy, ref_center_id,
                            cross_ref, sin_theta_ref, cross_approx_zero,
                            ref_dist_array, ref_dx_array, ref_dy_array,
                            ref_id_array):
        """Internal function finding matches for the remaining spokes of our
        candidate pinwheel.
        """
        # As before we first check references with matching distances, exiting
        # early if we find none.
        start_idx = np.searchsorted(ref_dist_array,
                                    cand_dist - self._dist_tol)
        end_idx = np.searchsorted(ref_dist_array, cand_dist + self._dist_tol,
                                  side='right')
        if start_idx == end_idx:
            return None
        if start_idx < 0:
            start_idx = 0
        if end_idx > self._dist_array.shape[0]:
            end_idx = self._dist_array.shape[0]
        # Loop over the posible matches and test them for quality.
        hold_id = -99
        for dist_idx in xrange(start_idx, end_idx):
            # Again the dot produt is a great way to test that our vector points
            # in the correct diretcion. 
            dot = (cand_dx*ref_dx_array[dist_idx] +
                   cand_dy*ref_dy_array[dist_idx])
            # If this vector is pointing counter to our spoke and the vector
            # does not point to the center we rejet this match.
            if (np.sign(dot) < 0 and not
                ref_id_array[dist_idx, 1] == ref_center_id):
                continue
            # Now if our rotation isn't within tolerance of zero we can use
            # the fact if the cross product is negative or positive to assure
            # ourselves that this spoke rotates in the same direction as our
            # first spoke.
            cross = np.sign(dot)*(cand_dx*ref_dy_array[dist_idx] -
                                  cand_dy*ref_dx_array[dist_idx])
            if not cross_approx_zero and np.sign(cross) != np.sign(cross_ref):
                continue
            # Again we test the aboslute shift to make sure that it is within
            # our max allowed.
            cos_theta_sq = dot*dot/(cand_dist*ref_dist_array[dist_idx])
            if cos_theta_sq < self._max_cos_sq:
                continue
            # Our last test is to check is if the two values of sine^2 for the
            # first spoke and this spoke match, that is if the two rotations
            # the of different spokes agree within tolerance. 
            sin_theta_sq = cross*cross/(cand_dist*ref_dist_array[dist_idx])
            # This limit relies on the small angle approximation
            # sin(theta)=theta and the fact that the difference between
            # theta_ref and theta_src should be at most +- theta_tolerance.
            ang_lim = (-self._ang_tol*(2*sin_theta_ref + self._ang_tol),
                       self._ang_tol*(2*sin_theta_ref - self._ang_tol))
            if ((not cross_approx_zero and 
                np.min(ang_lim) <= sin_theta_ref*sin_theta_ref - sin_theta_sq <
                np.max(ang_lim)) or
                (cross_approx_zero and sin_theta_sq <
                 self._ang_tol*self._ang_tol)):
                # We found a match, store the correct id and then exit the loop.
                if np.sign(dot) < 0:
                    hold_id =  ref_id_array[dist_idx, 0]
                else:
                    hold_id =  ref_id_array[dist_idx, 1]
                break
        # Return the id of our matched object that makes up this spoke if we
        # found it.
        if hold_id >= 0:
            return hold_id
        return None
        
    def _compute_shift_sources(self, source_catalog, center_ref, center_source,
                               cos_theta, sin_theta):
        """Given an input source catalog, pinwheel centers in the source and
        reference catalog, and a cosine and sine rotation return a shifted
        catalog for matching.
        """
        output_catalog = np.empty_like(source_catalog)
        
        tmp_x_array = source_catalog[:, 0] - center_source[0]
        tmp_y_array = source_catalog[:, 1] - center_source[1]
        
        output_catalog[:,0] = (
            (cos_theta*tmp_x_array - sin_theta*tmp_y_array) +
            self._reference_catalog[center_ref, 0])
        output_catalog[:,1] = (
            (sin_theta*tmp_x_array + cos_theta*tmp_y_array) +
            self._reference_catalog[center_ref, 1])
        
        return output_catalog
    
    def _compute_matches(self, shifted_sources):
        """Given a shifted source catalog, find matches in the reference
        catalog.
        """
        output_matches = np.empty((len(shifted_sources), 2), dtype=np.int_)
        output_matches[:, 0] = np.arange(len(shifted_sources), dtype=np.int_)
        tmp_ref_dist, tmp_ref_idx = self._ref_kdtree.query(
            shifted_sources[:,:2])
        output_matches[:, 1] = tmp_ref_idx
        dist_mask = np.where(tmp_ref_dist < self._max_match_dist)
        return output_matches[dist_mask], tmp_ref_dist[dist_mask] 
        
    def _test_valid_shift(self, shift):
        pass
    
    def match(self, source_catalog, n_match):
        # TODO:
        #     Create kdtree here on the larger of the two arrays to return
        #     unique mataches only.
        # Given our input source_catalog we sort on magnitude.
        sorted_catalog = source_catalog[source_catalog[:, 2].argsort()]
        n_source = len(sorted_catalog)
        # Loop through the sources from brightest to faintest grabbing a chucnk
        # of n_match each time.
        matches = None
        distances = None
        for pattern_idx in xrange(np.min((self._max_n_patterns,
                                          len(source_catalog) - n_match))):
            # Grab the sources
            pattern = sorted_catalog[pattern_idx: pattern_idx + n_match]
            ref_candidates, cos_theta, sin_theta = (
                self._construct_and_match_pattern(pattern))
            if len(ref_candidates) >= n_match:
                print('Shifting...')
                shifted_sources = self._compute_shift_sources(
                    sorted_catalog, ref_candidates[0], pattern[0], cos_theta,
                    sin_theta)
                print('Matching...')
                matches, distances = self._compute_matches(shifted_sources)
                print('Matches:', len(matches))
                if len(matches) > self._min_matches:
                    print("Succeeded after %i patterns." % pattern_idx)
                    break
        if matches is None:
            print("Failed after %i patterns." % pattern_idx)
        return matches, distances
