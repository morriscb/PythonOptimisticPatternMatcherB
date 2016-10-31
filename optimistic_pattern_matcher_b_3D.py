
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
        max_rotation_theta: The max "shift" distance allowed in degrees.
        max_rotation_phi: The max rotation allowed in degrees
        dist_tol: float epislon distance to consider a pair of objects the
        same. Units are in degrees.
        ang_tol: float max tolerance to consider the angles between two
        spokes in the pattern the same. Units are degrees.
        max_match_dist: float Maximum distance after shift and rotation are
        applied to consider two objects a match in the KDTree.
        min_matches: int minimum number of objects required to be matched to
        consider the match valid.
        max_n_patterns: int Number of patterns to attempt to create before
        exiting with a failure.
    """

    def __init__(self, reference_catalog, max_rotation_theta,
                 max_rotation_phi, dist_tol, ang_tol, max_match_dist,
                 min_matches, max_n_patterns):
        self._reference_catalog = copy(reference_catalog[:, :3])
        self._n_reference = len(self._reference_catalog)
        self._max_cos_theta = np.cos(max_rotation_theta*__deg_to_rad__)
        self._max_cos_phi_sq = np.cos(max_rotation_phi*__deg_to_rad__)**2
        self._max_sin_phi_sq = 1 - self._max_cos_phi_sq
        self._dist_tol = dist_tol*__deg_to_rad__
        self._ang_tol = ang_tol*__deg_to_rad__
        self._max_match_dist = max_match_dist*__deg_to_rad__
        self._min_matches = min_matches
        self._max_n_patterns = max_n_patterns

        self._is_valid_rotation = False

        self._build_distances_and_angles()

    def _build_distances_and_angles(self):
        """Internal function for constructing for searchable distances and
        angles between pairs of objects in the reference catalog.
        """
        self._id_array = np.empty(
            (int(self._n_reference*(self._n_reference - 1)/2), 2),
            dtype=np.int)
        self._dist_array = np.empty(
            int(self._n_reference*(self._n_reference - 1)/2),
            dtype=np.float64)
        self._delta_array = np.empty(
            (int(self._n_reference*(self._n_reference - 1)/2), 3),
            dtype=np.float64)

        start_idx = 0
        for ref_idx, ref_obj in enumerate(self._reference_catalog):
            end_idx = self._n_reference - 1 - ref_idx
            self._id_array[start_idx: start_idx + end_idx, 0] = ref_idx
            self._id_array[start_idx: start_idx + end_idx, 1] = np.arange(
                ref_idx + 1, self._n_reference, dtype=np.int_)
            self._delta_array[start_idx: start_idx + end_idx, 0] = (
                self._reference_catalog[ref_idx + 1:, 0] - ref_obj[0])
            self._delta_array[start_idx: start_idx + end_idx, 1] = (
                self._reference_catalog[ref_idx + 1:, 1] - ref_obj[1])
            self._delta_array[start_idx: start_idx + end_idx, 2] = (
                self._reference_catalog[ref_idx + 1:, 2] - ref_obj[2])
            self._dist_array[start_idx: start_idx + end_idx] = (
                self._delta_array[start_idx: start_idx + end_idx, 0]**2 +
                self._delta_array[start_idx: start_idx + end_idx, 1]**2 +
                self._delta_array[start_idx: start_idx + end_idx, 2]**2)
            start_idx += end_idx

        self._dist_array = np.sqrt(self._dist_array)
        self._sorted_args = self._dist_array.argsort()
        self._dist_array = self._dist_array[self._sorted_args]
        self._id_array = self._id_array[self._sorted_args]
        self._delta_array = self._delta_array[self._sorted_args]
        self._median_dist = self._dist_array[int(self._dist_array.shape[0]/2)]

        return None

    def _construct_and_match_pattern(self, source_candidates, n_match):
        """Given a list of source canidates we check the pinwheel pattern they
        create against the reference catalog by checking distances an angles.
        We keep the calculations as simple and fast as posible by keeping most
        distances squared and using cosines and sines.
        """
        matched_references = []
        # Create our vector and distances for the source object pinwheel.
        source_delta = np.empty((len(source_candidates) - 1, 3))
        source_delta[:, 0] = source_candidates[1:, 0] - source_candidates[0, 0]
        source_delta[:, 1] = source_candidates[1:, 1] - source_candidates[0, 1]
        source_delta[:, 2] = source_candidates[1:, 2] - source_candidates[0, 2]
        source_dist_array = np.sqrt(source_delta[:, 0]**2 +
                                    source_delta[:, 1]**2 +
                                    source_delta[:, 2]**2)
        # We first test if the distance of the first (AB) spoke of our source
        # pinwheel can be found in the array of reference catalog pairs.
        start_idx = np.searchsorted(
            self._dist_array,
            source_dist_array[0] - self._dist_tol)
        end_idx = np.searchsorted(
            self._dist_array,
            source_dist_array[0] + self._dist_tol,
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
        for dist_idx in xrange(start_idx, end_idx):
            # Reset the matched references to an empty list because we haven't
            # found any sure matches yet.
            matched_references = []
            # Compute the value of cos_theta between our source candidate and
            # both reference objects. We will pick the one with the smaller
            # value as our candidate.
            tmp_cos_theta_tuple = (
                np.dot(source_candidates[0],
                       self._reference_catalog[self._id_array[dist_idx][0]]),
                np.dot(source_candidates[0],
                       self._reference_catalog[self._id_array[dist_idx][1]]))
            tmp_cos_theta_tuple
            tmp_arg_max = np.argmax(tmp_cos_theta_tuple)
            cos_theta = tmp_cos_theta_tuple[tmp_arg_max]
            # Now we can test the displacement we have on the sky between the
            # centers of our source and reference pinwheels, exiting if it is
            # to distant.
            if cos_theta < self._max_cos_theta:
                continue
            if tmp_arg_max == 0:
                matched_references.append(self._id_array[dist_idx][0])
                matched_references.append(self._id_array[dist_idx][1])
                ref_delta = self._delta_array[dist_idx]
            else:
                matched_references.append(self._id_array[dist_idx][1])
                matched_references.append(self._id_array[dist_idx][0])
                ref_delta = -self._delta_array[dist_idx]
            ref_delta_dist = self._dist_array[dist_idx]
            # Since we already have the first two reference candidate ids we
            # can narrow our search to only those pairs that contain our
            # pinwheel reference and exclude the reference we have already used
            # to match the first spoke.
            id_mask = np.logical_or(
                np.logical_and(self._id_array[:, 0] == matched_references[0],
                               self._id_array[:, 1] != matched_references[1]),
                np.logical_and(self._id_array[:, 1] == matched_references[0],
                               self._id_array[:, 0] != matched_references[1]))
            tmp_ref_dist_arary = self._dist_array[id_mask]
            tmp_ref_delta_array = self._delta_array[id_mask]
            tmp_ref_id_array = self._id_array[id_mask]
            # Now we can start our loop to look for the remaining candidate
            # spokes of our pinwheel.
            n_failed = 0
            for cand_idx in xrange(1, len(source_dist_array)):
                match = self._pattern_spoke_test(
                    source_dist_array[cand_idx], source_delta[cand_idx],
                    source_candidates[0], source_delta[0],
                    source_dist_array[0], matched_references[0], ref_delta,
                    ref_delta_dist, tmp_ref_dist_arary, tmp_ref_delta_array,
                    tmp_ref_id_array)
                # If we don't find a mach for this spoke we can exit early.
                if match is None:
                    n_failed += 1
                    if n_failed >= len(source_candidates) - n_match:
                        break
                    continue
                matched_references.append(match)
                if len(matched_references) >= n_match:
                    break
            # If if we've found a match for each spoke we can exit early and
            # then return the matches. We can also send off the rotations we
            # have already computed.
            if len(matched_references) >= n_match:
                self._construct_rotation_matricies(
                    source_candidates[0],
                    self._reference_catalog[matched_references[0]],
                    source_delta[0], ref_delta, cos_theta)
                if self._is_valid_rotation:
                    break
        # Return the matches. If found.
        if len(matched_references) >= n_match:
            return matched_references
        return ([], None, None)

    def _pattern_spoke_test(self, cand_dist, cand_delta, source_center,
                            source_delta, source_delta_dist, ref_center_id,
                            ref_delta, ref_delta_dist, ref_dist_array,
                            ref_delta_array, ref_id_array):
        """Internal function finding matches for the remaining spokes of our
        candidate pinwheel.
        """
        # As before we first check references with matching distances, exiting
        # early if we find none.
        start_idx = np.searchsorted(
            ref_dist_array, cand_dist - self._dist_tol)
        end_idx = np.searchsorted(
            ref_dist_array, cand_dist + self._dist_tol,
            side='right')
        if start_idx == end_idx:
            return None
        if start_idx < 0:
            start_idx = 0
        if end_idx > ref_dist_array.shape[0]:
            end_idx = ref_dist_array.shape[0]
        # Loop over the posible matches and test them for quality.
        hold_id = -99
        for dist_idx in xrange(start_idx, end_idx):
            # First we compute the dot product between our delta
            # vectors in each of the source and reference pinwheels
            # and test that they are the same within tolerance. Since
            # we know the distances of these deltas already we can 
            # normalize the vectors and compare the values of cos_theta
            # between the two legs.
            ref_sign = 1
            if ref_id_array[dist_idx, 1] == ref_center_id:
                ref_sign = -1
            cos_theta_source = (np.dot(cand_delta, source_delta) /
                                (cand_dist*source_delta_dist))
            cos_theta_ref = ref_sign*(
                np.dot(ref_delta_array[dist_idx], ref_delta) /
                (ref_dist_array[dist_idx]*ref_delta_dist))
            # We need to test that the vectors are not completely aligned.
            # If they are our first test will be invalid thanks to
            # 1 - cos_theta**2 equaling zero.
            # Using a few trig relations and taylor expantions around
            # _ang_tol we compare the opening angles of our pinwheel
            # legs to see if they are within tolerance.
            if (cos_theta_ref < 1. and
                not ((cos_theta_source - cos_theta_ref)**2 /
                     (1 - cos_theta_ref**2) < self._ang_tol**2)):
                continue
            # Now we compute the cross product between the first
            # rungs of our spokes and our candidate rungs. We then
            # dot these into our center vector to make sure the
            # rotation direction and amount of rotation are correct.
            # If they are not we move on to the next candidate.
            cross_source = (np.cross(cand_delta, source_delta) /
                            (cand_dist*source_delta_dist))
            cross_ref = ref_sign*(
                np.cross(ref_delta_array[dist_idx], ref_delta) /
                (ref_dist_array[dist_idx]*ref_delta_dist))
            dot_cross_source = np.dot(cross_source, source_center)
            dot_cross_ref = np.dot(cross_ref,
                                   self._reference_catalog[ref_center_id])
            # 
            if not (-self._ang_tol <
                    (dot_cross_source - dot_cross_ref) / cos_theta_ref <
                    self._ang_tol):
                continue
            # Check to see which id we should return.
            if ref_sign == 1:
                hold_id = ref_id_array[dist_idx, 1]
            else:
                hold_id = ref_id_array[dist_idx, 0]
            break
        # Return the id of our matched object that makes up this spoke if we
        # found it.
        if hold_id >= 0:
            return hold_id
        return None

    def _construct_rotation_matricies(self, source_candidate, ref_candidate,
                                      source_delta, ref_delta, cos_theta):
        self._is_valid_rotation = False
        # First we compute the unit vector for the axis of rotation between
        # our two candidate centers. This gives us the overal shift.
        if cos_theta > 1.0:
            cos_theta = 1.
        elif cos_theta < -1.0:
            cos_theta = -1.
        sin_theta = np.sqrt(1. - cos_theta**2)
        # We need to test that we actually have to do this rotation. If the
        # vectors are already aligned we can skip the first rotation and just
        # store the identidity.
        if sin_theta != 0:
            rot_axis = np.cross(source_candidate, ref_candidate)
            rot_axis /= sin_theta
            # Now that we have our axis and cos_theta from before we can rotate
            # about it to align the source and candidate vectors. This is our
            # first rotation matrix.
            rot_cross_matrix = np.array(
                [[0., -rot_axis[2], rot_axis[1]],
                 [rot_axis[2], 0., -rot_axis[0]],
                 [-rot_axis[1], rot_axis[0], 0.]], dtype=np.float64)
            self.theta_rot_matrix = (
                cos_theta*np.identity(3) +
                sin_theta*rot_cross_matrix +
                (1. - cos_theta)*np.outer(rot_axis, rot_axis))
        else:
            self.theta_rot_matrix = np.identity(3)
        # Now we rotate our source delta to the frame of the reference.
        rot_source_delta = np.dot(self.theta_rot_matrix, source_delta)
        cos_phi_sq = (np.dot(rot_source_delta, ref_delta)**2 /
                      (np.dot(rot_source_delta, rot_source_delta) *
                       np.dot(ref_delta, ref_delta)))
        if cos_phi_sq < self._max_cos_phi_sq:
            self._is_valid_rotation = False
            return None
        cos_phi = np.sqrt(cos_phi_sq)
        delta_dot_cross = np.dot(np.cross(rot_source_delta, ref_delta),
                                 ref_candidate)
        sin_phi = np.sign(delta_dot_cross)*np.sqrt(1 - cos_phi_sq)
        ref_cross_matrix = np.array(
            [[0., -ref_candidate[2], ref_candidate[1]],
             [ref_candidate[2], 0., -ref_candidate[0]],
             [-ref_candidate[1], ref_candidate[0], 0.]], dtype=np.float64)
        self.phi_rot_matrix = (
            cos_phi*np.identity(3) +
            sin_phi*ref_cross_matrix +
            (1. - cos_phi)*np.outer(ref_candidate, ref_candidate))
        self.rot_matrix = np.dot(self.phi_rot_matrix, self.theta_rot_matrix)
        self._cos_theta = cos_theta
        self._cos_phi = cos_phi
        self._sin_phi = sin_phi
        self._is_valid_rotation = True
        return None

    def _compute_shift_and_match_sources(self, source_catalog):
        """Given an input source catalog, pinwheel centers in the source and
        reference catalog, and a cosine and sine rotation return a shifted
        catalog for matching.
        """
        if len(source_catalog) >= self._n_reference:
            shifted_references = np.dot(
                self.rot_matrix.transpose(),
                self._reference_catalog[:, :3].transpose()).transpose()

            output_matches = np.empty((len(shifted_references), 2),
                                      dtype=np.int_)
            output_matches[:, 1] = np.arange(len(shifted_references),
                                             dtype=np.int_)
            tmp_ref_dist, tmp_ref_idx = self._kdtree.query(
                shifted_references[:, :3])
            output_matches[:, 0] = tmp_ref_idx
            dist_mask = np.where(tmp_ref_dist < self._max_match_dist)
            return output_matches[dist_mask], tmp_ref_dist[dist_mask]
        else:
            shifted_sources = np.dot(
                self.rot_matrix, source_catalog[:, :3].transpose()).transpose()

            output_matches = np.empty((len(shifted_sources), 2), dtype=np.int_)
            output_matches[:, 0] = np.arange(len(shifted_sources),
                                             dtype=np.int_)
            tmp_ref_dist, tmp_ref_idx = self._kdtree.query(
                shifted_sources[:, :3])
            output_matches[:, 1] = tmp_ref_idx
            dist_mask = np.where(tmp_ref_dist < self._max_match_dist)
            return output_matches[dist_mask], tmp_ref_dist[dist_mask]

    def match(self, source_catalog, n_check, n_match):
        # Given our input source_catalog we sort on magnitude.
        sorted_catalog = source_catalog[source_catalog[:, -1].argsort()]
        n_source = len(sorted_catalog)
        # Loop through the sources from brightest to faintest grabbing a chucnk
        # of n_check each time.
        if n_source >= self._n_reference:
            self._kdtree = cKDTree(source_catalog[:, :3])
        else:
            self._kdtree = cKDTree(self._reference_catalog[:, :3])
        for pattern_idx in xrange(np.min((self._max_n_patterns,
                                          n_source - n_check))):
            matches = None
            distances = None
            # Grab the sources
            pattern = sorted_catalog[pattern_idx: pattern_idx + n_check, :3]
            ref_candidates = self._construct_and_match_pattern(pattern,
                                                               n_match)
            if len(ref_candidates) >= n_match:
                print('Matching...')
                matches, distances = self._compute_shift_and_match_sources(
                    source_catalog)
                print('Matches:', len(matches))
                if len(matches) > self._min_matches:
                    print("Succeeded after %i patterns." % pattern_idx)
                    print("\tShift %.4f arcsec" %
                          (np.arccos(self._cos_theta)*3600/__deg_to_rad__))
                    print("\tRotation: %.4f deg" %
                          (np.arcsin(self._sin_phi)/__deg_to_rad__))
                    break
        if matches is None:
            print("Failed after %i patterns." % pattern_idx)
        return matches, distances