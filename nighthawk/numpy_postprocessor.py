from collections import defaultdict
import itertools
import logging
import pickle

import numpy as np
import pandas as pd

from postprocessor import Postprocessor
import nighthawk_new as nh


_NO_TAXON = -1

_chain = itertools.chain.from_iterable


'''
Assumptions:
* Taxon names are distinct, both within and across ranks.
* Group members are all of the same family.

Questions:
* Is "rank" okay, or should we use "level" instead?
* Where to calibrators come from?
'''


# TODO: Write unit tests for `_get_detections` that use generated input.


class NumPyPostprocessor(Postprocessor):


    def __init__(self, input_taxa, taxa_of_interest, threshold):
        
        # Get tuple of taxon names for input logits concatenated
        # across taxonomic rank.
        all_taxa = tuple(_chain(input_taxa))

        # Get taxon and logit pruning indices.
        # `self._pruning_indices[i]` is the index in `all_taxa` of
        # taxon `self._taxa[i]` except if no taxa are to be pruned,
        # in which case `self._pruning_indices` is `None`.
        self._pruning_indices = \
            _get_pruning_indices(all_taxa, taxa_of_interest)

        # Get tuple of names of taxa of interest of all ranks, ordered
        # like pruned logits. The index of a taxon in this tuple is
        # the taxon's *taxon index*, and the taxon with taxon index
        # i is *taxon i*.
        self._taxa = tuple(all_taxa[i] for i in self._pruning_indices)

        # Get mapping from taxon name to pruned logit index.
        taxon_indices = {taxon: i for i, taxon in enumerate(self._taxa)}

        # Get 2-D NumPy array of taxonomic lineages.
        # `self._lineages[i, j]` is the taxon index of the rank `j`
        # taxon of taxon `i`'s lineage, or `_NO_TAXON` if there is no
        # such taxon. The *lineage* of a taxon comprises the taxon and
        # all of its taxonomic ancestors.
        self._lineages = _get_lineages(self._taxa, taxon_indices)

        # Get sets of taxon ancestors.
        # `self._ancestors[i]` is the set of taxon indices of all
        # taxonomic ancestors of taxon `i`.
        self._ancestors = _get_ancestors(self._lineages)

        # Get tuple of calibrators, ordered like `self._taxa`.
        self._calibrators = _get_calibrators(
            nh.CALIBRATORS_FILE_PATH, self._taxa)

        self._threshold = threshold


    def process(self, logits, start_frame_index):

        # Concatenate logits across taxonomic ranks.
        logits = np.concatenate(logits, axis=1)

        logits = self._prune_logits(logits)
        probs = self._get_probabilities(logits)
        return self._get_detections(probs, start_frame_index)


    def _prune_logits(self, logits):
        if self._pruning_indices is None:
            return logits
        else:
            return logits[:, self._pruning_indices]


    def _get_probabilities(self, logits):

        # Apply sigmoid function to logits.
        probs = 1 / (1 + np.exp(-logits))

        # Apply calibrators to probabilities if needed.
        if self._calibrators is not None:
            for i, calibrator in enumerate(self._calibrators):
                if calibrator is not None:
                    probs[:, i] = calibrator.predict(probs[:, i])

        return probs


    def _get_detections(self, probs, start_frame_index):

        # `probs` is an N x T NumPy array, where N is a number
        # of frames and T is the number of taxa of interest.
        # In the comments below we also use R as the number of
        # taxonomic ranks, i.e. `len(_TAXONOMIC_RANKS)`.

        # Get length N x T boolean array indicating for which taxa
        # probability is at least threshold.
        detections = np.where(probs >= self._threshold / 100, True, False)

        # Append column of `True` values to array. The new column
        # is used below in the construction of detection stacks,
        # where it is indexed by `_NO_TAXON` lineage array elements.
        trues = np.full((detections.shape[0], 1), True)
        detections = np.append(detections, trues, axis=1)

        # Get N x T x R boolean array of detection stacks. Elements
        # (i, j, :) are the detection stack for frame i and taxon j.
        # Element (i, j, k) is `True` if and only if for frame i the
        # taxon of rank k of taxon j's lineage was detected or there
        # is no taxon of rank k in the lineage.
        detection_stacks = detections[:, self._lineages]

        # Conjoin stack elements. Result is length N x T boolean
        # array that is `True` for a taxon of a frame if and only
        # if all taxa of its lineage were detected in that frame.
        detections = np.all(detection_stacks, axis=2)

        # Get frame and lineage indices of all detected lineages.
        # A lineage is detected if and only if all of its taxa are
        # detected. A lineage is represented by the index of its
        # lowest-rank taxon.
        frame_indices, lineage_indices = detections.nonzero()

        # Get mapping from frame index to set of detected lineage indices.
        frame_lineages = defaultdict(set)
        for frame_index, lineage_index in zip(frame_indices, lineage_indices):
            frame_lineages[frame_index].add(lineage_index)

        # Get tuple of (frame index, maximal detected lineage) pairs.
        # a detected lineage is *maximal* if and only if it is not a
        # proper sublineage of another detected lineage of the same
        # frame.
        maximal_detected_lineages = tuple(_chain(
            self._get_maximal_detected_lineages(*item)
            for item in frame_lineages.items()))

        # Return a detection tuple for each maximal detected lineage.
        return self._get_detection_tuples(
            maximal_detected_lineages, start_frame_index, probs)


    def _get_maximal_detected_lineages(self, frame_index, lineage_indices):

        # Get indices of detected lineages that are not maximal.
        nonmaximal_indices = set(_chain(
            self._ancestors[i] for i in lineage_indices))

        # Get set of indices of detected lineages that are maximal.
        maximal_indices = lineage_indices - nonmaximal_indices

        # Get tuple of (frame index, lineage index, lineage) triples.
        # Each lineage is a 1-D NumPy array of taxon indices, with one
        # index per taxonomic rank.
        return tuple(
            (frame_index, i, self._lineages[i])
            for i in maximal_indices)


    def _get_detection_tuples(self, detections, start_frame_index, probs):

        def get_taxon_info(taxon_index, probs):
            if taxon_index == _NO_TAXON:
                return ('', '')
            else:
                return (self._taxa[taxon_index], probs[taxon_index])

        def get_detection_tuple(frame_index, lineage_index, lineage):
            frame_probs = probs[frame_index]
            frame_index = (start_frame_index + frame_index,)
            lineage_info = tuple(_chain(
                get_taxon_info(i, frame_probs) for i in lineage))
            taxon = self._taxa[lineage_index]
            prob = frame_probs[lineage_index]
            taxon_info = (taxon, prob)
            return frame_index + lineage_info + taxon_info
            
        def get_detection_tuple_sort_key(t):
            return (t[0],) + tuple(t[1:9:2])    # frame index and taxon names

        return tuple(sorted(
            (get_detection_tuple(*d) for d in detections),
            key=get_detection_tuple_sort_key))


def _get_pruning_indices(all_taxa, taxa_of_interest):

    if taxa_of_interest is None:
        # detecting all taxa

        return None

    else:
        # detecting specified taxa

        # Get indices of taxa of interest in input logits
        # concatenated across taxonomic rank.
        return tuple(
            i for i, taxon in enumerate(all_taxa)
            if taxon in taxa_of_interest)


def _get_lineages(taxa, taxon_indices):

    # Get taxonomy as Pandas DataFrame with one row per species.
    taxonomy_df = _get_taxonomy()

    # Get per-rank mappings from taxon to lineage member of rank.
    lineage_member_dicts = _get_lineage_member_dicts(taxonomy_df, taxa)

    # Create lineage array filled with `_NO_TAXON` values.
    taxon_count = len(taxa)
    rank_count = len(nh.TAXONOMIC_RANKS)
    lineages = np.full((taxon_count, rank_count), _NO_TAXON)

    # Set non-`_NO_TAXON` lineage values.
    for i, taxon in enumerate(taxa):
        for j in range(rank_count):
            try:
                member = lineage_member_dicts[j][taxon]
            except:
                continue
            lineages[i][j] = taxon_indices[member]

    return lineages


def _get_taxonomy():

    # Load taxonomy into Pandas DataFrame with one row per species.
    df = pd.read_csv(nh.EBIRD_TAXONOMY_FILE_PATH)

    # Rename "code" column to "species".
    df.rename(columns={'code': 'species'}, inplace=True)

    # Remove parenthetical family descriptions.
    df['family'] = df['family'].str.replace(' \(.*\)', '', regex=True)

    # Get mapping from species to group.
    g = pd.read_csv(nh.GROUP_EBIRD_CODES_FILE_PATH)
    species_groups = dict(zip(g.ebird_code, g.group))

    # Add group column.
    df['group'] = df['species'].map(species_groups)

    # Replace `NaN` with `None`.
    df.replace({np.NaN: None}, inplace=True)

    return df


def _get_lineage_member_dicts(taxonomy_df, taxa):

    """
    Gets per-rank mappings from taxon to lineage member of rank.

    Returns
    -------
    tuple[dict[str, str]]
        Tuple of lineage member mappings.

        Element `i` of the returned tuple maps taxon to rank
        `_TAXONOMIC_RANKS[i]` taxon of the same lineage, or to
        `_NO_TAXON` if there is no such taxon.

        The mapping for a given rank includes items for all taxa whose
        lineages include a member of that rank. Accordingly, the mapping
        for a given rank includes the identity mapping for taxa of that
        rank.

        The mappings are restricted to the taxa of `taxa`.
    """

    ranks = nh.TAXONOMIC_RANKS

    # Make set of taxon names for efficient membership testing.
    taxa = frozenset(taxa)

    return tuple(
        _get_lineage_member_dicts_aux(taxonomy_df, ranks[i:], taxa)
        for i in range(len(ranks)))


def _get_lineage_member_dicts_aux(taxonomy_df, ranks, taxa):

    result = {}
    to_col = taxonomy_df[ranks[0]]

    for rank in ranks:

        # Update `result` to include mapping from rank `rank` taxa
        # to rank `ranks[0]` taxa. Include only taxa of interest and
        # taxa whose lineages include a member of rank `ranks[0]`.
        # The second condition is necessary since some species have
        # no group.
        from_col = taxonomy_df[rank]
        result.update(dict(
            p for p in zip(from_col, to_col)
            if p[0] in taxa and p[1] is not None))

    return result
    
    
def _get_ancestors(lineages):

    def get_ancestors(lineage):
        return frozenset(lineage[lineage != _NO_TAXON][:-1])

    taxon_count = lineages.shape[0]
    return tuple(get_ancestors(lineages[i]) for i in range(taxon_count))


def _get_calibrators(file_path, taxa):

    if file_path is None:
        return None

    else:

        # Load calibrators from pickle file.
        with open(file_path, 'rb') as f:
            calibrators = pickle.load(f)

        # Warn if any calibrators are missing.
        for taxon in taxa:
            if not taxon in calibrators:
                logging.warning(
                    f'Probability calibrator missing for taxon "{taxon}".')

        return tuple(calibrators.get(taxon) for taxon in taxa)
