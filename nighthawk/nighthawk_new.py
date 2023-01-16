"""Nighthawk NFC detector functions."""


from pathlib import Path
import itertools
import logging
import json

import librosa
import numpy as np
import pandas as pd


# TODO: Look into `librosa.load` sample value range.
# TODO: Consider different `librosa.load` resampling algorithms.
# TODO: Consider invoking model on blocks of input frames.


_PACKAGE_DIR_PATH = Path(__file__).parent
MODEL_DIR_PATH = _PACKAGE_DIR_PATH / 'saved_model_with_preprocessing'

TAXONOMY_DIR_PATH = _PACKAGE_DIR_PATH / 'taxonomy'
EBIRD_TAXONOMY_FILE_PATH = TAXONOMY_DIR_PATH / 'ebird_taxonomy.csv'
GROUP_EBIRD_CODES_FILE_PATH = TAXONOMY_DIR_PATH / 'groups_ebird_codes.csv'

_CONFIG_DIR_PATH = _PACKAGE_DIR_PATH / 'test_config'
CALIBRATORS_FILE_PATH = _CONFIG_DIR_PATH / 'calibrators_dict.obj'
TAXA_OF_INTEREST_FILE_PATH = _CONFIG_DIR_PATH / 'test_config.json'

TAXONOMIC_RANKS = ('order', 'family', 'group', 'species')
TAXONOMIC_RANK_PLURALS = ('orders', 'families', 'groups', 'species')


_chain = itertools.chain.from_iterable


def get_taxa_of_interest():

    # Load JSON that lists taxa of interest.
    with open(TAXA_OF_INTEREST_FILE_PATH) as f:
        config = json.load(f)

    # Return set of names of all taxa of interest.
    return frozenset(_chain(
        config['subselect_taxa'][rank] for rank in TAXONOMIC_RANKS))


def run_detector_on_file(
        audio_file_path, hop_size, block_length, model, postprocessor):

    logits = []
    start_frame_index = 0
    detections = []

    for i, samples in enumerate(generate_sample_frames(
            audio_file_path, model.input_length, hop_size,
            model.input_sample_rate)):

        if np.max(np.abs(samples)) > 1.1:

            logging.warning(
                f'Got out-of-range samples in input frame {i} of file '
                f'"{audio_file_path}". Will replace with zero frame.')

            samples = np.zeros(model.input_length, dtype='float32')

        # Apply model to this input.
        logits.append(model.process(samples))

        if len(logits) == block_length:
            # have a block of logits

            detections.append(
                _postprocess(logits, start_frame_index, postprocessor))

            logits = []
            start_frame_index += block_length

    if len(logits) != 0:
        # some logits have not yet been postprocessed

        detections.append(
            _postprocess(logits, start_frame_index, postprocessor))

    # Flatten list[tuple[tuple]] to tuple[tuple].
    detections = tuple(_chain(detections))

    # Put detections into Pandas DataFrame with timing and file info.
    detections = _get_result_dataframe(
        detections, audio_file_path, model.input_length, hop_size,
        model.input_sample_rate)

    return detections
        

def _get_result_dataframe(
        detections, audio_file_path, frame_length, hop_size, sample_rate):

    # Create DataFrame from detections.
    rank_columns = tuple(_chain((r, 'prob_' + r) for r in TAXONOMIC_RANKS))
    columns = ('frame',) + rank_columns + ('class', 'prob')
    df = pd.DataFrame(detections, columns=columns)

    # Prepend timing and file columns.
    frame_dur = frame_length / sample_rate
    hop_dur = frame_dur * hop_size / 100
    df.insert(0, 'start_sec', hop_dur * df['frame'])
    df.insert(1, 'end_sec', df['start_sec'] + frame_dur)
    df.insert(2, 'filename', audio_file_path.name)
    df.insert(3, 'path', audio_file_path)

    # Drop frame column.
    df.drop(columns='frame', inplace=True)

    return df


def generate_sample_frames(file_path, frame_length, hop_size, sample_rate):

    frame_dur = frame_length / sample_rate
    hop_length = int(round(frame_length * hop_size / 100))
    hop_dur = hop_length / sample_rate

    file_dur = librosa.get_duration(filename=file_path)

    load_length = 64        # sample frames
    load_dur = (load_length - 1) * hop_dur + frame_dur
    load_hop_dur = load_length * hop_dur

    load_offset = 0

    while load_offset < file_dur:

        samples, _ = librosa.load(
            file_path, sr=sample_rate, res_type='fft', offset=load_offset,
            duration=load_dur)

        sample_count = len(samples)
        start_index = 0
        end_index = frame_length

        while end_index <= sample_count:
            yield samples[start_index:end_index]
            start_index += hop_length
            end_index += hop_length

        load_offset += load_hop_dur


def _postprocess(logits, start_frame_index, postprocessor):

    # The input `logits` has type list[list[Tensor]]. The inner
    # lists have length four, with one element for taxonomic order,
    # family, group, and species, and each tensor is 2-D, with the
    # first dimension always one and the second dimension the number
    # of model taxa of the appropriate taxonomic rank.

    # Put order, family, group and species logit tensors into their
    # own 2-D NumPy arrays, squeezing out the first, unit tensor
    # dimension. The result is a list of four 2-D NumPy arrays, one
    # each for order, family, group, and species. The first index of
    # each array is for logit frame and the second is for taxon.
    logits = [np.squeeze(np.array(p), axis=1) for p in zip(*logits)]

    # Get detections.
    return postprocessor.process(logits, start_frame_index)
