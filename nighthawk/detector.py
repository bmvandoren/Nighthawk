"""Script that runs the Nighthawk NFC detector on specified audio files."""


from pathlib import Path
import time

import librosa
import numpy as np

import nighthawk.run_reconstructed_model as run_reconstructed_model


MODEL_SAMPLE_RATE = 22050         # Hz
MODEL_INPUT_DURATION = 1          # seconds

DEFAULT_HOP_DURATION = .2         # seconds
DEFAULT_THRESHOLD = 50            # percent
DEFAULT_MERGE_OVERLAPS = True
DEFAULT_DROP_UNCERTAIN = True
DEFAULT_CSV_OUTPUT = True
DEFAULT_RAVEN_OUTPUT = False
DEFAULT_OUTPUT_DIR_PATH = None

_PACKAGE_DIR_PATH = Path(__file__).parent
_MODEL_DIR_PATH = _PACKAGE_DIR_PATH / 'saved_model_with_preprocessing'
_TAXONOMY_DIR_PATH = _PACKAGE_DIR_PATH / 'taxonomy'
_CONFIG_DIR_PATH = _PACKAGE_DIR_PATH / 'test_config'


# TODO: Consider different `librosa.load` resampling algorithms.


def process_files(
        input_file_paths, hop_duration=DEFAULT_HOP_DURATION,
        threshold=DEFAULT_THRESHOLD, merge_overlaps=DEFAULT_MERGE_OVERLAPS,
        drop_uncertain=DEFAULT_DROP_UNCERTAIN, csv_output=DEFAULT_CSV_OUTPUT,
        raven_output=DEFAULT_RAVEN_OUTPUT,
        output_dir_path=DEFAULT_OUTPUT_DIR_PATH):
    
    print('Loading detector model...')
    model = _load_model()

    print('Getting detector configuration file paths...')
    config_file_paths = _get_configuration_file_paths()

    for input_file_path in input_file_paths:

        # Make sure input file path is absolute for messages.
        input_file_path = input_file_path.absolute()

        print(f'Running detector on audio file "{input_file_path}"...')
        
        detections = _process_file(
            input_file_path, model, config_file_paths, hop_duration,
            threshold, merge_overlaps, drop_uncertain)

        if csv_output:
            file_path = _prep_for_output(
                output_dir_path, input_file_path, hop_duration, threshold,
                merge_overlaps, drop_uncertain, '.csv')
            _write_detection_csv_file(file_path, detections)

        if raven_output:
            file_path = _prep_for_output(
                output_dir_path, input_file_path, hop_duration, threshold,
                merge_overlaps, drop_uncertain, '.txt')
            _write_detection_selection_table_file(file_path, detections)


def _load_model():

    # This is here instead of near the top of this file since it is
    # rather slow. Putting it here makes the script more responsive
    # if, say, the user just wants to display help or accidentally
    # specifies an invalid argument.
    import tensorflow as tf

    return tf.saved_model.load(_MODEL_DIR_PATH)


def _get_configuration_file_paths():

    paths = _Bunch()

    taxonomy = _TAXONOMY_DIR_PATH
    paths.species =  taxonomy / 'species_select_v5.txt'
    paths.groups =  taxonomy / 'groups_select_v5.txt'
    paths.families =  taxonomy / 'families_select_v5.txt'
    paths.orders =  taxonomy / 'orders_select_v5.txt'
    paths.ebird_taxonomy = taxonomy / 'ebird_taxonomy.csv'
    paths.group_ebird_codes = taxonomy / 'groups_ebird_codes.csv'
 
    config = _CONFIG_DIR_PATH
    paths.config = config / 'test_config.json'
    paths.calibrators = config / 'calibrators_dict.obj'

    return paths


def _process_file(
        audio_file_path, model, paths, hop_duration, threshold,
        merge_overlaps, drop_uncertain):

    p = paths
    
    # Change threshold from percentage to fraction.
    threshold /= 100

    return run_reconstructed_model.run_model_on_file(
        model, audio_file_path, MODEL_SAMPLE_RATE, MODEL_INPUT_DURATION,
        hop_duration, p.species, p.groups, p.families, p.orders,
        p.ebird_taxonomy, p.group_ebird_codes, p.calibrators, p.config,
        stream=False, threshold=threshold, quiet=True,
        model_runner=_get_model_predictions,
        postprocess_drop_singles_by_tax_level=drop_uncertain,
        postprocess_merge_overlaps=merge_overlaps,
        postprocess_retain_only_overlaps=drop_uncertain)


def _get_model_predictions(
        model, file_path, input_dur, hop_dur, target_sr=22050):
    
    start_time = time.time()

    # Get model predictions for sequence of model inputs. For each input
    # the model yields a list of four 1 x n tensors that hold order, family,
    # group, and species logits, respectively. So the result of the following
    # is a list of lists of four tensors.
    predictions = [
        model(samples) for samples in
        _generate_model_inputs(file_path, input_dur, hop_dur, target_sr)]

    # Put order, family, group and species logit tensors into their
    # own two-dimensional NumPy arrays, squeezing out the first tensor
    # dimension, which always has length one. The result is a list of four
    # two dimensional NumPy arrays, one each for order, family,
    # group, and species. The first index of each array is for input
    # and the second is for logit.
    predictions = [np.squeeze(np.array(p), axis=1) for p in zip(*predictions)]

    elapsed_time = time.time() - start_time
    _report_processing_speed(file_path, elapsed_time)

    input_count = len(predictions[0])
    return predictions, [], input_count


def _generate_model_inputs(file_path, input_dur, hop_dur, target_sr=22050):

    file_dur = librosa.get_duration(filename=file_path)

    load_size = 64        # model inputs
    load_dur = (load_size - 1) * hop_dur + input_dur
    load_hop_dur = load_size * hop_dur

    input_length = int(round(input_dur * target_sr))
    hop_length = int(round(hop_dur * target_sr))

    load_offset = 0

    while load_offset < file_dur:

        samples, _ = librosa.load(
            file_path, sr=target_sr, offset=load_offset, duration=load_dur)

        sample_count = len(samples)
        start_index = 0
        end_index = input_length

        while end_index <= sample_count:
            yield samples[start_index:end_index]
            start_index += hop_length
            end_index += hop_length

        load_offset += load_hop_dur


def _report_processing_speed(file_path, elapsed_time):
    file_dur = librosa.get_duration(filename=file_path)
    rate = file_dur / elapsed_time
    print(
        f'Processed {file_dur:.1f} seconds of audio in {elapsed_time:.1f} '
        f'seconds, {rate:.1f} times faster than real time.')


def _prep_for_output(
        output_dir_path, input_file_path, hop_duration, threshold,
        merge_overlaps, drop_uncertain, file_name_extension):

    # Get output file path.
    if output_dir_path is None:
        output_dir_path = input_file_path.parent
    file_name = (
        f'{input_file_path.stem}_{hop_duration:.2f}_{threshold:.1f}_'
        f'{merge_overlaps:d}_{drop_uncertain:d}{file_name_extension}')
    file_path = output_dir_path / file_name

    print(f'Writing output file "{file_path}"...')

    # Create parent directories if needed.
    file_path.parent.mkdir(parents=True, exist_ok=True)

    return file_path


def _write_detection_csv_file(file_path, detections):
    detections.to_csv(file_path, index=False, na_rep='')


def _write_detection_selection_table_file(file_path, detections):

    # Rename certain dataframe columns for Raven.
    columns = {
        'start_sec': 'Begin Time (s)',
        'end_sec': 'End Time (s)',
        'filename': 'Begin File'
    }
    selections = detections.rename(columns=columns)

    selections.to_csv(file_path, index=False, na_rep='', sep ='\t')


class _Bunch:
    pass