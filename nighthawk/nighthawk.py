"""Script that runs Nighthawk NFC detector on one audio file."""


from argparse import ArgumentParser, ArgumentTypeError
from pathlib import Path
import time

import librosa
import numpy as np
import tensorflow as tf

import run_reconstructed_model


PACKAGE_DIR_PATH = Path(__file__).parent
MODEL_DIR_PATH = PACKAGE_DIR_PATH / 'saved_model_with_preprocessing'
TAXONOMY_DIR_PATH = PACKAGE_DIR_PATH / 'taxonomy'
CONFIG_DIR_PATH = PACKAGE_DIR_PATH / 'test_config'

MODEL_SAMPLE_RATE = 22050       # Hz
MODEL_INPUT_DURATION = 1        # seconds


# TODO: Consider different `librosa.load` resampling algorithms.


def main():

    args = _parse_args()

    process_files(
        args.input_file_paths, args.threshold, args.hop_duration,
        args.merge_overlaps, args.drop_uncertain, args.csv_output,
        args.raven_output, args.output_dir_path)
    

def _parse_args():
    
    parser = ArgumentParser()
    
    parser.add_argument(
        'input_file_paths',
        help='paths of audio files on which to run the detector.',
        type=Path,
        nargs='*')
    
    parser.add_argument(
        '--threshold',
        help='the detection threshold, a number in [0, 100]. Default is 50.',
        type=_parse_threshold,
        default=50)
    
    parser.add_argument(
        '--hop-duration',
        help=(
            f'the hop duration in seconds, a number in the range '
            f'(0, {MODEL_INPUT_DURATION}]. Default is 0.2.'),
        type=_parse_hop_duration,
        default=0.2)    
    
    parser.add_argument(
        '--merge-overlaps',
        help='merge overlapping detections in output (the default).',
        action='store_true',
        default=True)

    parser.add_argument(
        '--no-merge-overlaps',
        help='do not merge overlapping detections in output.',
        action='store_false',
        dest='merge_overlaps')    

    parser.add_argument(
        '--drop-uncertain',
        help=(
            'apply postprocessing steps to only retain more confident '
            'predictions (the default).'),
        action='store_true',
        default=True)

    parser.add_argument(
        '--no-drop-uncertain',
        help=(
            'do not apply postprocessing steps that retain more confident '
            'predictions.'),
        action='store_false',
        dest='drop_uncertain')    
    
    parser.add_argument(
        '--csv-output',
        help='output detections to a CSV file (the default).',
        action='store_true',
        default=True)

    parser.add_argument(
        '--no-csv-output',
        help='do not output detections to a CSV file.',
        action='store_false',
        dest='csv_output')

    parser.add_argument(
        '--raven-output',
        help='output detections to a Raven selection table file.',
        action='store_true')

    parser.add_argument(
        '--no-raven-output',
        help=(
            'do not output detections to a Raven selection table file '
            '(the default).'),
        action='store_false',
        dest='raven_output')
        
    parser.add_argument(
        '--output-dir',
        help=(
            'directory in which to write output files. Default is '
            'input file directory.'),
        type=Path,
        dest='output_dir_path',
        default=None)
    
    return parser.parse_args()


def _parse_threshold(value):
    
    try:
        threshold = float(value)
    except Exception:
        _handle_threshold_error(value)
    
    if threshold < 0 or threshold > 100:
        _handle_threshold_error(value)
    
    return threshold
    
    
def _handle_threshold_error(value):
    raise ArgumentTypeError(
        f'Bad detection threshold "{value}". Threshold must be '
        f'a number in the range [0, 100].')
    

def _parse_hop_duration(value):
    
    try:
        hop = float(value)
    except Exception:
        _handle_hop_duration_error(value)

    if hop <= 0 or hop > MODEL_INPUT_DURATION:
        _handle_hop_duration_error(value)
    
    return hop


def _handle_hop_duration_error(value):
    raise ArgumentTypeError(
        f'Bad hop duration "{value}". Hop duration must be '
        f'a number in the range (0, {MODEL_INPUT_DURATION}].')    


def process_files(
        input_file_paths, threshold, hop_duration, merge_overlaps,
        drop_uncertain, csv_output, raven_output, output_dir_path):
    
    print('Loading detector model...')
    model = _load_model()

    print('Getting detector configuration file paths...')
    config_file_paths = _get_configuration_file_paths()

    for input_file_path in input_file_paths:

        # Make sure input file path is absolute for messages.
        input_file_path = input_file_path.absolute()

        print(
            f'Running detector on audio file "{input_file_path}" with '
            f'threshold {threshold}...')
        
        detections = _process_file(
            input_file_path, threshold, hop_duration, model,
            config_file_paths, merge_overlaps, drop_uncertain)

        if csv_output:
            file_path = _prep_for_output(
                output_dir_path, input_file_path, threshold, '.csv')
            _write_detection_csv_file(file_path, detections)

        if raven_output:
            file_path = _prep_for_output(
                output_dir_path, input_file_path, threshold, '.txt')
            _write_detection_selection_table_file(file_path, detections)


def _load_model():
    return tf.saved_model.load(MODEL_DIR_PATH)


def _get_configuration_file_paths():

    paths = _Bunch()

    taxonomy = TAXONOMY_DIR_PATH
    paths.species =  taxonomy / 'species_select_v5.txt'
    paths.groups =  taxonomy / 'groups_select_v5.txt'
    paths.families =  taxonomy / 'families_select_v5.txt'
    paths.orders =  taxonomy / 'orders_select_v5.txt'
    paths.ebird_taxonomy = taxonomy / 'ebird_taxonomy.csv'
    paths.group_ebird_codes = taxonomy / 'groups_ebird_codes.csv'
 
    config = CONFIG_DIR_PATH
    paths.config = config / 'test_config.json'
    paths.calibrators = config / 'calibrators_dict.obj'

    return paths


def _process_file(
        audio_file_path, threshold, hop_duration, model, paths,
        merge_overlaps, drop_uncertain):

    # Change threshold from percentage to fraction.
    threshold /= 100

    p = paths
    
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
        output_dir_path, input_file_path, threshold, file_name_extension):

    # Get output file path.
    if output_dir_path is None:
        output_dir_path = input_file_path.parent
    threshold_text = '' if threshold is None else f'_{int(threshold)}'
    file_name = input_file_path.stem + threshold_text + file_name_extension
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


if __name__ == '__main__':
    main()
