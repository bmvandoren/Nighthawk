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
INPUT_HOP_DURATION = 1          # seconds


# TODO: Create Python package.
# TODO: Create Vesper detector that invokes this script.
# TODO: Look into `librosa.load` sample value range.
# TODO: Consider different `librosa.load` resampling algorithms.


def main():

    args, input_file_path, output_dir_path = parse_args()
    
    print('Loading detector model...')
    model = load_model()

    print('Getting detector configuration file paths...')
    paths = get_configuration_file_paths()

    print(
        f'Running detector on audio file "{input_file_path}" with '
        f'threshold {args.threshold}...')
    detections = process_file(input_file_path, args.threshold, model, paths)

    if args.csv_output:
        file_path = prep_for_output(
            output_dir_path, input_file_path, args.threshold, '.csv')
        write_detection_csv_file(file_path, detections)

    if args.raven_output:
        file_path = prep_for_output(
            output_dir_path, input_file_path, args.threshold, '.txt')
        write_detection_selection_table_file(file_path, detections)


def parse_args():
    
    parser = ArgumentParser()
    
    parser.add_argument(
        'input_file_path',
        help='path of audio file on which to run the detector.')
    
    parser.add_argument(
        '--threshold',
        help='the detection threshold, a number in [0, 100]. Default is 50.',
        type=parse_threshold,
        default=50)
    
    parser.add_argument(
        '--output-dir',
        help=(
            'directory in which to write output files. Default is '
            'input file directory.'),
        default=None)
    
    parser.add_argument(
        '--csv-output',
        help=('output detections to a CSV file (the default).'),
        action='store_true',
        default=True)

    parser.add_argument(
        '--no-csv-output',
        help=('do not output detections to a CSV file.'),
        action='store_false',
        dest='csv_output')

    parser.add_argument(
        '--raven-output',
        help=('output detections to a Raven selection table file.'),
        action='store_true')

    parser.add_argument(
        '--no-raven-output',
        help=(
            'do not output detections to a Raven selection table file '
            '(the default).'),
        action='store_false',
        dest='raven_output')

    args = parser.parse_args()

    input_file_path = Path(args.input_file_path)

    if args.output_dir is None:
        output_dir_path = input_file_path.parent
    else:
        output_dir_path = Path(args.output_dir)

    return args, input_file_path, output_dir_path


def parse_threshold(value):
    
    try:
        threshold = float(value)
    except Exception:
        handle_threshold_error(value)
    
    if threshold < 0 or threshold > 100:
        handle_threshold_error(value)
    
    return threshold
    
    
def handle_threshold_error(value):
    raise ArgumentTypeError(
        f'Bad detection threshold "{value}". Threshold must be '
        f'a number in the range [0, 100].')


def load_model():
    return tf.saved_model.load(MODEL_DIR_PATH)


def get_configuration_file_paths():

    paths = Bunch()

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


def process_file(audio_file_path, threshold, model, paths):

    # Change threshold from percentage to fraction.
    threshold /= 100

    p = paths

    return run_reconstructed_model.run_model_on_file(
        model, audio_file_path, MODEL_SAMPLE_RATE, MODEL_INPUT_DURATION,
        INPUT_HOP_DURATION, p.species, p.groups, p.families, p.orders,
        p.ebird_taxonomy, p.group_ebird_codes, p.calibrators, p.config,
        stream=False, threshold=threshold, quiet=True,
        model_runner=get_model_predictions)


def get_model_predictions(
        model, file_path, input_dur, hop_dur, target_sr=22050):
    
    start_time = time.time()

    # Get model predictions for sequence of model inputs. For each input
    # the model yields a list of four 1 x n tensors that hold order, family,
    # group, and species logits, respectively. So the result of the following
    # is a list of lists of four tensors.
    predictions = [
        model(samples) for samples in
        generate_model_inputs(file_path, input_dur, hop_dur, target_sr)]

    # Put order, family, group and species logit tensors into their
    # own two-dimensional NumPy arrays, squeezing out the first tensor
    # dimension, which always has length one. The result is a list of four
    # two dimensional NumPy arrays, one each for order, family,
    # group, and species. The first index of each array is for input
    # and the second is for logit.
    predictions = [np.squeeze(np.array(p), axis=1) for p in zip(*predictions)]

    elapsed_time = time.time() - start_time
    report_processing_speed(file_path, elapsed_time)

    input_count = len(predictions[0])
    return predictions, [], input_count


def generate_model_inputs(file_path, input_dur, hop_dur, target_sr=22050):

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


def report_processing_speed(file_path, elapsed_time):
    file_dur = librosa.get_duration(filename=file_path)
    rate = file_dur / elapsed_time
    print(
        f'Processed {file_dur:.1f} seconds of audio in {elapsed_time:.1f} '
        f'seconds, {rate:.1f} times faster than real time.')


def prep_for_output(
        output_dir_path, input_file_path, threshold, file_name_extension):

    # Get output file path.
    threshold_text = '' if threshold is None else f'_{int(threshold)}'
    file_name = input_file_path.stem + threshold_text + file_name_extension
    file_path = output_dir_path / file_name

    print(f'Writing output file "{file_path}"...')

    # Create parent directories if needed.
    file_path.parent.mkdir(parents=True, exist_ok=True)

    return file_path


def write_detection_csv_file(file_path, detections):
    detections.to_csv(file_path, index=False, na_rep='')


def write_detection_selection_table_file(file_path, detections):

    # Rename certain dataframe columns for Raven.
    columns = {
        'start_sec': 'Begin Time (s)',
        'end_sec': 'End Time (s)',
        'filename': 'Begin File'
    }
    selections = detections.rename(columns=columns)

    selections.to_csv(file_path, index=False, na_rep='', sep ='\t')


class Bunch:
    pass


if __name__ == '__main__':
    main()
