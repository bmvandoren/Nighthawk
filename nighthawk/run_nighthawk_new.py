"""Script that runs the Nighthawk NFC detector on one audio file."""


from argparse import ArgumentParser, ArgumentTypeError
from pathlib import Path
import time

import librosa

from model import Model
from numpy_postprocessor import NumPyPostprocessor as Postprocessor
# from pandas_postprocessor import PandasPostprocessor as Postprocessor
import nighthawk_new as nh


DETECTION_HOP_SIZE = 100         # percent of input frame length
DETECTION_BLOCK_LENGTH = 128     # frames


def main():

    args, audio_file_path, output_dir_path = parse_args()

    print('Creating model...')
    model = Model()

    print('Creating postprocessor...')
    taxa_of_interest = nh.get_taxa_of_interest()
    postprocessor = Postprocessor(
        model.output_taxa, taxa_of_interest, args.threshold)

    print(
        f'Running detector on audio file "{audio_file_path}" with '
        f'threshold {args.threshold}...')
    start_time = time.time()
    detections = nh.run_detector_on_file(
        audio_file_path, DETECTION_HOP_SIZE, DETECTION_BLOCK_LENGTH, model,
        postprocessor)
    elapsed_time = time.time() - start_time
    report_processing_speed(audio_file_path, elapsed_time)

    if args.csv_output:
        file_path = prep_for_output(
            output_dir_path, audio_file_path, args.threshold, '.csv')
        write_detection_csv_file(file_path, detections)

    if args.raven_output:
        file_path = prep_for_output(
            output_dir_path, audio_file_path, args.threshold, '.txt')
        write_detection_selection_table_file(file_path, detections)
   

def parse_args():
    
    parser = ArgumentParser()
    
    parser.add_argument(
        'audio_file_path',
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

    audio_file_path = Path(args.audio_file_path)

    if args.output_dir is None:
        output_dir_path = audio_file_path.parent
    else:
        output_dir_path = Path(args.output_dir)

    return args, audio_file_path, output_dir_path


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


def report_processing_speed(file_path, elapsed_time):
    file_dur = librosa.get_duration(filename=file_path)
    speed = file_dur / elapsed_time
    print(
        f'Processed {file_dur:.1f} seconds of audio in {elapsed_time:.1f} '
        f'seconds, {speed:.1f} times faster than real time.')


def prep_for_output(
        output_dir_path, audio_file_path, threshold, file_name_extension):

    # Get output file path.
    threshold_text = '' if threshold is None else f'_{int(threshold)}'
    file_name = audio_file_path.stem + threshold_text + file_name_extension
    file_path = output_dir_path / file_name

    print(f'Writing output file "{file_path}"...')

    # Create parent directories if needed.
    file_path.parent.mkdir(parents=True, exist_ok=True)

    return file_path


def write_detection_csv_file(file_path, detections):
    detections.to_csv(file_path, index=False)


def write_detection_selection_table_file(file_path, detections):

    # Rename some DataFrame columns for Raven compatibility.
    columns = {
        'start_sec': 'Begin Time (s)',
        'end_sec': 'End Time (s)',
        'filename': 'Begin File'
    }
    selections = detections.rename(columns=columns)

    selections.to_csv(file_path, index=False, sep ='\t')


if __name__ == '__main__':
    main()
