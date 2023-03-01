"""Script that runs the Nighthawk NFC detector on specified audio files."""


from argparse import ArgumentParser, ArgumentTypeError, BooleanOptionalAction
from pathlib import Path

import nighthawk as nh


def main():

    args = _parse_args()

    nh.process_files(
        args.input_file_paths, args.hop_duration, args.threshold,
        args.merge_overlaps, args.drop_uncertain, args.csv_output,
        args.raven_output, args.output_dir_path)
    

def _parse_args():
    
    parser = ArgumentParser()
    
    parser.add_argument(
        'input_file_paths',
        help='paths of audio files on which to run the detector.',
        type=Path,
        nargs='+')
    
    parser.add_argument(
        '--hop-duration',
        help=(
            f'the hop duration in seconds, a number in the range '
            f'(0, {nh.MODEL_INPUT_DURATION}]. (default: 0.2)'),
        type=_parse_hop_duration,
        default=nh.DEFAULT_HOP_DURATION)    
    
    parser.add_argument(
        '--threshold',
        help='the detection threshold, a number in [0, 100]. (default: 50)',
        type=_parse_threshold,
        default=nh.DEFAULT_THRESHOLD)
    
    parser.add_argument(
        '--merge-overlaps',
        help='merge overlapping detections.',
        action=BooleanOptionalAction,
        default=nh.DEFAULT_MERGE_OVERLAPS)

    parser.add_argument(
        '--drop-uncertain',
        help='apply postprocessing steps to drop less certain detections.',
        action=BooleanOptionalAction,
        default=nh.DEFAULT_DROP_UNCERTAIN)

    parser.add_argument(
        '--csv-output',
        help='output detections to a CSV file.',
        action=BooleanOptionalAction,
        default=nh.DEFAULT_CSV_OUTPUT)

    parser.add_argument(
        '--raven-output',
        help='output detections to a Raven selection table file.',
        action=BooleanOptionalAction,
        default=nh.DEFAULT_RAVEN_OUTPUT)

    parser.add_argument(
        '--output-dir',
        help=(
            'directory in which to write output files. (default: input '
            'file directories)'),
        type=Path,
        dest='output_dir_path',
        default=nh.DEFAULT_OUTPUT_DIR_PATH)
    
    return parser.parse_args()


def _parse_hop_duration(value):
    
    try:
        hop = float(value)
    except Exception:
        _handle_hop_duration_error(value)

    if hop <= 0 or hop > nh.MODEL_INPUT_DURATION:
        _handle_hop_duration_error(value)
    
    return hop


def _handle_hop_duration_error(value):
    raise ArgumentTypeError(
        f'Bad hop duration "{value}". Hop duration must be '
        f'a number in the range (0, {nh.MODEL_INPUT_DURATION}].')    


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
    

if __name__ == '__main__':
    main()
