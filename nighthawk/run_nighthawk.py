"""Script that runs the Nighthawk NFC detector on specified audio files."""


from argparse import ArgumentParser, ArgumentTypeError, BooleanOptionalAction
from pathlib import Path

import nighthawk as nh


def main():

    args = _parse_args()
    
    nh.run_detector_on_files(
        args.input_file_paths, args.hop_size, args.threshold,
        args.merge_overlaps, args.drop_uncertain, args.csv_output,
        args.raven_output, args.audacity_output, args.duration_output,
        args.output_dir_path, args.ap_mask,
        args.tax_output, args.gzip_output, args.calibration,
        args.quiet, batch_size=args.batch_size,
        model_name=args.model,
        model_version=args.model_version,
        model_path=args.model_path,
        model_repo_url=args.model_repo_url,
        cache_dir=args.cache_dir,
        offline=args.offline)

def _parse_args():
    
    parser = ArgumentParser()
    
    parser.add_argument(
        'input_file_paths',
        metavar='input_file_path',
        help=(
            'path of audio file(s) on which to run the detector. '
            'Accepts directories and wildcards.'),
        type=Path,
        nargs='+')
    
    parser.add_argument(
        '--hop-size',
        help=(
            f'the hop size as a percentage of the model input '
            f'duration, a number in the range (0, 100]. (default: 20)'),
        type=_parse_hop_size,
        default=nh.DEFAULT_HOP_SIZE)    
    
    parser.add_argument(
        '--threshold',
        help='the detection threshold, a number in [0, 100]. (default: 80)',
        type=_parse_threshold,
        default=nh.DEFAULT_THRESHOLD)

    parser.add_argument(
        '--ap-mask',
        help='the AP mask threshold, a number in [0, 1]. (default: 0.7)',
        type=_parse_ap_mask,
        default=nh.DEFAULT_AP_MASK_THRESHOLD)    
    
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
        '--calibration',
        help='calibrate model outputs.',
        action=BooleanOptionalAction,
        default=nh.DEFAULT_DO_CALIBRATION)   

    parser.add_argument(
        '--quiet',
        help='Mask unnecessary console messages.',
        action=BooleanOptionalAction,
        default=nh.DEFAULT_QUIET)       
    
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
        '--audacity-output',
        help='output detections to an Audacity label file.',
        action=BooleanOptionalAction,
        default=nh.DEFAULT_AUDACITY_OUTPUT)  

    parser.add_argument(
        '--duration-output',
        help='output file duration in seconds to a txt file.',
        action=BooleanOptionalAction,
        default=nh.DEFAULT_DURATION_OUTPUT)      

    parser.add_argument(
        '--gzip-output',
        help='gzip all output files.',
        action=BooleanOptionalAction,
        default=nh.DEFAULT_GZIP_OUTPUT)  

    parser.add_argument(
        '--tax-output',
        help='save separate taxonomic output files.',
        action=BooleanOptionalAction,
        default=nh.DEFAULT_RETURN_TAX_LEVEL_PREDICTIONS)      

    parser.add_argument(
        '--output-dir',
        help=(
            'directory in which to write output files. (default: input '
            'file directories)'),
        type=Path,
        dest='output_dir_path',
        default=nh.DEFAULT_OUTPUT_DIR_PATH)

    parser.add_argument(
        '--batch-size',
        help=(
            f'number of 1-second windows to feed to the model per call. '
            f'Larger values improve throughput on GPU and CPU. (default: {nh.DEFAULT_BATCH_SIZE})'),
        type=int,
        default=nh.DEFAULT_BATCH_SIZE)

    parser.add_argument(
        '--model',
        help=(
            f'name of the model to use (e.g. "americas", "europe"). '
            f'(default: {nh.DEFAULT_MODEL_NAME})'),
        dest='model',
        default=nh.DEFAULT_MODEL_NAME)

    parser.add_argument(
        '--model-version',
        help=(
            f'version of the model to use, or "latest" to auto-select the '
            f'newest published version. Pin a version (e.g. "0.4.0") for '
            f'reproducible results. (default: {nh.DEFAULT_MODEL_VERSION})'),
        dest='model_version',
        default=nh.DEFAULT_MODEL_VERSION)

    parser.add_argument(
        '--model-path',
        help=(
            'path to a local model bundle directory or .tar.gz tarball. '
            'When given, no network access is performed and --model / '
            '--model-version are used only for display/validation.'),
        dest='model_path',
        type=Path,
        default=None)

    parser.add_argument(
        '--model-repo-url',
        help=(
            'base URL of the model repository '
            f'(default: {nh.DEFAULT_MODEL_REPO_URL!r}).'),
        dest='model_repo_url',
        default=nh.DEFAULT_MODEL_REPO_URL)

    parser.add_argument(
        '--cache-dir',
        help=(
            'directory used to cache downloaded models. '
            'Defaults to the platform user cache dir (e.g. ~/.cache/nighthawk). '
            'Override with env NIGHTHAWK_CACHE_DIR.'),
        dest='cache_dir',
        type=Path,
        default=None)

    parser.add_argument(
        '--offline',
        help=(
            'disable all network access. Serves models from the local cache; '
            'raises an error if the requested model is not cached. '
            'Also controlled via env NIGHTHAWK_OFFLINE=1.'),
        action=BooleanOptionalAction,
        default=False)

    return parser.parse_args()


def _parse_hop_size(value):
    
    try:
        hop = float(value)
    except Exception:
        _handle_hop_size_error(value)

    if hop <= 0 or hop > 100:
        _handle_hop_size_error(value)
    
    return hop


def _handle_hop_size_error(value):
    raise ArgumentTypeError(
        f'Bad hop size "{value}". Hop size must be a number in the '
        f'range (0, 100].')    


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



def _parse_ap_mask(value):
    
    try:
        ap_mask = float(value)
    except Exception:
        _handle_ap_mask_error(value)
    
    if ap_mask < 0 or ap_mask > 1:
        _handle_ap_mask_error(value)
    
    return ap_mask
    
    
def _handle_ap_mask_error(value):
    raise ArgumentTypeError(
        f'Bad AP mask threshold "{value}". Value must be '
        f'a number in the range [0, 1].')
    

if __name__ == '__main__':
    main()
