"""Functions and constants for the Nighthawk NFC detector."""


from pathlib import Path
import time

import librosa
import numpy as np

import nighthawk.run_reconstructed_model as run_reconstructed_model


MODEL_SAMPLE_RATE = 22050         # Hz
MODEL_INPUT_DURATION = 1          # seconds

DEFAULT_HOP_SIZE = 20             # percent of model input duration
DEFAULT_THRESHOLD = 80            # percent
DEFAULT_AP_MASK_THRESHOLD = 0.7
DEFAULT_MERGE_OVERLAPS = True
DEFAULT_DROP_UNCERTAIN = True
DEFAULT_CSV_OUTPUT = True
DEFAULT_RAVEN_OUTPUT = False
DEFAULT_AUDACITY_OUTPUT = False
DEFAULT_DURATION_OUTPUT = False
DEFAULT_OUTPUT_DIR_PATH = None
DEFAULT_RETURN_TAX_LEVEL_PREDICTIONS = False
DEFAULT_GZIP_OUTPUT = False
DEFAULT_DO_CALIBRATION = True
DEFAULT_QUIET = False

_PACKAGE_DIR_PATH = Path(__file__).parent
_MODEL_DIR_PATH = _PACKAGE_DIR_PATH / 'saved_model_with_preprocessing'
_TAXONOMY_DIR_PATH = _PACKAGE_DIR_PATH / 'taxonomy'
_CONFIG_DIR_PATH = _PACKAGE_DIR_PATH / 'test_config'


def run_detector_on_files(
        input_file_paths, hop_size=DEFAULT_HOP_SIZE,
        threshold=DEFAULT_THRESHOLD, merge_overlaps=DEFAULT_MERGE_OVERLAPS,
        drop_uncertain=DEFAULT_DROP_UNCERTAIN, csv_output=DEFAULT_CSV_OUTPUT,
        raven_output=DEFAULT_RAVEN_OUTPUT,
        audacity_output=DEFAULT_AUDACITY_OUTPUT,
        duration_output=DEFAULT_DURATION_OUTPUT,
        output_dir_path=DEFAULT_OUTPUT_DIR_PATH,
        mask_ap_threshold=DEFAULT_AP_MASK_THRESHOLD,
        return_tax_level_detections=DEFAULT_RETURN_TAX_LEVEL_PREDICTIONS,
        gzip_output=DEFAULT_GZIP_OUTPUT,
        do_calibration=DEFAULT_DO_CALIBRATION,
        quiet=DEFAULT_QUIET):
    
    input_file_paths = _expand_paths(input_file_paths)
    file_count = len(input_file_paths)
    if file_count == 0:
        print('No input files found.')
        return

    print('Loading detector model...')
    model = _load_model()

    print('Getting detector configuration file paths...')
    config_file_paths = _get_configuration_file_paths()

    for i, input_file_path in enumerate(input_file_paths):

        # Make sure input file path is absolute for messages.
        input_file_path = input_file_path.absolute()

        if len(input_file_paths) == 1:
            print(f'Running detector on audio file "{input_file_path}"...')
        else:
            print(
                f'Running detector on audio file {i + 1} of {file_count}: '
                f'"{input_file_path}"...')
        
        detections, detect_df_dict, diagnosis, drop_uncertain_used, short_no_drop = (
            _run_detector_on_file(
            input_file_path, model, config_file_paths, hop_size, threshold,
            merge_overlaps, drop_uncertain, mask_ap_threshold,
            return_tax_level_detections, do_calibration, quiet))

        if detections.empty:
            _explain_empty_detections(
                input_file_path, hop_size, threshold, drop_uncertain_used,
                quiet, diagnosis, short_no_drop)

        if duration_output:
            output_file_path = _prep_for_output(
                input_file_path, output_dir_path, '.txt',  descriptor='duration', gzip=False)
            input_file_duration_s = librosa.get_duration(path=input_file_path)
            _write_duration_txt_file(output_file_path, input_file_duration_s)
            

        if csv_output:
            output_file_path = _prep_for_output(
                input_file_path, output_dir_path, '.csv', gzip=gzip_output)
            _write_detection_csv_file(output_file_path, detections)

            if return_tax_level_detections:
                for tax_level, detect_df in detect_df_dict.items():
                    output_file_path = _prep_for_output(
                        input_file_path, output_dir_path, '.csv',  descriptor=tax_level, gzip=gzip_output)
                    _write_detection_csv_file(output_file_path, detect_df)

        if raven_output:
            output_file_path = _prep_for_output(
                input_file_path, output_dir_path, '.txt',  descriptor='raven', gzip=gzip_output)
            _write_detection_selection_table_file(output_file_path, detections)

        if audacity_output:
            output_file_path = _prep_for_output(
                input_file_path, output_dir_path, '.txt',  descriptor='audacity', gzip=gzip_output)
            _write_detection_audacity_label_file(output_file_path, detections)


def _expand_paths(paths):

    """
    If any wildcards or directories are supplied, expand them into a
    list of paths.
    """

    expanded_paths = []

    for path in paths:

        if path.is_dir():
            expanded_paths.extend(path.glob('*'))

        elif path.is_absolute():
            expanded_paths.extend(path.parent.glob(path.name))

        else:
            expanded_paths.extend(Path.cwd().glob(str(path)))

    return expanded_paths
    

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
    paths.species =  taxonomy / 'species.txt'
    paths.groups =  taxonomy / 'groups.txt'
    paths.families =  taxonomy / 'families.txt'
    paths.orders =  taxonomy / 'orders.txt'
    paths.ebird_taxonomy = taxonomy / 'ebird_taxonomy.csv'
    paths.group_ebird_codes = taxonomy / 'groups_ebird_codes.csv'
    paths.ibp_codes = taxonomy / 'IBP-AOS-LIST21.csv'

    config = _CONFIG_DIR_PATH
    paths.config = config / 'test_config.json'
    paths.test_set_performance = config / 'test_set_performance'
    paths.calibrators_from_probs = config / 'probability_calibrations.csv'
    paths.calibrators_from_logits = config / 'probability_calibrations_logistic_fromlogits.csv'

    return paths


def _stride_seconds(hop_size):
    """Sliding-window stride in seconds (hop_size is percent of model window)."""

    return hop_size / 100 * MODEL_INPUT_DURATION


def _min_duration_for_two_windows(hop_size):
    """Minimum duration for two overlapping analysis windows at this hop size."""

    return MODEL_INPUT_DURATION + _stride_seconds(hop_size)


def _resolve_drop_uncertain_for_file(audio_file_path, hop_size, drop_uncertain, quiet):
    """Disable --drop-uncertain on short recordings that cannot satisfy it.

    Returns (effective_drop_uncertain, relaxed_for_short_recording).
    """

    if not drop_uncertain:
        return drop_uncertain, False

    file_dur = librosa.get_duration(path=audio_file_path)
    min_dur = _min_duration_for_two_windows(hop_size)
    if file_dur >= min_dur:
        return drop_uncertain, False

    if not quiet:
        stride = _stride_seconds(hop_size)
        print(
            f'NOTE: Recording is short ({file_dur:.3f} s < '
            f'{MODEL_INPUT_DURATION} + {stride:.3f} s stride), applying '
            f'--no-drop-uncertain.')
    return False, True


def _run_detector_on_file(
        audio_file_path, model, paths, hop_size, threshold, merge_overlaps,
        drop_uncertain,mask_ap_threshold,return_tax_level_detections,do_calibration,
        quiet):

    p = paths

    calibrate_from_logits = True
    if do_calibration:
        if p.calibrators_from_logits.exists():
            print('Calibrating from logits.')            
            calib = p.calibrators_from_logits
        elif p.calibrators_from_probs.exists():
            print('Calibrating from probabilities.')
            calib = p.calibrators_from_probs
            calibrate_from_logits = False
        else:
            print('No calibration file found, proceeding without calibration.')
            calib = None
    else:
        calib = None      
    
    # Change hop size from percentage to seconds.
    hop_dur = _stride_seconds(hop_size)

    # Change threshold from percentage to fraction.
    threshold /= 100

    drop_uncertain, short_no_drop = _resolve_drop_uncertain_for_file(
        audio_file_path, hop_size, drop_uncertain, quiet)

    merged_df, detect_df_dict, diagnosis = run_reconstructed_model.run_model_on_file(
        model, audio_file_path, MODEL_SAMPLE_RATE, MODEL_INPUT_DURATION,
        hop_dur, p.species, p.groups, p.families, p.orders,
        p.ebird_taxonomy, p.group_ebird_codes, calib, calibrate_from_logits, 
        p.config, stream=False, threshold=threshold, quiet=quiet,
        model_runner=_get_model_predictions,
        postprocess_drop_singles_by_tax_level=drop_uncertain,
        postprocess_merge_overlaps=merge_overlaps,
        postprocess_retain_only_overlaps=drop_uncertain,
        mask_output_ap_threshold=mask_ap_threshold,
        test_set_performance_dir=p.test_set_performance,
        return_tax_level_detections=return_tax_level_detections)

    return merged_df, detect_df_dict, diagnosis, drop_uncertain, short_no_drop


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

    if not predictions:
        elapsed_time = time.time() - start_time
        _report_processing_speed(file_path, elapsed_time)
        return [[], [], [], []], [], 0

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

    file_dur = librosa.get_duration(path=file_path)

    load_size = 64        # model inputs
    load_dur = (load_size - 1) * hop_dur + input_dur
    load_hop_dur = load_size * hop_dur

    input_length = int(round(input_dur * target_sr))
    hop_length = int(round(hop_dur * target_sr))

    load_offset = 0

    while load_offset < file_dur:

        samples, _ = librosa.load(
            file_path, sr=target_sr, offset=load_offset, duration=load_dur,
            res_type='soxr_hq')

        sample_count = len(samples)
        if sample_count < input_length:
            if load_offset == 0:
                print(
                    f'Warning: audio duration ({file_dur:.3f} s) is less than '
                    f'the model input duration ({input_dur} s); padding with '
                    f'zeros on both sides to center the audio.')
            samples = _pad_audio_center(samples, input_length)
            sample_count = len(samples)

        start_index = 0
        end_index = input_length

        while end_index <= sample_count:
            yield samples[start_index:end_index]
            start_index += hop_length
            end_index += hop_length

        load_offset += load_hop_dur


def _pad_audio_center(samples, min_length):
    """Pad with zeros to at least min_length samples, keeping audio centered."""

    sample_count = len(samples)
    if sample_count >= min_length:
        return samples

    pad_total = min_length - sample_count
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(samples, (pad_left, pad_right), mode='constant')


def _explain_empty_detections(
        file_path, hop_size, threshold, drop_uncertain, quiet=False,
        diagnosis=None, short_no_drop=False):

    file_path = Path(file_path)
    file_dur = librosa.get_duration(path=file_path)

    if quiet:
        summary = _empty_detection_summary(diagnosis, threshold)
        print(
            f'No detections for "{file_path.name}" ({file_dur:.3f} s): {summary} '
            f'Rerun without --quiet for details.')
        return

    print()
    print(f'No detections for "{file_path.name}".')
    print(f'  Audio duration: {file_dur:.3f} s')
    if diagnosis is not None:
        print(f'  Analysis windows: {diagnosis["n_windows"]}')
    print('  The output CSV contains only a header row.')
    print()

    if diagnosis is None:
        print('  Could not determine a specific reason.')
        print(f'  Try lowering --threshold (current: {threshold:g}%).')
        print()
        return

    _print_empty_detection_reason(
        diagnosis, threshold, drop_uncertain, file_dur, hop_size, short_no_drop)
    _print_empty_detection_suggestions(
        diagnosis, threshold, drop_uncertain, file_dur, hop_size, short_no_drop)
    print()


def _empty_detection_summary(diagnosis, threshold):
    if diagnosis is None:
        return f'no detections (threshold {threshold:g}%).'
    reason = diagnosis['reason']
    thresh = diagnosis['threshold_pct']
    if reason == 'below_threshold':
        return (
            f'no taxon exceeded {thresh:g}% '
            f'(highest {diagnosis["max_prob_pct"]:.1f}%).')
    if reason == 'drop_uncertain':
        return (
            f'{diagnosis["n_threshold_hits"]} above {thresh:g}% removed by '
            f'--drop-uncertain.')
    if reason == 'no_windows':
        return 'no analysis windows were generated.'
    if reason == 'retain_only_overlaps':
        return f'detections above {thresh:g}% removed after merge filtering.'
    return f'no detections passed postprocessing (threshold {thresh:g}%).'


def _print_empty_detection_reason(
        diagnosis, threshold, drop_uncertain, file_dur, hop_size,
        short_no_drop=False):

    reason = diagnosis['reason']
    thresh = diagnosis['threshold_pct']

    if short_no_drop:
        print(
            '  Note: --no-drop-uncertain was applied automatically because '
            'this recording is shorter than 1 s + stride.')

    if file_dur < MODEL_INPUT_DURATION:
        print(
            f'  Note: audio was shorter than {MODEL_INPUT_DURATION} s and was '
            'zero-padded on both sides before analysis.')

    if reason == 'no_windows':
        print('  Reason: no analysis windows could be generated from this file.')
        return

    if reason == 'below_threshold':
        print(f'  Reason: no taxon exceeded the detection threshold ({thresh:g}%).')
        if diagnosis['max_prob'] > 0:
            print(
                f'  Highest calibrated probability: {diagnosis["max_prob_pct"]:.1f}% '
                f'({diagnosis["max_taxon"]}, {diagnosis["max_tax_level"]} level, '
                f'window starting at {diagnosis["max_start_sec"]:.1f} s).')
        else:
            print('  Highest calibrated probability: 0%.')
        if short_no_drop or file_dur < MODEL_INPUT_DURATION:
            print(
                '  This is common for clips much shorter than 1 s: the model '
                'still analyzes a full 1 s window, mostly silence after padding, '
                'so confidence usually stays below the default threshold.')
        return

    if reason == 'drop_uncertain':
        print(
            f'  Reason: {diagnosis["n_threshold_hits"]} taxon-window detection(s) '
            f'were above {thresh:g}% across '
            f'{diagnosis["n_windows_above_threshold"]} window(s), but '
            f'--drop-uncertain removed all of them.')
        detail = diagnosis.get('drop_uncertain_detail')
        if detail == 'single_window':
            print(
                f'  Detail: only {diagnosis["n_windows"]} analysis window was '
                'produced, so two overlapping agreeing windows were not possible.')
        elif detail == 'single_window_above_threshold':
            print(
                '  Detail: only one window had taxa above threshold; '
                '--drop-uncertain requires at least two overlapping windows '
                'with the same taxon.')
        else:
            print(
                '  Detail: multiple windows were analyzed, but no two '
                'overlapping windows had the same taxon above threshold.')
        print('  Detections removed by --drop-uncertain:')
        for hit in diagnosis['threshold_hits'][:8]:
            print(
                f'    - {hit["taxon"]} ({hit["tax_level"]}) at '
                f'{hit["start_sec"]:.1f}-{hit["end_sec"]:.1f} s, '
                f'prob {hit["prob"] * 100:.1f}%')
        if diagnosis['n_threshold_hits'] > 8:
            print(f'    ... and {diagnosis["n_threshold_hits"] - 8} more.')
        return

    if reason == 'retain_only_overlaps':
        print(
            f'  Reason: {diagnosis["n_threshold_hits"]} taxon-window detection(s) '
            f'were above {thresh:g}%, but they were removed because merged '
            'detections did not span long enough overlapping windows.')
        return

    if reason == 'taxonomic_merge':
        print(
            f'  Reason: {diagnosis["n_threshold_hits"]} taxon-window detection(s) '
            f'were above {thresh:g}%, but none survived taxonomic consistency '
            'merging across levels.')
        return

    print(
        f'  Reason: detections above {thresh:g}% were removed during '
        'postprocessing.')


def _print_empty_detection_suggestions(
        diagnosis, threshold, drop_uncertain, file_dur, hop_size,
        short_no_drop=False):

    if diagnosis is None:
        return

    reason = diagnosis['reason']
    min_dur_for_two_windows = _min_duration_for_two_windows(hop_size)

    print('  Suggestions:')
    if reason == 'below_threshold':
        print('    - Confirm the clip contains a clear bird vocalization.')
        if file_dur < MODEL_INPUT_DURATION:
            print(
                '    - Clips under 1 s are padded with silence; use a longer '
                'clip (ideally >= 1 s of vocalization) for reliable detections.')
        print(
            f'    - Lower --threshold (current: {threshold:g}%; highest signal was '
            f'{diagnosis["max_prob_pct"]:.1f}%).')
    elif reason == 'drop_uncertain' and not short_no_drop:
        if diagnosis['n_windows'] == 1 or file_dur < min_dur_for_two_windows:
            print(
                f'    - Use a clip at least {min_dur_for_two_windows:.1f} s long, or '
                'run with --no-drop-uncertain.')
        else:
            print(
                '    - Run with --no-drop-uncertain to keep single-window '
                'detections, or use a longer clip with repeated calls.')
    elif reason == 'retain_only_overlaps':
        print(
            '    - Use a longer clip, or run with --no-merge-overlaps and '
            '--no-drop-uncertain.')
    else:
        print('    - Confirm the clip contains a bird vocalization.')
        if drop_uncertain and not short_no_drop:
            print('    - Try --no-drop-uncertain.')
        print(f'    - Lower --threshold if appropriate (current: {threshold:g}%).')


def _report_processing_speed(file_path, elapsed_time):
    file_dur = librosa.get_duration(path=file_path)
    rate = file_dur / elapsed_time
    print(
        f'Processed {file_dur:.1f} seconds of audio in {elapsed_time:.1f} '
        f'seconds, {rate:.1f} times faster than real time.')


def _prep_for_output(input_file_path, output_dir_path, file_name_suffix, 
                     descriptor='detections',gzip=False):

    # Get output file path.
    if output_dir_path is None:
        output_dir_path = input_file_path.parent
    file_name = f'{input_file_path.stem}_{descriptor}{file_name_suffix}'
    file_path = output_dir_path / file_name

    # Create parent directories if needed.
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # add gzip extension if we are gzipping
    if gzip:
        file_path = file_path.parent / (file_path.name + '.gz')

    print(f'Writing output file "{file_path}"...')

    return file_path


def _write_detection_csv_file(file_path, detections):
    detections.to_csv(file_path, index=False, na_rep='')

def _write_duration_txt_file(file_path, duration):
     # write path and duration
    text_file = open(file_path, "w")
    n = text_file.write("%s\n" % (duration))
    text_file.close()


def _write_detection_selection_table_file(file_path, detections):
    

    # Rename certain dataframe columns for Raven.
    columns = {
        'start_sec': 'Begin Time (s)',
        'end_sec': 'End Time (s)',
        'filename': 'Begin File'
    }
    selections = detections.rename(columns=columns)
    
    # insert low/high frequency columns after Time columns
    selections.insert(loc = 2,
          column = 'Low Freq (Hz)',
          value = 0)
    selections.insert(loc = 3,
          column = 'High Freq (Hz)',
          value = 11025)    

    selections.to_csv(file_path, index=False, na_rep='', sep ='\t')


def _write_detection_audacity_label_file(file_path, detections):

    detections['pred_cat_prob'] = detections['predicted_category'].astype(str) + ' (' + detections['prob'].astype(float).round(3).astype(str) + ')'
    
    aud_df = detections[['start_sec','end_sec','pred_cat_prob']].copy()

    aud_df.to_csv(file_path, index=False, na_rep='', sep ='\t',header=False)


class _Bunch:
    pass
