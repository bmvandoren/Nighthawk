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
        
        detections, detect_df_dict = _run_detector_on_file(
            input_file_path, model, config_file_paths, hop_size, threshold,
            merge_overlaps, drop_uncertain, mask_ap_threshold, return_tax_level_detections,
            do_calibration, quiet)

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
    paths.species =  taxonomy / 'species_select_v6.txt'
    paths.groups =  taxonomy / 'groups_select_v6.txt'
    paths.families =  taxonomy / 'families_select_v6.txt'
    paths.orders =  taxonomy / 'orders_select_v6.txt'
    paths.ebird_taxonomy = taxonomy / 'ebird_taxonomy.csv'
    paths.group_ebird_codes = taxonomy / 'groups_ebird_codes.csv'
    paths.ibp_codes = taxonomy / 'IBP-AOS-LIST21.csv'
 
    config = _CONFIG_DIR_PATH
    paths.config = config / 'test_config.json'
    paths.test_set_performance = config / 'test_set_performance'
    paths.calibrators = config / 'probability_calibrations.csv'

    return paths


def _run_detector_on_file(
        audio_file_path, model, paths, hop_size, threshold, merge_overlaps,
        drop_uncertain,mask_ap_threshold,return_tax_level_detections,do_calibration,
        quiet):

    p = paths

    if do_calibration:
        calib = p.calibrators
    else:
        calib = None      
    
    # Change hop size from percentage to seconds.
    hop_dur = hop_size / 100 * MODEL_INPUT_DURATION

    # Change threshold from percentage to fraction.
    threshold /= 100

    return run_reconstructed_model.run_model_on_file(
        model, audio_file_path, MODEL_SAMPLE_RATE, MODEL_INPUT_DURATION,
        hop_dur, p.species, p.groups, p.families, p.orders,
        p.ebird_taxonomy, p.group_ebird_codes, calib, p.config,
        stream=False, threshold=threshold, quiet=quiet,
        model_runner=_get_model_predictions,
        postprocess_drop_singles_by_tax_level=drop_uncertain,
        postprocess_merge_overlaps=merge_overlaps,
        postprocess_retain_only_overlaps=drop_uncertain,
        mask_output_ap_threshold=mask_ap_threshold,
        test_set_performance_dir=p.test_set_performance,
        return_tax_level_detections=return_tax_level_detections)


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
        start_index = 0
        end_index = input_length

        while end_index <= sample_count:
            yield samples[start_index:end_index]
            start_index += hop_length
            end_index += hop_length

        load_offset += load_hop_dur


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
