"""Functions and constants for the Nighthawk NFC detector."""


from pathlib import Path
import queue
import threading
import time


import librosa
import numpy as np


import nighthawk.run_reconstructed_model as run_reconstructed_model




MODEL_SAMPLE_RATE = 22050         # Hz
MODEL_INPUT_DURATION = 1          # seconds


_INFERENCE_BATCH_SIZE = 256        # samples per GPU batch (tune up if VRAM allows)


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
    paths.calibrators_from_logits = config / 'probability_calibrations_logistic_fromlogits.csv'
    paths.calibrators_from_probs = config / 'probability_calibrations.csv'


    return paths




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
    hop_dur = hop_size / 100 * MODEL_INPUT_DURATION

    # Change threshold from percentage to fraction.
    threshold /= 100

    return run_reconstructed_model.run_model_on_file(
        model, audio_file_path, MODEL_SAMPLE_RATE, MODEL_INPUT_DURATION,
        hop_dur, p.species, p.groups, p.families, p.orders,
        p.ebird_taxonomy, p.group_ebird_codes, calib, calibrate_from_logits, p.config,
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


    batch_runner = _make_batch_runner(model)


    # Prefetch audio batches from disk in a background thread so librosa.load()
    # overlaps with GPU inference rather than stalling it.
    q = queue.Queue(maxsize=4)


    def _load_worker():
        try:
            for batch in _generate_batched_inputs(
                    file_path, input_dur, hop_dur, target_sr,
                    _INFERENCE_BATCH_SIZE):
                q.put(batch)
        finally:
            q.put(None)  # sentinel


    loader = threading.Thread(target=_load_worker, daemon=True)
    loader.start()


    taxon_batches = []


    while True:
        batch = q.get()
        if batch is None:
            break
        import tensorflow as tf
        preds = batch_runner(tf.constant(batch, dtype=tf.float32))
        taxon_batches.append([np.squeeze(p.numpy(), axis=1) for p in preds])


    loader.join()


    if taxon_batches:
        n_taxon = len(taxon_batches[0])
        predictions = [
            np.concatenate([b[j] for b in taxon_batches], axis=0)
            for j in range(n_taxon)
        ]
    else:
        predictions = [np.empty((0,)) for _ in range(4)]


    elapsed_time = time.time() - start_time
    _report_processing_speed(file_path, elapsed_time)


    input_count = len(predictions[0])
    return predictions, [], input_count




def _make_batch_runner(model):
    import tensorflow as tf


    # Warmup: discover the shapes/dtypes of every output tensor.
    dummy = tf.zeros([22050], dtype=tf.float32)
    warmup_out = model(dummy)
    # model() returns a list of 4 tensors, each (1, n_classes).
    # fn_output_signature describes what the function returns for ONE element,
    # so we use these shapes directly.
    output_sig = [
        tf.TensorSpec(shape=t.shape, dtype=t.dtype) for t in warmup_out
    ]


    @tf.function
    def run_batch(samples_batch):
        return tf.map_fn(model, samples_batch, fn_output_signature=output_sig)


    # Trace now so the first real call pays no tracing penalty.
    run_batch(tf.zeros([1, 22050], dtype=tf.float32))


    return run_batch




def _generate_batched_inputs(file_path, input_dur, hop_dur, target_sr,
                              batch_size):
    batch = []
    for sample in _generate_model_inputs(
            file_path, input_dur, hop_dur, target_sr):
        batch.append(sample)
        if len(batch) == batch_size:
            yield np.stack(batch)
            batch = []
    if batch:
        yield np.stack(batch)




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
