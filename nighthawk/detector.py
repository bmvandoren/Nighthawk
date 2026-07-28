"""Functions and constants for the Nighthawk NFC detector."""


from functools import partial
from pathlib import Path
import time

import librosa
import numpy as np
import soundfile as sf

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
DEFAULT_BATCH_SIZE = 64           # windows per model call

# Model repository settings.
DEFAULT_MODEL_NAME = 'americas'
DEFAULT_MODEL_VERSION = 'latest'   # tracks the newest published version
DEFAULT_MODEL_REPO_URL = 'https://nighthawk-models.s3.us-east-1.amazonaws.com/'

# Canonical output level order matching nh2 model heads.
_CANONICAL_LEVELS = ['order', 'family', 'group', 'species']


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
        quiet=DEFAULT_QUIET,
        batch_size=DEFAULT_BATCH_SIZE,
        model_name=DEFAULT_MODEL_NAME,
        model_version=DEFAULT_MODEL_VERSION,
        model_path=None,
        model_repo_url=DEFAULT_MODEL_REPO_URL,
        cache_dir=None,
        offline=False):

    input_file_paths = _expand_paths(input_file_paths)
    file_count = len(input_file_paths)
    if file_count == 0:
        print('No input files found.')
        return

    print('Resolving detector model...')
    from nighthawk.model_manager import resolve_model, NighthawkModelError
    try:
        resolved = resolve_model(
            name=model_name,
            version=model_version,
            model_path=model_path,
            repo_url=model_repo_url or None,
            cache_dir=cache_dir,
            offline=offline,
        )
    except NighthawkModelError as e:
        print(f'Error: {e}')
        return
    if not quiet:
        print(f'Using model {resolved.name}@{resolved.version} '
              f'(source: {resolved.source})')

    print('Loading detector model...')
    model = _load_model(resolved.saved_model_dir)

    print('Getting detector configuration file paths...')
    config_file_paths = _get_configuration_file_paths(resolved)

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
            do_calibration, quiet, batch_size)

        # For sub-1s recordings, write the zero-padded clip so the user can
        # hear exactly what the model analyzed.
        if librosa.get_duration(path=input_file_path) < MODEL_INPUT_DURATION:
            _export_zero_padded_audio(input_file_path, output_dir_path)

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
    

def _load_model(saved_model_dir):

    # TF is imported here (not at module level) since it is slow to load.
    # This keeps the script responsive if the user just wants --help or
    # accidentally specifies an invalid argument.
    import tensorflow as tf

    return tf.saved_model.load(str(saved_model_dir))


def _get_configuration_file_paths(resolved):
    """Build a _Bunch of config file paths from a ResolvedModel.

    Attribute names are preserved exactly so _run_detector_on_file and
    run_reconstructed_model.run_model_on_file remain untouched.
    """
    paths = _Bunch()

    tax = resolved.taxonomy
    paths.species           = tax['species']
    paths.groups            = tax['groups']
    paths.families          = tax['families']
    paths.orders            = tax['orders']
    paths.ebird_taxonomy    = tax['ebird_taxonomy']
    paths.group_ebird_codes = tax['group_ebird_codes']

    tc = resolved.test_config
    paths.config                  = tc['config']
    paths.test_set_performance    = tc['test_set_performance']
    paths.calibrators_from_logits = tc['calibrators_from_logits']

    # model_type drives runner selection: 'nh2' (batched, named-dict outputs)
    # or 'legacy' (unbatched, positional-list outputs).  Default to 'nh2' when
    # the manifest is absent (will be confirmed via _has_nh2_signature at
    # runtime if the auto-detect path fires).
    paths.model_type = (resolved.manifest or {}).get('model_type', 'nh2')

    return paths


def _stride_seconds(hop_size):
    """Sliding-window stride in seconds (hop_size is percent of model window)."""
    return hop_size / 100 * MODEL_INPUT_DURATION


def _min_duration_for_two_windows(hop_size):
    """Minimum file duration that can yield two overlapping analysis windows."""
    return MODEL_INPUT_DURATION + _stride_seconds(hop_size)


def _resolve_drop_uncertain_for_file(audio_file_path, hop_size, drop_uncertain, quiet):
    """Return the effective drop_uncertain setting for this file.

    If the recording is too short to ever produce two overlapping windows,
    ``--drop-uncertain`` cannot be satisfied and is automatically disabled
    for that file only.

    Returns (effective_drop_uncertain, was_relaxed).
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
            f'{MODEL_INPUT_DURATION} s + {stride:.3f} s stride); '
            f'applying --no-drop-uncertain automatically.')
    return False, True


def _run_detector_on_file(
        audio_file_path, model, paths, hop_size, threshold, merge_overlaps,
        drop_uncertain,mask_ap_threshold,return_tax_level_detections,do_calibration,
        quiet, batch_size=DEFAULT_BATCH_SIZE):

    p = paths

    if do_calibration and p.calibrators_from_logits.exists():
        print('Calibrating from logits.')
        calib = p.calibrators_from_logits
    else:
        if do_calibration:
            print('No calibration file found, proceeding without calibration.')
        calib = None
    
    # Change hop size from percentage to seconds.
    hop_dur = _stride_seconds(hop_size)

    # Change threshold from percentage to fraction.
    threshold /= 100

    # For recordings shorter than one model window + stride, --drop-uncertain
    # can never be satisfied (only one window is produced), so disable it for
    # this file and let the single-window detection through.
    drop_uncertain, _ = _resolve_drop_uncertain_for_file(
        audio_file_path, hop_size, drop_uncertain, quiet)

    # Select the runner based on model_type from the manifest.  When the
    # manifest is absent (--model-path to a legacy tree without manifest.json)
    # the model_type defaults to 'nh2', so we probe the actual signature and
    # fall back to the legacy runner if needed.
    model_type = getattr(p, 'model_type', 'nh2')
    if model_type != 'legacy' and not _has_nh2_signature(model):
        model_type = 'legacy'
    runner_fn = (
        _get_model_predictions_legacy if model_type == 'legacy'
        else _get_model_predictions
    )
    if model_type == 'legacy':
        print('Using legacy (unbatched) inference runner.')
    model_runner = partial(runner_fn, batch_size=batch_size)

    return run_reconstructed_model.run_model_on_file(
        model, audio_file_path, MODEL_SAMPLE_RATE, MODEL_INPUT_DURATION,
        hop_dur, p.species, p.groups, p.families, p.orders,
        p.ebird_taxonomy, p.group_ebird_codes, calib,
        p.config, stream=False, threshold=threshold, quiet=quiet,
        model_runner=model_runner,
        postprocess_drop_singles_by_tax_level=drop_uncertain,
        postprocess_merge_overlaps=merge_overlaps,
        postprocess_retain_only_overlaps=drop_uncertain,
        mask_output_ap_threshold=mask_ap_threshold,
        test_set_performance_dir=p.test_set_performance,
        return_tax_level_detections=return_tax_level_detections)


def _get_model_predictions(
        model, file_path, input_dur, hop_dur, target_sr=22050, batch_size=64):
    """Run the model on all windows of an audio file using batched inference.

    Feeds windows in batches of ``batch_size`` to the nh2 SavedModel, which
    accepts a batched waveform input ``[B, 22050]`` and returns a dict of named
    logit tensors ``{order, family, group, species}``.  Collecting windows into
    batches substantially improves throughput on both GPU and CPU compared to
    single-window inference.

    Returns:
        predictions: list of four numpy arrays ``[order, family, group, species]``,
            each of shape ``(num_windows, n_taxa)`` — same contract as before.
        bad_inds: empty list (no bad-index detection in this runner).
        input_count: total number of windows processed.
    """
    import tensorflow as tf

    start_time = time.time()

    sig = model.signatures['serving_default']

    # Accumulate per-level results across batches.
    level_arrays = {lv: [] for lv in _CANONICAL_LEVELS}
    input_count = 0
    pending = []

    def _flush(windows):
        batch = tf.constant(np.stack(windows), dtype=tf.float32)  # [B, 22050]
        out = sig(waveform=batch)
        for lv in _CANONICAL_LEVELS:
            level_arrays[lv].append(out[lv].numpy())  # [B, n_taxa]

    for samples in _generate_model_inputs(file_path, input_dur, hop_dur, target_sr):
        pending.append(samples)
        input_count += 1
        if len(pending) == batch_size:
            _flush(pending)
            pending = []

    if pending:  # partial final batch
        _flush(pending)

    elapsed_time = time.time() - start_time
    _report_processing_speed(file_path, elapsed_time)

    if input_count == 0:
        # No windows were generated (e.g. a truly empty or zero-length file).
        # Zero-padding in _generate_model_inputs handles the sub-1s case, so
        # this guard is a safety net for genuinely unreadable/empty files.
        return [np.zeros((0, 0))] * 4, [], 0

    # Concatenate batches and return in canonical order.
    predictions = [np.concatenate(level_arrays[lv], axis=0) for lv in _CANONICAL_LEVELS]
    return predictions, [], input_count


def _get_model_predictions_legacy(
        model, file_path, input_dur, hop_dur, target_sr=22050, batch_size=64):
    """Run a pre-nh2 (legacy) model on all windows of an audio file.

    Legacy SavedModels are directly callable with a single unbatched
    ``(22050,)`` waveform tensor and return a positional list of four
    ``1 x n_taxa`` logit tensors ``[order, family, group, species]``.

    The ``batch_size`` parameter is accepted but ignored (legacy models process
    one window at a time).

    Returns:
        predictions: list of four numpy arrays ``[order, family, group, species]``,
            each of shape ``(num_windows, n_taxa)`` — same contract as the nh2
            runner so run_reconstructed_model stays untouched.
        bad_inds: empty list.
        input_count: total number of windows processed.
    """
    import tensorflow as tf

    start_time = time.time()

    window_preds = [
        model(tf.constant(samples, dtype=tf.float32))
        for samples in _generate_model_inputs(file_path, input_dur, hop_dur, target_sr)
    ]

    elapsed_time = time.time() - start_time
    _report_processing_speed(file_path, elapsed_time)

    if not window_preds:
        # No windows (very short file) — return empty arrays.
        return [np.zeros((0, 0))] * 4, [], 0

    # window_preds is a list of per-window outputs.  Each per-window output is
    # either a list/tuple of four [1, n_taxa] tensors (positional) or a tensor
    # directly.  We transpose to get one array per level.
    predictions = [
        np.squeeze(np.array(level_outputs), axis=1)
        for level_outputs in zip(*window_preds)
    ]
    return predictions, [], len(window_preds)


def _has_nh2_signature(model):
    """Return True if the loaded model has the nh2 batched waveform signature."""
    try:
        sig = model.signatures.get('serving_default')
        if sig is None:
            return False
        input_keys = list(sig.structured_input_signature[1].keys())
        return 'waveform' in input_keys
    except Exception:
        return False


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
            # File is shorter than one model window.  Warn once (on the first
            # load chunk) and center-pad with zeros so exactly one window is
            # produced.  The padded WAV is exported by run_detector_on_files
            # for user inspection.
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


def _export_zero_padded_audio(
        file_path, output_dir_path=None, target_sr=MODEL_SAMPLE_RATE,
        input_dur=MODEL_INPUT_DURATION):
    """Load audio, center-pad to input_dur, and write a *_padded.wav file.

    Allows the user to hear exactly what the model analyzed for a short
    recording.  Returns the output path, or None if no padding was needed.
    """
    samples, _ = librosa.load(file_path, sr=target_sr, res_type='soxr_hq')
    input_length = int(round(input_dur * target_sr))
    if len(samples) >= input_length:
        return None
    padded = _pad_audio_center(samples, input_length)
    output_file_path = _prep_for_output(
        Path(file_path), output_dir_path, '.wav', descriptor='padded',
        gzip=False)
    sf.write(str(output_file_path), padded, target_sr, 'PCM_16')
    return output_file_path


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
