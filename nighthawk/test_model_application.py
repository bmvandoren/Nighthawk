import argparse
import gc
import time

import numpy as np
import tensorflow as tf


RECORDING_DURATION = 36000
HOP_DURATION = .2
RECORD_COUNT = int(round(RECORDING_DURATION / HOP_DURATION))
RECORD_SIZE = 22050
BATCH_SIZE = 64
MESSAGE_PERIOD = 64 * BATCH_SIZE  # print roughly every 64 batches
CHUNK_DURATION = 1000
CHUNK_RECORD_COUNT = int(round(CHUNK_DURATION / HOP_DURATION))


def main():
    parser = argparse.ArgumentParser(
        description='Nighthawk model throughput benchmark.')
    parser.add_argument(
        '--model-path', default=None,
        help='Path to a local model bundle directory or tarball. '
             'Omit to use the default model (auto-downloaded if needed).')
    args = parser.parse_args()

    print('Initializing...')
    model = load_model(args.model_path)
    samples = get_samples()

    time_processing(apply_model_and_retain_results, model, samples)
    # time_processing(apply_model_and_retain_result_chunks, model, samples)
    # time_processing(apply_model_and_discard_results, model, samples)


def load_model(model_path=None):
    from nighthawk.model_manager import resolve_model
    resolved = resolve_model(model_path=model_path)
    return tf.saved_model.load(resolved.saved_model_dir)


def get_samples():
    """Return a single 1-second waveform.  Batching is done in apply_model_*."""
    return tf.random.uniform((RECORD_SIZE,), minval=-.9, maxval=.9)


def time_processing(function, *args):

    print('Applying model to samples...')

    start_time = time.time()

    function(*args)

    end_time = time.time()
    elapsed_time = end_time - start_time
    speed = RECORDING_DURATION / elapsed_time

    print(
        f'Processed {RECORDING_DURATION} seconds of audio in '
        f'{elapsed_time} seconds, {speed:.1f} times faster than '
        f'real time.')


def apply_model_and_retain_results(model, samples):
    """Run the model on RECORD_COUNT windows using batched inference."""
    sig = model.signatures['serving_default']
    # Pre-build a full batch of identical samples (simulates sliding-window input).
    batch = tf.stack([samples] * BATCH_SIZE)  # [BATCH_SIZE, RECORD_SIZE]

    results = []
    num_full_batches = RECORD_COUNT // BATCH_SIZE
    remainder = RECORD_COUNT % BATCH_SIZE

    for i in range(num_full_batches):
        if i != 0 and (i * BATCH_SIZE) % MESSAGE_PERIOD == 0:
            print(i * BATCH_SIZE * HOP_DURATION)
        results.append(sig(waveform=batch))

    if remainder:
        small_batch = tf.stack([samples] * remainder)
        results.append(sig(waveform=small_batch))


def apply_model_and_retain_result_chunks(model, samples):
    """Run model in chunks, periodically discarding results to limit memory."""
    sig = model.signatures['serving_default']
    batch = tf.stack([samples] * BATCH_SIZE)
    results = []

    num_full_batches = RECORD_COUNT // BATCH_SIZE
    for i in range(num_full_batches):
        if i != 0 and (i * BATCH_SIZE) % MESSAGE_PERIOD == 0:
            print(i * BATCH_SIZE * HOP_DURATION)

        if i % (CHUNK_RECORD_COUNT // BATCH_SIZE) == 0 and i != 0:
            print('discarding results...')
            results = []
            gc.collect()

        results.append(sig(waveform=batch))


def apply_model_and_discard_results(model, samples):
    """Run model and discard all results (measures pure inference throughput)."""
    sig = model.signatures['serving_default']
    batch = tf.stack([samples] * BATCH_SIZE)

    num_full_batches = RECORD_COUNT // BATCH_SIZE
    for i in range(num_full_batches):
        if i != 0 and (i * BATCH_SIZE) % MESSAGE_PERIOD == 0:
            print(i * BATCH_SIZE * HOP_DURATION)
        sig(waveform=batch)


if __name__ == '__main__':
    main()
