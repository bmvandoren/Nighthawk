"""
Script that plots speed of Nighthawk NFC detector for two different
postprocessors.
"""


from pathlib import Path
import time

import librosa
import matplotlib.pyplot as plt
import pandas as pd

from model import Model
from pandas_postprocessor import PandasPostprocessor
from numpy_postprocessor import NumPyPostprocessor
import nighthawk_new as nh


PROJECT_DIR_PATH = Path(__file__).parent.parent
AUDIO_FILE_PATH =  PROJECT_DIR_PATH / 'test_inputs' / 'Ithaca.22050.wav'
OUTPUT_DIR_PATH = PROJECT_DIR_PATH / 'test_outputs'
CSV_FILE_PATH = OUTPUT_DIR_PATH / 'Detection Speeds.csv'
PLOT_FILE_PATH = OUTPUT_DIR_PATH / 'Detection Speeds.pdf'
TRIAL_COUNT = 10
DETECTION_THRESHOLD = 50      # percent
DETECTION_HOP_SIZE = 100      # percent
MIN_BLOCK_LENGTH_LOG = 0
MAX_BLOCK_LENGTH_LOG = 10


def main():
    
    time_detection()
    plot_detection_speeds()


def time_detection():

    print('Creating model...')
    start_time = time.time()
    model = Model()
    _report_time('Model load', start_time)
 
    taxa_of_interest = nh.get_taxa_of_interest()

    print('Creating Pandas postprocessor...')
    start_time = time.time()
    pandas_postprocessor = PandasPostprocessor(
        model.output_taxa, taxa_of_interest, DETECTION_THRESHOLD)
    _report_time('Pandas postprocessor creation', start_time)

    print('Creating NumPy postprocessor...')
    start_time = time.time()
    numpy_postprocessor = NumPyPostprocessor(
        model.output_taxa, taxa_of_interest, DETECTION_THRESHOLD)
    _report_time('NumPy postprocessor creation', start_time)

    print('Detecting...')

    audio_file_duration = librosa.get_duration(filename=AUDIO_FILE_PATH)

    # Order block lengths from largest to smallest. This makes it easier
    # to monitor script progress initially since detection is faster for
    # larger block lengths.
    block_length_logs = \
        range(MAX_BLOCK_LENGTH_LOG, MIN_BLOCK_LENGTH_LOG - 1, -1)
    block_lengths = tuple(2 ** i for i in block_length_logs)

    postprocessors = (
        ('Pandas', pandas_postprocessor),
        ('NumPy', numpy_postprocessor))

    results = []

    for trial_num in range(TRIAL_COUNT):

        for block_length in block_lengths:

            for name, postprocessor in postprocessors:

                start_time = time.time()

                detections = nh.run_detector_on_file(
                    AUDIO_FILE_PATH, DETECTION_HOP_SIZE, block_length, model,
                    postprocessor)

                detection_count = detections.shape[0]
                elapsed_time = time.time() - start_time
                speed = audio_file_duration / elapsed_time
                result = (
                    name, block_length, trial_num, detection_count,
                    elapsed_time, speed)

                print(result)

                results.append(result)

    results.sort()

    columns = (
        'postprocessor', 'block_length', 'trial', 'detections',
        'elapsed_time', 'speed')

    df = pd.DataFrame(results, columns=columns)

    df.to_csv(CSV_FILE_PATH, index=False)


def _report_time(operation, start_time):
    elapsed_time = time.time() - start_time
    print(f'{operation} took {elapsed_time:.3f} seconds.')


def plot_detection_speeds():

    df = pd.read_csv(CSV_FILE_PATH)

    _, axes = plt.subplots()

    for postprocessor in ('NumPy', 'Pandas'):
        block_lengths, speeds = get_speed_data(df, postprocessor)
        axes.semilogx(block_lengths, speeds, label=postprocessor)

    axes.set_ylim(bottom=0)
    axes.set_title('Nighthawk Detector Speed for Different Postprocessors')
    axes.set_xlabel('Block Length (sample frames)')
    axes.set_ylabel('Speed (x faster than real time)')
    axes.legend(loc='lower right')
    axes.grid()

    plt.savefig(PLOT_FILE_PATH)


def get_speed_data(df, postprocessor):

    # Get rows for only this postprocessor.
    df = df[df.postprocessor == postprocessor]

    # Retain only block length and speed columns.
    df = df[['block_length', 'speed']]

    # Get median speeds.
    df = df.groupby('block_length').median()

    return df.index, df.speed


if __name__ == '__main__':
    main()
