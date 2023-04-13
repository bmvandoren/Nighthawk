"""
Compare scikit-learn probability calibrations to ours to make sure they
are equivalent.
"""


from pathlib import Path
import pickle

import numpy as np

import nighthawk.probability_calibration_utils as probability_calibration_utils


DATA_DIR_PATH = Path(__file__).parent / 'test_config'
SKLEARN_CALIBRATION_FILE_PATH = DATA_DIR_PATH / 'calibrators_dict.obj'
OUR_CALIBRATION_FILE_PATH = DATA_DIR_PATH / 'probability_calibrations.csv'
    

def main():

    # Get scikit-learn calibrations.
    with open(SKLEARN_CALIBRATION_FILE_PATH, 'rb') as file:
        sklearn_calibrations = pickle.load(file)

    # Get our calibrations.
    our_calibrations = probability_calibration_utils.load_calibrations(
        OUR_CALIBRATION_FILE_PATH)
    
    # Get test input.
    x = np.random.uniform(size=1000000)


    # Compare scikit-learn calibrations to our calibrations.

    diff_count = 0
    
    for taxon, sklearn_calibration in sklearn_calibrations.items():

        sklearn_y = sklearn_calibration.predict(x)

        our_calibration = our_calibrations[taxon]
        our_y = our_calibration.predict(x)

        max_abs_diff = np.max(np.abs(sklearn_y - our_y))

        if max_abs_diff != 0:
            print(
                f'WARNING: Taxon "{taxon}" calibrated probability '
                f'differences had nonzero maximum absolute value of '
                f'{max_abs_diff}.')
            diff_count += 1

    if diff_count == 0:
        print('Calibrated probabilities were the same for all taxa.')
    

if __name__ == '__main__':
    main()
