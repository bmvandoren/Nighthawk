"""
Test Nighthawk taxon probability calibrations to make sure we understand
what they compute.
"""


from pathlib import Path
import pickle

import numpy as np


CALIBRATIONS_FILE_PATH = \
    Path(__file__).parent / 'test_config' / 'calibrators_dict.obj'


def main():

    # Get calibrations from file.
    with open(CALIBRATIONS_FILE_PATH, 'rb') as file:
        calibrations = pickle.load(file)

    # Get test input.
    x = .01 * np.arange(101)

    # Compare scikit-learn calibrations to our calibrations.
    for taxon, calibration in calibrations.items():

        sklearn_y = calibration.predict(x)

        a = calibration.a_
        b = calibration.b_
        our_y = 1 / (1 + np.exp(a * x + b))

        diff_norm = np.linalg.norm(sklearn_y - our_y)
        flag = '' if diff_norm == 0 else '<<<<<<<<<<'
        print(taxon, diff_norm, flag)
    

if __name__ == '__main__':
    main()
