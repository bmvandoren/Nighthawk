"""
Utility functions concerning probability calibrations.

The `load_calibrations` function of this file loads sigmoid probability
calibrations (one for each Nighthawk taxon) from a CSV file. The
calibration coefficients stored in that file must be computed using
scikit-learn, written to a pickle file `calibrators_dict.obj`, and then
put into a CSV file by running the script
`create_probability_calibration_file.py`.

This system was implemented to eliminate warnings about potential version
incompatibilities that scikit-learn issues when it loads probability
calibration objects (of scikit-learn class `_SigmoidCalibration`) from a
pickle file that was created by a different scikit-learn version. The
possibility of version incompatibilities is still there, but it is small
enough (given the simplicity of the sigmoid probability calibration model)
that warnings about it are unwarranted.

The script `compare_calibrations.py` compares calibrated probabilities
computed by scikit-learn to those computed by this module. It should
be run whenever the calibrations are updated, i.e. whenever the sigmoid
probability calibration CSV file changes.
"""


import numpy as np


def load_calibrations(csv_file_path):

    with open(csv_file_path, encoding='utf-8') as file:
        contents = file.read()

    lines = contents.strip().split('\n')[1:]
    triples = [line.split(',') for line in lines]

    return {
        taxon: _SigmoidProbabilityCalibration(float(a), float(b))
        for taxon, a, b in triples
    }


class _SigmoidProbabilityCalibration:

    """
    Sigmoid probability calibration.
     
    This class implements sigmoid probability calibration as in scikit-learn:
    see https://scikit-learn.org/stable/modules/calibration.html#sigmoid.
    """

    def __init__(self, a, b):
        self._a = a
        self._b = b

    def predict(self, x):
        return 1 / (1 + np.exp(self._a * x + self._b))