"""Tests for calibrating model logits without a probability round-trip.

Covers the data-flow change: apply from-logits calibrators to raw logits
directly (no sigmoid -> prob_to_logit -> calibrate). Run with:

    python -m unittest tests.test_calibration_dataflow -v
"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd


_NIGHTHAWK_DIR = Path(__file__).resolve().parents[1] / 'nighthawk'


def _load_module(module_name, file_name):
    """Load a nighthawk module by path, avoiding package __init__ (detector/TF)."""

    module_path = _NIGHTHAWK_DIR / file_name
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    # Register before exec so sibling imports resolve when other modules load.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# librosa is only needed for audio I/O elsewhere in run_reconstructed_model.
sys.modules.setdefault('librosa', MagicMock())
# Minimal package stub so ``import nighthawk....`` does not run __init__.py.
sys.modules.setdefault('nighthawk', types.ModuleType('nighthawk'))

_calib_utils = _load_module(
    'nighthawk.probability_calibration_utils',
    'probability_calibration_utils.py')
_load_module('nighthawk.tensor_flow_debug', 'tensor_flow_debug.py')
_rrm = _load_module(
    'nighthawk.run_reconstructed_model', 'run_reconstructed_model.py')

_SigmoidProbabilityCalibration = _calib_utils._SigmoidProbabilityCalibration
prob_to_logit = _calib_utils.prob_to_logit
logits_to_calibrated_probabilities = _rrm.logits_to_calibrated_probabilities
sigmoid_then_calibrate_probabilities = _rrm.sigmoid_then_calibrate_probabilities

SPECIES = 'comyel'
A = -1.5
B = 0.2


def expected_calibrated(x, a=A, b=B):
    """Plattscaling formula used by _SigmoidProbabilityCalibration.predict."""

    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(a * x + b))


def make_logit_df(logits, taxon=SPECIES):
    n = len(logits)
    return pd.DataFrame({
        taxon: list(logits),
        'start_sec': [float(i) for i in range(n)],
        'end_sec': [float(i + 1) for i in range(n)],
    })


def old_buggy_calibrate_from_logits(logits, a=A, b=B):
    """Former path: sigmoid -> clip/invert to logit -> calibrate.

    For extreme logits, clipping in ``prob_to_logit`` changes the value that
    reaches the calibrator, so this differs from calibrating raw logits.
    """

    probs = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))
    recovered = prob_to_logit(probs)
    return expected_calibrated(recovered, a=a, b=b)


class TestCalibrateDirectlyFromLogits(unittest.TestCase):
    """from-logits CSV path: calibrator sees raw model logits."""

    def setUp(self):
        self.calibrators = {
            SPECIES: _SigmoidProbabilityCalibration(A, B),
        }

    def test_matches_platt_formula_on_raw_logits(self):
        logits = [-2.0, 0.0, 2.0]
        pred_df_dict = {'species': make_logit_df(logits)}

        out = logits_to_calibrated_probabilities(
            pred_df_dict, self.calibrators)['species']

        np.testing.assert_allclose(
            out[SPECIES].to_numpy(),
            expected_calibrated(logits),
            rtol=0, atol=1e-12,
        )

    def test_extreme_logits_differ_from_old_prob_roundtrip(self):
        """Clipping in prob_to_logit made the old path diverge for large |logit|."""

        logits = np.array([-20.0, 20.0])
        probs = 1.0 / (1.0 + np.exp(-logits))
        recovered = prob_to_logit(probs)

        # Saturated probs clip, so the recovered "logits" are not the originals.
        self.assertFalse(np.allclose(recovered, logits, rtol=0, atol=1e-6))

        new = logits_to_calibrated_probabilities(
            {'species': make_logit_df(logits)},
            self.calibrators,
        )['species'][SPECIES].to_numpy()
        old = old_buggy_calibrate_from_logits(logits)

        np.testing.assert_allclose(
            new, expected_calibrated(logits), rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            old, expected_calibrated(recovered), rtol=0, atol=1e-12)
        # New path uses raw logits; old path used clipped recovered logits.
        self.assertGreater(np.max(np.abs(new - old)), 0.0)

    def test_outputs_are_probabilities_in_unit_interval(self):
        logits = [-5.0, 0.0, 5.0]
        out = logits_to_calibrated_probabilities(
            {'species': make_logit_df(logits)}, self.calibrators)['species']

        self.assertTrue(((out[SPECIES] >= 0) & (out[SPECIES] <= 1)).all())

    def test_preserves_time_columns(self):
        df = make_logit_df([1.0])
        out = logits_to_calibrated_probabilities(
            {'species': df.copy()}, self.calibrators)['species']

        self.assertEqual(out['start_sec'].tolist(), [0.0])
        self.assertEqual(out['end_sec'].tolist(), [1.0])


class TestSigmoidThenCalibrate(unittest.TestCase):
    """from-probs CSV path: sigmoid first, then calibrators fit on probabilities."""

    def setUp(self):
        self.calibrators = {
            SPECIES: _SigmoidProbabilityCalibration(A, B),
        }

    def test_calibrator_sees_sigmoid_of_logits(self):
        logits = [-1.0, 0.5, 3.0]
        probs = 1.0 / (1.0 + np.exp(-np.asarray(logits)))

        out = sigmoid_then_calibrate_probabilities(
            {'species': make_logit_df(logits)}, self.calibrators)['species']

        np.testing.assert_allclose(
            out[SPECIES].to_numpy(),
            expected_calibrated(probs),
            rtol=0, atol=1e-12,
        )


if __name__ == '__main__':
    unittest.main()
