"""
Read scikit-learn sigmoid probability calibrations from pickle file
and write coefficients to CSV file.

See https://scikit-learn.org/stable/modules/calibration.html#sigmoid
for a description of scikit-learn's sigmoid probability calibration.
"""


from pathlib import Path
import pickle


DATA_DIR_PATH = Path(__file__).parent / 'test_config'
INPUT_CALIBRATION_FILE_PATH = DATA_DIR_PATH / 'calibrators_dict.obj'
OUTPUT_CALIBRATION_FILE_PATH = DATA_DIR_PATH / 'probability_calibrations.csv'
OUTPUT_FILE_HEADER = 'Taxon,A,B\n'


def main():

    # Read scikit-learn calibrations from pickle file.
    with open(INPUT_CALIBRATION_FILE_PATH, 'rb') as file:
        calibrations = pickle.load(file)

    # Get output file data lines.
    taxa = sorted(calibrations.keys())
    get_line = lambda t, c: f'{t},{c.a_},{c.b_}\n'
    lines = [get_line(t, calibrations[t]) for t in taxa]

    # Write output CSV file.
    with open(OUTPUT_CALIBRATION_FILE_PATH, 'w', encoding='utf-8') as file:
       file.write(OUTPUT_FILE_HEADER)
       file.writelines(lines)


if __name__ == '__main__':
    main()
