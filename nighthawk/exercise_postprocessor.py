"""Script that exercises postprocessor for various inputs."""


from collections import defaultdict
from pathlib import Path

import pandas as pd

import run_reconstructed_model


PACKAGE_DIR_PATH = Path(__file__).parent
TAXONOMY_DIR_PATH = PACKAGE_DIR_PATH / 'taxonomy'
TAXONOMY_FILE_PATH = TAXONOMY_DIR_PATH / 'ebird_taxonomy.csv'
GROUP_FILE_PATH = TAXONOMY_DIR_PATH / 'groups_ebird_codes.csv'

FRAME_DURATION = 1
HOP_DURATION = .2
RANKS = ('order', 'family', 'group', 'species')
TAXA = {'order': 'Passeriformes', 'family': 'Parulidae'}
COLUMN_NAMES = ('start_sec', 'end_sec', 'filename', 'path', 'class', 'prob')
COLUMN_DTYPES = ('float', 'float', 'str', 'str', 'str', 'float')
FILE_NAME = 'bobo.wav'
FILE_PATH = '/' + FILE_NAME
PROBABILITY = .99

# Detection patterns to exercise postprocessor on.
# Patterns must be separated by blank lines. Each pattern represents a
# series of consecutive, thresholded detection probabilities of one or
# more taxonomic ranks. The horizontal axis of a pattern is time,
# increasing to the right. The vertical axis is taxonomic rank,
# increasing upward. A pattern must include lines for all taxonomic
# ranks (order, family, group, species) from order down to the pattern's
# lowest rank. A "." character in a pattern represents a probability
# below the detection threshold. A non-"." character represents a
# probability at or above the detection threshold. For clarity, we use
# the first letter of a taxonomic rank's name (e.g. "o" for order and
# "f" for family) as the non-"." letter in pattern lines for that rank.
DETECTION_PATTERNS = '''

    o....o
    ..ff..

    oooooo
    ..ff..

    oo
    .f

    ooo
    .ff

    oooo
    ..ff

    o....o
    f....f

    o....o
    ..f...

    oo

    o.o
    
    o..o

    o...o

    o....o

    o.....o

    o....o....o

'''


def main():

    # Load taxonomy.
    (species_group_map, species_family_map, group_family_map,
        family_order_map) = run_reconstructed_model.load_taxonomy(
            TAXONOMY_FILE_PATH, GROUP_FILE_PATH)

    patterns = parse_detection_patterns(DETECTION_PATTERNS)

    for pattern in patterns:

        show_pattern(pattern)

        # Create input DataFrames, one for each taxonomic rank.
        input_dfs = create_input_dfs(pattern)
        # show_input_dfs(input_dfs)

        # Apply postprocessing.
        output_df = run_reconstructed_model.postprocess(
            input_dfs, FRAME_DURATION, HOP_DURATION, family_order_map,
            group_family_map, species_group_map, species_family_map,
            True, True, True, True)
        
        show_output_df(output_df)


def parse_detection_patterns(s):

    # Get lines, excluding leading and trailing whitespace.
    lines = s.strip().split('\n')

    # Strip leading and trailing whitespace from each line.
    lines = [line.strip() for line in lines]

    # Append empty line to mark end of last pattern.
    if len(lines) != 0:
        lines.append('')

    patterns = []
    pattern = []

    for line in lines:

        if len(line) == 0:
            # `line` marks end of current pattern

            patterns.append(pattern)
            pattern = []

        else:
            # `line` is next line of current pattern

            pattern.append(line)

    return patterns


def parse_detection_pattern(s):
    return tuple(line.strip() for line in s.split('\n'))


def create_input_dfs(pattern):

    detections = defaultdict(list)

    for rank, symbols in zip(RANKS, pattern):
        for i, symbol in enumerate(symbols):
            if symbol != '.':
                detection = create_detection(i, rank)
                detections[rank].append(detection)

    return {rank: create_detection_df(detections[rank]) for rank in RANKS}

    
def create_detection(index, rank):
    start_time = index * HOP_DURATION
    end_time = start_time + FRAME_DURATION
    taxon = TAXA[rank]
    return [start_time, end_time, FILE_NAME, FILE_PATH, taxon, PROBABILITY]


def create_detection_df(detections):

    if len(detections) == 0:

        data = {
            name: pd.Series(dtype=dtype)
            for name, dtype in zip(COLUMN_NAMES, COLUMN_DTYPES)
        }

    else:
        data = detections

    return pd.DataFrame(data, columns=COLUMN_NAMES)


def show_pattern(pattern):
    print()
    print()
    for symbols in pattern:
        print(f'{symbols}')


def show_input_dfs(dfs):
    print()
    for rank in RANKS:
        df = dfs[rank]
        if df.shape[0] != 0:
            print(f'{rank}:')
            print(df)
        # print(df.dtypes)


def show_output_df(df):
    print()
    print(df)


if __name__ == '__main__':
    main()
