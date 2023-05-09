Nighthawk
=========

## Overview

Nighthawk is a machine learning model for acoustic monitoring of nocturnal bird migration. 

Nighthawk is trained on recordings of nocturnal flight calls (NFCs) from the Americas, with the greatest coverage in eastern North America. It processes mono-channel audio files in .wav format and returns detections in a tabular format (.csv and .txt). Nighthawk currently includes training data from 82 species, 18 families, and 4 orders of birds that vocalize during nocturnal migration.

Nighthawk runs in a Python environment. The underlying model was trained with [TensorFlow](tensorflow.org). 

For details on Nighthawk training and performance, see the following paper:
**<CITATION FOR PAPER ONCE ON BIORXIV>**

## Usage

The simplest possible command is to run Nighthawk on a single wav file and output .csv files into the same directory as the input file:
```
python nighthawk/run_nighthawk.py INPUT.wav
```

The following code runs Nighthawk on a test file included in the repo:
```
python nighthawk/run_nighthawk.py test_inputs/test1.wav
```
  
Adding the `--raven-output` flag will also export a .txt file that can be read as a selection table by [Raven Pro](https://ravensoundsoftware.com/software/raven-pro/):
```
python nighthawk/run_nighthawk.py test_inputs/test1.wav --raven-output
```
  
To view full usage help, run:
```
python nighthawk/run_nighthawk.py --help
```
  
```
usage: run_nighthawk.py [-h] [--hop-size HOP_SIZE] [--threshold THRESHOLD]
                        [--merge-overlaps | --no-merge-overlaps] [--drop-uncertain | --no-drop-uncertain]
                        [--csv-output | --no-csv-output] [--raven-output | --no-raven-output]
                        [--output-dir OUTPUT_DIR_PATH]
                        input_file_paths [input_file_paths ...]

positional arguments:
  input_file_paths      paths of audio files on which to run the detector.

options:
  -h, --help            show this help message and exit
  --hop-size HOP_SIZE   the hop size as a percentage of the model input duration, a number in the range (0, 100].
                        (default: 20)
  --threshold THRESHOLD
                        the detection threshold, a number in [0, 100]. (default: 80)
  --merge-overlaps, --no-merge-overlaps
                        merge overlapping detections. (default: True)
  --drop-uncertain, --no-drop-uncertain
                        apply postprocessing steps to drop less certain detections. (default: True)
  --csv-output, --no-csv-output
                        output detections to a CSV file. (default: True)
  --raven-output, --no-raven-output
                        output detections to a Raven selection table file. (default: False)
  --output-dir OUTPUT_DIR_PATH
                        directory in which to write output files. (default: input file directories)
```
  
## Installation

Users can install Nighthawk via pip 
**<HAROLD, COULD YOU PLEASE ADD RECOMMEND INSTALLATION INSTRUCTIONS FOR THE PACKAGE WITH ANACONDA?>**

### Vesper plugin
  
**HAROLD TODO**
  
  
## Licensing and Citation

Nighthawk is provided under a [Creative Commons BY-NC 4.0 License](https://creativecommons.org/licenses/by-nc/4.0/). Please read the [license terms](https://creativecommons.org/licenses/by-nc/4.0/legalcode). Under this license, you are free to share and adapt this material, but you must provide appropriate attribution, and you may not use the material for commercial purposes.
  
If you use Nighthawk, please use the following citation:
**<CITATION FOR PAPER ONCE ON BIORXIV>**
  
## Contact and Collaborations

Please contact Benjamin Van Doren (vandoren@cornell.edu) with questions about Nighthawk. We are always open to scientific collaborations.  

<!-- ![Image of Zenodo DOI badge](https://zenodo.org/badge/DOI/DOIHERE) -->
