Nighthawk
=========

## Overview

Nighthawk is a machine learning model for acoustic monitoring of nocturnal bird migration. 

Nighthawk is trained on recordings of nocturnal flight calls (NFCs) from the Americas, with the greatest coverage in eastern North America. It processes mono-channel audio files in .wav format and returns detections in a tabular format (.csv and .txt). Nighthawk currently includes training data from 82 species, 18 families, and 4 orders of birds that vocalize during nocturnal migration.

Nighthawk runs in a Python environment. The underlying model was trained with [TensorFlow](tensorflow.org). 

<!-- For details on Nighthawk training and performance, see the following [paper](https://doi.org/10.1101/541336): -->

Van Doren, B. M., A. Farnsworth, K. Stone, D. M. Osterhaus, J. Drucker, and G. Van Horn. 2023. Nighthawk: acoustic monitoring of nocturnal bird migration in the Americas. 

<!-- bioRxiv. doi:[10.1101/541336](https://doi.org/10.1101/541336). -->

<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.

## Installation

We recommend installing and using Nighthawk in its own Python environment.
If you're new to Python environments, we suggest installing either
[Anaconda](https://www.anaconda.com/download) or
[Miniconda](https://docs.conda.io/en/latest/miniconda.html) and using
the [`conda`](https://docs.conda.io/projects/conda/en/stable/) program
that comes with them to manage your environments. For help choosing
between Anaconda and Miniconda, see
[here](https://conda.io/projects/conda/en/stable/user-guide/install/download.html).

The following instructions assume that you're using `conda` to manage
your Python environments. If you're using anything else (such as `venv`,
for example), you'll need to modify the instructions accordingly.

On Windows, type the commands below in an Anaconda Prompt. Anaconda
Prompt is a program that comes with Anaconda and Miniconda. It is a
lot like the built-in Windows Command Prompt program, but it is
customized for Anaconda and Miniconda. Once you've installed Anaconda
or Miniconda, you can find the Anaconda Prompt program in your Windows
Start menu or by typing "Anaconda Prompt" in the search field of the
Windows taskbar. On macOS and Linux, type the commands below in a regular
terminal.

To install Nighthawk, first create a new Python environment named
`nighthawk-0.1.0` that uses Python 3.10:

    conda create -n nighthawk-0.1.0 python=3.10

Then activate the environment:

    conda activate nighthawk-0.1.0

and install the Nighthawk Python package and its dependencies:

    pip install nighthawk

For more about `conda`, including additional `conda` commands, see its
[documentation](https://docs.conda.io/projects/conda/en/stable/).

### `tensorflow-metal` for Apple Silicon Macs

On Apple computers with Apple silicon (e.g. M1 and M2)
processors, installing the ``tensorflow-metal`` Python package
into your Nighthawk Python environment with:

    pip install tensorflow-metal

can accelerate Nighthawk considerably. Unfortunately, however,
version 0.8.0 of the ``tensorflow-metal`` package (the most recent
version as of this writing) also appears to leak memory and cause
Nighthawk to hang when processing large (e.g. full-night) audio files
on our test machine, an M2 Mac mini with 8 GB of memory. For this
reason the package is not installed with Nighthawk by default.
Feel free to install it yourself, however, and test it for your
particular situation.

## Command Line Usage

Once you've installed Nighthawk, the simplest way to run it is to invoke
the `nighthawk` program from the command line. At an Anaconda Prompt
(on Windows) or terminal (on macOS or Linux), first activate your
Nighthawk environment with:

    conda activate nighthawk-0.1.0

Then, to run the `nighthawk` program on an audio file, say `my_file.wav`:

    nighthawk my_file.wav

`nighthawk` will output detections to the file `my_file.csv` in the
same directory as the input.

Adding the `--raven-output` flag will also export a `my_file.txt` file
that can be read as a selection table by
[Raven Pro](https://ravensoundsoftware.com/software/raven-pro/):

    nighthawk my_file.wav --raven-output

You can also specify more than one input file for `nighthawk` to process,
and specify relative or absolute file paths as well as just file names.

`nighthawk` has several command line arguments for configuring detection
parameters and controlling output. For full `nighthawk` help, including a
list of all command line arguments, run the command:

    nighthawk --help
  
to produce the output:

    usage: nighthawk [-h] [--hop-size HOP_SIZE] [--threshold THRESHOLD] [--merge-overlaps | --no-merge-overlaps]
                    [--drop-uncertain | --no-drop-uncertain] [--csv-output | --no-csv-output]
                    [--raven-output | --no-raven-output] [--output-dir OUTPUT_DIR_PATH]
                    input_file_path [input_file_path ...]

    positional arguments:
    input_file_path       path of audio file on which to run the detector.

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

## Use with Vesper
  
You can also use Nighthawk from Vesper, for example if you would like to
view and interact with spectrograms of its detections in Vesper clip albums.
For more on this see the
[Vesper documentation](https://vesper.readthedocs.io/en/latest/).

## Improving Nighthawk by correcting mistakes

See [this page](instructions/feedback/model_feedback.md) for instructions on how to review Nighthawk's results on your data and submit files to us to improve the model.

## Licensing and Citation

<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.

Under this license, you are free to share and adapt this material, but you must provide appropriate attribution, and you may not use the material for commercial purposes.
  
To reference Nighthawk, please use the following citation:

Van Doren, B. M., A. Farnsworth, K. Stone, D. M. Osterhaus, J. Drucker, and G. Van Horn. 2023. Nighthawk: acoustic monitoring of nocturnal bird migration in the Americas. 

<!-- bioRxiv. doi:[10.1101/541336](https://doi.org/10.1101/541336). 

BibTex:
    
    @article{vandorenNighthawkAcousticMonitoring2023,
        title = {Nighthawk: acoustic monitoring of nocturnal bird migration in the Americas},
        url = {https://www.biorxiv.org/content/10.1101/541336},
        doi = {10.1101/541336},
        journaltitle = {{bioRxiv}},
        author = {Van Doren, Benjamin M and Farnsworth, Andrew and Stone, Kate and Osterhaus, Dylan M and Drucker, Jacob and Van Horn, Grant},
        date = {2023},
    }
-->

## Code of Conduct

See our [code of conduct](CODE_OF_CONDUCT.md).

[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](code_of_conduct.md) 

## Contact and Collaborations

Please contact Benjamin Van Doren (vandoren@cornell.edu) with questions about Nighthawk. We are always open to scientific collaborations.  

<!-- ![Image of Zenodo DOI badge](https://zenodo.org/badge/DOI/DOIHERE) -->
