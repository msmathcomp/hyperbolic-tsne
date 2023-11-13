# HyperbolicTSNE

This repository contains the code of the paper ... TODO

## Setup

1. Install a conda (we recommend using [miniconda](https://docs.conda.io/projects/miniconda/en/latest/))
2. Create environment: `conda create --name=htsne python=3.9.16`
3. Activate environment: `conda activate htsne`
4. Install dependencies with pip: `pip install -r requirements.txt`
5. Build Cython extensions: `python setup.py build_ext --inplace`
6. Install hyperbolic-tsne package: `pip install .`
7. To test installation run `python -c "from hyperbolicTSNE import HyperbolicTSNE"`. No errors should be raised and you should see the output `Please note that 'empty_sequence' uses the KL divergence with Barnes-Hut approximation (angle=0.5) by default.`.

Note: on macOS, the build process of the Cython extensions might yield an error if it cannot find OpenMP.
This error can be ignored and the package will still be correctly installed and able to run. 
The main consequence of this error is that the optimization iterations are going to run slower.

## Data

You can run hyperbolic TSNE on your high-dimensional data. 
Nevertheless, the examples and experiments in this repository rely on specific datasets. 
Below, we provide downloading and processing instructions for each. 
We recommend putting all datasets in a `datasets` directory at the root of this repository.
The `load_data` function expects this path (`data_home`) to resolve the dataset.

Individual instructions per dataset:
- LUKK
- MYELOID8000
- PLANARIA: https://shiny.mdc-berlin.de/psca/ 
- MNIST
- WORDNET
- C_ELEGANS: https://data.caltech.edu/records/1945 

## First steps

There are two ways of getting stated with the `hyperbolicTSNE` package after setting it up.
First, `example_basic_usage.ipynb` offers a step-by-step guide showing how to use the HyperbolicTSNE package to embed a high-dimensional dataset. 
Second, the `example_different_params.py` script shows how to setup a script for quick experimentation. In this case, to compare the effect of different parameters.


## Replicating the paper results

TODO

## References

TODO