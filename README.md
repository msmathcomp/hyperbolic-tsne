# HyperbolicTSNE

This repository contains the code of the paper ... TODO

## Setup

1. Install a conda (we recommend using [miniconda](https://docs.conda.io/projects/miniconda/en/latest/))
2. Create environment: `conda create --name=htsne python=3.9.16`
3. Activate environment: `conda activate htsne`
4. Install dependencies with pip: `pip install -r requirements.txt`
5. Build Cython extensions: `python setup.py build_ext --inplace`
6. Install hyperbolic-tsne package: `pip install .`
7. Remove unnecessary files: `rm -r hyperbolic_tsne.egg-info build` and `rm hyperbolicTSNE/tsne_barnes_hut_hyperbolic.c* hyperbolicTSNE/tsne_barnes_hut.c* hyperbolicTSNE/tsne_utils.c*`
8. Test installation: `cd examples && python run_example.py && cd ..` TODO: some function to generate embedding


## First steps

`example_basic_usage.ipynb` is a step and step guide that shows how to use the HyperbolicTSNE package to embed a high-dimensional dataset. `example_different_params.py` is a script that permits shows how to set is a script for quick experimentation.


## Replicating the paper results

TODO

## References

TODO