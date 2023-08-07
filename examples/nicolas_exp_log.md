# Experiment log
Nicolas

These are some notes as to how I am conducting the experiments. 

## How to run the code

The following code runs hyperbolic tSNE on a given dataset and generates some images of the run and a mp4 video.
`cd examples`
`python run_example.py`

Don't forget to:
- Check which dataset is being used.
- Check the running parameters.
- Recompile stuff if something changed in the library.

To recompile stuff run the following commands from the root:
- `conda deactivate`
- `conda activate hdeo`
- `pip uninstall hyperbolic-tsne`
- `cd hyperbolic-tsne`
- `rm -r hyperbolic_tsne.egg-info build`
- `rm hyperbolicTSNE/tsne_barnes_hut_hyperbolic.c* hyperbolicTSNE/tsne_barnes_hut.c* hyperbolicTSNE/tsne_utils.c*`
- `python setup.py build_ext --inplace`
- `pip install .`

To run these commands all at once, do:
- `cd examples`
- `python cleanup_mac.py`

## Things to change

Things that can change in the code:
- Version 1: this was the version of the code we used for the first iteration of the paper. Parameters should be fixed to obtain the same results.
    - Perplexity: 30
    - Optimization: vanilla gradient descent.
    - Learning rate: `N/(12 * 50)` (Hunter's heuristic)
    - Early exaggeration: `12` for `250` iterations
    - Regular optimization: `750` iterations
- Version 2a: Only gradient fix (in `tsne_barrnes_hut_hyperbolic.pyx`) (same parameters as version 1)
- Version 2b: this is the version of the code after fixing the gradient implementation. In addition to the gradient fix of version 2a it also has the optimization scale fix (in `solver_.py`) (same parameters as version 2)
    - Gradient fix: `yes`
    - Scale fix: `yes`
    - Perplexity: `[1, 30, 100, 1000]`
    - Optimization: `[vanilla, momentum vdm, momentum hyperbolic]`
    - Learning rate: `N/(12 * 50) * [0.2, 0.5, 1, 10, 100]`
    - Early exaggeration
    - Regular optimization

## Observations

If we run V1 on on a `N=10k` sample of MNIST we obtain nice separation of the clusters.