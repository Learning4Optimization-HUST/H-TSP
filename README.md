# H-TSP

This is the implementation of our work, *H-TSP: Hierarchically Solving the Large-Scale Traveling Salesman Problem* (AAAI 2023)

https://www.microsoft.com/en-us/research/publication/h-tsp-hierarchically-solving-the-large-scale-traveling-salesman-problem/

## Operating System

Currently only Linux system is supported.

## Dependencies

### Main Dependency Packages

* [python](https://www.python.org/) >= 3.8
* [pytorch](https://pytorch.org/) == 1.10.0
* [cuda](https://developer.nvidia.com/cuda-toolkit) == 11.3
* [pytorch-lightning](pytorch-lightning) == 1.5.2
* [pytorch_geometric](https://pyg.org/)
* [dgl](https://www.dgl.ai/)
* [numba](https://numba.pydata.org/)
* [wandb](https://wandb.ai/)
* [hydra-core](https://hydra.cc/)
* [tensorboardX](https://github.com/lanpa/tensorboardX)
* [scikit-learn](https://scikit-learn.org/stable/index.html)
* [tensor-sensor\[torch\]](https://github.com/parrt/tensor-sensor)
* [LKH-3](http://akira.ruc.dk/~keld/research/LKH-3/)
* [tsplib95](https://github.com/rhgrant10/tsplib95)
* [lkh](https://github.com/ben-hudson/pylkh)
* [word2vec](https://github.com/danielfrg/word2vec)

Note that [LKH-3](http://akira.ruc.dk/~keld/research/LKH-3/) is a TSP/VRP solver written in C language, and [lkh](https://github.com/ben-hudson/pylkh) is its Python wrapper, please refer to [lkh](https://github.com/ben-hudson/pylkh) for more detail about the installation.


### Using Conda (Not tested)

One can use the [conda](https://anaconda.org/) package management system to build the development environment:

```bash
conda create --name htsp --file requirements.txt
```

And use
```bash
conda activate htsp
```
to activate this environment.

Note that [LKH-3](http://akira.ruc.dk/~keld/research/LKH-3/) still needs to be installed manully.

### Using Docker (Recommended)

[Docker](https://www.docker.com/) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) is required.

One can build a development environment for the `H-TSP` by using Docker with the `Dockerfile`:
```bash
docker build -t htsp .
```
Or use the existing docker image:
```bashs
docker pull neopan97/htsp:v0.3
# add a tag for consistency
docker tag neopan97/htsp:v0.3 htsp
```
After the htsp docker image is built or pulled from dockerhub, start the docker container with:
```bash
docker run -it --runtime=nvidia --mount type=bind,source=/path/to/source/code,target=/workspace htsp
```

## Code Structure

The main code of the upper-level model structure and the TSP environment locate in `h_tsp.py`.
Details about the neural networks and other miscellaneous are in `rl_models.py` and `rl_utils.py`.

The deep reinforcement learning training is in `train.py`, while the code for evaluation is in `evaluate.py`.
The experiment hyperparameters are in `config_ppo.yaml`.

Codes of the lower model are in the `rl4cop` folder. Refer to `README` in the folder for more details.

## Basic Usage

To train a model, firstly you need to modify the config file `config_ppo.yaml`, then run

```bash
python train.py
```

To test trained model, run
```bash
python evaluate.py --help
```
This will show you the parameters that need to be set.

**For more details about the parameters, please refer to the source code and our paper.**
