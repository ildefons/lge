# Latent Go-Explore


[![codecov](https://codecov.io/gh/qgallouedec/go-explore/branch/main/graph/badge.svg?token=f0yjhgL1nj)](https://codecov.io/gh/qgallouedec/go-explore)

Official implementation of Latent Go-Explore (LGE) algorithm.

Paper: [Cell-Free Latent Go-Explore](https://arxiv.org/abs/2208.14928)


## Installation

```bash
git clone https://github.com/qgallouedec/lge
cd lge
pip install -e .
```


### Installation CUDA driver (added by Ildefons)

* sudo apt update
* sudo apt install nvidia-driver-525  # Example for driver version 525
* sudo reboot


### Installation (added by Ildefons)

* create conda environment python 3.9
* python -m ipykernel install --user --name=lge39 --display-name "lge39"
* conda install ipykernel

* pip install setuptools==65.5.0 "wheel<0.40.0"
* python -m pip uninstall pip
* python -m ensurepip --upgrade
* python -m pip install pip==24.0 --force-reinstall
* pip install gym==0.21
 

* pip install tensorflow
* pip install opencv-python
* pip install git+https://github.com/qgallouedec/gym-continuous-maze.git
* pip install wandb
* pip install qdarkstyle
* pip install aiohttp

* pip install -e . 

## Usage


```python
from stable_baselines3 import SAC

from lge import LatentGoExplore

lge = LatentGoExplore(SAC, "MountainCarContinuous-v0")
lge.explore(total_timesteps=10_000)
```

Or via command line

```shell
python experiments/explore_lge.py --env MountainCarContinuous-v0 --algo sac
```

Supported envrionments specifications:

| Space           | Observation space  |
| --------------- | ------------------ |
| `Discrete`      | :heavy_check_mark: |
| `Box`           | :heavy_check_mark: |
| Image           | :heavy_check_mark: |
| `MultiDiscrete` | :x:                |
| `MultiBinary`   | :heavy_check_mark: |

| Space      | Action space       |
| ---------- | ------------------ |
| `Box`      | :heavy_check_mark: |
| `Discrete` | :heavy_check_mark: |


Image is actually a multi-dimensonal uint8 `Box`.
