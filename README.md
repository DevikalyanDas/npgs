# Neural Parametric Gaussians for Monocular Non-Rigid Object Reconstruction (CVPR 2024)

[Devikalyan Das](https://devikalyandas.github.io/), [Christopher Wewer](https://geometric-rl.mpi-inf.mpg.de/people/Wewer.html), [Raza Yunus](https://www.utn.de/person/raza-yunus/), [Eddy Ilg](https://www.utn.de/person/eddy-ilg/), [Jan Eric Lenssen](https://geometric-rl.mpi-inf.mpg.de/people/lenssen.html)

Saarland University, Saarland Informatics Campus, Germany and Max Planck Institute for Informatics, Saarland Informatics Campus, Germany
### [Project Page](https://geometric-rl.mpi-inf.mpg.de/npg/)| [arXiv Paper](https://arxiv.org/abs/2312.01196)

#### The source code release is ongoing. Sorry for the delay. Please have patience!!!

## Setup (with [conda](https://docs.conda.io/en/latest/))
### 1. Install dependencies
Use the below command
```
mamba env create --file env.yaml
```
### 2. Download model weights
You can check the test images used in the paper and also download model weights from this [link](https://drive.google.com/drive/folders/1CeRQDJ5hJXXtYYf3AzDzMpsE8cEQ8qO1?usp=sharing)

### 3. Download data
You can download D-NeRF data from the [here](https://www.dropbox.com/scl/fi/cdcmkufncwcikk1dzbgb4/data.zip?rlkey=n5m21i84v2b2xk6h7qgiu8nkg&e=1&dl=0) and UB4D data from [here](https://drive.google.com/drive/folders/1lFhLqeNjslqgIuRpQnUlHbd5-56vaDNE)

### 4. TODO.
To train on a video
```
python run_train.py \
    -c <path to the config file> \
    -w <path to save the experiments/logs> \
    -d <path to the dataset directory> \
    -g 0 \
    -m train
```
