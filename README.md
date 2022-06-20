# OpenSRH: optimizing brain tumor surgery using intraoperative stimulated Raman histology

Companion code for OpenSRH. Paper submitted to NeurIPS 2022
Datasets and Benchmarks Track.

[**Paper Website**](https://opensrh.mlins.org) /
[**arXiv**](https://arxiv.org/abs/2206.08439) /
[**MLiNS Lab**](https://mlins.org)

## Installation

1. Clone OpenSRH github repo
   ```console
   git clone git@github.com:MLNeurosurg/opensrh.git
   ```
2. Install miniconda: follow instructions
    [here](https://docs.conda.io/en/latest/miniconda.html)
3. Create conda environment
    ```console
    conda create -n opensrh python=3.9
    ```
4. Activate conda environment
    ```console
    conda activate opensrh
    ```
5. Install package and dependencies
    ```console
    <cd /path/to/opensrh/repo/dir>
    pip install -e .
    ```

## Directory organization
- opensrh: the library for training with OpenSRH
    - datasets: PyTorch datasets to work with the data release
    - losses: loss functions for contrastive learning
    - models: PyTorch networks for training and evaluation
    - train: training and evaluation scrpits
- README.md
- LICENSE

# Training / evaluation instructions

The code base is written using PyTorch Lightning, with custom network and
datasets.

## Cross entropy experiments
1. Download and uncompress the data.
2. Update the sample config file in `train/config/train_ce.yaml` with desired
    configurations.
3. Change directory to `train` and activate the conda virtual environment.
4. Use `train/train_ce.py` to start training:
    ```console
    python train_ce.py -c config/train_ce.yaml
    ```

## Contrastive learning experiments
1. Download and uncompress the data.
2. Update the sample config file in `train/config/train_contrastive.yaml` with
    desired configurations.
3. Change directory to `train` and activate the conda virtual environment.
4. Use `train/train_contrastive.py` to start training:
    ```console
    python train_contrastive.py -c config/train_contrastive.yaml
    ```
5. To run linear or finetuning protocol, update the config file
    `train/config/train_finetune.yaml` and continue training using
    `train/train_finetune.py`:
    ```console
    python train_finetune.py -c config/train_finetune.yaml
    ```

## Model evaluation
1. Update the sample config file in `train/config/eval.yaml` with desired
    configurations, including the PyTorch Lightning checkpoint you would like
    to use.
2. Change directory to `train` and activate the conda virtual environment.
3. Use `train/train_ce.py` to start training:
    ```console
    python eval.py -c config/eval.yaml
    ```

## License Information
OpenSRH data is released under Attribution-NonCommercial-ShareAlike 4.0
International (CC BY-NC-SA 4.0), and the code is licensed under the MIT License.
See LICENSE for license information and THIRD\_PARTY for third party notices.
