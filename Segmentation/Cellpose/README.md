# Cellpose

## Overview

This repository enhances the approach provided by [this source](https://github.com/simula/cellular/tree/main), facilitating easy training and evaluation of [Cellpose 2.0](https://www.cellpose.org/).

## Installation

Please follow the installation guidelines available at the [original repository](https://github.com/simula/cellular/tree/main).

## Data

Ensure the data is structured into `train`, `valid`, and `test` directories, each containing the respective images and their masks.

## Usage Guide

### Model Download
Acquire the pre-trained model weights from [this link](https://drive.google.com/file/d/1zHGFYCqRCTwTPwgEUMNZu0EhQy2zaovg/view), utilizing the `cyto2torch_3` model for this project.

### Training and Evaluation Commands

```python
python experiments/train.py --train_dir experiments/my_data/train --valid_dir experiments/my_data/valid --experiment_name {}
```

```python
python experiments/eval.py --test_dir experiments/my_data/test --model_path {}  --output_dir {}
```
