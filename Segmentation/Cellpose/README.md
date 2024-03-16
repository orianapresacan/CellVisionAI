# Cellpose

## Overview

This repository is based on this [repository](https://github.com/simula/cellular/tree/main), which provides a simple way of training and evaluating [Cellpose 2.0](https://www.cellpose.org/).

## Installation

Follow the installition information from [repository](https://github.com/simula/cellular/tree/main).

## Data

- The data must be divided into train, valid, and test folders. Each folder should contain the images and the corresponding masks.

## Usage Guide

- **Model Download:** Download the weights of the pre-trained model from [here](https://drive.google.com/file/d/1zHGFYCqRCTwTPwgEUMNZu0EhQy2zaovg/view). For this project, cyto2torch_3 model was used.
  
The following commands were used for training and testing the model:

```python
python experiments/train.py --train_dir experiments/my_data/train --valid_dir experiments/my_data/valid --experiment_name {}
```

```python
python experiments/eval.py --test_dir experiments/my_data/test --model_path {}  --output_dir {}
```
