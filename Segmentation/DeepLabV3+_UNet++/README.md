# U-Net++ and DeepLabV3+ segmentation models

This code is based on the [Segmentation Models Pytorch (SMP)](https://github.com/qubvel/segmentation_models.pytorch/tree/master) library.

## Prerequisites

```bash
pip install segmentation_models_pytorch
```

Download the pre-trained autoencoder from [here](https://drive.google.com/file/d/1vxMdSfCnnCA1U1R2OuBOxdAGXBxFWqCN/view?usp=sharing) and add it to the pretrained_autoencoder directory.


## Dataset preparation

The data must have the following structure:

```bash
data
│
├── train
│   ├── images
│   └── masks
├── val
│   ├── images
│   └── masks
├── test
    ├── images
    └── masks
```

The images and the corresponding masks should have the same name.


## Usage examples

Training:

* 2 different models are available: UNET++, DEEPLABV3+
- use `main.py` to train the models

Testing:

- use `test.py` to evaluate the models

## Model Checkpoints

The model checkpoints for the DeepLabV3+ and U-Net++ trained on the CELLULAR data set can be found [here](https://drive.google.com/drive/folders/1d4dgP2NLLR83QRsSNcr5zh9OFS065QaD?usp=sharing).
