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

Add the images and the masks to the train, test, and val directories based on the `data_division.txt` (you can download it from [here](https://drive.google.com/file/d/1impULoCal0-gGriwiZh4ROhIlgFzwOrO/view?usp=sharing). The images and the corresponding masks should have the same name.

Next, we need to crop each image based on the mask. To do this, run `crop_images_masks.py`, which will crop all images from the `data` folder. The script will create 2 new folders: cropped_images and cropped_masks.

## Usage examples

Training:

* 2 different models are available: UNET++, DEEPLABV3+
- use `main.py` to train the models

Testing:

- use `test.py` to evaluate the models

## Model Checkpoints

The model checkpoints for the DeepLabV3+ and U-Net++ trained on the CELLULAR data set can be found [here](https://drive.google.com/drive/folders/1d4dgP2NLLR83QRsSNcr5zh9OFS065QaD?usp=sharing).
