# Segmentation models

This code is based on the [Segmentation Models Pytorch (SMP)](https://github.com/qubvel/segmentation_models.pytorch/tree/master) library.


## Prerequisites

```bash
pip install segmentation_models_pytorch
```

Download the pretrained autoencoder from [here](https://drive.google.com/file/d/1vxMdSfCnnCA1U1R2OuBOxdAGXBxFWqCN/view?usp=sharing) and add it in the pretrained_autoencoder directory.


## Dataset preparation

The kvasir dataset provided in the real_data/kvasir directory is used for validation and testing. Custom synthetic datasets can be used for training. These must be added in the synthetic_data directory and should have the following structure:

```bash
synthetic_data
│
├── model1_name
│   │
│   ├── images
│   │   ├── train
│   │
│   └── masks
│       ├── train
│
├── model2_name
│   │
│   ├── images
│   │   ├── train
│   │
│   └── masks
│       ├── train
│
└── ... (more models)
```

The images and the corresponding masks should have the same name.


## Usage examples

Training:

* 3 different models are available: UNET++, FPN, DEEPLABV3+
* one or more folders containing images can be used for training (validation and testing are done on the kvasir dataset)
* a custom number of images to use for training can be selected (by default it will use all images from the given folder)

```bash
    python main.py --model UNET++ --batch_size 2 --folders real_data/kvasir synthetic_data/LDM --number_images 100
```

* train with wandb:

```bash
python main.py --model FPN --folders real_data/kvasir --wandb True --wandb_api_key {api_key}
```

Testing:

* to visualize random samples together with the corresponding masks from the test dataset:

```bash
    python test.py --model FPN --checkpoint {path_to_trained_model}.pt --action get_samples --samples 5
```

* to get the values of the metrics for the whole dataset: 

```bash
    python test.py --model FPN --checkpoint {path_to_trained_model}.pt --action get_results
```

* to get both the metrics and the samples

```bash
    python test.py --model FPN --checkpoint {path_to_trained_model}.pt --samples 5
```
