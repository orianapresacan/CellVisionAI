# tSNE

## Virtual Environment
Go to the project folder. Create a virtual environment:
```bash
python -m venv .venv
```

Activate the virtual environment:
```bash
source ./.venv/Scripts/activate
```

Install the requirements:
```bash
pip install -r requirements.txt
```

## Dataset Preparation
Download the dataset from [here](https://drive.google.com/file/d/1Ip1zDlZwIdZMy80kIBmps5sVu6uuG8K_/view?usp=sharing). This is the original data.

### Segment and Crop the Images

Start by running the `crop_and_segment_images.py` script to segment and crop all cells based on the provided masks. After segmentation, run the `rename_mix.py` script to shuffle the images and rename them by appending the corresponding cell class to each filename.

## Exatract Features

Once the images are prepared, extract their features using a pre-trained encoder model. You can choose from the following models: pre-trained DINOv2, pre-trained ResNet, or a UNet trained on your cell images.

To extract features, use the `extract_features.py` script and set the __model_type__ parameter to one of the following:
- unet
- resnet
- dinov2

For using the UNet, first train it using the `unet_training.py` script or download the pre-trained weights from [here](https://drive.google.com/file/d/1D6ME42dwBzFeugtOkVL-rnEsRMH3peQC/view?usp=sharing). To train the UNet, combine all cropped and segmented images into a single folder using the `merge_folders.py` script.

## Visualize with t-SNE

To visualize the extracted features, use t-SNE:

- Run `tsne_.py` to generate a t-SNE plot.
- Use `tsne_different_parameters.py` to experiment with different t-SNE parameters and find the best visualization settings.
- Run `tsne_with_tracking.py` 



