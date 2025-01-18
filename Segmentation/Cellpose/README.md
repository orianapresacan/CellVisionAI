# Cellpose

This repository builds upon [this source](https://github.com/simula/cellular/tree/main), facilitating easy training and evaluation of [Cellpose 2.0](https://www.cellpose.org/).

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

Ensure that you have the original directories for the images and their corresponding masks. Use the `prepare_data.py` script to process the mask subfolders and create a single combined mask image per original image. It also divides the images into train, test, and valid folders. 

```bash
python prepare_data.py --input_path data --image_dirname images --mask_dirname masks --output_dir data_processed --split_file data_division.txt
```

If you would like to skip these steps, you can download our preprocessed dataset from [here](https://drive.google.com/file/d/1Sky-aXSKRp55PnARnVIrHt6teGpjNkpA/view?usp=sharing). 


## Training the model 

```python
python train.py --train_dir data_processed/train --valid_dir data_processed/valid --experiment_name <experiment_name>
```

## Evaluation

### Evaluating the fine-tuned model 

```python
python eval.py --test_dir data_processed/test --model_path <path/to/the/model/checkpoints> --output_dir <output_dir>
```

### Evaluating the pre-trained model 

```python
python eval_pretrained.py --test_dir data_processed/test --model_type cyto2 --output_dir <output_dir>
```

## Model Checkpoints
You can download our fine-tuned model weights from [here](https://drive.google.com/file/d/1X4g5hxLI2eUfRkcdlKhvcR9eVjQu0KHD/view?usp=sharing). 
