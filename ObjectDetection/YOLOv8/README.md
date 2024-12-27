# YOLO

This guide for fine-tuning object detection models follows instructions from [Ultralytics tutorials](https://docs.ultralytics.com/tasks/detect/).

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

## Data
### Understand the Annotation Format 
The CELLULAR dataset provides bounding box annotations in YOLO format, which is compatible with YOLOv8. Each line in the annotation file follows this structure:
```bash
<class_id> <x_center> <y_center> <width> <height>
```
where the <class_id> shows which class the cell belongs to. 

### Remove Classification 
By default, YOLO models perform object classification. However, we want to treat all objects (e.g., cells) as belonging to a single class. Thus, we must modify the <class_id> values in the annotation files to 0 (from 1, 2, etc.). Use `modify_class_id.py` to automate this process.

### Organize the Dataset
Next, structure your dataset as follows, ensuring that the labels folder contains text files with the modified bounding box annotations. 

    data
    ├── train
    │   ├── images
    │   └── labels
    └── val
        ├── images
        └── labels

Place training images in train/images and their corresponding annotation text files in train/labels. Do the same for validation data, placing images in val/images and annotations in val/labels. Ensure that each image has a corresponding text file with the same name (e.g., image1.jpg and image1.txt).

Refer to `data_division.txt` from the GitHub repository for guidance on which data belongs in train and val. 

I have provided the data in a ready-to-use folder. See the `data` directory.

### Add Configuration File
Copy the `cell.yaml` file from the repository and paste it into your project directory. Verify that the folder paths in `cell.yaml` are correct.

You might need to modify the "dataset_dir" path in the `settings.json` file from the Ultralytics directory, usually installed in 'C:/Users/your_user/AppData/Roaming/Ultralytics'.

## Train the YOLOv8 Model

- **Model Fine-Tuning:**

```python
yolo train model=yolov8x.pt data=cell.yaml epochs=500 imgsz=2048 batch=4
```
```python
yolo train model=yolov8l.pt data=cell.yaml epochs=500 imgsz=2048 batch=4
```

## Evaluate the fine-tuned YOLOv8 Model
  
```python
yolo detect val model=path/to/checkpoints/best.pt data=cell.yaml
```
This will give you metrics such as Precision, Recall, mAP50, and mAP50-95 on the test data.


- **Model Checkpoints:**

You can download our trained yolo8x model from [here](https://drive.google.com/drive/folders/1ns0jNeTzDgscYFeK1cU-nZG7AMwvMIKt?usp=sharing).
