import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*(list(model.children())[:-1]))
model.eval()

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4437, 0.4503, 0.2327], std=[0.2244, 0.2488, 0.0564]), 
])

def extract_features(image):
    with torch.no_grad():
        feature_vector = model(image)
    return feature_vector.view(1, -1)


def process_data(detections_dir, images_dir):
    for detection_file in os.listdir(detections_dir):
        if detection_file.endswith(".txt"):
            detection_path = os.path.join(detections_dir, detection_file)
            image_file = os.path.splitext(detection_file)[0] + ".jpg"
            image_path = os.path.join(images_dir, image_file)
            
            image = Image.open(image_path)
            image = transform(image).unsqueeze(0)  
            
            with open(detection_path, 'r') as f:
                detections = f.readlines()
            
            updated_detections = []
            for detection in detections:
                parts = detection.strip().split()
                x_center, y_center, width, height = map(float, parts[1:])
                
                left = int((x_center - width / 2) * image.size()[3])
                top = int((y_center - height / 2) * image.size()[2])
                right = int((x_center + width / 2) * image.size()[3])
                bottom = int((y_center + height / 2) * image.size()[2])
                cropped_image = image[:, :, top:bottom, left:right]
                
                cropped_features = extract_features(cropped_image)
                
                updated_detection = detection.strip() + ' ' + ' '.join(map(str, cropped_features.squeeze().numpy())) + '\n'
                updated_detections.append(updated_detection)
            
            with open(detection_path, 'w') as f:
                f.writelines(updated_detections)

detections_dir = 'data/C03_s3'
images_dir = detections_dir
process_data(detections_dir, images_dir)
