import torch
import numpy as np
import cv2
from torchvision.models import vit_b_32
from utils import resource_path


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    model = vit_b_32(pretrained=True)
    model.heads.head = torch.nn.Linear(in_features=model.heads.head.in_features, out_features=3)
    model = torch.load(resource_path('checkpoints/vit'), map_location=torch.device(DEVICE))
    model.eval()  
    return model


def get_classification(cropped_image, resized_mask):
    model = load_model()  

    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    segmented_image = cropped_image.copy()
    for c in range(3): 
        segmented_image[:, :, c] = np.where(resized_mask > 0, segmented_image[:, :, c], 0)
    # PLOT
    # plt.imshow(segmented_image)
    # plt.title('Segmentation Mask')
    # plt.axis('off')  # Hide the axis
    # plt.show()
    # Normalize the image
    segmented_image = segmented_image.astype(np.float32) / 255.0
    mean = np.array([0.4437, 0.4503, 0.2327]).reshape(1, 1, 3)
    std = np.array([0.2244, 0.2488, 0.0564]).reshape(1, 1, 3)
    normalized_image = (segmented_image - mean) / std
    resized_image = cv2.resize(normalized_image, (224, 224), interpolation=cv2.INTER_LINEAR)

    image_tensor = torch.from_numpy(resized_image).to(DEVICE).permute(2, 0, 1).unsqueeze(0)  # CHW format
    image_tensor = image_tensor.type(torch.float32)

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = predicted.item()

    return predicted_class
