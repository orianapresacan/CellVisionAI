import torch
import numpy as np
import segmentation_models_pytorch as smp
import cv2
from utils import resource_path


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model():
    ENCODER = 'se_resnext50_32x4d'
    LOCAL_ENCODER_WEIGHTS = resource_path('checkpoints/se_resnext50_32x4d-a260b3a4.pth')
    CLASSES = ['cell']
    ACTIVATION = 'sigmoid'

    model = smp.UnetPlusPlus(
            encoder_name=ENCODER,
            encoder_weights=None,
            classes=len(CLASSES),
            activation=ACTIVATION,
        )
    model.encoder.load_state_dict(torch.load(LOCAL_ENCODER_WEIGHTS))
    model.load_state_dict(torch.load(resource_path("checkpoints/ckpt_UNET++.pt"), map_location=torch.device(DEVICE)))
    return model


def get_segmentation(cropped_image):
    model = load_model()

    resized_cropped_image = cv2.resize(cropped_image, (64, 64))
    image_tensor = normalize(resized_cropped_image, [0.4437, 0.4503, 0.2327], [0.2244, 0.2488, 0.0564])
    image_tensor = torch.from_numpy(image_tensor).to(DEVICE).unsqueeze(0)
    image_tensor = image_tensor.type(torch.float32)

    with torch.no_grad():
        pr_mask = model.predict(image_tensor)
    pr_mask = pr_mask.squeeze(0).cpu().numpy().round()
    pr_mask = pr_mask.reshape(64, 64, 1)

    mask = pr_mask.astype(np.uint8) * 255 
    mask = mask.squeeze()
    return mask


def normalize(image, mean, std):
    image = image.astype(np.float32) / 255.0
    mean = np.array(mean).reshape(1, 1, -1)
    std = np.array(std).reshape(1, 1, -1)
    normalized = (image - mean) / std
    return normalized.transpose(2, 0, 1)

def denormalize(tensor, mean, std):
    mean = np.array(mean).reshape(-1, 1, 1)
    std = np.array(std).reshape(-1, 1, 1)
    denormalized = tensor.numpy() * std + mean
    denormalized = np.clip(denormalized, 0, 1) * 255
    return denormalized.astype(np.uint8).transpose(1, 2, 0)

# TEST THE CODE
# import classification
# image = cv2.imread('Timepoint_001_220518-ST_C03_s3.jpg')
# image_height, image_width, _ = image.shape
# _, xc, yc, w, h = 3, 0.1279296875, 0.15185546875, 0.076171875, 0.07421875
# x = int((xc - w / 2) * image_width)
# y = int((yc - h / 2) * image_height)
# w = int(w * image_width)
# h = int(h * image_height)
# cropped_image = image[y:y+h, x:x+w]
# mask = get_segmentation(cropped_image)
# mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

# full_mask = np.zeros_like(image, dtype=np.uint8)
# # Expand the resized_mask to 3 channels
# expanded_mask = np.stack([mask]*3, axis=-1)

# # Apply the expanded mask to the full_mask
# full_mask[y:y+h, x:x+w] = expanded_mask 

# # predicted_class = classification.get_classification(cropped_image, mask)

# plt.imshow(expanded_mask)
# plt.title('Segmentation Mask')
# plt.axis('off')  # Hide the axis
# plt.show()