import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any
import model_sam
from torchvision.transforms.functional import to_tensor
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


def build_totalmask(pred: List[Dict[str, Any]], shape) -> np.ndarray:
    """
    Builds a total mask from a list of segmentations.

    Args:
        pred (list): List of dicts with keys 'segmentation' and others.
        shape (Tuple[int, int]): The shape of the output mask.

    Returns:
        total_mask (np.ndarray): Total mask.
    """
    total_mask = np.zeros(shape, dtype=np.uint8)
    for seg in pred:
        mask = seg['segmentation'].astype(np.uint8)
        total_mask = np.maximum(total_mask, mask * 255)  
    return total_mask

def overlay_mask_on_image(image: np.ndarray, total_mask: np.ndarray, color: tuple, alpha: float = 0.5) -> np.ndarray:
    """
    Adjusted to overlay masks on top of the real images.
    
    Args:
        image (np.ndarray): Original image in RGB format.
        total_mask (np.ndarray): Binary mask for all segmentations.
        color (tuple): Color for the masks in (R, G, B).
        alpha (float): Transparency factor for the overlay.
    
    Returns:
        overlaid_image (np.ndarray): Image with mask overlaid.
    """
    overlay = np.zeros_like(image)
    overlay[:, :] = color
    
    where_mask = total_mask.astype(bool)
    
    image[where_mask] = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)[where_mask]
    
    return image


def main():
    FINETUNED = 1
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sample_img = cv2.imread('data/sample_image/image/Timepoint_001_220518-ST_C03_s3.jpg')
    sample_img_rgb = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)

    if not FINETUNED:
        sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth").to(DEVICE)
        mask_generator = SamAutomaticMaskGenerator(sam)

        masks = mask_generator.generate(sample_img)

        mask_shape = sample_img_rgb.shape[:2] 
        predicted_mask = build_totalmask(masks, mask_shape)
    else:
        model = model_sam.ModelSimple()
        model.setup()
        model.load_state_dict(torch.load('best_model.pth'))
        model.to(DEVICE)

        sample_img_rgb = cv2.resize(sample_img_rgb, (1024, 1024))
        sample_img_rgb_tensor = to_tensor(sample_img_rgb)  # This also converts the image to [0, 1] range

        sample_img_rgb_tensor = sample_img_rgb_tensor.unsqueeze(0).to(DEVICE)  # shape [1, 3, 1024, 1024]

        predicted_masks, iou = model(sample_img_rgb_tensor)

        preds_sigmoid = torch.sigmoid(predicted_masks)  
        threshold = 0.5  
        predicted_mask = (preds_sigmoid > threshold).float() 
        # preds_binary = resize(preds_binary,[2048,2048])
        predicted_mask = predicted_mask.detach().cpu().squeeze().numpy()

    mask_color = (128, 0, 128)
    mask_transparency = 0.5

    overlaid_image = overlay_mask_on_image(sample_img_rgb, predicted_mask, mask_color, mask_transparency)

    plt.figure(figsize=(10, 10))
    plt.imshow(overlaid_image)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()