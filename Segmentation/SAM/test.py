from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from torch.utils.data import DataLoader
import cv2
import numpy as np
import torch
from typing import List, Dict, Any
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import dataset
import losses
import model_sam
import matplotlib.pyplot as plt


def get_total_mask(images, batched_masks):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = images.shape[0]
    _, H, W = images.shape[1:] 
    total_masks = torch.zeros((batch_size, H, W), device=DEVICE) 

    for idx, masks in enumerate(batched_masks):  
        for mask in masks:
            total_masks[idx] += mask.to(DEVICE)
    return total_masks

def build_totalmask(pred: List[Dict[str, Any]]) -> np.ndarray:
    """Builds a total mask from a list of segmentations
    ARGS:
        pred (list): list of dicts with keys 'segmentation' and others
    RETURNS:
        total_mask (np.ndarray): total mask

    """

    total_mask = np.zeros(pred[0]['segmentation'].shape, dtype=np.uint8)
    for seg in pred:
        total_mask += seg['segmentation']
    _, total_mask = cv2.threshold(total_mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


    return total_mask

def calculate_metrics(y_true, y_pred):
    """Calculate segmentation metrics."""
    y_true = y_true.squeeze().squeeze().detach().cpu().flatten().numpy()
    if FINETUNED:
        y_pred = y_pred.detach().cpu().flatten().numpy()
    else:
        y_pred = y_pred.flatten()

    iou = losses.binary_iou(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true.flatten(), y_pred.flatten(), average='binary')
    f1 = f1_score(y_true.flatten(), y_pred.flatten(), average='binary')
    accuracy = accuracy_score(y_true.flatten(), y_pred.flatten())
    return iou, precision, recall, f1, accuracy

def custom_collate_fn(batch):
    images, image_paths, masks = zip(*batch)  
    images = torch.stack(images)
    return images, image_paths, masks  

def plot(y_true, y_pred):
    y_true_squeezed = y_true.squeeze()
    y_pred_squeezed = y_pred.squeeze()

    y_true_np = y_true_squeezed.cpu().numpy() if torch.is_tensor(y_true_squeezed) else y_true_squeezed
    y_pred_np = y_pred_squeezed.cpu().numpy() if torch.is_tensor(y_pred_squeezed) else y_pred_squeezed

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(y_true_np, cmap='gray')
    axes[0].set_title('True Mask')
    axes[0].axis('off')

    axes[1].imshow(y_pred_np, cmap='gray')
    axes[1].set_title('Predicted Mask')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global FINETUNED 
    FINETUNED = 1

    test = "data/test"
    transform = dataset.ResizeAndPad(1024)
    testdata = dataset.COCODataset(root_dir=test,
                                annotation_file="data/test/annotations.json", 
                                transform=transform)
    test_dataloader = DataLoader(testdata,
                                batch_size=1,
                                shuffle=True,
                                num_workers=1, 
                                collate_fn=custom_collate_fn)

    iou_list, precision_list, recall_list, f1_list, accuracy_list = [], [], [], [], []

    if FINETUNED:
        model = model_sam.ModelSimple()
        model.setup()
        model.load_state_dict(torch.load('best_model.pth'))
        model = model.to(DEVICE)
    else:
        model = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth").to(DEVICE)
        mask_generator = SamAutomaticMaskGenerator(model)

    with torch.no_grad():
        model.eval()

        for images, _, batched_masks in test_dataloader:
            images = images.to(DEVICE)

            total_mask = get_total_mask(images, batched_masks)
            total_mask = (total_mask > 0).float()
            real_masks = total_mask.unsqueeze(0)

            if FINETUNED:
                predicted_masks, iou = model(images)
                preds_sigmoid = torch.sigmoid(predicted_masks)  
                predicted_masks = (preds_sigmoid > 0.5).float() 
            
            else:
                image_np = images.squeeze(0).cpu().numpy().transpose((1, 2, 0))
                image_np = image_np[:, :, ::-1]
                image_np = (image_np * 255).astype(np.uint8)
                masks = mask_generator.generate(image_np)
                predicted_masks = build_totalmask(masks)
                predicted_masks = (predicted_masks > 0.5).astype(np.int32)


            iou, precision, recall, f1, accuracy = calculate_metrics(real_masks, predicted_masks)
            iou_list.append(iou)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            accuracy_list.append(accuracy)

    mean_iou = np.mean(iou_list)
    mean_precision = np.mean(precision_list)
    mean_recall = np.mean(recall_list)
    mean_f1 = np.mean(f1_list)
    mean_accuracy = np.mean(accuracy_list)

    print(f"Mean IoU: {mean_iou}")
    print(f"Mean Precision: {mean_precision}")
    print(f"Mean Recall: {mean_recall}")
    print(f"Mean F1 Score: {mean_f1}")
    print(f"Mean Accuracy: {mean_accuracy}")


if __name__ == '__main__':
    main()