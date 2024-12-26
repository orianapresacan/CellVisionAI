import os
import torch
import argparse
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import train
import metrics
from PIL import Image
from dataset import CellDataset
import helpers 


ENCODER = 'se_resnext50_32x4d'
LOCAL_ENCODER_WEIGHTS = 'pretrained_encoder/se_resnext50_32x4d-a260b3a4.pth'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['cell']
ACTIVATION = 'sigmoid'


def visualize_samples(number_of_samples):
    for i in range(number_of_samples):
        # n = np.random.choice(len(test_dataset))
        
        image, gt_mask = test_dataset[i]
        denormalized_image = helpers.denormalize(image, [0.4437, 0.4503, 0.2327], [0.2244, 0.2488, 0.0564])

        x_tensor = image.to(DEVICE).unsqueeze(0)
        pr_mask = model.predict(x_tensor)
        
        pr_mask = pr_mask.squeeze(0).cpu().numpy().round()

        grid = helpers.visualize(
            image=denormalized_image, 
            ground_truth_mask=np.transpose(gt_mask*255, (1, 2, 0)), 
            predicted_mask=np.transpose(pr_mask*255, (1, 2, 0)), 
        )
        grid.save(f"test_outputs/sample_{i}.png")

def get_segmented_cells(main_folder_path, get_only_mask):
    subfolders = ['frame_0', 'frame_1', 'frame_2', 'frame_3', 'frame_4'] 

    for subfolder in subfolders:
        input_folder_path = os.path.join(main_folder_path, subfolder)
        output_folder_path = os.path.join(main_folder_path, f"cropped_{subfolder}")

        # Create output directory if it doesn't exist
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)

        # Iterate over each image file in the subfolder
        for file in os.listdir(input_folder_path):
            # Construct the full file path
            file_path = os.path.join(input_folder_path, file)
            
            # Load the image
            image = Image.open(file_path)
            image = np.array(image)

            # Normalize and prepare the image tensor
            image_tensor = helpers.normalize(image, mean=[0.4437, 0.4503, 0.2327], std=[0.2244, 0.2488, 0.0564])
            image_tensor = torch.from_numpy(image_tensor).to(DEVICE).unsqueeze(0)
            image_tensor = image_tensor.type(torch.float32)

            # Get the binary mask from the model
            with torch.no_grad():
                pr_mask = model.predict(image_tensor)
            pr_mask = pr_mask.squeeze(0).cpu().numpy().round()
           
            pr_mask = pr_mask.reshape(64, 64, 1)

            if get_only_mask:
                save_image = pr_mask.astype(np.uint8) * 255  # Scale up to make the mask visible
                save_image = save_image.squeeze()
            else:
                # Crop the image based on the mask and add a black background
                masked_image = image * pr_mask
                background = np.zeros_like(image)
                cropped_cell = np.where(pr_mask, masked_image, background)

                # Ensure that cropped_cell is in 'uint8' format
                save_image = cropped_cell.astype(np.uint8)

            # Save the cropped cell image in the corresponding output subfolder
            cropped_cell_image = Image.fromarray(save_image)
            cropped_cell_image.save(os.path.join(output_folder_path, os.path.basename(file)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=__name__, description='cell')
    parser.add_argument('--action', type=str, choices=["get_samples", "get_results", "get_samples_and_results", 'get_segmented_cells'], default=("get_samples_and_results"))
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--model', type=str, choices=["FPN", "UNET++", "DEEPLABV3+"], required=True)
    options, unknown = parser.parse_known_args()

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 0
    BATCH_SIZE = 1
    MODEL = options.model 

    model_types = { 
        "UNET++": smp.UnetPlusPlus,
        "FPN": smp.FPN, 
        "DEEPLABV3+": smp.DeepLabV3Plus,
    }

    model = model_types[MODEL](
        encoder_name=ENCODER,
        encoder_weights=None,
        classes=len(CLASSES),
        activation=ACTIVATION,
    )
    model.encoder.load_state_dict(torch.load(LOCAL_ENCODER_WEIGHTS))
    model.load_state_dict(torch.load(f"checkpoints/ckpt_{MODEL}.pt"))
    
    # Test dataset is always the same
    test_dataset = CellDataset(["data/test/cropped_images"], ["data/test/cropped_masks"])
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    loss = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
    metrics = [metrics.IoU(threshold=0.5), metrics.Fscore(), metrics.Precision(), metrics.Recall(), metrics.Accuracy()]

    test_epoch = train.ValidEpoch(
        model=model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
    )

    if options.action == 'get_samples':
        visualize_samples(options.samples)
    elif options.action == 'get_results':
        logs = test_epoch.run(test_loader)
    elif options.action == 'get_segmented_cells':
        get_segmented_cells('data/test/Deeplab_segm', get_only_mask=True)
    else:
        logs = test_epoch.run(test_loader)
        visualize_samples(options.samples)