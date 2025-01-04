import argparse
import os
import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM, XGradCAM, EigenCAM, AblationCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu', help='Torch device to use')
    parser.add_argument('--image-folder', type=str, required=True, help='Folder containing input images')
    parser.add_argument('--output-folder', type=str, required=True, help='Folder to save XAI images')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'gradcam++', 'scorecam', 'xgradcam', 'eigencam', 'kpcacam', 'ablationcam'],
                        help='XAI method to use')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the pretrained model')
    return parser.parse_args()

def reshape_transform(tensor):
    tensor = tensor[:, 1:, :]  # Remove class token
    h = w = int(tensor.shape[1]**0.5)
    return tensor.reshape(tensor.size(0), h, w, tensor.size(2)).permute(0, 3, 1, 2)

if __name__ == '__main__':
    args = get_args()

    methods = {
        'gradcam': GradCAM,
        'gradcam++': GradCAMPlusPlus,
        'scorecam': ScoreCAM,
        'xgradcam': XGradCAM,
        'eigencam': EigenCAM,
        'ablationcam': AblationCAM
    }

    model = torch.load(args.model_path).to(torch.device(args.device)).eval()
    target_layers = [model.encoder.layers[11].ln_1]  # Choose the appropriate layer

    if args.method == "ablationcam":
        cam = methods[args.method](model=model,
                                   target_layers=target_layers,
                                   reshape_transform=reshape_transform,
                                   ablation_layer=AblationLayerVit())
    else:
        cam = methods[args.method](model=model,
                                   target_layers=target_layers,
                                   reshape_transform=reshape_transform)

    os.makedirs(args.output_folder, exist_ok=True)

    for image_name in os.listdir(args.image_folder):
        image_path = os.path.join(args.image_folder, image_name)
        if not image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # Load and preprocess the image
        rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]  # BGR to RGB
        rgb_img = cv2.resize(rgb_img, (224, 224))
        rgb_img = np.float32(rgb_img) / 255
        input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]).to(args.device)

        # Generate CAM
        grayscale_cam = cam(input_tensor=input_tensor)[0, :]

        # Overlay CAM on the image
        cam_image = show_cam_on_image(rgb_img, grayscale_cam)

        # Save the CAM image
        output_path = os.path.join(args.output_folder, f"{os.path.splitext(image_name)[0]}_{args.method}.jpg")
        cv2.imwrite(output_path, cam_image)

        print(f"Saved: {output_path}")
