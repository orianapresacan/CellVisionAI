import os
from ultralytics import YOLO
from  tracking.tracking import *
import shutil
import torch 
from torchvision import transforms
from PIL import Image
import segmentation_models_pytorch as smp
from tracking.delete_inconsistent_cells import delete_incomplete_cells


DATA_DIR = "data"
DEVICE = "cuda"


def normalize(image, mean, std):
    image = image.astype(np.float32) / 255.0
    mean = np.array(mean).reshape(1, 1, -1)
    std = np.array(std).reshape(1, 1, -1)
    normalized = (image - mean) / std
    return normalized.transpose(2, 0, 1)
    
def get_bounding_boxes():
    model = YOLO('checkpoints/best.pt')
    output_dir = os.path.join(os.getcwd(), "yolo_results")  

    os.makedirs(output_dir, exist_ok=True)

    for image_file in os.listdir(DATA_DIR):
        if image_file.lower().endswith((".jpg", ".jpeg", ".png")):  
            image_path = os.path.join(DATA_DIR, image_file)

            results = model(
                source=image_path,
                conf=0.55,
                save=False,  
                imgsz=2048,
                save_txt=True,
                save_crop=False,
                show_labels=False,
                show_conf=True,
                project=output_dir, 
                name="detections", 
                exist_ok=True 
            )

    print(f"Processing complete. Results saved in: {output_dir}")

def move_annotations():
    source_dir = "yolo_results/detections/labels"
    destination_dir = DATA_DIR
    root_dir = "yolo_results"

    if os.path.exists(source_dir):
        for file in os.listdir(source_dir):
            source_path = os.path.join(source_dir, file)
            destination_path = os.path.join(destination_dir, file)
            
            shutil.move(source_path, destination_path)
            print(f"Moved: {file} -> {destination_dir}")

        shutil.rmtree(root_dir)
        print(f"Deleted directory: {root_dir}")

    else:
        print(f"Source folder does not exist: {source_dir}")

def crop_objects_from_yolo_annotations():
    """
    Reads YOLO annotation files, crops objects from corresponding images, 
    and saves them in separate folders named after the original image.

    Args:
        images_dir (str): Path to the directory containing images and annotation (.txt) files.
    """
    for file in os.listdir(DATA_DIR):
        if file.endswith(('.jpg', '.jpeg', '.png')): 
            image_path = os.path.join(DATA_DIR, file)
            annotation_path = os.path.join(DATA_DIR, os.path.splitext(file)[0] + ".txt") 
            
            if not os.path.exists(annotation_path):
                print(f"Skipping {file}: No annotation file found.")
                continue

            image = cv2.imread(image_path)
            if image is None:
                print(f"Error loading image: {image_path}")
                continue

            img_height, img_width, _ = image.shape

            image_name = os.path.splitext(file)[0]
            cropped_dir = os.path.join(DATA_DIR, image_name)
            os.makedirs(cropped_dir, exist_ok=True)

            with open(annotation_path, "r") as f:
                lines = f.readlines()

            for idx, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 5:
                    print(f"Skipping line {idx + 1} in {annotation_path}: Invalid format")
                    continue

                _, x_center, y_center, width, height = map(float, parts)

                x_center *= img_width
                y_center *= img_height
                width *= img_width
                height *= img_height

                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)

                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img_width, x2), min(img_height, y2)

                cropped_img = image[y1:y2, x1:x2]

                if cropped_img.size > 0:
                    crop_filename = os.path.join(cropped_dir, f"{image_name}_{idx+1}.jpg")
                    cv2.imwrite(crop_filename, cropped_img)
                    print(f"Saved: {crop_filename}")
                else:
                    print(f"Skipping empty crop from {file}")

def get_segmentation_masks():
    OUTPUT_DIR = "data/segmentation_masks"  # Output root folder

    ENCODER = 'se_resnext50_32x4d'
    LOCAL_ENCODER_WEIGHTS = 'checkpoints/pretrained_encoder/se_resnext50_32x4d-a260b3a4.pth'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ['cell']
    ACTIVATION = 'sigmoid'

    model = smp.UnetPlusPlus(
        encoder_name=ENCODER,
        encoder_weights=None,
        classes=len(CLASSES),
        activation=ACTIVATION,
    )
    model.encoder.load_state_dict(torch.load(LOCAL_ENCODER_WEIGHTS))
    model.load_state_dict(torch.load(f"checkpoints/ckpt_UNET++.pt"))
    model.to(DEVICE)

    # Ensure the root output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for folder in os.listdir(DATA_DIR):  # Iterate through each folder
        folder_path = os.path.join(DATA_DIR, folder)
        
        if os.path.isdir(folder_path):  # Check if it's a directory
            subfolder_output_path = os.path.join(OUTPUT_DIR, folder)  # Correct path for subfolders
            os.makedirs(subfolder_output_path, exist_ok=True)  # Create subfolder if it doesn’t exist

            for file in os.listdir(folder_path): 
                file_path = os.path.join(folder_path, file)

                if os.path.isfile(file_path): 
                    image = Image.open(file_path)
                    image = image.resize((64, 64))
                    image = np.array(image)

                    image_tensor = normalize(image, mean=[0.4437, 0.4503, 0.2327], std=[0.2244, 0.2488, 0.0564])
                    image_tensor = torch.from_numpy(image_tensor).to(DEVICE).unsqueeze(0)
                    image_tensor = image_tensor.type(torch.float32)

                    with torch.no_grad():
                        pr_mask = model.predict(image_tensor)
                    pr_mask = pr_mask.squeeze(0).cpu().numpy().round()
                    pr_mask = pr_mask.reshape(64, 64, 1)

                    save_image = (pr_mask.astype(np.uint8) * 255).squeeze()
                    cropped_cell_image = Image.fromarray(save_image)

                    save_path = os.path.join(subfolder_output_path, file)  # Save inside the correct folder
                    cropped_cell_image.save(save_path)

                    print(f"Saved mask: {save_path}")

def restructure_folders():
    SEG_MASKS_DIR = os.path.join(DATA_DIR, "segmentation_masks")

    # Define the folders for grouping
    group_folders = {
        "C03_s3": os.path.join(DATA_DIR, "C03_s3"),
        "I06_s2": os.path.join(DATA_DIR, "I06_s2")
    }

    # Create target folders if they don't exist
    for folder in group_folders.values():
        os.makedirs(folder, exist_ok=True)
        os.makedirs(os.path.join(folder, "segmentation_masks"), exist_ok=True)

    # Iterate over files and folders in DATA_DIR (excluding the new group folders)
    for item in os.listdir(DATA_DIR):
        item_path = os.path.join(DATA_DIR, item)

        # **Skip moving the group folders themselves**
        if item in group_folders:
            continue  # Skip processing "C03_s3" and "I06_s2" folders

        # Move matching files and folders
        if "C03_s3" in item:
            shutil.move(item_path, os.path.join(group_folders["C03_s3"], item))

        elif "I06_s2" in item:
            shutil.move(item_path, os.path.join(group_folders["I06_s2"], item))

    # Move segmentation mask folders into the correct segmentation_masks subfolders
    if os.path.exists(SEG_MASKS_DIR):
        for mask_folder in os.listdir(SEG_MASKS_DIR):
            mask_folder_path = os.path.join(SEG_MASKS_DIR, mask_folder)

            if "C03_s3" in mask_folder:
                shutil.move(mask_folder_path, os.path.join(group_folders["C03_s3"], "segmentation_masks", mask_folder))
            
            elif "I06_s2" in mask_folder:
                shutil.move(mask_folder_path, os.path.join(group_folders["I06_s2"], "segmentation_masks", mask_folder))

        shutil.rmtree(SEG_MASKS_DIR)

def get_tracking():
    use_features = False

    for folder in os.listdir(DATA_DIR):
        folder_path = os.path.join(DATA_DIR, folder)
        print()
        
        if os.path.isdir(folder_path):  
            print(f"\n🔹 Processing folder: {folder}\n")
            
            MASK_DIR = os.path.join(folder_path, "segmentation_masks")
            active_ids_per_frame = {}

            prev_bboxes = None 
            consistently_active_ids = None

            all_images = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
            all_texts = sorted([f for f in os.listdir(folder_path) if f.endswith('.txt')])

            for frame_number, (image_file, text_file) in enumerate(zip(all_images, all_texts)):
                image_path = os.path.join(folder_path, image_file)
                text_path = os.path.join(folder_path, text_file)
                base_name = os.path.splitext(image_file)[0]
                mask_folder = os.path.join(MASK_DIR, base_name)
                
                print(f"--- Processing frame {frame_number} ---")
                print(f"Image: {image_file}, Text: {text_file}, Mask folder: {mask_folder}")
                
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Error: Could not read image {image_file}")
                    continue  
                
                current_bboxes, current_features = read_bboxes(text_path, use_features=use_features)
                current_masks, current_mask_bboxes = load_masks_and_bboxes(mask_folder)
                
                print(f"Frame {frame_number}: Detected {len(current_bboxes)} cells.")
                
                if frame_number == 0:
                    if use_features and current_features:
                        feature_iter = current_features
                    else:
                        feature_iter = [None] * len(current_bboxes)  
                    
                    prev_bboxes = [
                        {
                            'id': idx + 1,
                            'bbox': bbox,
                            'mask': msk,
                            'features': feat,
                            'merged': False,
                            'active': True,
                            'history': [bbox],
                            'age': 0
                        }
                        for idx, (bbox, msk, feat) in enumerate(zip(current_bboxes, current_masks, feature_iter))
                    ]
                    
                    consistently_active_ids = {b['id'] for b in prev_bboxes}
                    active_ids = consistently_active_ids  
                else:
                    prev_bboxes = track_bboxes_with_masks(
                        prev_bboxes,
                        current_bboxes,
                        current_masks,
                        current_features,
                        distance_threshold=30.0,  
                        max_age=0, 
                        use_features=use_features
                    )
                    
                    active_ids = {bbox['id'] for bbox in prev_bboxes if bbox.get('active', False)}
                    
                    if consistently_active_ids is not None:
                        consistently_active_ids &= active_ids
                    else:
                        consistently_active_ids = active_ids.copy()
                
                output_folder = os.path.join(folder_path, f'tracked/frame_{frame_number+1}')
                os.makedirs(output_folder, exist_ok=True)
                
                process_bboxes(image, prev_bboxes, output_folder, image_file, image_path, consistent_ids=consistently_active_ids, use_features=use_features)
                
                active_ids_per_frame[frame_number] = active_ids  

            if active_ids_per_frame:
                final_consistent_ids = set.intersection(*active_ids_per_frame.values())
                print(f"\nTotal consistently active cells across all frames in {folder}: {len(final_consistent_ids)}")
            else:
                print(f"No frames processed in {folder}.")

            delete_incomplete_cells(os.path.join(folder_path, 'tracked'))

def segment_overlay_tracked_frames():
    """
    Processes images inside 'C03_s3' and 'I06_s2':
    - Moves files from 'tracked' into their parent folders.
    - Iterates through 'frame_1' to 'frame_5' and applies segmentation.
    - Overlays the mask with a black background while keeping the real image in the white area.
    - Saves the modified images back to their original location.

    :param DATA_DIR: The root directory containing C03_s3 and I06_s2.
    :param DEVICE: The computation device (default: 'cuda').
    """

    # Model Configuration
    ENCODER = 'se_resnext50_32x4d'
    LOCAL_ENCODER_WEIGHTS = 'checkpoints/pretrained_encoder/se_resnext50_32x4d-a260b3a4.pth'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ['cell']
    ACTIVATION = 'sigmoid'

    # Load Model
    model = smp.UnetPlusPlus(
        encoder_name=ENCODER,
        encoder_weights=None,
        classes=len(CLASSES),
        activation=ACTIVATION,
    )
    model.encoder.load_state_dict(torch.load(LOCAL_ENCODER_WEIGHTS))
    model.load_state_dict(torch.load(f"checkpoints/ckpt_UNET++.pt"))
    model.to(DEVICE)

    # Move tracked contents into their parent folders
    for folder in ["C03_s3", "I06_s2"]:
        folder_path = os.path.join(DATA_DIR, folder)
        tracked_folder = os.path.join(folder_path, "tracked")

        if os.path.exists(tracked_folder):
            for subfolder in os.listdir(tracked_folder):
                src = os.path.join(tracked_folder, subfolder)
                dst = os.path.join(folder_path, subfolder)
                
                if os.path.isdir(src):
                    shutil.move(src, dst)
                    print(f"Moved {src} → {dst}")

            shutil.rmtree(tracked_folder)
            print(f"Deleted empty 'tracked' folder: {tracked_folder}")

    # Process images in frame_1 to frame_5
    for folder in ["C03_s3", "I06_s2"]:
        folder_path = os.path.join(DATA_DIR, folder)

        for frame_num in range(1, 6):  # Iterate over frame_1 to frame_5
            frame_folder = os.path.join(folder_path, f"frame_{frame_num}")
            if not os.path.exists(frame_folder):
                continue

            for file in os.listdir(frame_folder):
                file_path = os.path.join(frame_folder, file)

                if os.path.isfile(file_path) and file.endswith((".jpg", ".png")):
                    # Read and preprocess the image
                    image = Image.open(file_path).convert("RGB")
                    image = image.resize((64, 64))  
                    image_np = np.array(image)  # Shape: (64, 64, 3)

                    image_tensor = normalize(image_np, mean=[0.4437, 0.4503, 0.2327], std=[0.2244, 0.2488, 0.0564])
                    image_tensor = torch.from_numpy(image_tensor).to(DEVICE).unsqueeze(0).type(torch.float32)

                    # Predict mask
                    with torch.no_grad():
                        pr_mask = model.predict(image_tensor)

                    # Convert mask to binary
                    pr_mask = pr_mask.squeeze(0).cpu().numpy().round()  # Shape: (64, 64)
                    pr_mask = (pr_mask * 255).astype(np.uint8)  # Convert to binary (0=black, 255=white)

                    # Ensure mask matches (64, 64, 3)
                    pr_mask = np.expand_dims(pr_mask, axis=-1)  # (64, 64, 1)
                    pr_mask = np.repeat(pr_mask, 3, axis=-1)    # (64, 64, 3)

                    # Overlay mask onto original image
                    masked_image = np.where(pr_mask == 255, image_np, 0)  # Keep real image inside white mask

                    # Remove batch dimension (Fix for TypeError)
                    masked_image = masked_image.squeeze()  # Ensure shape is (64, 64, 3)

                    # Save the modified image (overwrite original)
                    output_image = Image.fromarray(masked_image.astype(np.uint8))
                    output_image.save(file_path)

                    print(f"Saved masked image: {file_path}")

def delete_folders():
    """
    Deletes all contents inside 'C03_s3' and 'I06_s2' except for 'frame_1' to 'frame_5'.
    
    :param DATA_DIR: Root directory containing 'C03_s3' and 'I06_s2'.
    """
    folders_to_keep = {f"frame_{i}" for i in range(1, 6)}  # Set of allowed folders: frame_1 to frame_5

    for folder in ["C03_s3", "I06_s2"]:
        folder_path = os.path.join(DATA_DIR, folder)

        if not os.path.exists(folder_path):
            print(f"Skipping {folder} (does not exist).")
            continue

        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)

            if os.path.isdir(item_path) and item in folders_to_keep:
                print(f"Keeping: {item_path}")  # Keep frame_1 to frame_5
            else:
                # Delete everything else
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    print(f"Deleted folder: {item_path}")
                else:
                    os.remove(item_path)
                    print(f"Deleted file: {item_path}")

def get_classification_labels():
    """
    Iterates through two parent folders inside DATA_DIR, classifies images using the trained model,
    and renames them by appending their class name.

    :param DATA_DIR: The root directory containing the parent folders.
    :param model_path: Path to the trained model.
    """
    class_labels = {0: "basal", 1: "activated", 2: "unidentified"}

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4437, 0.4503, 0.2327], std=[0.2244, 0.2488, 0.0564]), 
    ])

    model = torch.load('checkpoints/vit', map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()

    parent_folders = ["C03_s3", "I06_s2"]
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for folder in parent_folders:
        parent_path = os.path.join(DATA_DIR, folder)

        if not os.path.exists(parent_path):
            print(f"Skipping {parent_path} (does not exist).")
            continue

        for subdir, _, files in os.walk(parent_path):
            for file_name in files:
                if file_name.endswith('.png'):  
                    image_path = os.path.join(subdir, file_name)

                    image = Image.open(image_path).convert('RGB')
                    image = transform(image).unsqueeze(0).to(DEVICE)

                    with torch.no_grad():
                        outputs = model(image)
                        _, predicted = torch.max(outputs, 1)
                        predicted_class = predicted.item()

                    class_name = class_labels.get(predicted_class, "unknown")
                    base_name, ext = os.path.splitext(file_name)
                    new_file_name = f"{base_name}_{class_name}{ext}"
                    new_file_path = os.path.join(subdir, new_file_name)

                    os.rename(image_path, new_file_path)

                    print(f"Renamed: {file_name} -> {new_file_name}")


