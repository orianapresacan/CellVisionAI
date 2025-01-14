import os
import cv2
import shutil 


def convert_and_mix_masks(parent_folder, output_folder):
    """
    Convert all mask images in the parent folder's subfolders to binary masks,
    keep the 5 main folders, and delete their subfolders, mixing the images into the main folders.

    :param parent_folder: Path to the parent folder containing main folders with subfolders of mask images.
    :param output_folder: Path to the output folder where processed masks will be saved.
    """
    for main_folder in os.listdir(parent_folder):
        main_folder_path = os.path.join(parent_folder, main_folder)
        if os.path.isdir(main_folder_path):
            output_main_folder = os.path.join(output_folder, main_folder)
            os.makedirs(output_main_folder, exist_ok=True)

            for root, _, files in os.walk(main_folder_path):
                for file in files:
                    if file.endswith('.png') or file.endswith('.jpg'):
                        input_path = os.path.join(root, file)
                        output_path = os.path.join(output_main_folder, file)

                        base_name, ext = os.path.splitext(file)
                        counter = 1
                        while os.path.exists(output_path):
                            output_path = os.path.join(output_main_folder, f"{base_name}_{counter}{ext}")
                            counter += 1

                        mask = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
                        if mask is None:
                            print(f"Warning: Could not read {input_path}")
                            continue

                        _, binary_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

                        cv2.imwrite(output_path, binary_mask)
                        print(f"Processed and saved: {output_path}")

            for subfolder in os.listdir(main_folder_path):
                subfolder_path = os.path.join(main_folder_path, subfolder)
                if os.path.isdir(subfolder_path):
                    shutil.rmtree(subfolder_path)
                    print(f"Deleted subfolder: {subfolder_path}")

parent_folder = "data/I06_s2"  
output_folder = "data/I06_s2_binary_mixed"   
convert_and_mix_masks(parent_folder, output_folder)
