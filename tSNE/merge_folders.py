import os
import shutil

def merge_images_from_folders(base_dir, output_dir):
    """
    Merge images from several subfolders across multiple folders into a single destination folder.
    
    Args:
        base_dir (str): Path to the base directory containing subfolders.
        output_dir (str): Path to the output directory where all images will be merged.
    """
    os.makedirs(output_dir, exist_ok=True)

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                source_path = os.path.join(root, file)
                dest_path = os.path.join(output_dir, file)

                if os.path.exists(dest_path):
                    base_name, ext = os.path.splitext(file)
                    counter = 1
                    while os.path.exists(dest_path):
                        dest_path = os.path.join(output_dir, f"{base_name}_{counter}{ext}")
                        counter += 1
                
                shutil.copy2(source_path, dest_path)
                print(f"Copied {source_path} to {dest_path}")

    print(f"All images have been merged into: {output_dir}")

merge_images_from_folders(base_dir="output", output_dir="mixed")
