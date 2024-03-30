import os
import shutil

def process_subfolders(base_folder):
    for subfolder in os.listdir(base_folder):
        subfolder_path = os.path.join(base_folder, subfolder)
        
        if os.path.isdir(subfolder_path):
            frame_paths = [os.path.join(subfolder_path, f"frame_{i}") for i in range(1, 6)]

            if all(os.path.exists(frame_path) for frame_path in frame_paths):
                for cell_image in os.listdir(frame_paths[0]):
                    corresponding_images = [cell_image.replace("frame_1", f"frame_{i}") for i in range(2, 6)]

                    if all(os.path.exists(os.path.join(frame_paths[i], corresponding_images[i-1])) for i in range(1, 5)):
                        new_folder_name = f"{subfolder}_{cell_image[:-4]}"
                        new_folder_path = os.path.join(base_folder, new_folder_name)
                        os.makedirs(new_folder_path, exist_ok=True)

                        shutil.copy(os.path.join(frame_paths[0], cell_image), os.path.join(new_folder_path, "frame_1.png"))
                        for i in range(1, 5):
                            shutil.copy(os.path.join(frame_paths[i], corresponding_images[i-1]), os.path.join(new_folder_path, f"frame_{i+1}.png"))

base_folder = 'train'  
process_subfolders(base_folder)
