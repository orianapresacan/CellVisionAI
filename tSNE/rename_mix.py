import os
import shutil


def consolidate_and_rename_images(parent_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    folder_classes = {
        'fed': '_class_1',
        'unfed': '_class_2',
        'unidentified': '_class_3'
    }

    for root, dirs, files in os.walk(parent_folder):
        for dir_name in dirs:
            if dir_name in folder_classes:  
                suffix = folder_classes[dir_name]
                full_dir_path = os.path.join(root, dir_name)
                
                for file_name in os.listdir(full_dir_path):
                    file_path = os.path.join(full_dir_path, file_name)
                    
                    if not os.path.isfile(file_path):
                        continue
                    
                    file_name_without_ext, ext = os.path.splitext(file_name)
                    
                    new_file_name = f"{file_name_without_ext}{suffix}{ext}"
                    new_file_path = os.path.join(output_folder, new_file_name)
                    
                    shutil.move(file_path, new_file_path)

parent_folder = 'data/output'  
output_folder = 'data/mixed_real_images'  

consolidate_and_rename_images(parent_folder, output_folder)
