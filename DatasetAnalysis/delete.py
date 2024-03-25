import os
import shutil

root_dir = 'masks' 

for dirpath, dirnames, filenames in os.walk(root_dir):
    if 'Unidentified' in dirnames:
        path_to_delete = os.path.join(dirpath, 'Unidentified')
        
        # Delete the 'unidentified' subdirectory and all its contents
        shutil.rmtree(path_to_delete)
        print(f"Deleted {path_to_delete}")
