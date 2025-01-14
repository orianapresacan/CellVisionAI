import os
import re

def delete_incomplete_cells(main_dir):
    """
    Scans all subfolders of 'main_dir' for images named like 'cell_X_frame_Y.png'.
    Keeps only those cells that appear in ALL subfolders. Deletes the rest.
    
    :param main_dir: Path to the main directory containing multiple subfolders.
    """
    subfolders = [
        os.path.join(main_dir, d) for d in os.listdir(main_dir)
        if os.path.isdir(os.path.join(main_dir, d))
    ]
    subfolders = sorted(subfolders)  # optional sort

    folder_cell_sets = []
    file_pattern = re.compile(r'^cell_([^_]+)_frame_\d+\.(png|jpg|jpeg|tif|bmp)$', re.IGNORECASE)
    
    for folder in subfolders:
        cell_set = set()
        for fname in os.listdir(folder):
            match = file_pattern.match(fname)
            if match:
                cell_id = match.group(1)  
                cell_set.add(cell_id)
        folder_cell_sets.append(cell_set)

    if not folder_cell_sets:
        print("No subfolders or no matching cell images found. Exiting.")
        return

    cells_in_all_folders = set.intersection(*folder_cell_sets)
    print("Cells in all folders:", cells_in_all_folders)

    for folder, cell_set in zip(subfolders, folder_cell_sets):
        for fname in os.listdir(folder):
            match = file_pattern.match(fname)
            if match:
                cell_id = match.group(1)
                if cell_id not in cells_in_all_folders:
                    file_path = os.path.join(folder, fname)
                    print(f"Deleting {file_path} (not in intersection).")
                    os.remove(file_path)

    print("Cleanup complete. Only cells found in all folders have been preserved.")

if __name__ == "__main__":
    main_directory = "data/I06_s2/tracked"  
    delete_incomplete_cells(main_directory)
