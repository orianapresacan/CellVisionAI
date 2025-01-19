import os
import re

def delete_incomplete_cells(main_dir):
    """
    Scans all subfolders of 'main_dir' for images named like 'Timepoint_X-frame_Y.png'.
    Keeps only those cells that appear in ALL subfolders, ignoring differences in 'Timepoint_XXX'.
    
    :param main_dir: Path to the main directory containing multiple subfolders.
    """
    subfolders = [
        os.path.join(main_dir, d) for d in os.listdir(main_dir)
        if os.path.isdir(os.path.join(main_dir, d))
    ]
    subfolders = sorted(subfolders)  

    folder_cell_sets = []
    
    # Updated regex to extract everything after 'ST_' and ignore 'Timepoint_XXX'
    file_pattern = re.compile(r'^Timepoint_\d{3,}_\d{6}-ST_([\w\d]+_cell_\d+)\.(png|jpg|jpeg|tif|bmp)$', re.IGNORECASE)

    for folder in subfolders:
        cell_set = set()
        for fname in os.listdir(folder):
            match = file_pattern.match(fname)
            if match:
                cell_id = match.group(1)  # Extract "ST_XXX_cell_YYY"
                cell_set.add(cell_id)
        folder_cell_sets.append(cell_set)

    if not folder_cell_sets:
        print("No subfolders or no matching cell images found. Exiting.")
        return

    # Find cells that exist in all folders (ignoring 'Timepoint_XXX')
    cells_in_all_folders = set.intersection(*folder_cell_sets)
    print(f"Cells in all folders ({len(cells_in_all_folders)}): {cells_in_all_folders}")

    # Delete images that do not appear in all subfolders
    for folder, cell_set in zip(subfolders, folder_cell_sets):
        for fname in os.listdir(folder):
            match = file_pattern.match(fname)
            if match:
                cell_id = match.group(1)  # Extract the common part
                
                if cell_id not in cells_in_all_folders:
                    file_path = os.path.join(folder, fname)
                    
                    # Ensure file exists before attempting deletion
                    if os.path.isfile(file_path):
                        print(f"Deleting {file_path} (not in all folders).")
                        os.remove(file_path)
                    else:
                        print(f"File not found: {file_path}, skipping.")

    print("Cleanup complete. Only cells found in all folders have been preserved.")

if __name__ == "__main__":
    main_directory = "data/I06_s2/tracked"  
    delete_incomplete_cells(main_directory)
