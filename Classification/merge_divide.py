import os
import shutil
from pathlib import Path

data_division_file = "data_division.txt"
output_folder = "data/output"
final_dataset_folder = "final_dataset"

classes = ["fed", "unfed", "unidentified"]

def create_final_structure():
    for split in ["train", "val", "test"]:
        for cls in classes:
            os.makedirs(Path(final_dataset_folder) / split / cls, exist_ok=True)

def parse_data_division(file_path):
    divisions = {"train": [], "val": [], "test": []}
    current_section = None

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("Train"):
                current_section = "train"
            elif line.startswith("Validation"):
                current_section = "val"
            elif line.startswith("Test"):
                current_section = "test"
            elif current_section:
                divisions[current_section].append(line)
    return divisions

def copy_files_by_division(divisions):
    for split, parent_folders in divisions.items():
        for parent_folder in parent_folders:
            parent_path = Path(output_folder) / parent_folder
            if not parent_path.exists():
                print(f"Warning: {parent_path} does not exist.")
                continue

            for cls in classes:
                class_folder = parent_path / cls
                if not class_folder.exists():
                    print(f"Warning: {class_folder} does not exist in {parent_path}.")
                    continue

                destination = Path(final_dataset_folder) / split / cls

                for file in class_folder.iterdir():
                    if file.is_file():
                        shutil.copy(file, destination / file.name)

if __name__ == "__main__":
    create_final_structure()

    data_divisions = parse_data_division(data_division_file)

    copy_files_by_division(data_divisions)

    print("Data reorganization complete.")
