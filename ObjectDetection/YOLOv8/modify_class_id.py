import os


def replace_in_file(file_path):
    print(f"Processing file: {file_path}")
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        with open(file_path, 'w') as file:
            for line in lines:
                if line.startswith("1 ") or line.startswith("2 "):
                    file.write("0" + line[1:])
                else:
                    file.write(line)
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

def process_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".txt"):
                replace_in_file(os.path.join(root, file))


main_folder_path = 'data/labels'
process_folder(main_folder_path)
