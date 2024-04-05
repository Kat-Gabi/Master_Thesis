import os
from glob import glob

import argparse

def main(folder_path):
    "Search for images and save to img.list type, used by MagFace"
    # Use glob to find all image files matching the pattern
    file_paths = [os.path.join(root, filename) for root, dirs, files in os.walk(folder_path) for filename in files if filename.lower().endswith(('.jpg', '.png'))]

    if file_paths:
        # Save the list of file paths to a text file
        output_folder = "/".join(folder_path.split("/")[:-1])
        print("out", output_folder)
        output_file_path = os.path.join(output_folder, 'img.list')
        with open(output_file_path, 'w') as file:
            for path in file_paths:
                file.write(path + '\n')
        print(f"File paths saved to: {output_file_path}")
    else:
        print(f"No images found in {folder_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Creates a list of image paths in .file format") #description of parser
    parser.add_argument("folder_path", type=str, help="Path to the folder containing image files or folders with image files.") # pass eg /work3/s174139/Master_Thesis/data/raw/YLFW_bench/data_p2
    args = parser.parse_args() #save the arguments with name folde the added argument to access
    main(args.folder_path)

