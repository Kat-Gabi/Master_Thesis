import os
from glob import glob
import pandas as pd

import argparse

def main(folder_path, balanced_csv_path, output_name):
    balanced_df = pd.read_csv(balanced_csv_path)
    
    balanced_file_names = balanced_df.image_name.tolist()
    "Search for images and save to img.list type, used by MagFace"
    # Use glob to find all image files matching the pattern
    file_paths = [os.path.join(root, filename) for root, dirs, files in os.walk(folder_path) for filename in files if filename.lower().endswith(('.jpg', '.png'))
                  and filename[:-4] in balanced_file_names]

    if file_paths:
        # Save the list of file paths to a text file
        output_folder = "/".join(folder_path.split("/")[:-1])
        print("out", output_folder)
        output_file_path = os.path.join(output_folder, "{}.list".format(output_name)) 
        with open(output_file_path, 'w') as file:
            for path in file_paths:
                file.write(path + '\n')
        print(f"File paths saved to: {output_file_path}")
    else:
        print(f"No images found in {folder_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Creates a list of image paths in .file format") #description of parser
    parser.add_argument("folder_path", type=str, help="Path to the folder containing image files or folders with image files.") # pass eg /work3/s174139/Master_Thesis/data/data_full/children_filtered_bibel_FINAL_INFERENCE #/work3/s174139/Master_Thesis/data/data_full/adults_filtered_bibel_cropped_resized_retina #/work3/s174139/Master_Thesis/data/data_full/RFW/data
    parser.add_argument("balanced_csv_path", type=str, help="Path to the csv containing the balanced dataset.") # pass eg  /work3/s174139/Master_Thesis/data/image_info_csvs/final_filtered_children_df_BIBEL.csv
    parser.add_argument("output_name", type=str, help="Name of output file for run of balanced data.") # pass eg children_filtered_bibel_FINAL_INFERENCE

    args = parser.parse_args() #save the arguments with name folde the added argument to access
    main(args.folder_path, args.balanced_csv_path, args.output_name)