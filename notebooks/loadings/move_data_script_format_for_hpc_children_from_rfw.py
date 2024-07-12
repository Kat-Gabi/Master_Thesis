# function to check number of images in folders - adults

import os
import shutil
import pandas as pd



###### CROP ADULTS

# Load packages
from retinaface import RetinaFace
import cv2
import os
import numpy as np
import sys
sys.path.append('../../MagFace-main/utils/')
import face_align
import shutil


def extract_landmarks(image_path):
    landmark_values = RetinaFace.detect_faces(image_path)["face_1"]["landmarks"].values()
    #print("yes")
    return np.array([sublist for sublist in landmark_values])


# MOVE CHILDREN FROM RFW: Only move files from YLFW full and RFW full also in BIBEL
def move_children_from_rfw(image_dir_rfw, dst_root, children_df):
    image_size=112

    # First for RFW
    rfw_csv = children_df[children_df["image_name"].str.contains('\.')]
    image_file_names = rfw_csv['image_name'].tolist()

    # Get the list of folders in the image directory
    folders = [os.path.join(image_dir_rfw, d) for d in os.listdir(image_dir_rfw) if os.path.isdir(os.path.join(image_dir_rfw, d))]
    # Iterate through each folder
    for folder in folders:
        # Get the list of subfolders
        folders2 = [os.path.join(folder, d) for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
        for folder2 in folders2:
        # Get the list of image files in the folder
            files = [f for f in os.listdir(folder2) if os.path.isfile(os.path.join(folder2, f))]

            # Check if any of the image file names from the CSV are in the folder
            for file in files:
                if file[:-4] in image_file_names:
                    # Move the image file to the output directory
                    
                    # Make folder with ethnicityname_imageid
                    folder2_add = folder2.split("/")[-2:]
                    folder2_add_real = '_'.join(folder2_add)

                    dst_root = dst_root
                    src = os.path.join(folder2, file)
                    dst = os.path.join(dst_root, folder2_add_real)
                    os.makedirs(dst,exist_ok=True)
                    
                    #CROP IMAGE AND MOVE TO FOLDER
                    input_image_path = src
                    
                    cv_image = cv2.imread(input_image_path)
                        
                    # Use MagFace function for alignment (utils.face_align.py)
                    #landmarks_np = extract_landmarks(input_image_path)
                    try:
                        landmarks_np = extract_landmarks(cv_image)

                        cv_image = face_align.norm_crop(cv_image, landmarks_np, image_size, mode='arcface') 
                    except:
                        print("input path could not be loaded using landmarks", input_image_path)
                        pass
                    
                    # # Construct the output path
                    # relative_path = os.path.relpath(root, input_folder)
                    # output_dir = os.path.join(output_folder, relative_path)
                    
                    # if not os.path.exists(output_dir):
                    #     os.makedirs(output_dir)
                    
                    # output_image_path = os.path.join(output_dir, file)
                    output_image_path = os.path.join(dst, file)
                    
                    if cv_image is not None:
                        # Resize the face image to the target size if needed
                        # resized_face_img = cv2.resize(face_img, target_size)
                        
                        # Save the cropped face image
                        cv2.imwrite(output_image_path, cv_image)
                    else:
                        print(f"No face found in image, copying original: {input_image_path}")
                        # Copy the original image to the output folder
                        shutil.copy2(input_image_path, dst)

    print('Done moving RFW files')


def count_images_in_directory(directory):
    """
    Count the number of image files in a directory.
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']  # Add more extensions if needed
    image_count = 0

    # Iterate through all files in the directory
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)

        # Check if the file is a regular file (not a directory)
        if os.path.isfile(file_path):
            # Check if the file has a recognized image extension
            if any(file_name.lower().endswith(ext) for ext in image_extensions):
                image_count += 1

    return image_count

def count_images_in_all_directories(root_directory):
    """
    Count the number of image files in all directories and subdirectories.
    """
    total_image_count = 0

    # Walk through all directories and subdirectories
    for dirpath, dirnames, filenames in os.walk(root_directory):
        # Count the number of image files in the current directory
        current_dir_image_count = count_images_in_directory(dirpath)

        # Add the count to the total image count
        total_image_count += current_dir_image_count

        # Print the directory path and the number of images
        #print(f"Directory: {dirpath}, Images: {current_dir_image_count}")

    return total_image_count

# Where's the duplicates?

# Function to find duplicate file names within a folder and remove the first instance
def remove_first_instance_of_duplicate_file_names(folder_path):
    file_names = {}  # Dictionary to store file names and their counts

    # Traverse each folder recursively
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            # Increment count for file name in dictionary
            # if file_name in file_names:
            #     # If the file name already exists, delete the first occurrence
            #     os.remove(file_path)
            #     print(f"Removed first instance of duplicate file: {file_path}")
            # else:
            #     file_names[file_name] = file_path

            if file_name in file_names:
                # If the file name already exists, remove the entire containing folder
                os.system(f"rm -rf '{os.path.dirname(file_path)}'")
                print(f"Removed folder containing duplicate file: {os.path.dirname(file_path)}")
                break  # Once a duplicate is found and removed, exit the loop
            else:
                file_names[file_name] = file_path

# Main function to check for duplicate file names in all folders
def check_for_duplicate_file_names_and_remove_first_instance(root_folder):
    for root, dirs, files in os.walk(root_folder):
        print(f"Checking duplicates in folder: {root}")
        remove_first_instance_of_duplicate_file_names(root)



if __name__ == '__main__':
    
    ### FROM RFW TO CHILDREN
    
    # Directories to move from and to
    image_dir_rfw = '/work3/s174139/Master_Thesis/data/data_full/RFW/data'
    dst_root = '/work3/s174139/Master_Thesis/data/data_full/children_filtered_bibel_FINAL_INFERENCE' 
    
    # DF to work with
    children_df = pd.read_csv('/work3/s174139/Master_Thesis/data/image_info_csvs/final_filtered_children_df_BIBEL.csv')
    df_w_duplicates = children_df[children_df.files_list.duplicated()==True]
    adults_children = children_df[children_df["image_name"].str.contains('\.')] #~
    print("length of rfw in children bibel to be moved",adults_children[~adults_children["Unnamed: 0"].isin(df_w_duplicates["Unnamed: 0"])])

    ### Only run if you need to move data
    move_children_from_rfw(image_dir_rfw, dst_root, children_df)


    ## Check number of images in new destination folder
    root_directory = '/work3/s174139/Master_Thesis/data/data_full/children_filtered_bibel_FINAL_INFERENCE'
    total_images = count_images_in_all_directories(root_directory)
    print(f"Total number of images in new destination folder: {total_images}")

    # Example usage:
    print("CHECKS FOR DUPLICATES")
    check_for_duplicate_file_names_and_remove_first_instance(root_directory)

    ## Check number of images in folder now
    total_images = count_images_in_all_directories(root_directory)
    print(f"Total number of images: {total_images}") 