# function to check number of images in folders - adults

import os
import shutil
import pandas as pd


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


def move_adults(image_dir, dst_root):
    image_file_names = adults_csv['image_name'].tolist()

    # Get the list of folders in the image directory
    folders = [os.path.join(image_dir, d) for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]
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
                    shutil.copy(src, dst)
    print('Done moving files')
    
    
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
    
    # Filtered csv you want to move
    adults_csv = pd.read_csv('/work3/s174139/Master_Thesis/data/image_info_csvs/final_filtered_adults_df_BIBEL.csv')
    print("Length of csv filtered",len(adults_csv))

    # Directories to move from and to
    image_dir = '/work3/s174139/Master_Thesis/data/data_full/RFW/data'
    dst_root = '/work3/s174139/Master_Thesis/data/data_full/adults_filtered_bibel' #../data/raw_full/adults'


    ### Only run if you need to move data
    move_adults(image_dir, dst_root)


    ## Check number of images in new destination folder
    root_directory = '/work3/s174139/Master_Thesis/data/data_full/adults_filtered_bibel'
    total_images = count_images_in_all_directories(root_directory)
    print(f"Total number of images in new destination folder: {total_images}")


    # Example usage:
    print("CHECKS FOR DUPLICATES")
    check_for_duplicate_file_names_and_remove_first_instance(root_directory)


    ## Check number of images in folder now
    total_images = count_images_in_all_directories(root_directory)
    print(f"Total number of images: {total_images}") 