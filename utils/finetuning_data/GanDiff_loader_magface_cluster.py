# from https://github.com/IrvingMeng/MagFace faces should be aligned to 112x112 with 5 landmarks, and save a .list file with image information
# This can be done using MagFace function for alignment (utils.face_align.py)


############################# OBS remember to change path when we have updated git" ########################################


# Load packages
from retinaface import RetinaFace
import cv2
import os
import numpy as np
import sys
sys.path.append('../../MagFace-main/utils/')
import face_align
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def extract_landmarks(image_path):
    landmark_values = RetinaFace.detect_faces(image_path)["face_1"]["landmarks"].values()
    #print("yes")
    return np.array([sublist for sublist in landmark_values])

def change_id_incremental(file_path):
    "file_path: your_file_path_here.list"
    # Read the contents of the .list file
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Extract IDs and create a mapping
    id_mapping = {}
    for line in lines:
        parts = line.strip().split()
        image_path = parts[0]
        id_value = int(parts[-1])
        if id_value not in id_mapping:
            id_mapping[id_value] = len(id_mapping)
        parts[-1] = str(id_mapping[id_value])
        lines[lines.index(line)] = ' '.join(parts) + '\n'

    # Write the updated contents back to the .list file
    with open(file_path, "w") as file:
        file.writelines(lines)

if __name__ == "__main__":
        
    # With help from franck hda synth loader
    data_root_folder = '../../data/'
    main_dataset_folder = data_root_folder +"data_full/GanDiffFace"
    output_folder = data_root_folder + "data_full/GanDiffFace_processed_cluster_magface"
    imgs_output_folder = output_folder + '/images'
    image_size=112
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(imgs_output_folder, exist_ok=True)
    # Initialize a dictionary to keep track of the unique IDs and their corresponding integers
    id_map = {}
    current_id = 0


    with open(os.path.join(output_folder, 'fine_tune_train_list_magface_all_GanDiffFace.list'), 'w') as train_list_file:
        for age_group in range(4):
            im_counter = 0
            print("age group ", age_group)
            
            
            skipped_images_count = 0
            id_counter = 0
            age_group_folder = os.path.join(main_dataset_folder, f'age_group_{age_group}') 

                
            # Walk through the directory structure
            for root, dirs, files in os.walk(age_group_folder):
                for file in files:
                    if file.endswith('.png') or file.endswith('.jpg'):
                        try: 
                        # Construct the full path to the input image
                            input_image_path = os.path.join(root, file)

                            #print("INPUT", input_image_path)
                            cv_image = cv2.imread(input_image_path)
                            #print("CV")
                            
                            # Use MagFace function for alignment (utils.face_align.py)
                            #landmarks_np = extract_landmarks(input_image_path)
                            landmarks_np = extract_landmarks(cv_image)
                    

                            #print("landmark")
                            aligned_resized_image = face_align.norm_crop(cv_image, landmarks_np, image_size, mode='arcface') 
                            #print("aligned")
                                                
                            
                            
                            
                            person_id = root.split('/')[-1].split('_')[0]
                            
                            # Check if this ID has been seen before
                            if person_id not in id_map:
                                id_map[person_id] = current_id
                                current_id += 1
                            
                            # Get the unique integer ID for the current original ID
                            incrementally_integer_id = id_map[person_id]
                            
                            output_image_path = os.path.join(imgs_output_folder, os.path.basename(file))
                            #print("output")
                            cv_image_written = cv2.imwrite(output_image_path, aligned_resized_image)
                            #print("cv2 write")


                            output_image_path_write = os.path.join(imgs_output_folder, file) 
                            train_list_file.write(f'{output_image_path_write} 0 {incrementally_integer_id}\n') #corresponding to id. list format required by magface. First id should be 0. 
                            #print("written!")
                        except:
                            print("Error processing image, possibly due to low image quality:", file)
                        pass
                        im_counter += 1

            print("\n*** {im_counter} images were preprocessed and saved in new directory age_group_{ag} :) ***".format(im_counter=im_counter, ag=age_group))
                    
            

                

    