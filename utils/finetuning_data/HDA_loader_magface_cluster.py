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

def extract_landmarks(image_path):
    landmark_values = RetinaFace.detect_faces(image_path)["face_1"]["landmarks"].values()
    print("yes")
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
    main_dataset_folder = data_root_folder +"data_full/HDA_database"
    output_folder = data_root_folder + "data_full/HDA_processed_cluster_magface"
    imgs_output_folder = output_folder + '/images'
    rest_path = '/probes/images'
    image_size=112
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(imgs_output_folder, exist_ok=True)
    # Initialize a dictionary to keep track of the unique IDs and their corresponding integers
    id_map = {}
    current_id = 0


    with open(os.path.join(output_folder, 'fine_tune_train_list_magface_all_HDA.list'), 'w') as train_list_file:
        for age_group in range(5): #normally 5
            im_counter = 0
            # Create output directory for the current age group   
            age_group_folder = os.path.join(main_dataset_folder, f'age_group_{age_group}' + rest_path)
            all_images = sorted([image for image in os.listdir(age_group_folder) if image.endswith('.png') or image.endswith('.jpg')]) #Jpg images are bad image quality
            #num_images = len(all_images)    
            #print("Number of images in original directory age_group_{}:".format(age_group), num_images)
                        
            # Iterate over each image path
            for img in all_images:
                try: 
                        
                    # Extract the original ID from the image filename
                    original_id = img.split("_")[0]
                    
                    # Check if this ID has been seen before
                    if original_id not in id_map:
                        id_map[original_id] = current_id
                        current_id += 1
                    
                    # Get the unique integer ID for the current original ID
                    incrementally_integer_id = id_map[original_id]
                        
                    
                    
                    input_image_path = os.path.join(age_group_folder, img)
                    #print("INPUT", input_image_path)
                    cv_image = cv2.imread(input_image_path)
                    #print("CV")
                    

                    # Use MagFace function for alignment (utils.face_align.py)
                    #landmarks_np = extract_landmarks(input_image_path)
                    landmarks_np = extract_landmarks(cv_image)
            

                    #print("landmark")
                    aligned_resized_image = face_align.norm_crop(cv_image, landmarks_np, image_size, mode='arcface') 
                    #print("aligned")
                    

                    output_image_path = os.path.join(imgs_output_folder, os.path.basename(img))
                    #print("output")
                    cv_image_written = cv2.imwrite(output_image_path, aligned_resized_image)
                    #print("cv2 write")


                    output_image_path_write = os.path.join(imgs_output_folder, img) 
                    train_list_file.write(f'{output_image_path_write} 0 {incrementally_integer_id}\n') #corresponding to id. list format required by magface. First id should be 0. 
                    #print("written!")
                except:
                    print("Error processing image, possibly due to low image quality:", img)
                pass
                im_counter += 1

            print("\n*** {im_counter} images were preprocessed and saved in new directory age_group_{ag} :) ***".format(im_counter=im_counter, ag=age_group))
            
    # # Check number of ids in list 
    # list_folder = "../../data/data_full/HDA_processed_cluster_magface/fine_tune_train_list_magface_correct_index_tester.list"
    # change_id_incremental(list_folder) #OBS lidt i tvivl om de skal være sorteret så alle 0'ere er  samlet, men Francks er ligesom vores..
    # with open(list_folder, 'r') as f:
    #     lines = f.readlines()
    # imids = []
    # for line in lines:
    #     parts = line.strip().split(' ')
    #     imids.append(parts[-1])
    # print("number of unique ids: ", len(set(imids)) )


                

    