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
    return np.array([sublist for sublist in landmark_values])


if __name__ == "__main__":
        
    # With help from franck hda synth loader
    data_root_folder = '../../data/'
    main_dataset_folder = data_root_folder +"raw/HDA"
    output_folder = data_root_folder + "raw_full/HDA_aligned_resized"
    imgs_output_folder = output_folder + '/images'
    rest_path = '/probes/images'
    image_size=112

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(imgs_output_folder, exist_ok=True)

    with open(os.path.join(data_root_folder, 'fine_tune_train_list.list'), 'w') as train_list_file:
        for age_group in range(5):
            id_counter = 0
            # Create output directory for the current age group   
            age_group_folder = os.path.join(main_dataset_folder, f'age_group_{age_group}' + rest_path)
            all_images = sorted([image for image in os.listdir(age_group_folder) if image.endswith('.png')]) #Jpg images are bad image quality
            num_images = len(all_images)    
            print("Number of images in original directory age_group_{}:".format(age_group), num_images)
                        
            # Iterate over each image path
            for img in all_images:
                try: 
                    input_image_path = os.path.join(age_group_folder, img)
                    cv_image = cv2.imread(input_image_path)
                    # Use MagFace function for alignment (utils.face_align.py)
                    landmarks_np = extract_landmarks(input_image_path)
                    aligned_resized_image = face_align.norm_crop(cv_image, landmarks_np, image_size, mode='arcface') 
                    output_image_path = os.path.join(imgs_output_folder, os.path.basename(img))
                    cv2.imwrite(output_image_path, aligned_resized_image)
                    train_list_file.write(f'{input_image_path} 0 {img.split("_")[0]}\n') #corresponding to id. list format required by magface
                except:
                    print("Error processing image, possibly due to low image quality:", img)
                pass
                id_counter += 1

            print("\n*** {id_counter} images were preprocessed and saved in new directory age_group_{ag} :) ***".format(id_counter=id_counter, ag=age_group))
            
    # Check number of ids in list 
    list_folder = "../../data/fine_tune_train_list.list"
    with open(list_folder, 'r') as f:
        lines = f.readlines()
    imids = []
    for line in lines:
        parts = line.strip().split(' ')
        imids.append(parts[-1])
    print("number of unique ids: ", len(set(imids)) )
                        
                

    