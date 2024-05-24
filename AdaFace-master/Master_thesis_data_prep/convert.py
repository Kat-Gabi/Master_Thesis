## Convert script according to https://github.com/mk-minchul/AdaFace/blob/master/README_TRAIN.md
## I.e. prepare images in folder (as label) structure 

# To count number of folders in a folder in terminal:
# find imgs -mindepth 1 -maxdepth 1 -type d | wc -l = 91


import os
import cv2
from PIL import Image
import numpy as np
import sys
sys.path.append('..')
from face_alignment import align

def main():
    # move images to folder structure with id as folder name
    main_dataset_folder = '../../data/data_full/HDA_database'
    output_folder = '../../data/data_full/HDA_processed_AdaFace/imgs'
    rest_path = '/probes/images'

    for age_group in range(5): #normally 5
        
        print("age group ", age_group)
        skipped_images_count = 0
        id_counter = 0
        age_group_folder = os.path.join(main_dataset_folder, f'age_group_{2}' + rest_path)
        all_images = sorted([image for image in os.listdir(age_group_folder) if image.endswith('.png') or image.endswith('.jpg')]) #Jpg images are bad image quality
   
        # Iterate over each image path
        for img in all_images:
            try:
                input_image_path = os.path.join(age_group_folder, img)
                #print("Input", input_image_path) 
                aligned_face = align.get_aligned_face(input_image_path)
                if aligned_face:
                    #print("aligned!")
                    person_id = img.split('_')[0]
                    person_folder = os.path.join(output_folder, person_id)
                    #print("person_folder", person_folder )
                    # make directory with id as folder name
                    os.makedirs(person_folder, exist_ok=True)
                    
                    # Convert to PIL and BRG
                    aligned_face_bgr = cv2.cvtColor(np.array(aligned_face), cv2.COLOR_RGB2BGR)
                    #print("aligned_face_bgr", aligned_face_bgr.shape)

                    output_image_path = os.path.join(person_folder, os.path.basename(img))
                    cv2.imwrite(output_image_path, aligned_face_bgr)
                    #print("written?",output_image_path)
                    id_counter += 1
            except:
                print("skipped image:", img)
                skipped_images_count += 1

    print(f"\n*** {id_counter} images were preprocessed and saved.***")
    print(f"*** {skipped_images_count} images were skipped.***\n")
    
    
if __name__ == "__main__":
    print("start")
    main()
