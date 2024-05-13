# Load packages
from retinaface import RetinaFace
import cv2
import os
import numpy as np
import sys
sys.path.append('../../MagFace-main/utils/')
import face_align

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

    list_folder = "../../data/data_full/HDA_processed_local/fine_tune_train_list_TEST_local.list"
    #change_id_incremental(list_folder) #OBS lidt i tvivl om de skal være sorteret så alle 0'ere er  samlet, men Francks er ligesom vores..
    with open(list_folder, 'r') as f:
        lines = f.readlines()
    imids = []
    for line in lines:
        print("LINE")
        parts = line.strip().split(' ')
        print("IMSID", parts, parts[-1])
        imids.append(parts[-1])
        
    print("number of unique ids: ", len(set(imids)) )


                