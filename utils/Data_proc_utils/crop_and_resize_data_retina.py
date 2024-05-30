# from https://github.com/IrvingMeng/MagFace face_align

############################# OBS remember to change path when we have updated git" ########################################

######## IF FILES ARE MISSING FROM ORG

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


def process_images(input_folder, output_folder, file_list): 
    file_set = set(file_list)
    image_size=112
    
    # Walk through the directory structure
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file in file_set and (file.endswith('.png') or file.endswith('.jpg')):
                # Construct the full path to the input image
                input_image_path = os.path.join(root, file)
                
                cv_image = cv2.imread(input_image_path)
                    
                # Use MagFace function for alignment (utils.face_align.py)
                #landmarks_np = extract_landmarks(input_image_path)
                try:
                    landmarks_np = extract_landmarks(cv_image)

                    cv_image = face_align.norm_crop(cv_image, landmarks_np, image_size, mode='arcface') 
                except:
                    print("input path could not be loaded using landmarks", input_image_path)
                    pass
                
                # Construct the output path
                relative_path = os.path.relpath(root, input_folder)
                output_dir = os.path.join(output_folder, relative_path)
                
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                output_image_path = os.path.join(output_dir, file)
                
                if cv_image is not None:
                    # Resize the face image to the target size if needed
                    # resized_face_img = cv2.resize(face_img, target_size)
                    
                    # Save the cropped face image
                    cv2.imwrite(output_image_path, cv_image)
                else:
                    print(f"No face found in image, copying original: {input_image_path}")
                    # Copy the original image to the output folder
                    shutil.copy2(input_image_path, output_image_path)
                    
if __name__ == '__main__':
    # Example usage
    
    file_list = ['m.05zx6h_0002.jpg', 'm.05ry0p_0005.jpg', 'm.03cn6js_0004.jpg', 'm.0hntxx8_0001.jpg', 'm.06zpjqp_0002.jpg', 'm.01mx_5r_0003.jpg', 'm.0gfdvnv_0002.jpg', 'm.0bgtjb_0004.jpg', 'm.06g9t5_0003.jpg', 'm.04zvqgd_0004.jpg', 'm.0gbz836_0004.jpg', 'm.04tkfj_0004.jpg', 'm.0hnc76t_0003.jpg', 'm.0833bt_0002.jpg', 'm.04n2tgy_0002.jpg', 'm.0b38gz_0001.jpg', 'm.052g6_0002.jpg', 'm.04gpr5w_0002.jpg', 'm.04zvqgd_0003.jpg', 'm.09k7405_0003.jpg', 'm.0f4q6y_0001.jpg', 'm.03bzhk8_0001.jpg', 'm.07x_rw_0002.jpg', 'm.0469sp_0001.jpg', 'm.07td21_0003.jpg', 'm.03cn6js_0007.jpg', 'm.0bs90ts_0002.jpg', 'm.06224x_0004.jpg', 'm.03cn6js_0006.jpg', 'm.05zx6h_0003.jpg', 'm.03bzhk8_0002.jpg', 'm.0bjtyg_0002.jpg', 'm.01qhc8_0005.jpg', 'm.04zwfrk_0001.jpg', 'm.05m0xf_0004.jpg', 'm.05ry0p_0003.jpg', 'm.0ds0311_0001.jpg', 'm.03cyr2_0002.jpg', 'm.04jx1g_0003.jpg', 'm.0j63pr1_0003.jpg', 'm.0dryqdt_0002.jpg', 'm.04rlb8_0006.jpg', 'm.04zwfrk_0004.jpg', 'm.04ygtw6_0002.jpg', 'm.07x_rw_0001.jpg', 'm.09xmkg_0005.jpg', 'm.06224x_0001.jpg', 'm.0fhrbz_0004.jpg', 'm.01w312b_0001.jpg', 'm.0bdw4v3_0001.jpg', 'm.0j63pr1_0001.jpg', 'm.04d50l_0001.jpg', 'm.052_wzm_0005.jpg', 'm.03hgcwv_0003.jpg', 'm.0b38gz_0003.jpg', 'm.0dryqdt_0001.jpg', 'm.0dk12xn_0004.jpg', 'm.04gpr5w_0001.jpg', 'm.0bbcqb_0003.jpg', 'm.03wdy1j_0001.jpg', 'm.0261yk1_0001.jpg', 'm.0h3p11z_0002.jpg', 'm.02z35g0_0003.jpg', 'm.0gr9gr_0002.jpg', 'm.04n2tgy_0004.jpg', 'm.05b598q_0001.jpg', 'm.0fhrbz_0001.jpg', 'm.028116t_0003.jpg', 'm.025xdfy_0001.jpg', 'm.04n2tgy_0003.jpg', 'm.04zwfrk_0003.jpg', 'm.02vgny_0001.jpg', 'm.0gyw0dd_0002.jpg', 'm.05hrs7_0001.jpg', 'm.05b598q_0003.jpg', 'm.03dmy0_0002.jpg', 'm.04y08n_0003.jpg', 'm.0fyy_y_0002.jpg', 'm.03c0pqf_0003.jpg', 'm.03cdj9p_0003.jpg', 'm.0dr046_0002.jpg', 'm.065fyq_0003.jpg', 'm.05m0xf_0002.jpg', 'm.0hnc76t_0004.jpg', 'm.0fhrbz_0005.jpg', 'm.06yg_v_0002.jpg', 'm.0cz8tfb_0003.jpg', 'm.0gbz836_0003.jpg', 'm.02ns0x_0003.jpg', 'm.025xdfy_0004.jpg', 'm.0gg6jmx_0002.jpg', 'm.043p03w_0001.jpg', 'm.07td21_0001.jpg', 'm.02x68cc_0001.jpg', 'm.02rxjrz_0002.jpg', 'm.06zt05_0003.jpg', 'm.05bp47_0002.jpg', 'm.019hz7_0001.jpg', 'm.03sl3x_0002.jpg', 'm.0gg60_t_0003.jpg', 'm.03cyr2_0001.jpg', 'm.0fhrbz_0002.jpg', 'm.0gr9gr_0001.jpg', 'm.043p03w_0002.jpg', 'm.02vgny_0003.jpg', 'm.0bhcm0z_0003.jpg', 'm.07x_rw_0003.jpg', 'm.0bbcqb_0002.jpg', 'm.02h2my_0001.jpg', 'm.07r4j1_0001.jpg', 'm.04gpr5w_0004.jpg', 'm.0dk12xn_0003.jpg', 'm.07y_bb_0003.jpg', 'm.0j33kpx_0003.jpg', 'm.0h6wyf_0003.jpg', 'm.0ds0311_0002.jpg', 'm.0ds0311_0003.jpg', 'm.0hntxx8_0003.jpg', 'm.02z7gwf_0001.jpg', 'm.04rlb8_0005.jpg', 'm.06nvhd_0003.jpg', 'm.05bzwz0_0001.jpg', 'm.03sl3x_0003.jpg', 'm.04rlb8_0004.jpg', 'm.08g5y5_0001.jpg', 'm.0gyv__4_0003.jpg', 'm.03cn6js_0003.jpg', 'm.0f5j55_0002.jpg', 'm.07r4j1_0004.jpg', 'm.0h6wyf_0001.jpg', 'm.08051l0_0005.jpg', 'm.09mcr7_0001.jpg', 'm.03cn6js_0001.jpg', 'm.05687l1_0002.jpg', 'm.03hgcwv_0005.jpg', 'm.02qwxhj_0003.jpg', 'm.01sh3s__0001.jpg', 'm.08g5y5_0004.jpg', 'm.04gpr5w_0003.jpg', 'm.04tkfj_0001.jpg', 'm.0d69k__0001.jpg', 'm.02qqv8w_0002.jpg', 'm.0gr9gr_0004.jpg', 'm.0fyy_y_0003.jpg', 'm.0fmnry_0002.jpg', 'm.0bbcqb_0001.jpg', 'm.05687l1_0001.jpg', 'm.0b38gz_0002.jpg', 'm.05b598q_0002.jpg', 'm.09xmkg_0003.jpg', 'm.0j33kpx_0002.jpg', 'm.02rxjrz_0001.jpg', 'm.0bhcm0z_0001.jpg', 'm.02qdbpn_0002.jpg', 'm.0dm1ts_0005.jpg', 'm.0dm1ts_0004.jpg', 'm.09ggr80_0002.jpg', 'm.0bs90ts_0004.jpg', 'm.025xdfy_0003.jpg', 'm.07kks3_0001.jpg', 'm.07r4j1_0003.jpg', 'm.0gyv__4_0002.jpg', 'm.03wdy1j_0003.jpg', 'm.06224x_0003.jpg', 'm.04zvqgd_0002.jpg', 'm.0fmnry_0003.jpg', 'm.04lz00_0002.jpg', 'm.0f5j55_0003.jpg', 'm.04rlb8_0001.jpg', 'm.02rxjrz_0004.jpg', 'm.02qdbpn_0001.jpg', 'm.0bwh1fb_0001.jpg', 'm.03cn6js_0005.jpg', 'm.02qqv8w_0004.jpg', 'm.04d50l_0002.jpg', 'm.05zx6h_0001.jpg', 'm.01451__0003.jpg', 'm.07x_rw_0004.jpg', 'm.02qqv8w_0005.jpg', 'm.01sh3s__0005.jpg', 'm.02rxjrz_0003.jpg', 'm.06yg_v_0003.jpg', 'm.04c1fd_0002.jpg', 'm.01mx_5r_0005.jpg', 'm.03cyr2_0007.jpg', 'm.0fyy_y_0004.jpg', 'm.02b_dr_0001.jpg', 'm.0833bt_0004.jpg', 'm.05bp47_0003.jpg', 'm.0dk12xn_0005.jpg', 'm.0dk12xn_0001.jpg', 'm.05f4w4t_0001.jpg', 'm.03cyr2_0004.jpg', 'm.0bhcm0z_0002.jpg', 'm.0fyy_y_0005.jpg', 'm.0dr046_0001.jpg', 'm.028116t_0004.jpg', 'm.027677__0002.jpg', 'm.04d50l_0003.jpg', 'm.02ns0x_0004.jpg', 'm.025xdfy_0005.jpg', 'm.0fyy_y_0006.jpg', 'm.02z7gwf_0003.jpg', 'm.02qwxhj_0002.jpg', 'm.0bgtjb_0001.jpg', 'm.09xmkg_0004.jpg', 'm.02x68cc_0002.jpg', 'm.05ry0p_0002.jpg', 'm.04n2tgy_0001.jpg', 'm.05hrs7_0003.jpg', 'm.0cz8tfb_0002.jpg', 'm.05ry0p_0004.jpg', 'm.0bs90ts_0001.jpg', 'm.02b_dr_0002.jpg', 'm.03wdy1j_0002.jpg', 'm.05bp47_0001.jpg', 'm.0gg6jmx_0004.jpg', 'm.03cyr2_0005.jpg', 'm.0cz8tfb_0001.jpg', 'm.09k7405_0002.jpg', 'm.02z35g0_0001.jpg', 'm.01w312b_0002.jpg', 'm.06yg_v_0001.jpg', 'm.08051l0_0002.jpg', 'm.03cdj9p_0004.jpg', 'm.0f4q6y_0002.jpg']

    
    input_folder = '/work3/s174139/Master_Thesis/data/data_full/adults_filtered_bibel'
    output_folder = '/work3/s174139/Master_Thesis/data/data_full/adults_filtered_bibel_cropped_resized_retina'

    process_images(input_folder, output_folder, file_list)
    print("done")



######## ORIGIANL
# # Load packages
# from retinaface import RetinaFace
# import cv2
# import os
# import numpy as np
# import sys
# sys.path.append('../../MagFace-main/utils/')
# import face_align
# import shutil


# def extract_landmarks(image_path):
#     landmark_values = RetinaFace.detect_faces(image_path)["face_1"]["landmarks"].values()
#     #print("yes")
#     return np.array([sublist for sublist in landmark_values])


# def process_images(input_folder, output_folder): 
#     image_size=112
    
#     # Walk through the directory structure
#     for root, dirs, files in os.walk(input_folder):
#         for file in files:
#             if file.endswith('.png') or file.endswith('.jpg'):
#                 # Construct the full path to the input image
#                 input_image_path = os.path.join(root, file)
                
#                 cv_image = cv2.imread(input_image_path)
                    
#                 # Use MagFace function for alignment (utils.face_align.py)
#                 #landmarks_np = extract_landmarks(input_image_path)
#                 landmarks_np = extract_landmarks(cv_image)
                

#                 aligned_resized_image = face_align.norm_crop(cv_image, landmarks_np, image_size, mode='arcface') 
                
                
#                 # Construct the output path
#                 relative_path = os.path.relpath(root, input_folder)
#                 output_dir = os.path.join(output_folder, relative_path)
                
#                 if not os.path.exists(output_dir):
#                     os.makedirs(output_dir)
                
#                 output_image_path = os.path.join(output_dir, file)
                
#                 if aligned_resized_image is not None:
#                     # Resize the face image to the target size if needed
#                     # resized_face_img = cv2.resize(face_img, target_size)
                    
#                     # Save the cropped face image
#                     cv2.imwrite(output_image_path, aligned_resized_image)
#                 else:
#                     print(f"No face found in image, copying original: {input_image_path}")
#                     # Copy the original image to the output folder
#                     shutil.copy2(input_image_path, output_image_path)
                    
# if __name__ == '__main__':
#     # Example usage
#     input_folder = '/work3/s174139/Master_Thesis/data/data_full/adults_filtered_bibel'
#     output_folder = '/work3/s174139/Master_Thesis/data/data_full/adults_filtered_bibel_cropped_resized_retina'

#     process_images(input_folder, output_folder)
#     print("done")