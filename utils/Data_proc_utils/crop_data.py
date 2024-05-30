import os
import cv2
import numpy as np
from PIL import Image
import shutil

def crop_face_cv2(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Load the pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    try:
        if len(faces) > 0:
            # Assuming only one face is present, extract the first face
            (x, y, w, h) = faces[0]
            
            # Crop the face from the image
            face_image = image[y:y+h, x:x+w]
            
            return face_image
    except Exception as e:
        print(f"No face found in the image: {image_path}. Error: {e}")
        pass #Instead of None

def process_images(input_folder, output_folder): 
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Walk through the directory structure
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.png') or file.endswith('.jpg'):
                # Construct the full path to the input image
                input_image_path = os.path.join(root, file)
                
                # Crop the face
                face_img = crop_face_cv2(input_image_path)

                # Construct the output path
                relative_path = os.path.relpath(root, input_folder)
                output_dir = os.path.join(output_folder, relative_path)
                
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                output_image_path = os.path.join(output_dir, file)
                
                if face_img is not None:
                    # Resize the face image to the target size if needed
                    # resized_face_img = cv2.resize(face_img, target_size)
                    
                    # Save the cropped face image
                    cv2.imwrite(output_image_path, face_img)
                else:
                    print(f"No face found in image, copying original: {input_image_path}")
                    # Copy the original image to the output folder
                    shutil.copy2(input_image_path, output_image_path)
                
               # Worked before
                # # Crop the face
                # face_img = crop_face_cv2(input_image_path)

                # if face_img is not None:
                #     # Resize the face image to the target size
                #     #resized_face_img = cv2.resize(face_img, target_size)
                    
                #     # Construct the output path
                #     relative_path = os.path.relpath(root, input_folder)
                #     output_dir = os.path.join(output_folder, relative_path)
                    
                #     if not os.path.exists(output_dir):
                #         os.makedirs(output_dir)
                    
                #     output_image_path = os.path.join(output_dir, file)
                    
                #     # Save the cropped and resized face image
                #     cv2.imwrite(output_image_path, face_img)
                # else:
                #     print(f"Skipping image due to no face found: {input_image_path}")
                    
if __name__ == '__main__':
    # Example usage
    input_folder = '/work3/s174139/Master_Thesis/data/data_full/adults_filtered_bibel'
    output_folder = '/work3/s174139/Master_Thesis/data/data_full/adults_filtered_bibel_cropped'

    process_images(input_folder, output_folder)
    print("done")
    
    
# Count images in folder
