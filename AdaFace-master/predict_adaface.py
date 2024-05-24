## Source: https://github.com/mk-minchul/AdaFace/blob/master/inference.py

# Initial necessary imports
import sys
import os

# Specify the desired directory
desired_directory = '/work3/s174139/Master_Thesis/AdaFace-master' 

# Change the current working directory
print("WORKING DIR HERE","/".join(os.path.realpath(__file__).split("/")[0:-2]) + "/AdaFace-master")
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]) + "/AdaFace-master")

# Model specific imports
import net
import torch
from face_alignment import align
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

torch.cuda.empty_cache()
import cv2



##% Model and Training starts here

adaface_models = {
    'ir_18': "/work3/s174139/Master_Thesis/AdaFace-master/pretrained/adaface_ir18_casia.ckpt", #"pretrained/adaface_ir50_ms1mv2.ckpt",
}

def load_pretrained_model(architecture='ir_18'):
    # load model and pretrained statedict
    assert architecture in adaface_models.keys()
    model = net.build_model(architecture)
    statedict = torch.load(adaface_models[architecture])['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model

class FilteredImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.dataset = datasets.ImageFolder(root, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        while True:
            image, label = self.dataset[idx]
            if image is not None:
                return image, label
            else:
                # Randomly select another index to avoid None
                idx = (idx + 1) % len(self.dataset)

class ResizeCropFaceAndToTensorTransform:
    def __init__(self, face_cascade_path, target_size):
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        self.target_size = target_size

    def __call__(self, pil_rgb_image):
        # Resize the image to the target size
        #resized_img = pil_rgb_image.resize(self.target_size)

        # Convert PIL image to NumPy array
        
        np_img = np.array(pil_rgb_image)

        # Crop the face
        face_img = self.crop_face_cv2(np_img)

        if face_img is not None:
            # Normalize and convert the image to PyTorch tensor
            resized_face_img = cv2.resize(face_img, self.target_size)
            brg_img = ((resized_face_img[:, :, ::-1] / 255.) - 0.5) / 0.5  # expects BGR
            tensor = torch.tensor([brg_img.transpose(2, 0, 1)]).float()
            return tensor
        else:
            return None
        
    def crop_face_cv2(self, image):
        faces = self.face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face_image = image[y:y+h, x:x+w]
            return face_image
        return None

    # def crop_face_cv2(self, image):
    #     faces = self.face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    #     try:
    #         if len(faces) > 0:
    #             (x, y, w, h) = faces[0]
    #             face_image = image[y:y+h, x:x+w]
    #             return face_image
    #     except:
    #         print("No face found in the image.")
    #         pass
# Worked before
# # image transformations by adaface
# class ToInputTransform:
#     def __call__(self, pil_rgb_image):
        
#         try:         
#             np_img = np.array(pil_rgb_image)        
#             np_img = crop_face_cv2(np_img)    
#             np_img = np.array(np_img)    
#             #np_img = np.array(np_img)
#             brg_img = ((np_img[:, :, ::-1] / 255.) - 0.5) / 0.5  # expects BGR
#             tensor = torch.tensor([brg_img.transpose(2, 0, 1)]).float()
#             #print("no typeerror")
#             return tensor
#         except:
#             pass
#             #print("Type error for im:", pil_rgb_image)
#             # np_img = np.array(pil_rgb_image)        
#             # brg_img = ((np_img[:, :, ::-1] / 255.) - 0.5) / 0.5  # expects BGR
#             # tensor = torch.tensor([brg_img.transpose(2, 0, 1)]).float()
#             # return tensor
            
    
# def crop_face_cv2(image):
#     # Load the image
#     #image = cv2.imread(image_path)
    
#     # Convert the image to grayscale
#     #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # Load the pre-trained face detector
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
#     # Detect faces in the image
#     faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#     try:
#         if len(faces) > 0:
#             # Assuming only one face is present, extract the first face
#             (x, y, w, h) = faces[0]
            
#             # Crop the face from the image
#             face_image = image[y:y+h, x:x+w]
            
#             return face_image
#     except:
#         print("No face found in the image.")
#         pass

if __name__ == '__main__':

    model = load_pretrained_model('ir_18')
    
    
    #    Path to the face cascade XML file
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    target_size=(112,112)


    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        ResizeCropFaceAndToTensorTransform(face_cascade_path, target_size)
        ])

    test_image_path = '/work3/s174139/Master_Thesis/data/data_full/adults_filtered_bibel' #RFW/data' #/work3/s174139/Master_Thesis/AdaFace-master/face_alignment/test_images'
    print("at this step")
    #dataset_in = datasets.ImageFolder(test_image_path, transform=transform)
    dataset = FilteredImageFolder(test_image_path, transform=transform)

    print("in")
    
    # b_size = 10
    # dataloader = DataLoader(dataset_in, batch_size=b_size, shuffle=True)
    
    # d=[]
    # for images, labels in dataloader:
    #     print(f"Batch of {images.size(0)} images and their labels:")
    #     if images is not None:
    #         print(images)
    #         print(labels)
            
    #     break  # Exit after the first batch
    
    
    #dataset = [(img, label) for img, label in dataloader if img is not None]
    #print("over this step..")

    
    batch_size = 512  # Define your batch size


    # WITH BATCH
    features = []
    image_ids = []
    for i in range(0, len(dataset), batch_size):
        batch_images = []
        batch_ids = []
        for j in range(batch_size):
            if i + j < len(dataset):
                image, id = dataset[i + j]
                batch_images.append(image)
                batch_ids.append(id)
        
        if batch_images:
            batch_images = torch.stack(batch_images) # Convert list of images to tensor
            with torch.no_grad():  # Disable gradient calculation for efficiency
                features_batch, _ = model(batch_images.squeeze(1))  # Forward pass through the model # Removes extra batch dimension from Image_Folder
            
            # Normalize the feature vectors
            features_batch = torch.nn.functional.normalize(features_batch, p=2, dim=1)

            features.extend(features_batch.cpu()) 
            image_ids.extend(batch_ids)
            print("Batch {}/{}".format(round(i/batch_size), round(len(dataset)/512)))


    print("Saving dictionary")
    
    data_dict = {
    'file_name': [sample[0] for sample in dataset.dataset.samples], # dataset.imgs,
    'image_id': image_ids,
    'feature_vectors': features,
    }
    print(data_dict["image_id"])
    print("SET", set(data_dict["image_id"]))
    
    print(len(data_dict["feature_vectors"]))

    # Save the dictionary
    torch.save(data_dict, '/work3/s174139/Master_Thesis/data/data_full/feature_vectors/adaface_feature_vectors/similarity_scores_adults_all_baseline1.pt')

    
    
    
    # Before batch:
    
    # features = []
    # image_id = []
    # for images, ids in dataset:
    #     # output of forward pass
    #     feature, _ = model(images) 
    #     features.append(feature)
    #     image_id.append(ids)
    
    # features = []
    # image_path = []
    # for fname in sorted(os.listdir(test_image_path)):
    #     path = os.path.join(test_image_path, fname)
        
    #     # apply transformations
    #     input_tensor = transform(path)
    
    #     # output of forward pass
    #     feature, _ = model(input_tensor) 
        
    #     features.append(feature)
    #     image_path.append(path)

    # similarity_scores = torch.cat(features) @ torch.cat(features).T
    
    # print(similarity_scores)
    
    # data_dict = {
    # 'image_paths': image_path,
    # 'feature_vectors': features,
    # 'similarity_scores': similarity_scores
    # }

    # # Save the dictionary
    # torch.save(data_dict, 'image_data_similarity_scores.pt')
        
        
    
    
    #måkse til træning
    
    # Load the dataset with transformations
    #dataset = datasets.ImageFolder(test_image_path, transform=transform)
    
    # Define a DataLoader to batch and shuffle the data
    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    
    # # Lists to store features, labels, and image paths
    # predictions = []
    # id_class = []
    # image_paths = []

    # # Make features on the dataset
    # with torch.no_grad():
    #     for images, ids in dataloader:
    #         feature, _ = model(images) # output of forward pass
    #         predictions.extend(feature.cpu().numpy())
    #         id_class.extend(ids.cpu().numpy())
    #         image_paths.extend(dataset.imgs)


