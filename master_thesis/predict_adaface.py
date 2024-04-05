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

# image transformations by adaface
class ToInputTransform:
    def __call__(self, pil_rgb_image):
        np_img = np.array(pil_rgb_image)
        brg_img = ((np_img[:, :, ::-1] / 255.) - 0.5) / 0.5  # expects BGR
        tensor = torch.tensor([brg_img.transpose(2, 0, 1)]).float()
        return tensor

if __name__ == '__main__':

    model = load_pretrained_model('ir_18')
    
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        ToInputTransform()
        ])

    test_image_path = '/work3/s174139/Master_Thesis/data/raw_full/adults' #/work3/s174139/Master_Thesis/AdaFace-master/face_alignment/test_images'
    
    dataset = datasets.ImageFolder(test_image_path, transform=transform)
    
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
                features_batch, _ = model(batch_images.squeeze(1))  # Forward pass through the model
            features.extend(features_batch.cpu()) 
            image_ids.extend(batch_ids)
            print("Batch {}/{}".format(round(i/batch_size), round(len(dataset)/512)))

        
    similarity_scores = torch.cat(features) @ torch.cat(features).T
    
    print("similarity scores calculated! Saving dictionary")
    
    data_dict = {
    'file_name': dataset.imgs,
    'image_id': image_ids,
    'feature_vectors': features,
    'similarity_scores': similarity_scores
    }

    # Save the dictionary
    torch.save(data_dict, '/work3/s174139/Master_Thesis/master_thesis/saved_predictions/similarity_scores_adults.pt')

    
    
    
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


