## Source: https://github.com/mk-minchul/AdaFace/blob/master/inference.py

import os
# Specify the desired directory
desired_directory = '/work3/s174139/Master_Thesis/AdaFace-master' 

# Change the current working directory
#os.chdir(desired_directory)
import sys
import sys
print("HELLO","/".join(os.path.realpath(__file__).split("/")[0:-2]) + "/AdaFace-master")
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]) + "/AdaFace-master")

import net
import torch
from face_alignment import align
import numpy as np

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

def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    brg_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
    tensor = torch.tensor([brg_img.transpose(2,0,1)]).float()
    return tensor

if __name__ == '__main__':

    model = load_pretrained_model('ir_18')
    feature, norm = model(torch.randn(2,3,112,112))

    test_image_path = '/work3/s174139/Master_Thesis/AdaFace-master/face_alignment/test_images'
    features = []
    for fname in sorted(os.listdir(test_image_path)):
        path = os.path.join(test_image_path, fname)
        aligned_rgb_img = align.get_aligned_face(path)
        bgr_tensor_input = to_input(aligned_rgb_img)
        feature, _ = model(bgr_tensor_input)
        features.append(feature)

    similarity_scores = torch.cat(features) @ torch.cat(features).T
    print(similarity_scores)
    

