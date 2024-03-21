from torch.utils.data import Dataset, DataLoader, RandomSampler
from torchvision import transforms

import json
import numpy as np
import torch
from os import path

# YLFW Source: https://github.com/JessyFrish/YLFW_Links?tab=readme-ov-file

# Make suitable for AdaFace and MagFace (112x112)



class YLFWDataset(Dataset):
    """Dataset for the YLFW images.
    The images are saved in the following structure:
        data/
            processed/
                id1/
                    face1_0.jpg
                        ...
                id2/
                    face2_0.jpg
                        ...
                ...
    Args:
        dataname: Name of the dataset. The dataset is saved in data/processed/dataname/
        preprocesser: Preprocesser for the images (if model is specified).
                    'create_transform(**resolve_data_config(model.pretrained_cfg))' from timm.data library should be used
                    if preprocesser is none reads the images as numpy arrays and returns them as torch tensors
                    (assumes that the images are already preprocessed)
        NB: See example below!
    """

    def __init__(self, dataname="sample", datapath="data/processed", preprocesser=None) -> None:
        super().__init__()
        self.info = json.load(open(path.join(datapath, f"{dataname}.json"), "r"))
        self.images = self.info["images"]
        self.categories = self.info["categories"]
        self.annotations = self.info["annotations"]
        self.categories_dict = np.load(path.join(datapath, "categories.npy"), allow_pickle=True).item()
        self.preprocesser = preprocesser
        self.datapath = datapath

    def __len__(self) -> int:
        return len(self.info["images"])

    def __getitem__(self, index: int):
        """Returns image, category and super category for the given index.

        Args:
            index: Index of the image to be returned.

        Returns:
            img: Image as a torch tensor of shape (3, 224, 224) if preprocesser is None. Otherwise the shape is (3, W, H)
            category: Category of the image as a torch tensor of shape (N_CLASSES)
        """

        filename = self.images[index]["file_name"]
        if not path.exists(filename):
            filename = path.join(self.datapath, filename)

        try:
            img = image_to_tensor(filename, self.preprocesser)
        except Exception:
            return self.__getitem__((index + 1) % len(self))

        label_dict = [cat for cat in self.categories if cat["id"] == self.annotations[index]["category_id"]].pop()

        category = label_dict["name"]

        return img


if __name__ == "__main__":
    
    # Apply transformations here from the respective models. 
    
    # OBS: Husk at brug enten Random Sampler eller Weighted Random Sampler
    sampler = RandomSampler()
    
    
    trainTransforms = transforms.Compose([resize, hFlip, vFlip, rotate,
        transforms.ToTensor()])

    model = timm.create_model("mobilenetv3_large_100", pretrained=True)
    preprocesser = create_transform(**resolve_data_config(model.pretrained_cfg))

    dataset = ShroomDataset("sample", preprocesser)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=8)
    for img, category, super_category in dataloader:
        print(img.shape)
        print(category.shape)
        print(super_category.shape)