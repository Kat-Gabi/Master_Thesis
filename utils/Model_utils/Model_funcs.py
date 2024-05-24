# Load libraries
import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize
import torch
import seaborn as sns
import pandas as pd
import os
sns.set(style="white")



# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def convert_unique_ids(ids):
    "For all Ids, get last id name and convert to unique ids"
    unique_ids_list = []
    for id in ids:
        im_name = id.split("/")[-1][:-4]
        if '.' in im_name:
            un_id = im_name[:-5]
        else:
            un_id = "_".join(im_name.split("_")[:-1])

        unique_ids_list.append(un_id)
    return unique_ids_list

# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


def factorize_ids(ids):
    "Returns a list of factors and a dictionary mapping each unique ID name to a unique index"
    unique_ids = {}
    factors = []
    for id in ids:
        if id not in unique_ids:
            unique_ids[id] = len(unique_ids)  # Assign a unique index for each unique ID
        factors.append(unique_ids[id])  # Append the index corresponding to the ID
    return factors, unique_ids


# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def load_magface_vectors(feature_list, canonical=False, df_c_can=None):
    """
    Input: Feature list from magface (str), eg.: '../data/feat_children.list'
    Output: Normalized Feature vectors, numerical ids e.g. 0 for African_49 id
    """

    with open(feature_list, 'r') as f:
        lines = f.readlines()

    img_2_feats = {}
    img_2_mag = {}
    for line in lines:
        parts = line.strip().split(' ')
        imgname = parts[0]
        imgname = "/"+"/".join(imgname.split("/")[4:])
        print(imgname)
        feats = [float(e) for e in parts[1:]]
        mag = np.linalg.norm(feats)
        img_2_feats[imgname] = feats/mag # computes normalized feature vectors
        img_2_mag[imgname] = mag # magnitude of the feature vector

    file_name = np.array(list(img_2_feats.keys()))

    if canonical:
        file_name = [file_name[ele] for ele in range(len(lines)) if file_name[ele].split("/")[-1] in np.array(df_c_can.Filename)]
        norm_feature_vectors = np.array([img_2_feats[file_name[ele]] for ele in range(len(file_name))])
    else:
        norm_feature_vectors = np.array([img_2_feats[file_name[ele]] for ele in range(len(lines))])

    image_names = [full_name.split("/")[-1][:-4] for full_name in file_name]
    identity_names = convert_unique_ids(file_name) # from /data/Indian_682/Indian_682_0 to just Indian_682
    factors_c, unique_ids = factorize_ids(identity_names) #Factorized list: [0, 1, 2, 2], Image IDs mapping: {'Indian_682': 0, 'Asian_504': 1,..}
    num_ids = np.array(factors_c)

    return image_names, identity_names, num_ids, norm_feature_vectors

# Example usage:
# image_names_c, ids_c, num_ids_c, norm_feats_c = load_magface_vectors('../data/feat_children.list')
# sim_mat = np.dot(norm_feats_c, norm_feats_c.T)

#df_c_can = pd.read_csv("../data/OFIQ_results/canonical_children.csv", sep=";")
#ids_can, num_ids_can, norm_feats_can = load_magface_vectors('../data/feat_children.list', canonical=True, df_c_can=df_c_can)


# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

##### OBS perhaps this shohuld only be used for canonical loading
def load_enrolled_magface_vectors(feature_list, enrolled_img_names, canonical=False, df_c_can=None):
    """
    Input: Feature list from magface (str), eg.: '../data/feat_children.list'
    Output: Only for enrolled ids: Normalized Feature vectors, numerical ids e.g. 0 for African_49 id
    """

    with open(feature_list, 'r') as f:
        lines = f.readlines()

    img_2_feats = {}
    img_2_mag = {}
    for line in lines:
        parts = line.strip().split(' ')
        imgname = parts[0]
        imgname = "/"+"/".join(imgname.split("/")[4:])
        feats = [float(e) for e in parts[1:]]
        mag = np.linalg.norm(feats)
        img_2_feats[imgname] = feats/mag # computes normalized feature vectors
        img_2_mag[imgname] = mag # magnitude of the feature vector

    mated_feature_dict = {key: value for key, value in img_2_feats.items() if key.split("/")[-1][:-4] in enrolled_img_names}
    file_name = np.array(list(mated_feature_dict.keys()))

    if canonical:
        # only get mated canonical image names and feature vectors
        file_name = [file_name[ele] for ele in range(len(lines)) if file_name[ele].split("/")[-1] in np.array(df_c_can.Filename)]
        norm_feature_vectors = np.array([mated_feature_dict[file_name[ele]] for ele in range(len(file_name))])
    else:
        norm_feature_vectors = np.array([mated_feature_dict[file_name[ele]] for ele in range(len(file_name))])

    image_names = [full_name.split("/")[-1][:-4] for full_name in file_name]
    identity_names = convert_unique_ids(file_name) # from /data/Indian_682/Indian_682_0 to just Indian_682
    factors_c, unique_ids = factorize_ids(identity_names) #Factorized list: [0, 1, 2, 2], Image IDs mapping: {'Indian_682': 0, 'Asian_504': 1,..}
    num_ids = np.array(factors_c)

    return image_names, identity_names, num_ids, norm_feature_vectors

# Example usage
# enrolled_img_names_c = c_df[c_df.enrolled == "enrolled"].img_name.to_list()
# image_names_c, ids_c, enrolled_num_ids_c, enrolled_norm_feats_c = load_enrolled_magface_vectors(feature_list, enrolled_img_names_c, canonical=False, df_c_can=None):
#
# _, _, num_ids_a_enrolled, norm_feats_a_enrolled = load_enrolled_magface_vectors(feature_list_adults, enrolled_image_names_a, canonical=False, df_c_can=None)
# sim_mat_a_enrolled = np.dot(norm_feats_a_enrolled, norm_feats_a_enrolled.T)


# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def load_adaface_vectors(file_path, canonical=False, df_c_can=None):
    """
    Input: Feature list from adaface (str), eg.: '../saved_predictions/similarity_scores_children_full_baseline1.pt'
    Output: Normalized Feature vectors, numerical ids e.g. 0 for African_49 id
    """

    # Load the file
    data = torch.load(file_path)

    identity_names = [os.path.basename(os.path.dirname(path)) for path, _ in data["file_name"]]
    image_names = [os.path.splitext(os.path.basename(path))[0] for path, _ in data["file_name"]]
    norm_feature_vectors = np.array(data["feature_vectors"]) # feature vectors are normalized in adaface
    num_ids = np.array(data["image_id"])

    file_name = np.array([n for n, _ in data["file_name"]])

    # convert to dict as magface:
    file_names_vectors_dict = {}

    # Iterate over the names and values
    for file_name_i, feature_vect in zip(file_name, norm_feature_vectors):
        # Assign the name as key and the corresponding list of values as value
        file_names_vectors_dict[file_name_i] = feature_vect

    if canonical:
        file_name = [file_name[ele] for ele in range(len(file_name)) if file_name[ele].split("/")[-1] in np.array(df_c_can.Filename)]
        norm_feature_vectors = np.array([file_names_vectors_dict[file_name[ele]] for ele in range(len(file_name))]) #unsorted image quality
        image_names = [full_name.split("/")[-1][:-4] for full_name in file_name]
        identity_names = convert_unique_ids(file_name)
        factors_c, unique_ids = factorize_ids(identity_names) #Factorized list: [0, 1, 2, 2], Image IDs mapping: {'Indian_682': 0, 'Asian_504': 1,..}
        num_ids = np.array(factors_c)

    return image_names, identity_names, num_ids, norm_feature_vectors


# Example usage:
#ids_c, num_ids_c, norm_feats_c = load_adaface_vectors('../master_thesis/saved_predictions/similarity_scores_children_baseline1.pt')
#df_c_can = pd.read_csv("../data/OFIQ_results/canonical_children.csv", sep=";")
#ids_can, num_ids_can, norm_feats_can = load_magface_vectors('../data/feat_children.list', canonical=True, df_c_can=df_c_can)

# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def load_adaface_vectors_adults(file_path, canonical=False, df_c_can=None):
    """
    Input: Feature list from adaface (str), eg.: '../saved_predictions/similarity_scores_children_full_baseline1.pt'
    Output: Normalized Feature vectors, numerical ids e.g. 0 for African_49 id
    """

    # Load the file
    data = torch.load(file_path)

    # Adjusted based on assumption that data["file_name"] contains only paths
    identity_names = [os.path.basename(os.path.dirname(path)) for path in data["file_name"]]
    image_names = [os.path.splitext(os.path.basename(path))[0] for path in data["file_name"]]
    norm_feature_vectors = np.array(data["feature_vectors"])  # feature vectors are normalized in adaface
    num_ids = np.array(data["image_id"])

    file_name = np.array([n for n in data["file_name"]])

    # convert to dict as magface:
    file_names_vectors_dict = {}

    # Iterate over the names and values
    for file_name_i, feature_vect in zip(file_name, norm_feature_vectors):
        # Assign the name as key and the corresponding list of values as value
        file_names_vectors_dict[file_name_i] = feature_vect

    if canonical:
        file_name = [file_name[ele] for ele in range(len(file_name)) if file_name[ele].split("/")[-1] in np.array(df_c_can.Filename)]
        norm_feature_vectors = np.array([file_names_vectors_dict[file_name[ele]] for ele in range(len(file_name))])  # unsorted image quality
        image_names = [full_name.split("/")[-1][:-4] for full_name in file_name]
        identity_names = convert_unique_ids(file_name)
        factors_c, unique_ids = factorize_ids(identity_names)  # Factorized list: [0, 1, 2, 2], Image IDs mapping: {'Indian_682': 0, 'Asian_504': 1,..}
        num_ids = np.array(factors_c)

    return image_names, identity_names, num_ids, norm_feature_vectors


# -----------------------------------------------------------


def load_enrolled_adaface_vectors(file_path, enrolled_img_names, canonical=True, df_c_can=None):
    """
    Input: Feature list from adaface (str), eg.: '../saved_predictions/similarity_scores_children_full_baseline1.pt'
    Output: Only for enrolled ids: Normalized Feature vectors, numerical ids e.g. 0 for African_49 id
    """
    # Load the file
    data = torch.load(file_path)

    file_name = np.array([n for n, _ in data["file_name"]])

    # convert to dict as magface:
    file_names_vectors_dict = {}

    # Iterate over the names and values
    for file_name, feature_vect in zip(file_name, norm_feature_vectors):
        # Assign the name as key and the corresponding list of values as value
        file_names_vectors_dict[file_name] = feature_vect


    mated_feature_dict = {key: value for key, value in file_names_vectors_dict.items() if key.split("/")[-1][:-4] in enrolled_img_names}
    file_name = np.array(list(mated_feature_dict.keys()))

    if canonical:
        # only get mated canonical image names and feature vectors
        file_name = [file_name[ele] for ele in range(len(data)) if file_name[ele].split("/")[-1] in np.array(df_c_can.Filename)]
        norm_feature_vectors = np.array([mated_feature_dict[file_name[ele]] for ele in range(len(file_name))])
    else:
        norm_feature_vectors = np.array([mated_feature_dict[file_name[ele]] for ele in range(len(data))])

    image_names = [full_name.split("/")[-1][:-4] for full_name in file_name]
    identity_names = convert_unique_ids(file_name)
    factors_c, unique_ids = factorize_ids(identity_names) #Factorized list: [0, 1, 2, 2], Image IDs mapping: {'Indian_682': 0, 'Asian_504': 1,..}
    num_ids = np.array(factors_c)


    return image_names, identity_names, num_ids, norm_feature_vectors
