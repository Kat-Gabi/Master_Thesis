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
    for file_name, feature_vect in zip(file_name, norm_feature_vectors):
        # Assign the name as key and the corresponding list of values as value
        file_names_vectors_dict[file_name] = feature_vect
    
    if canonical:
        file_name = [file_name[ele] for ele in range(len(data["image_id"])) if file_name[ele].split("/")[-1] in np.array(df_c_can.Filename)]
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
        file_name = [file_name[ele] for ele in range(len(data["image_id"])) if file_name[ele].split("/")[-1] in np.array(df_c_can.Filename)]
        norm_feature_vectors = np.array([mated_feature_dict[file_name[ele]] for ele in range(len(file_name))]) 
    else: 
        norm_feature_vectors = np.array([mated_feature_dict[file_name[ele]] for ele in range(len(data["image_id"]))]) 
        
    image_names = [full_name.split("/")[-1][:-4] for full_name in file_name]
    identity_names = convert_unique_ids(file_name)
    factors_c, unique_ids = factorize_ids(identity_names) #Factorized list: [0, 1, 2, 2], Image IDs mapping: {'Indian_682': 0, 'Asian_504': 1,..}
    num_ids = np.array(factors_c)
    
 
    return image_names, identity_names, num_ids, norm_feature_vectors

# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def GARBE(fnir_c, fnir_a, fpir_c, fpir_a, alpha=0.5):
    """
    Function calculates GARBe score based on ISO standard ISO/IEC DIS 19795-10
    """
    
    FPD = fpir_c/fpir_a
    print("FPD result: ", FPD)


    FND = fnir_c/fnir_a
    print("FND result: ", FND)
    
    GARBE = alpha * FPD + (1 - alpha) * FND
    print("GARBE result, GARBE close to 1 means more unfair: ", GARBE)

    return FPD, FND, GARBE

# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def remove_probeid_in_classification(arr, value):
    "Removes probe unique id in array"
    for i, v in enumerate(arr):
        if v == value:
            return np.delete(arr, i)
    return arr  # Value not found in the array, return the original array

def compute_fpir(non_enrolled_sim_score, num_ids_non_enrolled, num_ids_all, thold=0.5):
    """
    FPIR formula from ISO standard ISO/IEC 19795-1:2021
    """

    # U_D: set of non-mated identification transactions with reference database D. I.e. equal to number IDs with no enrolled ids.
    U_d_set_len = len(non_enrolled_sim_score)
    cand_list_returned = 0

    for i in range(U_d_set_len):
        probe = num_ids_non_enrolled[i] # probe corresponding to the current sample similarity matrix

        # for the non enrolled probe id, check if any of its similarity scores are above thold
        classified_pos_list = non_enrolled_sim_score[i] > thold
        classified_pos_idx = list(np.where(classified_pos_list)[0]) # get indexes where the score is above threshold
        face_idx_pos_class = num_ids_all[classified_pos_idx] # get numerical ids in the positive class
        # remove instance of probe id in classification list
        face_idx_pos_class_filtered = remove_probeid_in_classification(face_idx_pos_class, probe)

        # if length of candidate list (filtered, i.e. without the probe itsef) is greater than 0, count 1
        if len(face_idx_pos_class_filtered) > 0:
            cand_list_returned += 1

    fpir = cand_list_returned/U_d_set_len

    return fpir

# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

## Function for calculating FNIR
def compute_fnir_utils(enrolled_sim_score, enrolled_num_id, thold=0.5):
    """
    FNIR formula from ISO standard ISO/IEC 19795-1:2021
    """
    # M_D: set of mated identification transactions with reference database. - i.e. there can be multiple ids?
    M_d_set_len = len(enrolled_sim_score)
    neg_ref = 0
    
    # For each id corresponding to the id in the set, check if one of it's corresponding ids are above threshold
    
    # Iterate over each enrolled reference for transaction i
    for i in range(M_d_set_len):
        probe = enrolled_num_id[i] # numerical id by magface, e.g. str value "African_244" becomes num. value 35. 
        
        # Check if the reference probe id is in negative list/below threshold
        classified_negative_list = enrolled_sim_score[i] <= thold
        classified_negative_idx = list(np.where(classified_negative_list)[0])  # Get indexes where the score is below threshold
        face_idx_neg_class = enrolled_num_id[classified_negative_idx]  # Get numerical ids in the negative class
        # If numerical id in negative list is equal to the probe id, count 1
        if probe in face_idx_neg_class:
            neg_ref += 1

    # Calculate FNIR
    fnir = neg_ref / M_d_set_len

    return fnir
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def remove_ones(matrix, reshape=False):
    "Function for removing ones to get similarity scores flattened for plots"
    
    # number of 1s to remove
    n_remove = len(matrix)
    # Flatten the matrix
    flattened_matrix = matrix.flatten()

    # Sort the flattened matrix in descending order
    sorted_indices = np.argsort(flattened_matrix)[::-1]

    # Get indices of the 10 highest values
    top_n_indices = sorted_indices[:n_remove]

    # Remove the indices of the 10 highest values from the flattened matrix
    filtered_indices = np.delete(np.arange(len(flattened_matrix)), top_n_indices)
    final_matrix = flattened_matrix[filtered_indices]
    if reshape:
    # Reshape the modified flattened matrix back into the original shape
        final_matrix = final_matrix.reshape(matrix.shape)

    return final_matrix
#Example usage:
#enrolled_sim_scores_flattened_without_ones = remove_ones(enrolled_sim_score)