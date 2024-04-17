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
    Output: Normalized Feature vectors, ids
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
#ids_c, num_ids_c, norm_feats_c = load_magface_vectors('../data/feat_children.list')
#df_c_can = pd.read_csv("../data/OFIQ_results/canonical_children.csv", sep=";")
#ids_can, num_ids_can, norm_feats_can = load_magface_vectors('../data/feat_children.list', canonical=True, df_c_can=df_c_can)


# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


def load_enrolled_magface_vectors(feature_list, enrolled_img_names, canonical=False, df_c_can=None):
    """
    Input: Feature list from magface (str), eg.: '../data/feat_children.list' 
    Output: Normalized Feature vectors, ids
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
        norm_feature_vectors = np.array([mated_feature_dict[file_name[ele]] for ele in range(len(lines))]) 
    
    image_names = [full_name.split("/")[-1][:-4] for full_name in file_name]
    identity_names = convert_unique_ids(file_name) # from /data/Indian_682/Indian_682_0 to just Indian_682
    factors_c, unique_ids = factorize_ids(identity_names) #Factorized list: [0, 1, 2, 2], Image IDs mapping: {'Indian_682': 0, 'Asian_504': 1,..}
    num_ids = np.array(factors_c)
    
    return image_names, identity_names, num_ids, norm_feature_vectors

# Example usage
# enrolled_img_names_c = c_df[c_df.enrolled == "enrolled"].img_name.to_list()
# image_names_c, ids_c, enrolled_num_ids_c, enrolled_norm_feats_c = load_enrolled_magface_vectors(feature_list, enrolled_img_names_c, canonical=False, df_c_can=None):


# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def load_adaface_vectors(file_path, canonical=True, df_c_can=None):
    
    # Load the file
    data = torch.load(file_path)
    
    identity_names = [os.path.basename(os.path.dirname(path)) for path, _ in data["file_name"]]
    image_names = [os.path.splitext(os.path.basename(path))[0] for path, _ in data["file_name"]]
    norm_feature_vectors = np.array(data["feature_vectors"]) # feature vectors are normalized in adaface
    num_ids = data["image_id"]
    
    
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
    "TODO check if correct"
    
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

def compute_fnir(mated_df, sim_scores, im_ids, ids, thold=0.5):
    # M_D: set of mated identification transactions with reference database.
    M_d_set = set(mated_df)
    M_d_set_len = len(M_d_set)
    neg_ref = 0

    # Initialize the list to store the mated similarity scores
    mated_sim_scores = []

    # Iterate over each mated reference for transaction i
    for m_i, id_now in enumerate(ids):
        probe = im_ids[m_i]
        id_name = ids[m_i]

        # Check if the identification is mated
        if id_now in M_d_set:
            mated_ids_exact = [id == id_now for id in ids]
            mated_sim_scores_slice = sim_scores[m_i]
            mated_sim_scores_slice_slice = [value for value, keep in zip(mated_sim_scores_slice, mated_ids_exact) if keep]
            mated_sim_scores.extend(mated_sim_scores_slice_slice)

            # Check if the reference probe id is in negative list/below threshold
            classified_negative_list = sim_scores[m_i] <= thold
            classified_negative_idx = list(np.where(classified_negative_list)[0])  # Get indexes where the score is below threshold
            face_idx_neg_class = im_ids[classified_negative_idx]  # Get numerical ids in the negative class

            # If numerical id in negative list is equal to the probe id, count 1
            if probe in face_idx_neg_class:
                neg_ref += 1

    # Calculate FNIR
    fnir = neg_ref / M_d_set_len

    # Convert list to numpy array
    mated_sim_scores_final = np.array(mated_sim_scores)
    mated_sim_scores_final = mated_sim_scores_final[mated_sim_scores_final<0.999]


    # Return FNIR and the array of mated similarity scores
    return fnir, mated_sim_scores_final



# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

## Function for calculating FPIR
def remove_probeid_in_classification(arr, value):
    for i, v in enumerate(arr):
        if v == value:
            return np.delete(arr, i)
    return arr  # Value not found in the array, return the original array


def compute_fpir(non_mated_df, sim_scores, im_ids, ids, thold=0.5):

    # U_D: set of non-mated identification transactions with reference database D. I.e. equal to number IDs with no enrolled ids.
    U_d_set = set(non_mated_df)
    U_d_set_len = len(U_d_set)
    cand_list_returned = 0

    # for each m_i (mated reference for transaction i), get score of a reference in identification transaction i. I.e. number of transactions equal to number of mated ids**2
    # Initialize the list to store the mated similarity scores
    nonmated_sim_scores = []

    for t_i in range(0,len(sim_scores)):
        probe = im_ids[t_i]
        id_name = ids[t_i]

        # check if probe is in mated
        if id_name in U_d_set:
            # Save the similarity scores of mated identifications
            nonmated_sim_scores.append(sim_scores[t_i])
            # nonmated_sim_scores.append(sim_scores[t_i])


            # if reference probe id is in negative list/below treshold, count 1
            classified_pos_list = sim_scores[t_i] > thold
            classified_pos_idx = list(np.where(classified_pos_list)[0]) # get indexes where the score is above threshold
            face_idx_pos_class = im_ids[classified_pos_idx] # get numerical ids in the positive class
            # remove first instance of probe id in classification list
            face_idx_pos_class_filtered = remove_probeid_in_classification(face_idx_pos_class, probe)

            # if length of candidate list (filtered, i.e. without the probe itsef) is greater than 0, count 1
            if len(face_idx_pos_class_filtered) > 0:
                cand_list_returned += 1

    fnir = cand_list_returned/U_d_set_len
    # Convert list to numpy array
    nonmated_sim_scores_final = np.array(nonmated_sim_scores).flatten()
    nonmated_sim_scores_final = nonmated_sim_scores_final[nonmated_sim_scores_final<0.99]

    return fnir, nonmated_sim_scores_final, sim_scores

# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


## Function for calculating confusion matrix scores

# False Positives = Number of instances belonging to the negative class but are classified as the positive class
# True Negatives = Number of instances belonging to the negative class that are correctly classsified as the negative class
# True positives: N instances belonging to the positive class that were also correctly classified as the positive class
# False negatives: N instances belonging to the positive class that were incorrectly classified to the negative class


def remove_probeid_in_classification(arr, value):
    for i, v in enumerate(arr):
        if v == value:
            return np.delete(arr, i)
    return arr  # Value not found in the array, return the original array

def confusion_matrix_scores(thold, sim_scores, im_ids, ids):
    "ids corresponds to identity class that is either mated or non-mated"
    tps = []
    fps = []
    tns = []
    fns = []

    for i in range(0,len(sim_scores)):
        probe = im_ids[i]
        print(probe)

        # for probe i
        if ids[i] in non_mated_ids:
            a = 1
        else:
            # Positive classification list
            classified_positive_list = sim_scores[i] >= thold
            classified_positive_idx = list(np.where(classified_positive_list)[0])

            # get the classified identities - positive
            face_idx_pos_class = im_ids[classified_positive_idx]
            print("correct ids",face_idx_pos_class)

            # remove first instance of probe id in classification list
            face_idx_pos_class_filtered = remove_probeid_in_classification(face_idx_pos_class, probe)
            print("filtered ids",face_idx_pos_class_filtered)

            ### Only for closed set
            # get TP
            tp = np.sum(face_idx_pos_class_filtered == probe)
            tps.append(tp)

            # get FP (classified as correct but not equal to probe)
            fp = len(face_idx_pos_class_filtered) - tp
            fps.append(fp)

            # Negative classification list
            classified_negative_list = sim_scores[i] < thold
            classified_negative_idx = list(np.where(classified_negative_list)[0])

            # get the classified identities - negative
            face_idx_neg_class = im_ids[classified_negative_idx]
            face_idx_neg_class_filtered = remove_probeid_in_classification(face_idx_neg_class, probe)


            # get TN (classified as false and not equal to probe)
            tn = np.sum(face_idx_neg_class_filtered != probe)
            tns.append(tn)

            # get FN (classified as false, but is actually equal to probe)
            fn = len(face_idx_neg_class_filtered) - tn
            fns.append(fn)

    return tps, fps, tns, fns


# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def GARBE(fnir_c, fnir_a, fpir_c, fpir_a, alpha=0.5):
    """
    Function calculates GARBe score based on ISO standard ISO/IEC DIS 19795-10
    """
    
    FPD = fpir_c/fpir_a
    print("FPD result: ", FPD)


    FND = fnir_c/fnir_a
    print("FND result: ", FND)
    
    GARBE = alpha_val * FPD + (1 - alpha_val) * FND
    print("GARBE result MagFace: ", GARBE)

    return FPD, FND, GARBE

# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
