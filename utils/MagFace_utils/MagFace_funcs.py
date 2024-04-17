# Load libraries
import numpy as np
import cv2
from matplotlib import pyplot as plt
# import torch
import seaborn as sns
import pandas as pd
sns.set(style="white")




# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


# For all Ids, get last id name and convert to unique ids
def convert_unique_ids(ids):
    unique_ids_list = []
    for id in ids:
        im_id = id.split("/")[-1][:-4]
        if '.' in im_id:
            un_id = im_id.split("_")[0]
        else:
            un_id = '_'.join(im_id.split("_")[:-1])

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
