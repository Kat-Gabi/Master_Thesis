
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
## Function for calculating FNIR
def compute_fnir(enrolled_sim_mat, sim_mat, enrolled_ids, enrolled_num_id, ids, thold=0.5):
    """
    FNIR formula from ISO standard ISO/IEC 19795-1:2021
    ids: unique ids for all images in results
    enrolled_ids: ids for the enrolled images
    enrolled_sim_mat: enrolled similarity matrix
    sim_mat: all similarity scores
    """
    # M_D: set of mated identification transactions with reference database. - i.e. there can be multiple ids?
    M_d_set = set(enrolled_ids)
    M_d_set_len = len(enrolled_sim_mat)
    neg_ref = 0
    
    # For each id corresponding to the id in the set, check if one of it's corresponding ids are above threshold
    
    # Get enrolled similarity scores
    enrolled_sim_scores = []
    
    ## Iterate over each enrolled reference for transaction i
    for m_i, id_now in enumerate(ids):
        # Check if the identity is enrolled
        if id_now in M_d_set:
            mated_ids_exact = [id == id_now for id in ids] # Array of true and falses
            mated_sim_scores_slice = sim_mat[m_i] # Row corresponding to the enrolled probe id
            mated_sim_scores_slice_slice = [value for value, keep in zip(mated_sim_scores_slice, mated_ids_exact) if keep] #Only enrolled similarity scores for the same ids corresponding to the probe id
            enrolled_sim_scores.extend(mated_sim_scores_slice_slice)
    
    # Iterate over each enrolled reference for transaction i
    for i in range(M_d_set_len):
        probe = enrolled_num_id[i] # numerical id by magface, e.g. str value "African_244" becomes num. value 35. 
        
        # Check if the reference probe id is in negative list/below threshold
        classified_negative_list = enrolled_sim_mat[i] <= thold
        classified_negative_idx = list(np.where(classified_negative_list)[0])  # Get indexes where the score is below threshold
        face_idx_neg_class = enrolled_num_id[classified_negative_idx]  # Get numerical ids in the negative class
        # If numerical id in negative list is equal to the probe id, count 1
        if probe in face_idx_neg_class:
            neg_ref += 1

    # Calculate FNIR
    fnir = neg_ref / M_d_set_len
    
    enrolled_sim_scores_final = np.array(enrolled_sim_scores)
    enrolled_sim_scores_final = enrolled_sim_scores_final[enrolled_sim_scores_final<0.999]
    
    i = 0
    while len(enrolled_sim_scores_final) > (len(enrolled_sim_scores)-len(enrolled_ids)):
        i += 0.001
        print("NOT SAME LENGTH", len(enrolled_sim_scores_final), len(enrolled_sim_scores)-len(enrolled_ids))
        enrolled_sim_scores_final = enrolled_sim_scores_final[enrolled_sim_scores_final<0.999+i]
        
    return fnir, enrolled_sim_scores_final



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
