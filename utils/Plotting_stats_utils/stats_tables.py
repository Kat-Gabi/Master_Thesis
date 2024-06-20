import numpy as np
import pandas as pd



def generate_latex_table(df):
    latex_code = "\\begin{table}\n"
    latex_code += "\\caption{Descriptive Statistics} \n"
    latex_code += "\\label{table:descriptive_stats} \n"
    latex_code += "\\begin{tabular}{ccccccccc} \n"
    latex_code += "\\toprule\n"
    latex_code += "\\textbf{{Iter.}} & $FNIR_c$ & $FNIR_a$ & $FPIR_c$ & $FPIR_a$ & $FPD$ & $FND$ & $GARBE$ & $T$ \\\\\n"
    latex_code += "\\midrule\n"

    for idx, row in df.iterrows():
        row_data = " & ".join(row.values)
        latex_code += f"\\textbf{{{idx}}} & {row_data} \\\\\n"

    latex_code += "\\bottomrule\n"
    latex_code += "\\end{tabular} \n"
    latex_code += "\\end{table}\n"

    return latex_code


# def remove_probeid_in_classification(arr, score_arr, probe_id, score_threshold=0.9):
#     """
#     Removes probe unique id and any comparison with a score larger than score_threshold in the array
#     """
#     filtered_indices = [i for i, v in enumerate(arr) if score_arr[i] <= score_threshold]
#     return arr[filtered_indices]



# def compute_fpir_2(non_enrolled_sim_score, num_ids_non_enrolled, num_ids_all, thold=0.5, score_threshold=0.9):
#     """
#     FPIR formula from ISO standard ISO/IEC 19795-1:2021
#     """

#     # U_D: set of non-mated identification transactions with reference database D. I.e. equal to number IDs with no enrolled ids.
#     U_d_set_len = len(non_enrolled_sim_score)
#     cand_list_returned = 0

#     for i in range(U_d_set_len):
#         probe = num_ids_non_enrolled[i] # probe corresponding to the current sample similarity matrix

#         # for the non enrolled probe id, check if any of its similarity scores are above thold
#         classified_pos_list = non_enrolled_sim_score[i] > 0.32
#         classified_pos_idx = list(np.where(classified_pos_list)[0]) # get indexes where the score is above threshold
#         face_idx_pos_class = num_ids_all[classified_pos_idx] # get numerical ids in the positive class
#         similarity_scores = non_enrolled_sim_score[i][classified_pos_idx] # get similarity scores for the positive class
#         # remove instance of probe id in classification list
#         face_idx_pos_class_filtered = remove_probeid_in_classification(face_idx_pos_class, similarity_scores, probe)

#         # if length of candidate list (filtered, i.e. without the probe itself) is greater than 0, count 1
#         if len(face_idx_pos_class_filtered) > 0:
#             cand_list_returned += 1

#     fpir = cand_list_returned / 3000210 #U_d_set_len

#     return fpir

def evaluate_metrics_ex_1_1(random_states, children_all, adults_all, image_names_c, image_names_a, sim_mat_c, sim_mat_a, num_ids_c, num_ids_a, ids_c, ids_a, balance_child_data, balance_adults_data_enrolled, compute_fnir, compute_fpir_2, GARBE, remove_ones, threshold_number, alpha_garbe=0.25):
    sim_mat_dict_all = {}
    FNIR_c_list = []
    FNIR_a_list = []
    FPIR_c_list = []
    FPIR_a_list = []
    FPD_list = []
    FND_list = []
    GARBE_list = []
    threshold_list = []

    for random_state_i in random_states:
        ### Load children and adults balanced data ###
        children_balanced_df_i = balance_child_data(children_all, print_stats=False, random_state=random_state_i)
        adults_balanced_df_i = balance_adults_data_enrolled(children_balanced_df_i, adults_all, print_stats=False, random_state=random_state_i)

        ### All reference image names, enrolled and non-enrolled image names - children ###
        c_mates = children_balanced_df_i.groupby("identity_name").agg({'identity_name': ['count']})
        enrolled_identity_names_c = c_mates[c_mates[('identity_name', 'count')] > 1].index
        enrolled_image_names_c = list(children_balanced_df_i[children_balanced_df_i["identity_name"].isin(enrolled_identity_names_c)].image_name)
        non_enrolled_identity_names_c = c_mates[c_mates[('identity_name', 'count')] == 1].index
        non_enrolled_image_names_c = list(children_balanced_df_i[children_balanced_df_i["identity_name"].isin(non_enrolled_identity_names_c)].image_name)
        all_reference_image_names_c = list(children_balanced_df_i.image_name)

        ### All reference image names, enrolled and non-enrolled image names - adults ###
        a_mates = adults_balanced_df_i.groupby("identity_name").agg({'identity_name': ['count']})
        enrolled_identity_names_a = a_mates[a_mates[('identity_name', 'count')] > 1].index
        enrolled_image_names_a = list(adults_balanced_df_i[adults_balanced_df_i["identity_name"].isin(enrolled_identity_names_a)].image_name)
        non_enrolled_identity_names_a = a_mates[a_mates[('identity_name', 'count')] == 1].index
        non_enrolled_image_names_a = list(adults_balanced_df_i[adults_balanced_df_i["identity_name"].isin(non_enrolled_identity_names_a)].image_name)
        all_reference_image_names_a = list(adults_balanced_df_i.image_name)

        ### Similarity matrices for ids in reference database ###
        indices_c_all_reference = [image_names_c.index(name) for name in all_reference_image_names_c if image_names_c.index(name) < sim_mat_c.shape[1]]
        indices_a_all_reference = [image_names_a.index(name) for name in all_reference_image_names_a if image_names_a.index(name) < sim_mat_a.shape[1]]

        # Extract corresponding columns from the similarity matrix
        sim_mat_c_reference_cols = sim_mat_c[:, indices_c_all_reference]
        sim_mat_a_reference_cols = sim_mat_a[:, indices_a_all_reference]

        print(f'len(sim_mat_c) {len(sim_mat_c)}')
        print(f'len(sim_mat_c_reference_cols) {len(sim_mat_c_reference_cols)}')

        # Extract corresponding rows from the numerical ids
        num_ids_c_reference = num_ids_c[indices_c_all_reference]
        num_ids_a_reference = num_ids_a[indices_a_all_reference]

        ### Similarity matrices for non-enrolled ids ###
        indices_c_non_enrolled = [image_names_c.index(name) for name in non_enrolled_image_names_c if image_names_c.index(name) < sim_mat_c.shape[0]]
        indices_a_non_enrolled = [image_names_a.index(name) for name in non_enrolled_image_names_a if image_names_a.index(name) < sim_mat_a.shape[0]]

        sim_mat_c_non_enrolled_0 = sim_mat_c_reference_cols[indices_c_non_enrolled]
        sim_mat_a_non_enrolled_0 = sim_mat_a_reference_cols[indices_a_non_enrolled]

        num_ids_c_non_enrolled = num_ids_c[indices_c_non_enrolled]
        num_ids_a_non_enrolled = num_ids_a[indices_a_non_enrolled]

        ### Similarity matrices for enrolled ids ###
        indices_c_enrolled = [image_names_c.index(name) for name in enrolled_image_names_c if image_names_c.index(name) < sim_mat_c.shape[0]]
        indices_a_enrolled = [image_names_a.index(name) for name in enrolled_image_names_a if image_names_a.index(name) < sim_mat_a.shape[0]]

        sim_mat_c_enrolled_0 = sim_mat_c[np.ix_(indices_c_enrolled, indices_c_enrolled)]
        print(f'len(sim_mat_c) {len(sim_mat_c)}')
        print(f'len(sim_mat_c_enrolled_0) {len(sim_mat_c_enrolled_0)}')
        sim_mat_a_enrolled_0 = sim_mat_a[np.ix_(indices_a_enrolled, indices_a_enrolled)]

        num_ids_c_enrolled = num_ids_c[indices_c_enrolled]
        num_ids_a_enrolled = num_ids_a[indices_a_enrolled]

        thold = threshold_number

        ### Evaluation metrics ###
        # FNIR
        FNIR_c, sim_mat_c_enrolled = compute_fnir(sim_mat_c_enrolled_0, sim_mat_c, enrolled_identity_names_c, num_ids_c_enrolled, ids_c, thold=threshold_number)
        FNIR_a, sim_mat_a_enrolled = compute_fnir(sim_mat_a_enrolled_0, sim_mat_a, enrolled_identity_names_a, num_ids_a_enrolled, ids_a, thold=threshold_number)
        # FPIR

        print(f'len(children_all){len(children_all)}')
        print(f'len(sim_mat_c_non_enrolled_0) {len(sim_mat_c_enrolled_0)}')
        print(f'len(num_ids_c_non_enrolled) {len(num_ids_c_enrolled)}')
        print(f'len(num_ids_c_reference) {len(num_ids_c_reference)}')


        FPIR_c = compute_fpir_2(sim_mat_c_non_enrolled_0, num_ids_c_non_enrolled, num_ids_c_reference, thold=threshold_number)
        FPIR_a = compute_fpir_2(sim_mat_a_non_enrolled_0, num_ids_a_non_enrolled, num_ids_a_reference, thold=threshold_number)

        FPD_i, FND_i, GARBE_i = GARBE(FNIR_c, FNIR_a, FPIR_c, FPIR_a, alpha=alpha_garbe)

        FNIR_c_list.append(FNIR_c)
        FNIR_a_list.append(FNIR_a)
        FPIR_c_list.append(FPIR_c)
        FPIR_a_list.append(FPIR_a)
        FPD_list.append(FPD_i)
        FND_list.append(FND_i)
        GARBE_list.append(GARBE_i)
        threshold_list.append(thold)

        sim_mat_dict_all[f'sim_mat_c_enrolled_iteration_{random_state_i}'] = sim_mat_c_enrolled
        sim_mat_dict_all[f'sim_mat_a_enrolled_iteration_{random_state_i}'] = sim_mat_a_enrolled
        sim_mat_dict_all[f'sim_mat_c_non_enrolled_iteration_{random_state_i}'] = remove_ones(sim_mat_c_non_enrolled_0)
        sim_mat_dict_all[f'sim_mat_a_non_enrolled_iteration_{random_state_i}'] = remove_ones(sim_mat_a_non_enrolled_0)

        print("done")

    data = {
        'Iteration': random_states,
        'FNIR_c': FNIR_c_list,
        'FNIR_a': FNIR_a_list,
        'FPIR_c': FPIR_c_list,
        'FPIR_a': FPIR_a_list,
        'FPD': FPD_list,
        'FND': FND_list,
        'GARBE': GARBE_list,
        'Threshold': threshold_list
        }
    df_all_results = pd.DataFrame(data)
    return df_all_results, sim_mat_dict_all


# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def balance_child_data_can(y_df, print_stats=False, random_state=42):
    """
    Input: raw df for ylfw and rfw
    Returns: csvs with equally balanced children and adults
    Original child_balanced has random state 42
    """

    # Randomly sample 1000 identities from the entire dataset
    ylfw_witha_balanced = y_df.sample(n=2000, random_state=random_state)

    if print_stats:
        # Print the distribution of age groups and other relevant statistics
        print("Balanced data?:", ylfw_witha_balanced.children_agegroup.value_counts())

    return ylfw_witha_balanced


# for magface 1.2
def evaluate_metrics_ex_1_2(random_states, children_all, adults_all, image_names_c, image_names_a, sim_mat_c, sim_mat_a, num_ids_c, num_ids_a, ids_c, ids_a, balance_child_data, balance_adults_data_enrolled, compute_fnir, compute_fpir, GARBE, remove_ones, threshold_number, alpha_garbe=0.25):
    sim_mat_dict_all = {}
    FNIR_c_list = []
    FNIR_a_list = []
    FPIR_c_list = []
    FPIR_a_list = []
    FPD_list = []
    FND_list = []
    GARBE_list = []
    threshold_list = []

    for random_state_i in random_states:

        ### Load children and adults balanced data ###
        children_balanced_df_i = balance_child_data_can(children_all, print_stats=False, random_state=random_state_i)
        adults_balanced_df_i = balance_child_data(adults_all, print_stats=False, random_state=random_state_i)

        ### All reference image names, enrolled and non-enrolled image names - children ###
        c_mates = children_balanced_df_i.groupby("identity_name").agg({'identity_name': ['count']})
        enrolled_identity_names_c = c_mates[c_mates[('identity_name', 'count')] > 1].index
        enrolled_image_names_c = list(children_balanced_df_i[children_balanced_df_i["identity_name"].isin(enrolled_identity_names_c)].image_name)
        non_enrolled_identity_names_c = c_mates[c_mates[('identity_name', 'count')] == 1].index
        non_enrolled_image_names_c = list(children_balanced_df_i[children_balanced_df_i["identity_name"].isin(non_enrolled_identity_names_c)].image_name)
        all_reference_image_names_c = list(children_balanced_df_i.image_name)

        ### All reference image names, enrolled and non-enrolled image names - adults ###
        a_mates = adults_balanced_df_i.groupby("identity_name").agg({'identity_name': ['count']})
        enrolled_identity_names_a = a_mates[a_mates[('identity_name', 'count')] > 1].index
        enrolled_image_names_a = list(adults_balanced_df_i[adults_balanced_df_i["identity_name"].isin(enrolled_identity_names_a)].image_name)
        non_enrolled_identity_names_a = a_mates[a_mates[('identity_name', 'count')] == 1].index
        non_enrolled_image_names_a = list(adults_balanced_df_i[adults_balanced_df_i["identity_name"].isin(non_enrolled_identity_names_a)].image_name)
        all_reference_image_names_a = list(adults_balanced_df_i.image_name)

        ### Similarity matrices for ids in reference database ###
        indices_c_all_reference = [image_names_c.index(name) for name in all_reference_image_names_c]
        indices_a_all_reference = [image_names_a.index(name) for name in all_reference_image_names_a]

        # Extract corresponding columns from the similarity matrix
        sim_mat_c_reference_cols = sim_mat_c[:, indices_c_all_reference]
        sim_mat_a_reference_cols = sim_mat_a[:, indices_a_all_reference]

        # Extract corresponding rows from the numerical ids
        num_ids_c_reference = num_ids_c[indices_c_all_reference]
        num_ids_a_reference = num_ids_a[indices_a_all_reference]

        ### Similarity matrices for non-enrolled ids ###
        indices_c_non_enrolled = [image_names_c.index(name) for name in non_enrolled_image_names_c]
        indices_a_non_enrolled = [image_names_a.index(name) for name in non_enrolled_image_names_a]

        sim_mat_c_non_enrolled_0 = sim_mat_c_reference_cols[indices_c_non_enrolled]
        sim_mat_a_non_enrolled_0 = sim_mat_a_reference_cols[indices_a_non_enrolled]

        num_ids_c_non_enrolled = num_ids_c[indices_c_non_enrolled]
        num_ids_a_non_enrolled = num_ids_a[indices_a_non_enrolled]

        ### Similarity matrices for enrolled ids ###
        indices_c_enrolled = [image_names_c.index(name) for name in enrolled_image_names_c]
        indices_a_enrolled = [image_names_a.index(name) for name in enrolled_image_names_a]

        sim_mat_c_enrolled_0 = sim_mat_c[np.ix_(indices_c_enrolled, indices_c_enrolled)]
        sim_mat_a_enrolled_0 = sim_mat_a[np.ix_(indices_a_enrolled, indices_a_enrolled)]

        num_ids_c_enrolled = num_ids_c[indices_c_enrolled]
        num_ids_a_enrolled = num_ids_a[indices_a_enrolled]

        thold = threshold_number

        ### Evaluation metrics ###
        # FNIR
        FNIR_c, sim_mat_c_enrolled = compute_fnir(sim_mat_c_enrolled_0, sim_mat_c, enrolled_identity_names_c, num_ids_c_enrolled, ids_c, thold=threshold_number)
        FNIR_a, sim_mat_a_enrolled = compute_fnir(sim_mat_a_enrolled_0, sim_mat_a, enrolled_identity_names_a, num_ids_a_enrolled, ids_a, thold=threshold_number)
        # FPIR
        FPIR_c = compute_fpir(sim_mat_c_non_enrolled_0, num_ids_c_non_enrolled, num_ids_c_reference, thold=threshold_number)
        FPIR_a = compute_fpir(sim_mat_a_non_enrolled_0, num_ids_a_non_enrolled, num_ids_a_reference, thold=threshold_number)

        FPD_i, FND_i, GARBE_i = GARBE(FNIR_c, FNIR_a, FPIR_c, FPIR_a, alpha=alpha_garbe)

        FNIR_c_list.append(FNIR_c)
        FNIR_a_list.append(FNIR_a)
        FPIR_c_list.append(FPIR_c)
        FPIR_a_list.append(FPIR_a)
        FPD_list.append(FPD_i)
        FND_list.append(FND_i)
        GARBE_list.append(GARBE_i)
        threshold_list.append(thold)

        sim_mat_dict_all[f'sim_mat_c_enrolled_iteration_{random_state_i}'] = sim_mat_c_enrolled
        sim_mat_dict_all[f'sim_mat_a_enrolled_iteration_{random_state_i}'] = sim_mat_a_enrolled
        sim_mat_dict_all[f'sim_mat_c_non_enrolled_iteration_{random_state_i}'] = remove_ones(sim_mat_c_non_enrolled_0)
        sim_mat_dict_all[f'sim_mat_a_non_enrolled_iteration_{random_state_i}'] = remove_ones(sim_mat_a_non_enrolled_0)

        print("done")

    data = {
        'Iteration': random_states,
        'FNIR_c': FNIR_c_list,
        'FNIR_a': FNIR_a_list,
        'FPIR_c': FPIR_c_list,
        'FPIR_a': FPIR_a_list,
        'FPD': FPD_list,
        'FND': FND_list,
        'GARBE': GARBE_list,
        'Threshold': threshold_list
        }
    df_all_results = pd.DataFrame(data)
    return df_all_results, sim_mat_dict_all
