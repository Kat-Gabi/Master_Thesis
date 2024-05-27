import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def compute_metrics_ex_1_1(random_states, percentiles, children_all, adults_all, image_names_c, image_names_a, sim_mat_c, sim_mat_a, num_ids_c, num_ids_a, ids_c, ids_a, balance_child_data, balance_adults_data_enrolled, compute_fnir, compute_fpir, GARBE):
    FNIR_c_list = []
    FNIR_a_list = []
    FPIR_c_list = []
    FPIR_a_list = []
    FPD_list = []
    FND_list = []
    GARBE_list = []
    threshold_list = []

    for random_state_i in random_states:
        # Load children and adults balanced data
        children_balanced_df_i = balance_child_data(children_all, print_stats=False, random_state=random_state_i)
        adults_balanced_df_i = balance_adults_data_enrolled(children_balanced_df_i, adults_all, print_stats=False, random_state=random_state_i)

        # All reference image names, enrolled and non-enrolled image names - children
        c_mates = children_balanced_df_i.groupby("identity_name").agg({'identity_name': ['count']})
        enrolled_identity_names_c = c_mates[c_mates[('identity_name', 'count')] > 1].index
        enrolled_image_names_c = list(children_balanced_df_i[children_balanced_df_i["identity_name"].isin(enrolled_identity_names_c)].image_name)
        non_enrolled_identity_names_c = c_mates[c_mates[('identity_name', 'count')] == 1].index
        non_enrolled_image_names_c = list(children_balanced_df_i[children_balanced_df_i["identity_name"].isin(non_enrolled_identity_names_c)].image_name)
        all_reference_image_names_c = list(children_balanced_df_i.image_name)

        # All reference image names, enrolled and non-enrolled image names - adults
        a_mates = adults_balanced_df_i.groupby("identity_name").agg({'identity_name': ['count']})
        enrolled_identity_names_a = a_mates[a_mates[('identity_name', 'count')] > 1].index
        enrolled_image_names_a = list(adults_balanced_df_i[adults_balanced_df_i["identity_name"].isin(enrolled_identity_names_a)].image_name)
        non_enrolled_identity_names_a = a_mates[a_mates[('identity_name', 'count')] == 1].index
        non_enrolled_image_names_a = list(adults_balanced_df_i[adults_balanced_df_i["identity_name"].isin(non_enrolled_identity_names_a)].image_name)
        all_reference_image_names_a = list(adults_balanced_df_i.image_name)

        # Similarity matrices for ids in reference database
        indices_c_all_reference = [image_names_c.index(name) for name in all_reference_image_names_c]
        indices_a_all_reference = [image_names_a.index(name) for name in all_reference_image_names_a]

        # Extract corresponding columns from the similarity matrix
        sim_mat_c_reference_cols = sim_mat_c[:, indices_c_all_reference]
        sim_mat_a_reference_cols = sim_mat_a[:, indices_a_all_reference]

        # Extract corresponding rows from the numerical ids
        num_ids_c_reference = num_ids_c[indices_c_all_reference]
        num_ids_a_reference = num_ids_a[indices_a_all_reference]

        # Similarity matrices for non-enrolled ids
        indices_c_non_enrolled = [image_names_c.index(name) for name in non_enrolled_image_names_c]
        indices_a_non_enrolled = [image_names_a.index(name) for name in non_enrolled_image_names_a]

        sim_mat_c_non_enrolled_0 = sim_mat_c_reference_cols[indices_c_non_enrolled]
        sim_mat_a_non_enrolled_0 = sim_mat_a_reference_cols[indices_a_non_enrolled]

        num_ids_c_non_enrolled = num_ids_c[indices_c_non_enrolled]
        num_ids_a_non_enrolled = num_ids_a[indices_a_non_enrolled]

        # Similarity matrices for enrolled ids
        indices_c_enrolled = [image_names_c.index(name) for name in enrolled_image_names_c]
        indices_a_enrolled = [image_names_a.index(name) for name in enrolled_image_names_a]

        sim_mat_c_enrolled_0 = sim_mat_c[np.ix_(indices_c_enrolled, indices_c_enrolled)]
        sim_mat_a_enrolled_0 = sim_mat_a[np.ix_(indices_a_enrolled, indices_a_enrolled)]

        num_ids_c_enrolled = num_ids_c[indices_c_enrolled]
        num_ids_a_enrolled = num_ids_a[indices_a_enrolled]

        for percentile in percentiles:
            thold = (np.percentile(sim_mat_c, percentile) + np.percentile(sim_mat_a, percentile)) / 2

            # Evaluation metrics
            FNIR_c, sim_mat_c_enrolled = compute_fnir(sim_mat_c_enrolled_0, sim_mat_c, enrolled_identity_names_c, num_ids_c_enrolled, ids_c, thold=thold)
            FNIR_a, sim_mat_a_enrolled = compute_fnir(sim_mat_a_enrolled_0, sim_mat_a, enrolled_identity_names_a, num_ids_a_enrolled, ids_a, thold=thold)

            FPIR_c = compute_fpir(sim_mat_c_non_enrolled_0, num_ids_c_non_enrolled, num_ids_c_reference, thold=thold)
            FPIR_a = compute_fpir(sim_mat_a_non_enrolled_0, num_ids_a_non_enrolled, num_ids_a_reference, thold=thold)

            alpha_garbe = 0.25
            FPD_i, FND_i, GARBE_i = GARBE(FNIR_c, FNIR_a, FPIR_c, FPIR_a, alpha=alpha_garbe)

            FNIR_c_list.append(FNIR_c)
            FNIR_a_list.append(FNIR_a)
            FPIR_c_list.append(FPIR_c)
            FPIR_a_list.append(FPIR_a)
            FPD_list.append(FPD_i)
            FND_list.append(FND_i)
            GARBE_list.append(GARBE_i)
            threshold_list.append(thold)

            print(f"Done for random_state: {random_state_i}, percentile: {percentile}")

    data = {
        'Iteration': np.repeat(random_states, len(percentiles)),
        'Percentile': percentiles * len(random_states),
        'FNIR_c': FNIR_c_list,
        'FNIR_a': FNIR_a_list,
        'FPIR_c': FPIR_c_list,
        'FPIR_a': FPIR_a_list,
        'FPD': FPD_list,
        'FND': FND_list,
        'GARBE': GARBE_list,
        'Threshold': threshold_list
    }

    df_all_threshold_x = pd.DataFrame(data)

    return df_all_threshold_x

def plot_threshold_metrics_ex_1_1(df_all_threshold_x):
    # Set the aesthetics for the plots
    sns.set(style="whitegrid")

    plot_df = df_all_threshold_x.groupby('Threshold').mean()

    plt.figure(figsize=(14, 7))

    # Plot FNIR
    plt.subplot(1, 2, 1)
    plt.plot(plot_df.index, plot_df['FNIR_c'], 'r-', label='Children', linewidth=2)
    plt.plot(plot_df.index, plot_df['FNIR_a'], 'c-', label='Adults', linewidth=2)
    plt.xlabel('Threshold', fontsize=14)
    plt.ylabel('FNIR', fontsize=14)
    plt.title('FNIR vs Threshold', fontsize=16)
    plt.ylim(0.0, 0.125)  # Limit the FNIR y-axis
    plt.legend(fontsize=12)
    plt.grid(True)

    # Plot FPIR
    plt.subplot(1, 2, 2)
    plt.plot(plot_df.index, plot_df['FPIR_c'], 'r-', label='Children', linewidth=2)
    plt.plot(plot_df.index, plot_df['FPIR_a'], 'c-', label='Adults', linewidth=2)
    plt.xlabel('Threshold', fontsize=14)
    plt.ylabel('FPIR', fontsize=14)
    plt.title('FPIR vs Threshold', fontsize=16)
    plt.ylim(0.85, 1.0)  # Limit the FPIR y-axis
    plt.legend(fontsize=12)
    plt.grid(True)

    plt.tight_layout()

    # Show the plot
    plt.show()
# # Example usage:
# random_states = [1, 2, 3]
# percentiles = [74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98]

# df_all_threshold_x = compute_metrics(random_states, percentiles, children_all, adults_all, image_names_c, image_names_a, sim_mat_c, sim_mat_a, num_ids_c, num_ids_a, ids_c, ids_a, balance_child_data, balance_adults_data_enrolled, compute_fnir, compute_fpir, GARBE)

# plot_threshold_metrics(df_all_threshold_x)




# from Model_utils.Model_funcs import *
# from Result_metric_utils.result_metrics import *
# from Data_proc_utils.Data_proc_funcs import *

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


def compute_metrics_ex_1_2(random_states, percentiles, children_all, adults_all, image_names_c, image_names_a, sim_mat_c, sim_mat_a, num_ids_c, num_ids_a, ids_c, ids_a, balance_child_data, balance_adults_data_enrolled, compute_fnir, compute_fpir, GARBE):
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
        # All reference image names, enrolled and non-enrolled image names - children
        c_mates = children_balanced_df_i.groupby("identity_name").agg({'identity_name': ['count']})
        enrolled_identity_names_c = c_mates[c_mates[('identity_name', 'count')] > 1].index
        enrolled_image_names_c = list(children_balanced_df_i[children_balanced_df_i["identity_name"].isin(enrolled_identity_names_c)].image_name)
        non_enrolled_identity_names_c = c_mates[c_mates[('identity_name', 'count')] == 1].index
        non_enrolled_image_names_c = list(children_balanced_df_i[children_balanced_df_i["identity_name"].isin(non_enrolled_identity_names_c)].image_name)
        all_reference_image_names_c = list(children_balanced_df_i.image_name)

        # All reference image names, enrolled and non-enrolled image names - adults
        a_mates = adults_balanced_df_i.groupby("identity_name").agg({'identity_name': ['count']})
        enrolled_identity_names_a = a_mates[a_mates[('identity_name', 'count')] > 1].index
        enrolled_image_names_a = list(adults_balanced_df_i[adults_balanced_df_i["identity_name"].isin(enrolled_identity_names_a)].image_name)
        non_enrolled_identity_names_a = a_mates[a_mates[('identity_name', 'count')] == 1].index
        non_enrolled_image_names_a = list(adults_balanced_df_i[adults_balanced_df_i["identity_name"].isin(non_enrolled_identity_names_a)].image_name)
        all_reference_image_names_a = list(adults_balanced_df_i.image_name)

        # Similarity matrices for ids in reference database
        indices_c_all_reference = [image_names_c.index(name) for name in all_reference_image_names_c]
        indices_a_all_reference = [image_names_a.index(name) for name in all_reference_image_names_a]

        # Extract corresponding columns from the similarity matrix
        sim_mat_c_reference_cols = sim_mat_c[:, indices_c_all_reference]
        sim_mat_a_reference_cols = sim_mat_a[:, indices_a_all_reference]

        # Extract corresponding rows from the numerical ids
        num_ids_c_reference = num_ids_c[indices_c_all_reference]
        num_ids_a_reference = num_ids_a[indices_a_all_reference]

        # Similarity matrices for non-enrolled ids
        indices_c_non_enrolled = [image_names_c.index(name) for name in non_enrolled_image_names_c]
        indices_a_non_enrolled = [image_names_a.index(name) for name in non_enrolled_image_names_a]

        sim_mat_c_non_enrolled_0 = sim_mat_c_reference_cols[indices_c_non_enrolled]
        sim_mat_a_non_enrolled_0 = sim_mat_a_reference_cols[indices_a_non_enrolled]

        num_ids_c_non_enrolled = num_ids_c[indices_c_non_enrolled]
        num_ids_a_non_enrolled = num_ids_a[indices_a_non_enrolled]

        # Similarity matrices for enrolled ids
        indices_c_enrolled = [image_names_c.index(name) for name in enrolled_image_names_c]
        indices_a_enrolled = [image_names_a.index(name) for name in enrolled_image_names_a]

        sim_mat_c_enrolled_0 = sim_mat_c[np.ix_(indices_c_enrolled, indices_c_enrolled)]
        sim_mat_a_enrolled_0 = sim_mat_a[np.ix_(indices_a_enrolled, indices_a_enrolled)]

        num_ids_c_enrolled = num_ids_c[indices_c_enrolled]
        num_ids_a_enrolled = num_ids_a[indices_a_enrolled]

        for percentile in percentiles:
            thold = (np.percentile(sim_mat_c, percentile) + np.percentile(sim_mat_a, percentile)) / 2

            # Evaluation metrics
            FNIR_c, sim_mat_c_enrolled = compute_fnir(sim_mat_c_enrolled_0, sim_mat_c, enrolled_identity_names_c, num_ids_c_enrolled, ids_c, thold=thold)
            FNIR_a, sim_mat_a_enrolled = compute_fnir(sim_mat_a_enrolled_0, sim_mat_a, enrolled_identity_names_a, num_ids_a_enrolled, ids_a, thold=thold)

            FPIR_c = compute_fpir(sim_mat_c_non_enrolled_0, num_ids_c_non_enrolled, num_ids_c_reference, thold=thold)
            FPIR_a = compute_fpir(sim_mat_a_non_enrolled_0, num_ids_a_non_enrolled, num_ids_a_reference, thold=thold)

            alpha_garbe = 0.25
            FPD_i, FND_i, GARBE_i = GARBE(FNIR_c, FNIR_a, FPIR_c, FPIR_a, alpha=alpha_garbe)

            FNIR_c_list.append(FNIR_c)
            FNIR_a_list.append(FNIR_a)
            FPIR_c_list.append(FPIR_c)
            FPIR_a_list.append(FPIR_a)
            FPD_list.append(FPD_i)
            FND_list.append(FND_i)
            GARBE_list.append(GARBE_i)
            threshold_list.append(thold)

            print(f"Done for random_state: {random_state_i}, percentile: {percentile}")

    data = {
        'Iteration': np.repeat(random_states, len(percentiles)),
        'Percentile': percentiles * len(random_states),
        'FNIR_c': FNIR_c_list,
        'FNIR_a': FNIR_a_list,
        'FPIR_c': FPIR_c_list,
        'FPIR_a': FPIR_a_list,
        'FPD': FPD_list,
        'FND': FND_list,
        'GARBE': GARBE_list,
        'Threshold': threshold_list
    }

    df_all_threshold_x = pd.DataFrame(data)

    return df_all_threshold_x


def plot_threshold_metrics_ex_1_2(df_all_threshold_x):
    # Set the aesthetics for the plots
    sns.set(style="whitegrid")

    plot_df = df_all_threshold_x.groupby('Threshold').mean()

    plt.figure(figsize=(14, 7))

    # Plot FNIR
    plt.subplot(1, 2, 1)
    plt.plot(plot_df.index, plot_df['FNIR_c'], 'r-', label='Canonical - children')
    plt.plot(plot_df.index, plot_df['FNIR_a'], 'c-', label='Mixed quality - children')
    plt.xlabel('Threshold', fontsize=14)
    plt.ylabel('FNIR', fontsize=14)
    plt.title('FNIR vs Threshold', fontsize=16)
    plt.ylim(0.0, 0.125)  # Limit the FNIR y-axis
    plt.legend(fontsize=12)
    plt.grid(True)

    # Plot FPIR
    plt.subplot(1, 2, 2)
    plt.plot(plot_df.index, plot_df['FPIR_c'], 'r-', label='Canonical - children')
    plt.plot(plot_df.index, plot_df['FPIR_a'], 'c-', label='Mixed quality - children')
    plt.xlabel('Threshold', fontsize=14)
    plt.ylabel('FPIR', fontsize=14)
    plt.title('FPIR vs Threshold', fontsize=16)
    plt.ylim(0.85, 1.0)  # Limit the FPIR y-axis
    plt.legend(fontsize=12)
    plt.grid(True)

    plt.tight_layout()

    # Show the plot
    plt.show()
