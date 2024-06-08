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
            # thold = (np.percentile(sim_mat_c, percentile) + np.percentile(sim_mat_a, percentile)) / 2
            thold = percentile

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

    plt.figure(figsize=(12, 7))

    # Plot FNIR
    plt.subplot(1, 2, 1)
    plt.plot(plot_df.index, plot_df['FNIR_c'], 'r-', label='Children', linewidth=3)
    plt.plot(plot_df.index, plot_df['FNIR_a'], 'c-', label='Adults', linewidth=3)
    plt.xlabel('Threshold', fontsize=14)
    plt.ylabel('FNIR', fontsize=14)
    # plt.title('FNIR vs Threshold', fontsize=16)
    plt.ylim(0.0, 0.125)  # Limit the FNIR y-axis
    plt.legend(fontsize=12)
    plt.grid(True)

    # Plot FPIR
    plt.subplot(1, 2, 2)
    plt.plot(plot_df.index, plot_df['FPIR_c'], 'r-', label='Children', linewidth=3)
    plt.plot(plot_df.index, plot_df['FPIR_a'], 'c-', label='Adults', linewidth=3)
    plt.xlabel('Threshold', fontsize=14)
    plt.ylabel('FPIR', fontsize=14)
    # plt.title('FPIR vs Threshold', fontsize=16)
    plt.ylim(0.85, 1.0)  # Limit the FPIR y-axis
    plt.legend(fontsize=12)
    plt.grid(True)

    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_threshold_metrics_ex_1_1_together(df_all_threshold_x, title, save_fig_path):
    # Set the aesthetics for the plots
    sns.set(style="whitegrid")

    plot_df = df_all_threshold_x.groupby('Threshold').mean()

    plt.figure(figsize=(12, 7))

    # Plot FNIR
    plt.plot(plot_df.index, plot_df['FNIR_c'], color='#88E288', linestyle = '-', label='Children - FNIR', linewidth=2.7)
    plt.plot(plot_df.index, plot_df['FNIR_a'], color = '#95DFFF', linestyle = '-', label='Adults - FNIR',linewidth=2.7)
    plt.xlabel('Threshold', fontsize=14)
    plt.ylabel('FNIR and FPIR', fontsize=14)
    # plt.title('FNIR vs Threshold', fontsize=16)
    # plt.ylim(0.0, 0.125)  # Limit the FNIR y-axis
    plt.xlim(0.0, 1)
    plt.legend(fontsize=16)
    plt.grid(True)

    # Plot FPIR
    plt.plot(plot_df.index, plot_df['FPIR_c'], color='#88E288', linestyle = '--', label='Children - FPIR',linewidth=2.7)
    plt.plot(plot_df.index, plot_df['FPIR_a'], color = '#95DFFF', linestyle = '--', label='Adults - FPIR',linewidth=2.7)
    plt.xlabel('Threshold', fontsize=14)
    plt.ylabel('FPIR and FNIR', fontsize=14)
    # plt.title(f'FPIR and FNIR vs Threshold - {title}', fontsize=16)
    # plt.ylim(0.85, 1.0)  # Limit the FPIR y-axis
    plt.xlim(0.0, 1)
    plt.legend(fontsize=16)
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(f'{save_fig_path}TH.png')

    # Show the plot
    plt.show()





def plot_threshold_metrics_ex_1_1_together_w_conf(df_all_threshold_x, title, save_fig_path):
    # Set the aesthetics for the plots
    sns.set(style="whitegrid")

    plot_df_mean = df_all_threshold_x.groupby('Threshold').mean()
    plot_df_std = df_all_threshold_x.groupby('Threshold').std()
    count = df_all_threshold_x.groupby('Threshold').size().values

    # Calculate the 95% confidence intervals
    ci_factor = 1.96
    plot_df_ci = plot_df_std / np.sqrt(count[:, None]) * ci_factor

    plt.figure(figsize=(12, 7))

    # Plot FNIR with confidence intervals
    plt.plot(plot_df_mean.index, plot_df_mean['FNIR_c'], color='#88E288', linestyle='-', label='Children - FNIR', linewidth=2.7)
    plt.fill_between(plot_df_mean.index, plot_df_mean['FNIR_c'] - plot_df_ci['FNIR_c'], plot_df_mean['FNIR_c'] + plot_df_ci['FNIR_c'], color='#88E288', alpha=0.2)
    plt.plot(plot_df_mean.index, plot_df_mean['FNIR_a'], color='#95DFFF', linestyle='-', label='Adults - FNIR', linewidth=2.7)
    plt.fill_between(plot_df_mean.index, plot_df_mean['FNIR_a'] - plot_df_ci['FNIR_a'], plot_df_mean['FNIR_a'] + plot_df_ci['FNIR_a'], color='#95DFFF', alpha=0.2)

    plt.xlabel('Threshold', fontsize=14)
    plt.ylabel('FNIR and FPIR', fontsize=14)
    # plt.title('FNIR vs Threshold', fontsize=16)
    plt.xlim(0.0, 1)
    plt.legend(fontsize=16)
    plt.grid(True)

    # Plot FPIR with confidence intervals
    plt.plot(plot_df_mean.index, plot_df_mean['FPIR_c'], color='#88E288', linestyle='--', label='Children - FPIR', linewidth=2.7)
    plt.fill_between(plot_df_mean.index, plot_df_mean['FPIR_c'] - plot_df_ci['FPIR_c'], plot_df_mean['FPIR_c'] + plot_df_ci['FPIR_c'], color='#88E288', alpha=0.2)
    plt.plot(plot_df_mean.index, plot_df_mean['FPIR_a'], color='#95DFFF', linestyle='--', label='Adults - FPIR', linewidth=2.7)
    plt.fill_between(plot_df_mean.index, plot_df_mean['FPIR_a'] - plot_df_ci['FPIR_a'], plot_df_mean['FPIR_a'] + plot_df_ci['FPIR_a'], color='#95DFFF', alpha=0.2)

    plt.xlabel('Threshold', fontsize=14)
    plt.ylabel('FPIR and FNIR', fontsize=14)
    # plt.title(f'FPIR and FNIR vs Threshold - {title}', fontsize=16)
    plt.xlim(0.0, 1)
    plt.legend(fontsize=16)
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(f'{save_fig_path}TH.png')

    # Show the plot
    plt.show()



# -----------------------------------------------------------------------------------
# Hereunder is the experiment 1.2




def balance_child_data_can(y_df, print_stats=False, random_state=42):
    """
    Input: raw df for ylfw and rfw
    Returns: csvs with equally balanced children and adults
    Original child_balanced has random state 42
    """

    # Randomly sample 2000 identities from the entire dataset
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
            # thold = (np.percentile(sim_mat_c, percentile) + np.percentile(sim_mat_a, percentile)) / 2
            thold = percentile

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




def plot_threshold_metrics_ex_1_2(df_all_threshold_x, save_fig_path):
    # Set the aesthetics for the plots
    sns.set(style="whitegrid")

    plot_df = df_all_threshold_x.groupby('Threshold').mean()

    plt.figure(figsize=(12, 7))

    # Plot FNIR
    # plt.subplot(1, 2, 1)
    plt.plot(plot_df.index, plot_df['FNIR_c'], color='#FBC02D', linestyle = '-', label='Canonical - children - FNIR',linewidth=2.7)
    plt.plot(plot_df.index, plot_df['FNIR_a'], color='#88E288', linestyle = '-',label='Mixed quality - children - FNIR',linewidth=2.7)
    plt.xlabel('Threshold', fontsize=14)
    plt.ylabel('FNIR and FPIR', fontsize=14)
    # plt.title('FNIR vs Threshold', fontsize=16)
    # plt.ylim(0.0, 0.125)  # Limit the FNIR y-axis
    plt.xlim(0.0, 1)
    plt.legend(fontsize=16)
    plt.grid(True)

    # Plot FPIR
    # plt.subplot(1, 2, 2)
    plt.plot(plot_df.index, plot_df['FPIR_c'], color='#FBC02D', linestyle = '--', label='Canonical - children - FPIR',linewidth=2.7)
    plt.plot(plot_df.index, plot_df['FPIR_a'], color='#88E288', linestyle = '--', label='Mixed quality - children - FPIR',linewidth=2.7)
    plt.xlabel('Threshold', fontsize=14)
    plt.ylabel('FPIR and FNIR', fontsize=14)
    # plt.title('FPIR and FNIR vs Threshold', fontsize=16)
    # plt.ylim(0.85, 1.0)  # Limit the FPIR y-axis
    plt.xlim(0.0, 1)
    plt.legend(fontsize=16)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'{save_fig_path}TH.png')


    # Show the plot
    plt.show()
# plot_threshold_metrics_ex_1_2(df_all_threshold_x)



def plot_threshold_metrics_ex_1_2_w_conf(df_all_threshold_x, title,save_fig_path):
    # Set the aesthetics for the plots
    sns.set(style="whitegrid")

    plot_df_mean = df_all_threshold_x.groupby('Threshold').mean()
    plot_df_std = df_all_threshold_x.groupby('Threshold').std()
    count = df_all_threshold_x.groupby('Threshold').size().values

    # Calculate the 95% confidence intervals
    ci_factor = 1.96
    plot_df_ci = plot_df_std / np.sqrt(count[:, None]) * ci_factor

    plt.figure(figsize=(12, 7))

    # Plot FNIR with confidence intervals
    plt.plot(plot_df_mean.index, plot_df_mean['FNIR_c'], color='#FBC02D', linestyle='-', label='Canonical - children - FNIR', linewidth=2.7)
    plt.fill_between(plot_df_mean.index, plot_df_mean['FNIR_c'] - plot_df_ci['FNIR_c'], plot_df_mean['FNIR_c'] + plot_df_ci['FNIR_c'], color='#FBC02D', alpha=0.2)
    plt.plot(plot_df_mean.index, plot_df_mean['FNIR_a'], color='#88E288', linestyle='-', label='Mixed quality - children - FNIR', linewidth=2.7)
    plt.fill_between(plot_df_mean.index, plot_df_mean['FNIR_a'] - plot_df_ci['FNIR_a'], plot_df_mean['FNIR_a'] + plot_df_ci['FNIR_a'], color='#88E288', alpha=0.2)

    plt.xlabel('Threshold', fontsize=14)
    plt.ylabel('FNIR and FPIR', fontsize=14)
    # plt.title('FNIR vs Threshold', fontsize=16)
    plt.xlim(0.0, 1)
    plt.legend(fontsize=16)
    plt.grid(True)

    # Plot FPIR with confidence intervals
    plt.plot(plot_df_mean.index, plot_df_mean['FPIR_c'], color='#FBC02D', linestyle='--', label='Canonical - children - FPIR', linewidth=2.7)
    plt.fill_between(plot_df_mean.index, plot_df_mean['FPIR_c'] - plot_df_ci['FPIR_c'], plot_df_mean['FPIR_c'] + plot_df_ci['FPIR_c'], color='#FBC02D', alpha=0.2)
    plt.plot(plot_df_mean.index, plot_df_mean['FPIR_a'], color='#88E288', linestyle='--', label='Mixed quality - children - FPIR', linewidth=2.7)
    plt.fill_between(plot_df_mean.index, plot_df_mean['FPIR_a'] - plot_df_ci['FPIR_a'], plot_df_mean['FPIR_a'] + plot_df_ci['FPIR_a'], color='#88E288', alpha=0.2)

    plt.xlabel('Threshold', fontsize=14)
    plt.ylabel('FPIR and FNIR', fontsize=14)
    # plt.title('FPIR and FNIR vs Threshold', fontsize=16)
    plt.xlim(0.0, 1)
    plt.legend(fontsize=16)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'{save_fig_path}TH.png')

    # Show the plot
    plt.show()




def plot_threshold_metrics_ex_1_2_zoomed(df_all_threshold_x):
    # Set the aesthetics for the plots
    sns.set(style="whitegrid")

    plot_df = df_all_threshold_x.groupby('Threshold').mean()

    plt.figure(figsize=(12, 7))
    plt.plot(plot_df.index, plot_df['FNIR_c'], 'r-', label='Canonical - children - FNIR',linewidth=2.7)
    plt.plot(plot_df.index, plot_df['FNIR_a'], 'c-', label='Mixed quality - children - FNIR',linewidth=2.7)
    plt.xlabel('Threshold', fontsize=20)
    plt.ylabel('FNIR and FPIR', fontsize=20)
    # plt.title('FNIR vs Threshold', fontsize=20)
    plt.ylim(0.0, 0.07)  # Limit the FNIR y-axis
    plt.xlim(0.20, 0.3)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)

    plt.tight_layout()

    # Show the plot
    plt.show()



# -------------------------------------------------
# Experiment 2.3

import numpy as np
import pandas as pd

def compute_metrics_ex_2_3(random_states, percentiles, children_all, adults_all, image_names_c, image_names_a, sim_mat_c, sim_mat_a, num_ids_c, num_ids_a, ids_c, ids_a, balance_child_data, balance_adults_data_enrolled, compute_fnir, compute_fpir, GARBE, age_1, age_2):
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
        data_1 = balance_child_data_2_3(children_all, print_stats=False, random_state=random_state_i, age=age_1)
        data_2 = balance_child_data_2_3(children_all, print_stats=False, random_state=random_state_i, age=age_2)

        c_mates = data_1.groupby("identity_name").agg({'identity_name': ['count']})
        enrolled_identity_names_c = c_mates[c_mates[('identity_name', 'count')] > 1].index
        enrolled_image_names_c = list(data_1[data_1["identity_name"].isin(enrolled_identity_names_c)].image_name)
        non_enrolled_identity_names_c = c_mates[c_mates[('identity_name', 'count')] == 1].index
        non_enrolled_image_names_c = list(data_1[data_1["identity_name"].isin(non_enrolled_identity_names_c)].image_name)
        all_reference_image_names_c = list(data_1.image_name)

        a_mates = data_2.groupby("identity_name").agg({'identity_name': ['count']})
        enrolled_identity_names_a = a_mates[a_mates[('identity_name', 'count')] > 1].index
        enrolled_image_names_a = list(data_2[data_2["identity_name"].isin(enrolled_identity_names_a)].image_name)
        non_enrolled_identity_names_a = a_mates[a_mates[('identity_name', 'count')] == 1].index
        non_enrolled_image_names_a = list(data_2[data_2["identity_name"].isin(non_enrolled_identity_names_a)].image_name)
        all_reference_image_names_a = list(data_2.image_name)

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
            # thold = (np.percentile(sim_mat_c, percentile) + np.percentile(sim_mat_a, percentile)) / 2
            thold = percentile

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





def plot_threshold_metrics_ex_2_3(df_all_threshold_x, df_all_threshold_x_1, df_all_threshold_x_2, save_fig_path):
    # Set the aesthetics for the plots
    sns.set(style="whitegrid")

    plot_df = df_all_threshold_x.groupby('Threshold').mean()
    plot_df_1 = df_all_threshold_x_1.groupby('Threshold').mean()
    plot_df_2 = df_all_threshold_x_2.groupby('Threshold').mean()



    plt.figure(figsize=(12, 7))

    # Plot FNIR
    # plt.subplot(1, 2, 1)
    plt.plot(plot_df.index, plot_df['FNIR_c'], 'r-', label='1-3 years - FNIR',linewidth=2.7)
    plt.plot(plot_df.index, plot_df['FNIR_a'], 'y-', label='4-6 years - FNIR',linewidth=2.7)
    plt.plot(plot_df_1.index, plot_df_1['FNIR_c'], 'g-', label='7-9 years - FNIR',linewidth=2.7)
    plt.plot(plot_df_1.index, plot_df_1['FNIR_a'], 'b-', label='10-12 years - FNIR',linewidth=2.7)
    plt.plot(plot_df_2.index, plot_df_2['FNIR_c'], 'k-', label='13-15 years - FNIR',linewidth=2.7)

    plt.xlabel('Threshold', fontsize=14)
    plt.ylabel('FNIR and FPIR', fontsize=14)
    # plt.title('FNIR vs Threshold', fontsize=16)
    # plt.ylim(0.0, 0.125)  # Limit the FNIR y-axis
    plt.xlim(0.0, 1)
    plt.legend(fontsize=16)
    plt.grid(True)

    # Plot FPIR
    # plt.subplot(1, 2, 2)
    plt.plot(plot_df.index, plot_df['FPIR_c'], 'r--', label='1-3 years - FPIR',linewidth=2.7)
    plt.plot(plot_df.index, plot_df['FPIR_a'], 'y--', label='4-6 years - FPIR',linewidth=2.7)
    plt.plot(plot_df_1.index, plot_df_1['FPIR_c'], 'g--', label='7-9 years - FPIR',linewidth=2.7)
    plt.plot(plot_df_1.index, plot_df_1['FPIR_a'], 'b--', label='10-12 years - FPIR',linewidth=2.7)
    plt.plot(plot_df_2.index, plot_df_2['FPIR_c'], 'k--', label='13-15 years - FPIR',linewidth=2.7)

    plt.xlabel('Threshold', fontsize=14)
    plt.ylabel('FPIR and FNIR', fontsize=14)
    # plt.title('FPIR and FNIR vs Threshold', fontsize=16)
    # plt.ylim(0.85, 1.0)  # Limit the FPIR y-axis
    plt.xlim(0.0, 1)
    plt.legend(fontsize=16)
    plt.grid(True)


    plt.tight_layout()
    plt.savefig(f'{save_fig_path}TH.png')


    # Show the plot
    plt.show()



def plot_threshold_metrics_ex_2_3_w_conf(df_all_threshold_x, df_all_threshold_x_1, df_all_threshold_x_2, save_fig_path):
    # Set the aesthetics for the plots
    sns.set(style="whitegrid")

    def calculate_ci(df, column):
        """Calculate the 95% confidence interval for a column in the DataFrame."""
        mean = df[column].mean()
        std = df[column].std()
        count = df[column].count()
        ci = 1.96 * (std / count**0.5)
        return mean, ci

    # Prepare the data with confidence intervals
    plot_df = df_all_threshold_x.groupby('Threshold').agg(['mean', 'std'])
    plot_df_1 = df_all_threshold_x_1.groupby('Threshold').agg(['mean', 'std'])
    plot_df_2 = df_all_threshold_x_2.groupby('Threshold').agg(['mean', 'std'])

    plt.figure(figsize=(12, 7))

    # Plot FNIR with confidence intervals
    plt.plot(plot_df.index, plot_df['FNIR_c']['mean'], 'r-', label='1-3 years - FNIR', linewidth=2.7)
    plt.fill_between(plot_df.index, plot_df['FNIR_c']['mean'] - 1.96 * plot_df['FNIR_c']['std'] / plot_df['FNIR_c']['mean'].count()**0.5,
                     plot_df['FNIR_c']['mean'] + 1.96 * plot_df['FNIR_c']['std'] / plot_df['FNIR_c']['mean'].count()**0.5, color='r', alpha=0.2)
    plt.plot(plot_df.index, plot_df['FNIR_a']['mean'], 'y-', label='4-6 years - FNIR', linewidth=2.7)
    plt.fill_between(plot_df.index, plot_df['FNIR_a']['mean'] - 1.96 * plot_df['FNIR_a']['std'] / plot_df['FNIR_a']['mean'].count()**0.5,
                     plot_df['FNIR_a']['mean'] + 1.96 * plot_df['FNIR_a']['std'] / plot_df['FNIR_a']['mean'].count()**0.5, color='y', alpha=0.2)
    plt.plot(plot_df_1.index, plot_df_1['FNIR_c']['mean'], 'g-', label='7-9 years - FNIR', linewidth=2.7)
    plt.fill_between(plot_df_1.index, plot_df_1['FNIR_c']['mean'] - 1.96 * plot_df_1['FNIR_c']['std'] / plot_df_1['FNIR_c']['mean'].count()**0.5,
                     plot_df_1['FNIR_c']['mean'] + 1.96 * plot_df_1['FNIR_c']['std'] / plot_df_1['FNIR_c']['mean'].count()**0.5, color='g', alpha=0.2)
    plt.plot(plot_df_1.index, plot_df_1['FNIR_a']['mean'], 'b-', label='10-12 years - FNIR', linewidth=2.7)
    plt.fill_between(plot_df_1.index, plot_df_1['FNIR_a']['mean'] - 1.96 * plot_df_1['FNIR_a']['std'] / plot_df_1['FNIR_a']['mean'].count()**0.5,
                     plot_df_1['FNIR_a']['mean'] + 1.96 * plot_df_1['FNIR_a']['std'] / plot_df_1['FNIR_a']['mean'].count()**0.5, color='b', alpha=0.2)
    plt.plot(plot_df_2.index, plot_df_2['FNIR_c']['mean'], 'k-', label='13-15 years - FNIR', linewidth=2.7)
    plt.fill_between(plot_df_2.index, plot_df_2['FNIR_c']['mean'] - 1.96 * plot_df_2['FNIR_c']['std'] / plot_df_2['FNIR_c']['mean'].count()**0.5,
                     plot_df_2['FNIR_c']['mean'] + 1.96 * plot_df_2['FNIR_c']['std'] / plot_df_2['FNIR_c']['mean'].count()**0.5, color='k', alpha=0.2)

    plt.xlabel('Threshold', fontsize=14)
    plt.ylabel('FNIR and FPIR', fontsize=14)
    # plt.title('FNIR vs Threshold', fontsize=16)
    plt.xlim(0.0, 1)
    plt.legend(fontsize=16)
    plt.grid(True)

    # Plot FPIR with confidence intervals
    plt.plot(plot_df.index, plot_df['FPIR_c']['mean'], 'r--', label='1-3 years - FPIR', linewidth=2.7)
    plt.fill_between(plot_df.index, plot_df['FPIR_c']['mean'] - 1.96 * plot_df['FPIR_c']['std'] / plot_df['FPIR_c']['mean'].count()**0.5,
                     plot_df['FPIR_c']['mean'] + 1.96 * plot_df['FPIR_c']['std'] / plot_df['FPIR_c']['mean'].count()**0.5, color='r', alpha=0.2)
    plt.plot(plot_df.index, plot_df['FPIR_a']['mean'], 'y--', label='4-6 years - FPIR', linewidth=2.7)
    plt.fill_between(plot_df.index, plot_df['FPIR_a']['mean'] - 1.96 * plot_df['FPIR_a']['std'] / plot_df['FPIR_a']['mean'].count()**0.5,
                     plot_df['FPIR_a']['mean'] + 1.96 * plot_df['FPIR_a']['std'] / plot_df['FPIR_a']['mean'].count()**0.5, color='y', alpha=0.2)
    plt.plot(plot_df_1.index, plot_df_1['FPIR_c']['mean'], 'g--', label='7-9 years - FPIR', linewidth=2.7)
    plt.fill_between(plot_df_1.index, plot_df_1['FPIR_c']['mean'] - 1.96 * plot_df_1['FPIR_c']['std'] / plot_df_1['FPIR_c']['mean'].count()**0.5,
                     plot_df_1['FPIR_c']['mean'] + 1.96 * plot_df_1['FPIR_c']['std'] / plot_df_1['FPIR_c']['mean'].count()**0.5, color='g', alpha=0.2)
    plt.plot(plot_df_1.index, plot_df_1['FPIR_a']['mean'], 'b--', label='10-12 years - FPIR', linewidth=2.7)
    plt.fill_between(plot_df_1.index, plot_df_1['FPIR_a']['mean'] - 1.96 * plot_df_1['FPIR_a']['std'] / plot_df_1['FPIR_a']['mean'].count()**0.5,
                     plot_df_1['FPIR_a']['mean'] + 1.96 * plot_df_1['FPIR_a']['std'] / plot_df_1['FPIR_a']['mean'].count()**0.5, color='b', alpha=0.2)
    plt.plot(plot_df_2.index, plot_df_2['FPIR_c']['mean'], 'k--', label='13-15 years - FPIR', linewidth=2.7)
    plt.fill_between(plot_df_2.index, plot_df_2['FPIR_c']['mean'] - 1.96 * plot_df_2['FPIR_c']['std'] / plot_df_2['FPIR_c']['mean'].count()**0.5,
                     plot_df_2['FPIR_c']['mean'] + 1.96 * plot_df_2['FPIR_c']['std'] / plot_df_2['FPIR_c']['mean'].count()**0.5, color='k', alpha=0.2)

    plt.xlabel('Threshold', fontsize=14)
    plt.ylabel('FPIR and FNIR', fontsize=14)
    # plt.title('FPIR and FNIR vs Threshold', fontsize=16)
    plt.xlim(0.0, 1)
    plt.legend(fontsize=16)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'{save_fig_path}TH.png')

    # Show the plot
    plt.show()






def balance_child_data_2_3(y_df, print_stats=False, random_state=42, age = '1-3'):
    """
    Input: raw df for children
    Returns: balanced csvs for different age groups of children
    """
    # Randomly sample 100 from each age group
    data = y_df[y_df['children_agegroup'] == age].sample(n=400, random_state=random_state, replace = True)


    return data
