# Load libraries
import pandas as pd
import seaborn as sns
sns.set()
import random
random.seed(42)
import os
import shutil
import numpy as np
from collections import Counter



# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''



# Balance child data
def balance_child_data(y_df, print_stats=False, random_state=42):
    """
    Input: raw df for ylfw and rfw
    Returns: csvs with equally balanced children and adults
    Original child_balanced has random state 42
    """

    ### 1. Undersample based on the minority class in the children's age group in YLFW
    # - Keep racial distribution as in minority class
    # - Delete based on IDs
    # (obs be aware that each id have images in different age groups). I.e. The minority class will also have other ids in other age groups.
    # Therefore we sample based on number of images per age group.

    age_labels = ['1-3', '4-6', '7-9', '10-12', '13-15']

    # Take minority age group BASED ON N_IMAGES, and make dataframe
    min_agegroup = y_df.groupby('children_agegroup').image_name.count().sort_values(ascending=False).idxmin()
    minority_agedf = y_df[y_df.children_agegroup == min_agegroup]


    # Remove samples in the other age groups such that they have approximately the same amount of images and same racial distribution.
    # This is done by sampling the same amount of images within each ethnicity group.

    # Save variables of frequency within each ethnicity
    etnicities = list(y_df.ethnicity.unique())
    minority_etnicity_dist = minority_agedf.groupby('ethnicity').image_name.count().sort_values(ascending=False)

    # Get this distribution of frequency in ethnicities from each age group
    ylfw_witha_balanced = minority_agedf.copy()
    for agegroup in age_labels:
        if agegroup != min_agegroup:
            agegroup_df = y_df[y_df.children_agegroup == agegroup]

            # Get number of ids based on percentage dist of the minority age group
            for e in etnicities:
                freq_e = minority_etnicity_dist[e]

                # Randomly sample the images from the current age group from this ethnicity
                age_ethn_df = agegroup_df[agegroup_df.ethnicity == e]
                sample_eids = age_ethn_df.sample(n=freq_e, random_state=random_state)

                # Add theese to the minority dataset - to create a balanced dataset with the other age groups
                ylfw_witha_balanced = pd.concat([ylfw_witha_balanced, sample_eids], ignore_index=True)

    if print_stats:
        # Print Race distribution
        print("minority age group from childrens data: ", min_agegroup, "\nnumber of images: ", len(minority_agedf),
                "\n\nracial distribution:", minority_agedf.groupby('ethnicity').image_name.count().sort_values(ascending=False),
                "\n\nAll new groups should have same distribution")


        print("\nOther age group stats:","\n0-3\n",ylfw_witha_balanced[ylfw_witha_balanced.children_agegroup == "0-3"].groupby('ethnicity').image_name.count().sort_values(ascending=False))
        print("\n16-18",ylfw_witha_balanced[ylfw_witha_balanced.children_agegroup == "16-18"].groupby('ethnicity').image_name.count().sort_values(ascending=False))

        print("Balanced data?:", ylfw_witha_balanced.children_agegroup.value_counts())

    return ylfw_witha_balanced


## Example use:

# random_states = [1,2,3,4,5,6,7,8,9,10]
# children_all = pd.read_csv('../data/YLFW_full_info_including_adults.csv')
# children_balanced_df_1 = balance_child_data(children_all, print_stats=True, random_state=random_states[0])

# save as csv
#children_balanced_df_1.to_csv('../data/children_balanced_df_1.csv', index=False)


# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''



# Balance child data
# def balance_child_data_can(y_df, print_stats=False, random_state=42):
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


# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''







# Balance adults data
def balance_adults_data_enrolled(children_balanced_df_i, a_df, print_stats=False, random_state=42):
    """
    Input: adults full df and balanced child df at iteration i. Set random state equal to random state i for generation of children balanced df
    Returns: balanced adults df with equally balanced distribution of ethnicities and enrolled/non_enrolled ids as in children balanced df
    """

    # Remove identities with age standard deviation larger than 10 years.
    std_df = a_df.groupby("identity_name")['Age'].agg(['count', 'std']).reset_index().sort_values(by="std", ascending=False)
    high_std_ids = np.array(std_df[std_df["std"]>=10].identity_name)
    a_df = a_df[~a_df.identity_name.isin(high_std_ids)]

    random.seed(random_state)

    # Split in mated and non-mated ids
    c_mates = children_balanced_df_i.groupby("identity_name").agg({'identity_name': ['count']})
    c_enrolled_ids = c_mates[c_mates[('identity_name', 'count')] > 1].index
    c_non_enrolled_ids = c_mates[c_mates[('identity_name', 'count')] == 1].index


    a_mates = a_df.groupby("identity_name").agg({'identity_name': ['count']})
    a_enrolled_ids = a_mates[a_mates[('identity_name', 'count')] > 1].index

    # Get distribution to stratify on.
    c_enrolled_df = children_balanced_df_i[children_balanced_df_i["identity_name"].isin(set(c_enrolled_ids))]
    c_enrolled_ethnicity = c_enrolled_df.groupby('ethnicity').identity_name.nunique().sort_values(ascending=False)
    c_non_enrolled_df = children_balanced_df_i[children_balanced_df_i["identity_name"].isin(set(c_non_enrolled_ids))]
    c_non_enrolled_ethnicity = c_non_enrolled_df.groupby('ethnicity').identity_name.nunique().sort_values(ascending=False)
    #print(c_enrolled_ethnicity)

    etnicities = list(children_balanced_df_i.ethnicity.unique())
    a_balanced = pd.DataFrame()
    for e in etnicities:
        # a_df of etnicity group e
        a_ethnicity_df = a_df[a_df.ethnicity == e]


        ## For enrolled ids:
        n_enrolled_e = c_enrolled_ethnicity[e] # number of enrolled ids in ethnicity e in children

        # Randomly sample this number of ids and corresponding images from a_df in etnicity group e
        a_enrolled_ethnicity_ids = a_ethnicity_df[a_ethnicity_df["identity_name"].isin(set(a_enrolled_ids))].identity_name.unique()
        random_sample_enrolled_ids = random.sample(list(a_enrolled_ethnicity_ids), n_enrolled_e) # same size as enrolled ids in ethnicity e in children

        #print("is child ids same as adults ids number", n_enrolled_e,len(random_sample_enrolled_ids)  )

        a_enrolled_ethnicity_df = a_ethnicity_df[a_ethnicity_df["identity_name"].isin(set(random_sample_enrolled_ids))] # "final sampling"

        # Add theese to balanced adults dataset
        a_balanced = pd.concat([a_balanced, a_enrolled_ethnicity_df], ignore_index=True)



        ## For non-enrolled ids:
        n_non_enrolled_e = c_non_enrolled_ethnicity[e] # number of enrolled ids in ethnicity e in children

        # identities allowed to sample from
        a_non_enrolled_ethnicity_ids = a_ethnicity_df[~a_ethnicity_df["identity_name"].isin(set(random_sample_enrolled_ids))].identity_name.unique()


        # Shuffle the list
        random.shuffle(list(a_non_enrolled_ethnicity_ids))

        # Take the first n_non_en elements
        random_sample_non_enrolled_ids = a_non_enrolled_ethnicity_ids[:n_non_enrolled_e]

        # For each of these ids, take one image
        a_non_enrolled_ethnicity_ids = a_ethnicity_df[a_ethnicity_df["identity_name"].isin(random_sample_non_enrolled_ids)] # "final sampling"
        a_non_enrolled_ethnicity_image_names = a_non_enrolled_ethnicity_ids.groupby('identity_name')['image_name'].first().reset_index().image_name.unique()

        # Get org df with these img names
        a_non_enrolled_ethnicity_df = a_ethnicity_df[a_ethnicity_df["image_name"].isin(set(a_non_enrolled_ethnicity_image_names))] # "final sampling"

        # Count occurrences of each element
        counts = Counter(a_non_enrolled_ethnicity_df.identity_name)

        # Get the number of duplicates
        num_duplicates = sum(count for count in counts.values() if count > 1)

        # add theese to balanced adults dataset
        a_balanced = pd.concat([a_balanced, a_non_enrolled_ethnicity_df], ignore_index=True)

    if print_stats:
        a_bal_mates = a_balanced.groupby("identity_name").agg({'identity_name': ['count']})
        a_bal_enrolled_ids = a_bal_mates[a_bal_mates[('identity_name', 'count')] > 1].index
        a_bal_non_enrolled_ids = a_bal_mates[a_bal_mates[('identity_name', 'count')] == 1].index

        print("Balanced data between adults and children?:",
            "\n\nadults: ", a_balanced.groupby('ethnicity').identity_name.nunique().sort_values(ascending=False),
            "\nnumber of enrolled, and non-enrolled ids (a): ", len(set(a_bal_enrolled_ids)), len(set(a_bal_non_enrolled_ids)),
            "\n\nchildren: ", children_balanced_df_i.groupby('ethnicity').identity_name.nunique().sort_values(ascending=False),
            "\nnumber of enrolled, and non-enrolled ids (c): ", len(set(c_enrolled_ids)), len(set(c_non_enrolled_ids)))

        print("Duplicates?",num_duplicates)

        print("is child ids same as adults ids number non-enrolled?", n_non_enrolled_e,len(a_non_enrolled_ethnicity_df)  )


    return a_balanced

# Example usage
# random_states = [1,2,3,4,5,6,7,8,9,10]
# children_all = pd.read_csv('../data/YLFW_full_info_excluding_adults.csv')
# a_df = pd.read_csv('../data/RFW_full_info_excluding_children.csv')
# children_balanced_df_1 = balance_child_data(children_all, print_stats=False, random_state=random_states[0])


# balance_adults_data_enrolled(children_balanced_df_1, a_df, print_stats=True, random_state=random_states[0])

#a_balanced.to_csv('../data/adults_balanced.csv', index=False)



# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# Save dataframe as enrolled and non-enrolled

# def split_df_enrolled_non_enrolled(df_balanced):
#     """
#     Input: balanced df
#     Returns: df of enrolled ids and df with non-enrolled ids

#     """
#     # Mated and non-mated ids
#     df_mates = df_balanced.groupby("im_id").agg({'im_id': ['count']})
#     df_mated_ids = df_mates[df_mates[('im_id', 'count')] > 1].index
#     df_nonmated_ids = df_mates[df_mates[('im_id', 'count')] == 1].index

#     return df


# Example usage:
#y_df = pd.read_csv('../data/raw_full/raw_ylfw_df.csv')
#a_df = pd.read_csv('../data/raw_full/raw_rfw_df.csv')
#ylfw_witha_balanced = balance__child_data(y_df, a_df, print_stats=True)
#a_balanced = balance_adults_data(ylfw_witha_balanced, a_df, print_stats=False, random_state=42)

# enrolled_a, non_enrolled_a = split_df_enrolled_non_enrolled(a_balanced)
# Save to csv
#enrolled_a.to_csv('../data/adults_balanced_1_enrolled.csv', index=False)


# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# Check number of images in folders

def count_images_in_directory(directory):
    """
    Count the number of image files in a directory.
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']  # Add more extensions if needed
    image_count = 0

    # Iterate through all files in the directory
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)

        # Check if the file is a regular file (not a directory)
        if os.path.isfile(file_path):
            # Check if the file has a recognized image extension
            if any(file_name.lower().endswith(ext) for ext in image_extensions):
                image_count += 1

    return image_count

def count_images_in_all_directories(root_directory):
    """
    Count the number of image files in all directories and subdirectories.
    """
    total_image_count = 0

    # Walk through all directories and subdirectories
    for dirpath, dirnames, filenames in os.walk(root_directory):
        # Count the number of image files in the current directory
        current_dir_image_count = count_images_in_directory(dirpath)

        # Add the count to the total image count
        total_image_count += current_dir_image_count

        # Print the directory path and the number of images
        #print(f"Directory: {dirpath}, Images: {current_dir_image_count}")

    return total_image_count

# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# Move data for adults

def move_adults(image_dir, dst_root, adults_df):
    image_file_names = adults_df['img_name'].tolist()

    # Get the list of folders in the image directory
    folders = [os.path.join(image_dir, d) for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]
    # Iterate through each folder
    for folder in folders:
        # Get the list of subfolders
        folders2 = [os.path.join(folder, d) for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
        for folder2 in folders2:
        # Get the list of image files in the folder
            files = [f for f in os.listdir(folder2) if os.path.isfile(os.path.join(folder2, f))]

            # Check if any of the image file names from the CSV are in the folder
            for file in files:
                if file[:-4] in image_file_names:
                    # Move the image file to the output directory

                    # Make folder with ethnicityname_imageid
                    folder2_add = folder2.split("/")[-2:]
                    folder2_add_real = '_'.join(folder2_add)

                    dst_root = dst_root
                    src = os.path.join(folder2, file)
                    dst = os.path.join(dst_root, folder2_add_real)
                    os.makedirs(dst,exist_ok=True)
                    shutil.copy(src, dst)
    print('Done moving files')

# Example usage:
#adults_df = pd.read_csv('/Users/gabriellakierulff/Desktop/HCAI/speciale/Master_Thesis/data/adults_balanced.csv')
#image_dir = '../data/raw_full/RFW/data'
#dst_root = '../data/raw_full/adults' #../data/raw_full/adults'
#move_adults(image_dir, dst_root, adults_df)

## Check number of images in folder
#root_directory = '/Users/gabriellakierulff/Desktop/HCAI/speciale/Master_Thesis/data/raw_full/adults'
#total_images = count_images_in_all_directories(root_directory)
#print(f"Total number of images: {total_images}")

# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# Check for duplicates

def remove_first_instance_of_duplicate_file_names(folder_path):
    "Function to find duplicate file names within a folder and remove the first instance"
    file_names = {}  # Dictionary to store file names and their counts

    # Traverse each folder recursively
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)

            if file_name in file_names:
                # If the file name already exists, remove the entire containing folder
                os.system(f"rm -rf '{os.path.dirname(file_path)}'")
                print(f"Removed folder containing duplicate file: {os.path.dirname(file_path)}")
                break  # Once a duplicate is found and removed, exit the loop
            else:
                file_names[file_name] = file_path

# Main function to check for duplicate file names in all folders
def check_for_duplicate_file_names_and_remove_first_instance(root_folder):
    for root, dirs, files in os.walk(root_folder):
        print(f"Checking duplicates in folder: {root}")
        remove_first_instance_of_duplicate_file_names(root)

# Example usage:
#root_folder = '/Users/gabriellakierulff/Desktop/HCAI/speciale/Master_Thesis/data/raw_full/adults'
#check_for_duplicate_file_names_and_remove_first_instance(root_folder)
## Check number of images in folder now
#root_directory = '/Users/gabriellakierulff/Desktop/HCAI/speciale/Master_Thesis/data/raw_full/adults'
#total_images = count_images_in_all_directories(root_directory)
#print(f"Total number of images: {total_images}") # should be 3306


# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# CHILDREN: Only move files from YLFW full and RFW full also in balanced children csv folder
def move_children(image_dir_rfw, image_dir_ylfw, dst_root, children_df):

    # First for RFW
    #rfw_csv = children_csv[children_csv.img_org_name.notna()]
    #image_file_names = rfw_csv['img_org_name'].tolist()
    rfw_csv = children_df[children_df["img_name"].str.contains('\.')]
    image_file_names = rfw_csv['img_name'].tolist()

    # Get the list of folders in the image directory
    folders = [os.path.join(image_dir_rfw, d) for d in os.listdir(image_dir_rfw) if os.path.isdir(os.path.join(image_dir_rfw, d))]
    # Iterate through each folder
    for folder in folders:
        # Get the list of subfolders
        folders2 = [os.path.join(folder, d) for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
        for folder2 in folders2:
        # Get the list of image files in the folder
            files = [f for f in os.listdir(folder2) if os.path.isfile(os.path.join(folder2, f))]

            # Check if any of the image file names from the CSV are in the folder
            for file in files:
                if file[:-4] in image_file_names:
                    # Move the image file to the output directory

                    # Make folder with ethnicityname_imageid
                    folder2_add = folder2.split("/")[-2:]
                    folder2_add_real = '_'.join(folder2_add)

                    dst_root = dst_root
                    src = os.path.join(folder2, file)
                    dst = os.path.join(dst_root, folder2_add_real)
                    os.makedirs(dst,exist_ok=True)
                    shutil.copy(src, dst)
    print('Done moving RFW files')


    # THEN YLFW
    #ylfw_csv = children_csv[children_csv.img_org_name.isna()]
    ylfw_csv = children_df[~children_df["img_name"].str.contains('\.')]
    ylfw_image_file_names = ylfw_csv['img_name'].tolist()

    # Get the list of folders in the image directory
    folders = [os.path.join(image_dir_ylfw, d) for d in os.listdir(image_dir_ylfw) if os.path.isdir(os.path.join(image_dir_ylfw, d))]
    # Iterate through each folder
    for folder in folders:
        # Get the list of image files in the folder
        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        # Check if any of the image file names from the CSV are in the folder
        for file in files:
            if file[:-4] in ylfw_image_file_names:
                # Move the image file to same foldername in the output directory
                folder_dst = folder.split("/")[-1]

                dst_root = dst_root
                src = os.path.join(folder, file)
                dst = os.path.join(dst_root, folder_dst)
                os.makedirs(dst,exist_ok=True)
                shutil.copy(src, dst)

    print('Done moving YLFW files')

# Example usage:
# children_df = pd.read_csv('/Users/gabriellakierulff/Desktop/HCAI/speciale/Master_Thesis/data/child_balanced.csv')
# image_dir_rfw = "../data/raw_full/RFW/data"
# image_dir_ylfw = "../data/raw_full/YLFW/data_aligned"
# dst_root = "../data/raw_full/children"
# move_children(image_dir_rfw, image_dir_ylfw, dst_root, children_df)

## Check number of images in folder
# root_directory = '../data/raw_full/children'
# total_images = count_images_in_all_directories(root_directory)
# print(f"Total number of images: {total_images}")
