
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_demographic_distribution(group, df, title, column="Age", color = 'skyblue', bins=30, figsize=(10, 6)):
    """
    Plots an improved histogram with a kernel density estimate for the specified column
    in a style consistent with unified quality score visualizations.

    Parameters:
    - df: Pandas DataFrame.
    - column: str, default "Age"
        The column name to be used for the histogram.
    - bins: int, default 30
        The number of bins to use for the histogram.
    - figsize: tuple, default (10, 6)
        The dimensions for the figure size.
    """
    # Set the aesthetic style of the plots
    sns.set_style("whitegrid")

    plt.figure(figsize=figsize)  # Set figure size
    # Histogram with KDE in the chosen colors and style

    if group == 'children':

        age_bins = [1, 4, 7, 10, 13, 16]  # adding one more bin for 16-18
        age_labels = ['1-3', '4-6', '7-9', '10-12', '13-15']
        df['Age Group'] = pd.cut(df[column], bins=age_bins, labels=age_labels, right=False)

        ax = sns.histplot(data=df, x='Age Group', bins=age_bins, color=color, kde=False, label=title)
        # ax = sns.countplot(data=df, x='Age Group', palette='coolwarm')


        plt.title(f"{column} Distribution - {title}")  # Dynamic title based on the column name
        plt.xlabel("Age Group")  # Label for the x-axis
        plt.ylabel("Count")  # Label for the y-axis
        plt.xticks(rotation=-45)  # Rotate x-axis labels
        plt.legend()  # Display legend
        plt.grid(True)  # Enable grid lines for a whitegrid look
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.0f'),
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        xytext=(0, 5), textcoords='offset points', color='grey')
    else:
        # age_bins = [1, 4, 7, 10, 13, 16, 19]  # adding one more bin for 16-18
        # age_labels = ['1-3', '4-6', '7-9', '10-12', '13-15', '16-18']
        # df['Age Group'] = pd.cut(df[column], bins=age_bins, labels=age_labels, right=False)

        ax = sns.histplot(data=df, x=column, bins=bins, color=color, kde=False, label=title)
        # ax = sns.countplot(data=df, x='Age Group', palette='coolwarm')


        plt.title(f"{column} Distribution - {title}")  # Dynamic title based on the column name
        plt.xlabel(column)  # Label for the x-axis
        plt.ylabel("Count")  # Label for the y-axis
        plt.xticks(rotation=-45)  # Rotate x-axis labels
        plt.legend()  # Display legend
        plt.grid(True)  # Enable grid lines for a whitegrid look
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.0f'),
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        xytext=(0, 5), textcoords='offset points', color='grey')
    # Improve the layout and show the plot
    plt.tight_layout()
    plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_age_group_distribution_mated(df, title, figsize=(11, 7)):
    """
    Plots the distribution of age groups within the provided DataFrame.

    Parameters:
    - df: Pandas DataFrame containing an 'Age' column.
    - title: The title for the plot.
    - figsize: tuple, optional, default (11, 7)
        The dimensions for the figure size.
    """
    # Ensure 'Age' is of type integer
    df['Age'] = df['Age'].astype(int)

    # Define the age groups and corresponding labels
    age_bins = [1, 4, 7, 10, 13, 16]
    age_labels = ['1-3', '4-6', '7-9', '10-12', '13-15']

    enrolled_palette = {'Enrolled': '#1f77b4', 'Non-enrolled': '#ff7f0e'}

    # Update the DataFrame with age groupings
    df['Age Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)

    # Create the bar plot for the age groups
    plt.figure(figsize=figsize)
    ax = sns.countplot(data=df, x='Age Group', hue='Enrolled', palette=enrolled_palette, alpha=0.6)

    plt.title(f"Age Group Distribution - {title}")
    plt.xlabel("Age Group")
    plt.ylabel("# images")
    plt.xticks(rotation=45)
    plt.grid(axis='y')

    # Add value counts to the top of each bar
    for p in ax.patches:
        if p.get_height() > 0:  # Only annotate bars with height greater than 0
            ax.annotate(format(p.get_height(), '.0f'),
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        xytext=(0, 5), textcoords='offset points')

    # Show the plot
    plt.tight_layout()
    plt.show()

    return df

# Assuming df is your DataFrame
# df = pd.read_csv('your_file.csv')  # Load your DataFrame here
# plot_age_group_distribution_mated(df, "One instance of subset")



def plot_combined_ethnicity_distribution(df1, df2, title1, title2, title_all):
    sns.set_style("whitegrid")

    # Combining and sorting ethnicities
    all_ethnicities = pd.concat([df1['ethnicity'], df2['ethnicity']]).unique()
    all_ethnicities.sort()

    # Prepare data
    df1_counts = df1['ethnicity'].value_counts().reindex(all_ethnicities, fill_value=0).reset_index()
    df1_counts.columns = ['ethnicity', 'count']
    df2_counts = df2['ethnicity'].value_counts().reindex(all_ethnicities, fill_value=0).reset_index()
    df2_counts.columns = ['ethnicity', 'count']

    # Plot setup
    plt.figure(figsize=(10, 6))
    width = 0.35  # Width of the bars

    # Plotting
    x = np.arange(len(all_ethnicities))  # the label locations
    plt.bar(x - width/2, df1_counts['count'], width, label=title1, color='lightgreen', alpha=0.6)
    plt.bar(x + width/2, df2_counts['count'], width, label=title2, color='skyblue', alpha=0.6)

    # Labels, title and custom x-axis tick labels
    plt.ylabel('Image count')
    plt.title(f'Combined Ethnicity Distribution - {title_all}')
    plt.xticks(x, all_ethnicities, rotation=45)
    plt.xlabel('Ethnicity')

    # Adding a legend
    plt.legend(title="Dataset")

    plt.show()



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_age_group_distribution_mated_adults(df, title, figsize=(11, 7)):
    """
    Plots the distribution of age groups within the provided DataFrame.

    Parameters:
    - df: Pandas DataFrame containing an 'Age' column.
    - title: The title for the plot.
    - figsize: tuple, optional, default (11, 7)
        The dimensions for the figure size.
    """
    # Ensure 'Age' is of type integer
    df['Age'] = df['Age'].astype(int)

    # Define the age groups and corresponding labels
    age_bins = [20, 30, 40, 50, 60, 70, 80, 90]
    age_labels = ['20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89']

    enrolled_palette = {'Enrolled': '#1f77b4', 'Non-enrolled': '#ff7f0e'}

    # Update the DataFrame with age groupings
    df['Age Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)

    # Filter out any NaN values that might have been introduced in the 'Age Group' column
    df = df.dropna(subset=['Age Group'])

    # Create the bar plot for the age groups
    plt.figure(figsize=figsize)
    ax = sns.countplot(data=df, x='Age Group', hue='Enrolled', palette=enrolled_palette, alpha=0.6)

    plt.title(f"Age Group Distribution - {title}")
    plt.xlabel("Age Group")
    plt.ylabel("# images")
    plt.xticks(rotation=45)
    plt.grid(axis='y')

    # Add value counts to the top of each bar
    for p in ax.patches:
        if p.get_height() > 0:  # Only annotate bars with height greater than 0
            ax.annotate(format(p.get_height(), '.0f'),
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        xytext=(0, 5), textcoords='offset points')

    # Show the plot
    plt.tight_layout()
    plt.show()

    return df


def number_of_enrolled_ids_agebin(df, title, figsize=(11, 7)):
    """
    Plots the distribution of unique image IDs within the provided DataFrame,
    grouped by age bins and enrollment status, with specific colors for enrolled and non-enrolled.

    Parameters:
    - df: Pandas DataFrame containing 'Age' and 'im_id' columns.
    - title: str
        The title for the plot.
    - figsize: tuple, optional, default (10, 6)
        The dimensions for the figure size.
    """
    # Ensure 'Age' is of type integer
    df['Age'] = df['Age'].astype(int)

    # Define the age groups and corresponding labels
    age_bins = [1, 4, 7, 10, 13, 16]
    age_labels = ['1-3', '4-6', '7-9', '10-12', '13-15']

    # Update the DataFrame with age groupings
    df['Age Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)

    # Group by 'Age Group' and 'Enrolled', then count unique 'im_id'
    age_enrolled_counts = df.groupby(['Age Group', 'Enrolled'])['identity_name'].nunique().reset_index(name='Count')

    # Define the colors for the plot
    # enrolled_palette = {'Enrolled': 'cornflowerblue', 'Non-enrolled': 'orange'}
    # Define the colors for the plot
    enrolled_palette = {'Enrolled': '#1f77b4', 'Non-enrolled': '#ff7f0e'}




    # Create the bar plot for the age groups with count of unique image IDs
    plt.figure(figsize=figsize)
    ax = sns.barplot(data=age_enrolled_counts, x='Age Group', y='Count', hue='Enrolled', palette=enrolled_palette, alpha = 0.6)

    plt.title(f"Age Group Distribution of Enrollled IDs - {title}")
    plt.xlabel("Age Group")
    plt.ylabel("# Enrolled image IDs")
    plt.xticks(rotation=45)
    plt.grid(axis='y')

    # Customize the legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, ["Enrolled", "Non-Enrolled"], title='Enrolled Status')

    # Add value counts to the top of each bar
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.0f'),
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 5), textcoords='offset points')

    # Show the plot
    plt.tight_layout()
    plt.show()

    return df




def number_of_enrolled_ids_agebin(df, title, figsize=(11, 7)):
    """
    Plots the distribution of unique image IDs within the provided DataFrame,
    grouped by age bins and enrollment status, with specific colors for enrolled and non-enrolled.

    Parameters:
    - df: Pandas DataFrame containing 'Age' and 'im_id' columns.
    - title: str
        The title for the plot.
    - figsize: tuple, optional, default (10, 6)
        The dimensions for the figure size.
    """
    # Ensure 'Age' is of type integer
    df['Age'] = df['Age'].astype(int)

    # Define the age groups and corresponding labels
    age_bins = [20, 30, 40, 50, 60, 70, 80, 90]
    age_labels = ['20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89']

    enrolled_palette = {'Enrolled': '#1f77b4', 'Non-enrolled': '#ff7f0e'}

    # Update the DataFrame with age groupings
    df['Age Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)

    # Filter out any NaN values that might have been introduced in the 'Age Group' column
    df = df.dropna(subset=['Age Group'])

    # Group by 'Age Group' and 'Enrolled', then count unique 'im_id'
    age_enrolled_counts = df.groupby(['Age Group', 'Enrolled'])['identity_name'].nunique().reset_index(name='Count')

    # Define the colors for the plot
    # enrolled_palette = {'Enrolled': 'cornflowerblue', 'Non-enrolled': 'orange'}
    # Define the colors for the plot
    enrolled_palette = {'Enrolled': '#1f77b4', 'Non-enrolled': '#ff7f0e'}




    # Create the bar plot for the age groups with count of unique image IDs
    plt.figure(figsize=figsize)
    ax = sns.barplot(data=age_enrolled_counts, x='Age Group', y='Count', hue='Enrolled', palette=enrolled_palette, alpha = 0.6)

    plt.title(f"Age Group Distribution of Enrollled IDs - {title}")
    plt.xlabel("Age Group")
    plt.ylabel("# Enrolled image IDs")
    plt.xticks(rotation=45)
    plt.grid(axis='y')

    # Customize the legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, ["Enrolled", "Non-Enrolled"], title='Enrolled Status')

    # Add value counts to the top of each bar
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.0f'),
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 5), textcoords='offset points')

    # Show the plot
    plt.tight_layout()
    plt.show()

    return df
