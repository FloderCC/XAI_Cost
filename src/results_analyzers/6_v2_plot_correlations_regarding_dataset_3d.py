import copy
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.interpolate import griddata

from src.results_analyzers.utils import divide_dataframe_regarding_mcc

# Set default font size
mpl.rcParams['font.size'] = 10


def generate_correlation_matrices(json_file_path, mode):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    correlation_matrices = {}

    for first_key, second_level in data.items():
        correlation_matrices[first_key] = {}

        for second_key, third_level in second_level.items():
            for third_key, fourth_level in third_level.items():
                second_level[second_key][third_key] = second_level[second_key][third_key][mode]

        for second_key, third_level in second_level.items():
            # Convert the third level values into a DataFrame
            df = pd.DataFrame(third_level)

            # renaming columns
            df.rename(columns={'LIME (ALL)': 'LIME'}, inplace=True)
            df.rename(columns={'Permutation Importance': 'PI'}, inplace=True)

            # Calculate the correlation matrix
            correlation_matrix = df.corr()

            # Store the correlation matrix in the result dictionary
            correlation_matrices[first_key][second_key] = correlation_matrix

    return correlation_matrices

def get_correlation_level(value):
    if value > 0.75: return 'High Positive Correlation (p > 0.75)'
    if value > 0.25: return 'Positive Correlation (p>0.25)'
    if value < -0.75: return 'High Negative Correlation (p < -0.75)'
    if value < -0.25: return 'Negative Correlation (p<-0.25)'
    return 'No Meaningful Relationship (-0.25< p <0.25)'


json_file_path = 'results/backup/results_weights.json'

datasets = ['QoS-QoE', 'UNAC', '5G Slicing']
XAI_pairs = ['PI SHAP', 'PI LIME', 'SHAP LIME']
modes = ['TE']

results_file_path = 'results/backup/results_by_dataset_model_xai.csv'

for result_name, result_dataframe in divide_dataframe_regarding_mcc(pd.read_csv(results_file_path)).items():
    template = {
        'PI SHAP': {
            'High Positive Correlation (p > 0.75)': 0,
            'Positive Correlation (p>0.25)': 0,
            'No Meaningful Relationship (-0.25< p <0.25)': 0,
            'Negative Correlation (p<-0.25)': 0,
            'High Negative Correlation (p < -0.75)': 0
        },
        'PI LIME': {
            'High Positive Correlation (p > 0.75)': 0,
            'Positive Correlation (p>0.25)': 0,
            'No Meaningful Relationship (-0.25< p <0.25)': 0,
            'Negative Correlation (p<-0.25)': 0,
            'High Negative Correlation (p < -0.75)': 0
        },
        'SHAP LIME': {
            'High Positive Correlation (p > 0.75)': 0,
            'Positive Correlation (p>0.25)': 0,
            'No Meaningful Relationship (-0.25< p <0.25)': 0,
            'Negative Correlation (p<-0.25)': 0,
            'High Negative Correlation (p < -0.75)': 0
        }
    }

    json_data = {}

    dataset_values = []
    models_values = []
    xai_values = []
    correlation_values = []

    for mode in modes:
        result = generate_correlation_matrices(json_file_path, mode)
        dataset_counts = copy.deepcopy(template)
        for dataset in datasets:
            models_in_dataset = result_dataframe[result_dataframe['Dataset'] == dataset]['Model'].unique().tolist()
            for model in models_in_dataset:
                corr_matrix_as_df = result[dataset][model]

                for col_name in corr_matrix_as_df.columns:
                    for row_name in corr_matrix_as_df.index:
                        cell_value = corr_matrix_as_df.loc[row_name, col_name]
                        if col_name != row_name and f'{col_name} {row_name}' in dataset_counts:
                            dataset_counts[f'{col_name} {row_name}'][get_correlation_level(cell_value)] += 1

                            dataset_values += [dataset]
                            models_values += [model]
                            xai_values += [f'{col_name} {row_name}']
                            correlation_values += [cell_value]

        xs = [datasets.index(v) for v in dataset_values]
        ys = [XAI_pairs.index(v) for v in xai_values]

        zs = correlation_values

        fig = plt.figure(figsize=(16, 6))

        ax = fig.add_subplot(111, projection='3d')

        # surface
        # Create a 2D grid from the 1D arrays x, y
        X, Y = np.meshgrid(np.linspace(min(xs), max(xs), 100), np.linspace(min(ys), max(ys), 100))
        # Use griddata to interpolate the z values on the 2D grid
        Z = griddata((xs, ys), zs, (X, Y), method='linear')
        # Use plot_surface with the 2D grid to create the 3D surface plot
        surf = ax.plot_surface(X, Y, Z, cmap=cm.jet.reversed(), linewidth=0, antialiased=False, vmin=-1, vmax=1)

        ax.set_zlim(-1, 1)

        # Improve layout by adjusting the view
        # ax.view_init(elev=20, azim=30)

        ax.set_position([0.13, 0.1, 0.6, 0.8])
        ax.tick_params(axis='y', which='major', pad=3)
        ax.tick_params(axis='z', which='major', pad=10)
        ax.zaxis.labelpad = 18
        ax.yaxis.labelpad = 10

        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels(datasets)
        ax.set_yticks(range(len(XAI_pairs)))
        ax.set_yticklabels(XAI_pairs)

        ax.set_xlabel('Dataset')
        ax.set_ylabel('XAI Pair')
        ax.set_zlabel('Correlation')


        out_file_name = f"correlations_regarding_datasets_3d_{result_name}"

        plt.savefig(f"plots/png/{out_file_name}.png", format='png', bbox_inches='tight')
        # Save the plot to PDF
        plt.savefig(f"plots/pdf/{out_file_name}.pdf", format='pdf', bbox_inches='tight')
        # plt.show()
