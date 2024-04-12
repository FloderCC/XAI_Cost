import copy
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
import seaborn as sns

from src.results_analyzers.utils import divide_dataframe_regarding_mcc




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

    models = result_dataframe['Model'].unique().tolist()

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
                            xai_values += [f'{col_name} {row_name}'.replace(' ', ' - ')]
                            correlation_values += [cell_value]

        # xs = [models.index(v) for v in models_values]
        # ys = [XAI_pairs.index(v) for v in xai_values]
        # zs = correlation_values


        # create a dataframe with xs, ys, and zs
        df_p = pd.DataFrame({'Model': models_values, 'XAI pair': xai_values, 'Correlation': correlation_values})

        # Create a new plot
        fig = plt.figure(figsize=(8, 5.6))
        # Set default font size
        mpl.rcParams['font.size'] = 16

        plt.style.use('tableau-colorblind10')
        plt.rcParams['figure.autolayout'] = True
        plt.rcParams['figure.figsize'] = (4, 4)
        plt.rcParams['lines.linewidth'] = 2

        colormap = ['#377eb8', '#ff7f00', '#4daf4a']
        markers = ['o', '*', '.', 'x', '+', 's', 'd', 'h', 'v']
        lines = ['-', ':', '--', '-.']

        # rotate x-axis labels
        plt.xticks(rotation=30)

        #define e new color palette
        # custom_palette = ["#859ce0", "#8585e0", "#9c85e0"]

        barplot = sns.barplot(x='Model', y='Correlation', hue='XAI pair', errorbar='sd', data=df_p, capsize=.20)  # , palette=colormap

        # for p in barplot.patches:
        #     barplot.annotate(format(p.get_height(), '.3f'),
        #                      (p.get_x() + p.get_width() / 2., p.get_height()),
        #                      ha='center', va='center',
        #                      xytext=(0, 0),
        #                      textcoords='offset points')
        # aa_std = df_p.groupby(['Model', 'XAI pair'])['Correlation'].std().reset_index()
        # aa_avg = df_p.groupby(['Model', 'XAI pair'])['Correlation'].mean().reset_index()
        # #order by correlation
        # aa_avg = aa_avg.sort_values(by='Correlation', ascending=False)
        #
        # print(result_name)
        # print('mean:')
        # print(aa_avg)
        # print('std')
        # print(aa_std)
        #
        # print('Mean of the means')
        # print(aa_avg['Correlation'].mean())


        # plt.gca().set_ylim(-2, 2)
        plt.yticks(np.arange(-1.2, 1.3, 0.4))

        # set legend position by coordinates
        plt.legend(loc='lower right', bbox_to_anchor=(0.85, 0))

        # set Y axis label
        plt.ylabel('PCC')

        # add horizontal line and label at y 0.3
        plt.axhline(y=0.3, linestyle='--', color='#545353', linewidth=1.5)
        plt.text(9.3 if result_name == 'lp' else 11.6, 0.22, 'Weak\nPositive', fontsize=14, ha='center')
        plt.axhline(y=-0.3, linestyle='--', color='#545353', linewidth=1.5)
        plt.text(9.3 if result_name == 'lp' else 11.6, - 0.38, 'Weak\nNegative', fontsize=14, ha='center')

        # # add horizontal line and label at y 0.8
        # plt.axhline(y=0.8, linestyle='--', color='#545353', linewidth=1.5)
        # plt.text(9.1, 0.70, 'Moderate\nPositive', fontsize=10, ha='center')
        # plt.axhline(y=-0.8, linestyle='--', color='#545353', linewidth=1.5)
        # plt.text(9.1, - 0.86, 'Moderate\nNegative', fontsize=10, ha='center')
        #
        # # add horizontal line and label at y 1.0
        # plt.axhline(y=1.0, linestyle='--', color='#545353', linewidth=1.5)
        # plt.text(9.1, 0.94, 'Strong\nPositive', fontsize=10, ha='center')
        # plt.axhline(y=-1.0, linestyle='--', color='#545353', linewidth=1.5)
        # plt.text(9.1, - 1.1, 'Strong\nNegative', fontsize=10, ha='center')



        out_file_name = f"correlations_regarding_model_3d_{result_name}"

        plt.savefig(f"plots/png/{out_file_name}.png", format='png', bbox_inches='tight')  # , transparent=True
        # Save the plot to PDF
        plt.savefig(f"plots/pdf/{out_file_name}.pdf", format='pdf', bbox_inches='tight')
        # plt.show()
