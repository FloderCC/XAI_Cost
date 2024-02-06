import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy

from src.results_analyzers.utils import divide_dataframe_regarding_mcc

# Set default font size
mpl.rcParams['font.size'] = 14
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
modes = ['TE']

results_file_path = 'results/backup/results_by_dataset_model_xai.csv'

for result_name, result_dataframe in divide_dataframe_regarding_mcc(pd.read_csv(results_file_path)).items():
    template = {
        'PI': {
            'SHAP': {
                'High Positive Correlation (p > 0.75)': 0,
                'Positive Correlation (p>0.25)': 0,
                'No Meaningful Relationship (-0.25< p <0.25)': 0,
                'Negative Correlation (p<-0.25)': 0,
                'High Negative Correlation (p < -0.75)': 0
            },
            'LIME': {
                'High Positive Correlation (p > 0.75)': 0,
                'Positive Correlation (p>0.25)': 0,
                'No Meaningful Relationship (-0.25< p <0.25)': 0,
                'Negative Correlation (p<-0.25)': 0,
                'High Negative Correlation (p < -0.75)': 0
            },
        },
        'SHAP': {
            'LIME': {
                'High Positive Correlation (p > 0.75)': 0,
                'Positive Correlation (p>0.25)': 0,
                'No Meaningful Relationship (-0.25< p <0.25)': 0,
                'Negative Correlation (p<-0.25)': 0,
                'High Negative Correlation (p < -0.75)': 0
            },
        }
    }

    json_data = {}

    for mode in modes:
        result = generate_correlation_matrices(json_file_path, mode)
        dataset_counts = copy.deepcopy(template)
        for dataset in datasets:
            models = result_dataframe[result_dataframe['Dataset'] == dataset] ['Model'].unique().tolist()
            for model in models:
                corr_matrix_as_df = result[dataset][model]

                for col_name in corr_matrix_as_df.columns:
                    for row_name in corr_matrix_as_df.index:
                        cell_value = corr_matrix_as_df.loc[row_name, col_name]
                        if col_name != row_name and row_name in dataset_counts and col_name in dataset_counts[row_name]:
                            dataset_counts[row_name][col_name][get_correlation_level(cell_value)] += 1

        json_data[mode] = dataset_counts

        # plot_results(dataset_counts)

    with open(f'plots/weights_correlation_regarding_xai_{result_name}.json', 'w') as json_file:
        json.dump(json_data, json_file, indent=2)
