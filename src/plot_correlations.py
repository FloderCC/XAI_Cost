import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set default font size
mpl.rcParams['font.size'] = 14
def generate_correlation_matrices(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    correlation_matrices = {}

    for first_key, second_level in data.items():
        correlation_matrices[first_key] = {}
        for second_key, third_level in second_level.items():
            # Convert the third level values into a DataFrame
            df = pd.DataFrame(third_level)

            # removing sampled LIME
            df = df.drop('LIME', axis=1)


            # renaming columns
            df.rename(columns={'LIME (ALL)': 'LIME'}, inplace=True)
            df.rename(columns={'MorrisSensitivity': 'MS'}, inplace=True)
            df.rename(columns={'Permutation Importance': 'PI'}, inplace=True)

            # Calculate the correlation matrix
            correlation_matrix = df.corr()

            # Store the correlation matrix in the result dictionary
            correlation_matrices[first_key][second_key] = correlation_matrix

    return correlation_matrices


json_file_path = 'results/results_weights.json'
result = generate_correlation_matrices(json_file_path)

# Access correlation matrices
# corr_matrix_as_df = result["QOE_prediction_ICC2018"]["RF"]
# print(corr_matrix_as_df)

datasets = ['QOE_prediction_ICC2018', 'OtIPMB']
models = ['LR', 'RF', 'KNN', 'MLP']

for dataset in datasets:
    for model in models:
        corr_matrix_as_df = result[dataset][model]
        # Plot the correlation matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix_as_df, annot=True, cmap='crest', fmt=".5f", linewidths=.5, vmin=-1, vmax=1, center=0)
        plt.tight_layout()
        # plt.title(f'Correlation matrix ({dataset}, {model})')

        # Save the plot to PNG
        plt.savefig(f"plots/png/correlation_{dataset}_{model}.png", format='png', bbox_inches='tight')
        # Save the plot to PDF
        plt.savefig(f"plots/pdf/correlation_{dataset}_{model}.pdf", format='pdf', bbox_inches='tight')
        # plt.show()
