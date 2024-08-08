import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

from src.results_analyzers.utils import divide_dataframe_regarding_mcc

# Set default font size
mpl.rcParams['font.size'] = 14

results_file_path = 'results/results_by_dataset_model_xai.csv'

df = pd.read_csv(results_file_path)

# renaming values
df['XAI'] = df['XAI'].replace('LIME (ALL)', 'LIME')
df['XAI'] = df['XAI'].replace('Permutation Importance', 'PI')
df.rename(columns={f'XAI TE Energy (J)': 'XAI Energy (J)'}, inplace=True)
df.rename(columns={f'TE Energy (J)': 'Model Energy (J)'}, inplace=True)
df.rename(columns={'Model': 'ML model'}, inplace=True)
df.rename(columns={'XAI': 'XAI technique'}, inplace=True)

datasets = order = ['KPI-KQI', 'UNAC', 'NSR', 'QoS-QoE', '5G Slicing']
for result_name, result_dataframe in divide_dataframe_regarding_mcc(df).items():
    for dataset in datasets:
        df_d = result_dataframe[result_dataframe['Dataset'] == dataset]

        data = {
            'Model': [],
            'Model Energy (J)': [],
            'PI Energy (J)': [],
            'SHAP Energy (J)': [],
            'LIME Energy (J)': [],
        }

        for index, row in df_d.iterrows():
            model_name = row['ML model']
            model_energy = row['Model Energy (J)']
            xai_name = row['XAI technique']
            xai_energy = row['XAI Energy (J)']

            if model_name not in data['Model']:
                data['Model'] += [model_name]
                data['Model Energy (J)'] += [model_energy]

            xai_key_name = f'{xai_name} Energy (J)'

            data[f'{xai_name} Energy (J)'] += [xai_energy]

        df_to_plot = pd.DataFrame(data)

        plt.figure(figsize=(10, 6))

        sns.lineplot(x='Model', y='Model Energy (J)', data=df_to_plot, marker='o', label='Model')
        sns.lineplot(x='Model', y='PI Energy (J)', data=df_to_plot, marker='o', label='PI')
        sns.lineplot(x='Model', y='SHAP Energy (J)', data=df_to_plot, marker='o', label='SHAP')
        sns.lineplot(x='Model', y='LIME Energy (J)', data=df_to_plot, marker='o', label='LIME')

        # Set plot labels and title
        plt.xlabel('Models')
        plt.ylabel('Energy (J)')

        # Show legend
        plt.legend()

        # Show the plot
        plt.tight_layout()

        out_file_name = f"cost_relation_lines_{result_name}_energy_{dataset}"

        plt.savefig(f"plots/png/{out_file_name}.png", format='png', bbox_inches='tight')
        # Save the plot to PDF
        plt.savefig(f"plots/pdf/{out_file_name}.pdf", format='pdf', bbox_inches='tight')
        # plt.show()


