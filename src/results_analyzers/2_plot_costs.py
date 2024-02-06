import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

from src.results_analyzers.utils import divide_dataframe_regarding_mcc

# Set default font size
mpl.rcParams['font.size'] = 14

results_file_path = 'results/backup/results_by_dataset_model_xai.csv'

df = pd.read_csv(results_file_path)

mode = 'TE'

# renaming values
df['XAI'] = df['XAI'].replace('LIME (ALL)', 'LIME')
df['XAI'] = df['XAI'].replace('Permutation Importance', 'PI')
df.rename(columns={f'XAI {mode} TIME': 'Execution time (s)'}, inplace=True)
df.rename(columns={f'XAI {mode} Energy (J)': 'Energy (J)'}, inplace=True)
df.rename(columns={'Model': 'ML model'}, inplace=True)
df.rename(columns={'XAI': 'XAI technique'}, inplace=True)


datasets = ['QoS-QoE', 'UNAC', '5G Slicing']
costs = ['Energy (J)', 'Execution time (s)']





max_costs = {}
for dataset in datasets:
    if dataset not in max_costs:
        max_costs[dataset] = {}
    for cost in costs:
        df_d = df[df['Dataset'] == dataset]
        max_costs[dataset][cost] = max(df_d[cost])


dataset_divisions = divide_dataframe_regarding_mcc(df)
for result_name, result_dataframe in dataset_divisions.items():
    for dataset in datasets:
        for cost in costs:
            df_d = result_dataframe[result_dataframe['Dataset'] == dataset]

            plt.figure(figsize=(6.5, 4.5))
            sns.barplot(x=cost, y='ML model', hue='XAI technique', data=df_d, orient='h', palette='deep')  # ci=None disables error bars

            plt.gca().set_xlim(0, max_costs[dataset][cost])

            out_file_name = f"cost_bars_{result_name}_{'time' if cost == 'Execution time (s)' else 'energy'}_{dataset}"
            plt.savefig(f"plots/png/{out_file_name}.png", format='png', bbox_inches='tight')
            # Save the plot to PDF
            plt.savefig(f"plots/pdf/{out_file_name}.pdf", format='pdf', bbox_inches='tight')
            # plt.show()

