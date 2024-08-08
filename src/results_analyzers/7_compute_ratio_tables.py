import pandas as pd

from src.results_analyzers.utils import divide_dataframe_regarding_mcc

results_file_path = 'results/results_by_dataset_model_xai.csv'

df = pd.read_csv(results_file_path)

mode = 'TE'

# renaming values
df['XAI'] = df['XAI'].replace('LIME (ALL)', 'LIME')
df['XAI'] = df['XAI'].replace('Permutation Importance', 'PI')
df.rename(columns={f'{mode} Energy (J)': 'Inference Energy (J)'}, inplace=True)
df.rename(columns={f'XAI {mode} Energy (J)': 'XAI Energy (J)'}, inplace=True)
df.rename(columns={'Model': 'ML model'}, inplace=True)
df.rename(columns={'XAI': 'XAI technique'}, inplace=True)
df.rename(columns={'TE Energy (J)': 'Energy (J)'}, inplace=True)


datasets = order = ['KPI-KQI', 'UNAC', 'NSR', 'QoS-QoE', '5G Slicing']
costs = ['Energy (J)']

df['Ratio'] = df['XAI Energy (J)'] / df['Inference Energy (J)']

for dataset in datasets:
    for cost in costs:
        df_d = df[df['Dataset'] == dataset]

        df_d = df_d[['ML model', 'Inference Energy (J)', 'XAI technique', 'XAI Energy (J)', 'Ratio']]

        print(f"\nDataset: {dataset}\n")

        print(df_d.to_markdown(index=False, floatfmt=".2f"))
