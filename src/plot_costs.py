import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set default font size
mpl.rcParams['font.size'] = 14

results_file_path = 'results/results_by_dataset_model_xai.csv'

df = pd.read_csv(results_file_path)

# removing sampled LIME
df = df[df['XAI'] != 'LIME']
df['XAI'] = df['XAI'].replace('LIME (ALL)', 'LIME')

# renaming values
df['XAI'] = df['XAI'].replace('LIME (ALL)', 'LIME')
df['XAI'] = df['XAI'].replace('Permutation Importance', 'PI')
df['XAI'] = df['XAI'].replace('MorrisSensitivity', 'MS')

# renaming columns
df.rename(columns={'XAI-TIME': 'Execution time (s)'}, inplace=True)
df.rename(columns={'Model': 'ML model'}, inplace=True)
df.rename(columns={'XAI': 'XAI technique'}, inplace=True)


datasets = ['QOE_prediction_ICC2018', 'OtIPMB']
costs = ['Execution time (s)', 'Energy (J)']

for dataset in datasets:
    for cost in costs:
        df_d = df[df['Dataset'] == dataset]

        plt.figure(figsize=(10, 6))

        sns.barplot(x='XAI technique', y=cost, hue='ML model', data=df_d)  # ci=None disables error bars
        # plt.title(f'{cost} ({dataset})')
        # plt.show()
        # Save the plot to PNG
        out_file_name = f"cost_{'time' if cost == 'Execution time (s)' else 'energy'}_{dataset}"

        plt.savefig(f"plots/png/{out_file_name}.png", format='png', bbox_inches='tight')
        # Save the plot to PDF
        plt.savefig(f"plots/pdf/{out_file_name}.pdf", format='pdf', bbox_inches='tight')
        # plt.show()
