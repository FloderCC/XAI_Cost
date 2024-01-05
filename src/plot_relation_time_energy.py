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
    df_d = df[df['Dataset'] == dataset]
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Execution time (s)', y='Energy (J)', data=df_d, hue='XAI technique', palette='viridis')
    plt.title('Energy Consumption vs XAI-TIME')
    plt.xlabel('XAI-TIME')
    plt.ylabel('Energy (J)')
    plt.legend(title='XAI Technique', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    plt.show()

correlation = df['Execution time (s)'].corr(df['Energy (J)'])

print(f"Correlation between Execution time and Energy: {correlation}")

