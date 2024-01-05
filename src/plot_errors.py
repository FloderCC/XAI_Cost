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
df['Dataset'] = df['Dataset'].replace('QOE_prediction_ICC2018', 'QoS-QoE')

# renaming columns
df.rename(columns={'XAI-TIME': 'Execution time (s)'}, inplace=True)
df.rename(columns={'Model': 'ML model'}, inplace=True)
df.rename(columns={'XAI': 'XAI technique'}, inplace=True)
df.rename(columns={'R2': 'R^2'}, inplace=True)

errors = ['MSE','MAE','R^2','MAPE']

for error in errors:
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Dataset', y=error, hue='ML model', data=df, ci=None)  # ci=None disables error bars
    # plt.title(f'{error}')

    plt.legend(fontsize=13)
    # plt.show()
    # Save the plot to PNG
    plt.savefig(f"plots/png/error_{error}.png", format='png', bbox_inches='tight')
    # Save the plot to PDF
    plt.savefig(f"plots/pdf/error_{error}.pdf", format='pdf', bbox_inches='tight')
    # plt.show()