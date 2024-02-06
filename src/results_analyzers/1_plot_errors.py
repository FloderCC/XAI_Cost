import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set default font size
mpl.rcParams['font.size'] = 14

results_file_path = 'results/backup/results_by_dataset_model_xai.csv'

df = pd.read_csv(results_file_path)

# renaming values
df['XAI'] = df['XAI'].replace('LIME (ALL)', 'LIME')
df['XAI'] = df['XAI'].replace('Permutation Importance', 'PI')

# renaming columns
df.rename(columns={'Model': 'ML model'}, inplace=True)
df.rename(columns={'XAI': 'XAI technique'}, inplace=True)

errors = ['MCC']

for error in errors:
    plt.figure(figsize=(12, 4.8))

    custom_palette = ["#bad4be", "#8abd92", "#67bf77", "#54a161", "#417d4b",  "#d49d9d", "#cc8181", "#c25555", "#a84848",  "#93bddb", "#65a7d6", "#5588ad", "#35698f"]

    ax = sns.barplot(x='Dataset', y=error, hue='ML model', data=df, ci=None, palette=custom_palette)  # ci=None disables error bars

    # Create a custom legend and center it
    ax.legend(fontsize=12, loc='upper right', bbox_to_anchor=(1.15, 1))

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


    # plt.show()
    # Save the plot to PNG
    plt.savefig(f"plots/png/error_{error}.png", format='png', bbox_inches='tight')
    # Save the plot to PDF
    plt.savefig(f"plots/pdf/error_{error}.pdf", format='pdf', bbox_inches='tight')
    # plt.show()