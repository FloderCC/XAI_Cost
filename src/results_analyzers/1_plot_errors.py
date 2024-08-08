import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set default font size
mpl.rcParams['font.size'] = 16

results_file_path = 'results/results_by_dataset_model_xai.csv'

df = pd.read_csv(results_file_path)

# renaming values
df['XAI'] = df['XAI'].replace('LIME (ALL)', 'LIME')
df['XAI'] = df['XAI'].replace('Permutation Importance', 'PI')

# renaming columns
df.rename(columns={'Model': 'ML model'}, inplace=True)
df.rename(columns={'XAI': 'XAI technique'}, inplace=True)

# setting order
order = ['KPI-KQI', 'UNAC', 'NSR', 'QoS-QoE', '5G Slicing']
df['Dataset'] = pd.Categorical(df['Dataset'], categories=order, ordered=True)

order = ['DT', 'KNN', 'MLP', 'SGD', 'GNB', 'RF', 'VC', 'BC', 'ABC', 'DNN1v0', 'DNN1v1', 'DNN2v0', 'DNN2v1']
df['ML model'] = pd.Categorical(df['ML model'], order, ordered=True)

df = df.sort_values(by=['Dataset', 'ML model'])


errors = ['MCC']

for error in errors:
    plt.figure(figsize=(8, 6))

    ax = sns.barplot(x='Dataset', y=error, hue='ML model', data=df, errorbar=None, palette='tab20')  # ci=None disables error bars

    # add horizontal line 0.8 MCC as a cutoff (dashed)
    plt.axhline(y=0.8, color='#545353', linewidth=1.0, linestyle='--')

    # Create the legend and center it
    ax.legend(fontsize=16, loc='upper right', bbox_to_anchor=(1.30, 1))

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


    # plt.show()
    # Save the plot to PNG
    plt.savefig(f"plots/png/error_{error}.png", format='png', bbox_inches='tight')
    # Save the plot to PDF
    plt.savefig(f"plots/pdf/error_{error}.pdf", format='pdf', bbox_inches='tight')
    # plt.show()