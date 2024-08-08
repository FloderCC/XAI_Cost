import numpy as np
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
# renaming values
df['XAI'] = df['XAI'].replace('LIME (ALL)', 'LIME')
df['XAI'] = df['XAI'].replace('Permutation Importance', 'PI')

# setting order
order = ['KPI-KQI', 'UNAC', 'NSR', 'QoS-QoE', '5G Slicing']
df['Dataset'] = pd.Categorical(df['Dataset'], categories=order, ordered=True)

order = ['DT', 'KNN', 'MLP', 'SGD', 'GNB', 'RF', 'VC', 'BC', 'ABC', 'DNN1v0', 'DNN1v1', 'DNN2v0', 'DNN2v1']
df['ML model'] = pd.Categorical(df['ML model'], order, ordered=True)

df = df.sort_values(by=['Dataset', 'ML model'])


df['Ratio'] = df['XAI TE Energy (J)'] / df['TE Energy (J)']

df2 = df[['Dataset', 'ML model', 'XAI', 'Ratio']]

plt.figure(figsize=(8, 5))

# adding a regression line.
dataset_dimensions = {
    'KPI-KQI': 165 * 13,
    'UNAC': 389 * 22,
    'NSR': 31583 * 16,
    'QoS-QoE': 69129 * 50,
    '5G Slicing': 466739 * 8
}
# X will bte the dataset size (dataset_dimensions in values)
df_reg = pd.DataFrame()
df_reg['Dataset'] = df2['Dataset']
df_reg['X'] = df2['Dataset'].map(lambda x: dataset_dimensions[x] if x in dataset_dimensions else np.nan)
df_reg['Ratio'] = df2['Ratio']

# Normalize the X values to be between 0 and 5
df_reg['X'] = pd.to_numeric(df_reg['X'], errors='coerce')
df_reg['X_normalized'] = (df_reg['X'] / df_reg['X'].max()) * 5










# categories = list(dataset_dimensions.keys())
# category_positions = np.arange(len(categories))
# df2['X_normalized'] = df2['Dataset'].map(lambda x: category_positions[categories.index(x)] if x in categories else np.nan)

sns.regplot(x='X_normalized', y='Ratio', data=df_reg, scatter=False, color='black')

# # Plot the perfect regression line (slope=1, intercept=0)
# x_vals = np.linspace(df2['X_normalized'].min(), df2['X_normalized'].max(), 100)
# y_vals = x_vals  # Since slope=1 and intercept=0
# plt.plot(x_vals, y_vals, color='red', linestyle='--', label='Perfect Regression Line')


# adding the bars
# Define markers and colors for the ML models
markers = {'PI': 'o', 'SHAP': 'd', 'LIME': 'X'}
palette = sns.color_palette('tab20', n_colors=len(df['ML model'].unique()))

# Create a list to store handles for the custom marker legend
handles_marker = []
unique_ml_models = df['ML model'].unique()

# Plot each XAI group separately
for xai, marker in markers.items():
    sns.stripplot(x='Dataset', y='Ratio', hue='ML model', data=df[df['XAI'] == xai],
                  palette=palette, jitter=False, dodge=True, size=8, marker=marker, label=xai)

    # Create a custom legend handle for the marker legend
    handle_marker = plt.Line2D([0], [0], marker=marker, color='w', markerfacecolor='gray', markersize=10, label=xai)
    handles_marker.append(handle_marker)

# Create handles for ML model legend
handles_ml = []
for model in unique_ml_models:
    handle_ml = plt.Line2D([0], [0], color=palette[unique_ml_models.tolist().index(model)], lw=4, label=model)
    handles_ml.append(handle_ml)

# Remove the default legend
plt.legend([],[], frameon=False)

# Create the ML model legend
legend_ml = plt.legend(handles=handles_ml, title='Model', loc='upper left', bbox_to_anchor=(1, 1))

# Create the custom marker legend
legend_marker = plt.legend(handles=handles_marker, title='XAI', loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(markers))
# Add the legends to the plot
plt.gca().add_artist(legend_ml)
plt.gca().add_artist(legend_marker)


# plt.xticks(category_positions, categories)
plt.xlabel('Dataset')

# Set y-axis to use scientific notation
# formatter = ScalarFormatter()
# formatter.set_scientific(True)
# formatter.set_powerlimits((-3, 3))
# plt.gca().yaxis.set_major_formatter(formatter)

plt.yscale('log')

# plt.show()
# Save the plot to PNG
plt.savefig(f"plots/png/ratios.png", format='png', bbox_inches='tight')
# Save the plot to PDF
plt.savefig(f"plots/pdf/ratios.pdf", format='pdf', bbox_inches='tight')
# plt.show()