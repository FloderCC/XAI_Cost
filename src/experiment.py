"""File description:

Script to run the experiment.
"""
import json
import os

import keras
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from resources_monitor import monitor_tic, monitor_toc
from src.dnn_models import *
from src.energy_simulator import simulate_energy_consumption
from src.xai_methods import *

# /// Setup begin \\\
global_random_seed = 42
np.random.seed(global_random_seed)
tf.random.set_seed(global_random_seed)
tf.config.set_visible_devices([], 'GPU')
keras.utils.disable_interactive_logging()


# QOE_prediction_ICC2018 dataset dec http://jeremie.leguay.free.fr/files/QoE-prediction-ICC2018.pdf
# UNAC dataset dec https://www.researchgate.net/profile/Madhusanka-Liyanage/publication/372250269_From_Opacity_to_Clarity_Leveraging_XAI_for_Robust_Network_Traffic_Classification/links/64acf0aac41fb852dd67fa41/From-Opacity-to-Clarity-Leveraging-XAI-for-Robust-Network-Traffic-Classification.pdf
# 5G_Slicing dataset dec https://ieeexplore.ieee.org/document/8993066

# format: [name, [features to be removed], output]]
dataset_setup_list = [
    ['QOE_prediction_ICC2018', ['RebufferingRatio', 'AvgVideoBitRate', 'AvgVideoQualityVariation'], 'StallLabel'],
    ['UNAC', ['file'], 'output'],
    ['5G_Slicing', [], 'Slice Type (Output)']
]

models = {
    # Shallow
    'DT': DecisionTreeClassifier(random_state=global_random_seed),# fails with: MorrisSensitivity with IOT-DNL
    'KNN': KNeighborsClassifier(),
    'MLP': MLPClassifier(random_state=global_random_seed),
    'SGD': SGDClassifier(loss="modified_huber", random_state=global_random_seed),

    'GNB': GaussianNB(),  # fails with: MorrisSensitivity with QOE_prediction_ICC2018

    # Voting based/ensemble
    'RF': RandomForestClassifier(random_state=global_random_seed),
    'VC': VotingClassifier(estimators=[('DT', DecisionTreeClassifier(random_state=global_random_seed)), ('KNN', KNeighborsClassifier()), ('SGD', SGDClassifier(loss="modified_huber", random_state=global_random_seed))], voting='soft'),
    'BC': BaggingClassifier(DecisionTreeClassifier(random_state=global_random_seed), random_state=global_random_seed),
    'ABC': AdaBoostClassifier(DecisionTreeClassifier(random_state=global_random_seed), random_state=global_random_seed),

    # DNN
    'DNN1v0': DNNClassifier1v0(),
    'DNN1v1': DNNClassifier1v1(),
    'DNN2v0': DNNClassifier2v0(),
    'DNN2v1': DNNClassifier2v1(),
}

xai_algorithms = {
    'Permutation Importance': permutation_importance_explanation,
    'SHAP': shap_explanation,
    'LIME (ALL)': lime_explanation_all,
}

# \\\ Setup end ///

results_header = ['Dataset', 'Model', 'TR TIME', 'TR CPU%', 'TR Energy (J)', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'MCC', 'TE TIME', 'TE CPU%', 'TE Energy (J)', 'XAI', 'XAI TR TIME', 'XAI TR CPU%', ' XAI TR Energy (J)', 'XAI TE TIME', 'XAI TE CPU%', ' XAI TE Energy (J)']
results = []

# aux variable for saving the features weights
results_weights = {}

# count
explanation_number = 1
explanations_quantity = len(dataset_setup_list) * len(models) * len(xai_algorithms)

for dataset_setup in dataset_setup_list:

    dataset_name = dataset_setup[0]

    # loading the dataset
    dataset_folder = f"./datasets/{dataset_name}"
    df = pd.read_csv(f"{dataset_folder}/{os.listdir(dataset_folder)[0]}")

    print(f"Started execution with dataset {dataset_name} {df.shape}")

    # removing not useful columns
    df = df.drop(columns=dataset_setup[1])
    print(f"Dropping columns {dataset_setup[1]}")

    # replacing infinite values by the maximum allowed value
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if np.any(np.isinf(df[numeric_columns])):
        print("Replacing infinite values by the maximum allowed value")
        df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)

    # replacing missing values by mean
    if df.isnull().any().any():
        print("Replacing missing values by mean")
        df.fillna(df.mean(), inplace=True)

    # encoding all no numerical columns
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for column in df.columns:
        if not df[column].dtype.kind in ['i', 'f']:
            print(f"Encoding column {column}")
            df[column] = le.fit_transform(df[column].astype(str))

    # splitting features & label
    X = df.drop(dataset_setup[2], axis=1)
    y = df[dataset_setup[2]]

    # encoding Y to make it processable with DNN models
    y = pd.get_dummies(y)

    # splitting the dataset in train and test
    x_train, x_test, y_train_encoded, y_test_encoded = train_test_split(X, y, test_size=0.3, random_state=global_random_seed)

    # parsing y_test to a multiclass target
    y_test = y_test_encoded.idxmax(axis=1)

    results_weights[dataset_name] = {}
    for model_name, model in models.items():
        print(f"\nDataset {dataset_name}. Model: {model_name}")

        is_dnn = model_name.startswith("DNN")
        if is_dnn:
            y_train = y_train_encoded
            # building the tf models in case of DNN
            model.build(x_train.shape[1], y_train.shape[1])
        else:
            # parsing y_train to a multiclass target if the model is not DNN
            y_train = y_train_encoded.idxmax(axis=1)

        # training
        print("Training")
        monitor_tic()
        model.fit(x_train, y_train)
        tr_action_cpu_percent, tr_action_elapsed_time = monitor_toc()

        # testing
        print("Testing")
        monitor_tic()
        y_pred = model.predict(x_test)
        inf_action_cpu_percent, inf_action_elapsed_time = monitor_toc()

        if is_dnn:
            y_train = y_train_encoded.idxmax(axis=1)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        mcc = matthews_corrcoef(y_test, y_pred)

        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print(f"MCC: {mcc}")

        results_weights[dataset_name][model_name] = {}
        for xai_name, xai_algorithm in xai_algorithms.items():
            if is_dnn and xai_name == 'Permutation Importance':
                y_train_to_explain = y_train_encoded
                y_test_to_explain = y_test_encoded
                model.set_enabled_codified_predict(False)
            else:
                y_train_to_explain = y_train
                y_test_to_explain = y_test

            # xai
            print(f"\nExplanation  {explanation_number} / {explanations_quantity}")
            print(f"Explaining train dataset with {xai_name}")
            monitor_tic()
            tr_f_relevance = xai_algorithm(x_train, y_train_to_explain, model)
            tr_xai_action_cpu_percent, tr_xai_action_elapsed_time = monitor_toc()

            print(f"Explaining test dataset with {xai_name}")
            monitor_tic()
            te_f_relevance = xai_algorithm(x_test, y_test_to_explain, model)
            te_xai_action_cpu_percent, te_xai_action_elapsed_time = monitor_toc()

            # print(f"F Relevance: {f_relevance}")
            print(f"F Relevance computed")

            # restoring the codified prediction mode
            if is_dnn and xai_name == 'Permutation Importance':
                model.set_enabled_codified_predict(True)

            # saving the results
            results.append([
                dataset_name,
                model_name,
                tr_action_elapsed_time,
                tr_action_cpu_percent,
                simulate_energy_consumption(tr_action_elapsed_time, tr_action_cpu_percent),
                accuracy,
                precision,
                recall,
                f1,
                mcc,
                inf_action_elapsed_time,
                inf_action_cpu_percent,
                simulate_energy_consumption(inf_action_elapsed_time, inf_action_cpu_percent),
                xai_name,
                tr_xai_action_elapsed_time,
                tr_xai_action_cpu_percent,
                simulate_energy_consumption(tr_xai_action_elapsed_time, tr_xai_action_cpu_percent),
                te_xai_action_elapsed_time,
                te_xai_action_cpu_percent,
                simulate_energy_consumption(te_xai_action_elapsed_time, te_xai_action_cpu_percent)
            ])

            results_weights[dataset_name][model_name][xai_name] = {'TR': tr_f_relevance.tolist(), 'TE': te_f_relevance.tolist()}

            # dumping results for a file
            results_df = pd.DataFrame(results, index=None, columns=results_header)

            # Write to csv
            results_df.to_csv(f'results/results_by_dataset_model_xai.csv', index=False)

            # saving the features weights to json file
            with open(f'results/results_weights.json', 'w') as json_file:
                json.dump(results_weights, json_file, indent=4)

            explanation_number += 1

