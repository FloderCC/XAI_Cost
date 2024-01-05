import numpy as np
import pandas as pd
from shap import KernelExplainer, kmeans
from alibi.explainers import PartialDependenceVariance
from sklearn.inspection import permutation_importance
from interpret.blackbox import MorrisSensitivity
from sklearn.model_selection import train_test_split
import lime
import lime.lime_tabular

def shap_explanation(X, y, model):
    background_data = kmeans(X, 10)
    explainer = KernelExplainer(model.predict, background_data)
    shap_values = explainer.shap_values(X, nsamples=100)
    return np.mean(np.abs(shap_values), axis=0)


def pd_variance_explanation(X, y, model):

    # alibi only expects numpy arrays
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    explainer = PartialDependenceVariance(model)
    explanation = explainer.explain(X, method='importance')

    return explanation.data['feature_importance'][0]


def permutation_importance_explanation(X, y, model):
    feature_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    return feature_importance.importances_mean


def morris_sensitivity_explanation(X, y, model):
    explainer = MorrisSensitivity(model.predict, X)
    explanation = explainer.explain_global()
    explanation_results = explanation._internal_obj['specific']
    return np.array([r["mu_star"] for r in explanation_results])


# tmp solution
import warnings
warnings.filterwarnings("ignore")

def lime_explanation(X, y, model):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X.values,
        feature_names=X.columns,
        mode='classification'
    )

    # stratified sampling - 20% of the data
    _, X_, _, y_ = train_test_split_regression(X, y, test_size=0.2, b='auto', random_state=42)

    X_ = X_.values

    results = []
    for i in range(len(X_)):
        exp = explainer.explain_instance(
            data_row=X_[i],
            predict_fn=model.predict,
            num_features=X_.shape[1]
        )
        importances = exp.as_map()[1]
        importances = sorted(importances, key=lambda x: x[0])  # sort by feature index
        importances = [x[1] for x in importances]
        results.append(importances)

    return np.mean(np.abs(results), axis=0)


def train_test_split_regression(X, y, test_size=0.2, b='auto', random_state=42):
    if isinstance(b, str):
        bins = np.histogram_bin_edges(y, bins=b)
        bins = bins[:-1]
    elif isinstance(b, int):
        bins = np.linspace(min(y), max(y), num=b, endpoint=False)
    else:
        raise Exception(f'Undefined bins {b}')

    groups = np.digitize(y, bins)
    return train_test_split(X, y, test_size=test_size, stratify=groups, random_state=random_state)


def lime_explanation_all(X, y, model):
    X_ = X.values

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_,
        feature_names=X.columns,
        mode='classification'
    )

    results = []
    for i in range(len(X_)):
        exp = explainer.explain_instance(
            data_row=X_[i],
            predict_fn=model.predict,
            num_features=X_.shape[1]
        )
        importances = exp.as_map()[1]
        importances = sorted(importances, key=lambda x: x[0])  # sort by feature index
        importances = [x[1] for x in importances]
        results.append(importances)

    return np.mean(np.abs(results), axis=0)