import shap
import pandas as pd
import matplotlib.pyplot as plt

def explain_model_with_shap(model, X_sample, max_display=15):
    """
    Generates SHAP explainability plots for a trained pipeline model (tree or logistic).
    Works with Random Forest, XGBoost, or other supported estimators.
    
    Parameters:
        model: trained sklearn Pipeline with 'model' step
        X_sample: sample of features (DataFrame)
        max_display: number of top features to display in global plots
    """
    estimator = model.named_steps["model"]

    # Create SHAP explainer (TreeExplainer or LinearExplainer)
    if hasattr(estimator, "predict_proba") or hasattr(estimator, "feature_importances_"):
        explainer = shap.Explainer(estimator, X_sample)
    else:
        explainer = shap.LinearExplainer(estimator, X_sample)

    shap_values = explainer(X_sample)

    # Global importance summary
    shap.summary_plot(shap_values, X_sample, max_display=max_display)
    plt.show()

    # Average magnitude (importance ranking)
    shap.plots.bar(shap_values, max_display=max_display)
    plt.show()

    return shap_values


def explain_single_prediction(model, X_sample, index=0):
    """
    Displays SHAP values for a single observation (local interpretability).
    """
    estimator = model.named_steps["model"]
    explainer = shap.Explainer(estimator, X_sample)
    shap_values = explainer(X_sample)

    # Local (force) explanation
    shap.plots.waterfall(shap_values[index])
