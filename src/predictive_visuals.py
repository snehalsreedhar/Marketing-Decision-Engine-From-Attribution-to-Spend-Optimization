import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.inspection import PartialDependenceDisplay
import numpy as np

def plot_feature_importances(model, feature_names, top_n=15, title="Feature Importances"):
    """
    Plot feature importances for models that support the attribute `feature_importances_`.
    Works with tree-based models (Random Forest, XGBoost, etc.).
    """
    if not hasattr(model.named_steps["model"], "feature_importances_"):
        print("⚠️ This model does not provide feature importances.")
        return

    importances = model.named_steps["model"].feature_importances_
    feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    feat_df = feat_df.sort_values(by="Importance", ascending=False).head(top_n)

    plt.figure(figsize=(10, 5))
    sns.barplot(y="Feature", x="Importance", data=feat_df, palette="Blues_d")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_roc_curve(model, X_test, y_test):
    """
    Plot ROC curve for a binary classification model.
    """
    # Ensure binary classes
    unique_classes = sorted(y_test.unique())
    if len(unique_classes) != 2:
        print("⚠️ ROC curve only available for binary classification.")
        return

    # Get probabilities if available
    clf = model.named_steps["model"]
    if hasattr(clf, "predict_proba"):
        y_score = clf.predict_proba(X_test)[:, 1]
    elif hasattr(clf, "decision_function"):
        y_score = clf.decision_function(X_test)
    else:
        print("⚠️ Model doesn't provide decision scores or probabilities.")
        return

    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="Classifier").plot()
    plt.title(f"ROC Curve (AUC = {roc_auc:.3f})")
    plt.show()


def plot_precision_recall(model, X_test, y_test):
    """
    Plot Precision–Recall curve for binary classification problems.
    """
    clf = model.named_steps["model"]
    # only works for binary classification
    if len(np.unique(y_test)) != 2:
        print("⚠️ Precision–Recall curve available only for binary classification.")
        return
    if hasattr(clf, "predict_proba"):
        y_score = clf.predict_proba(X_test)[:, 1]
    elif hasattr(clf, "decision_function"):
        y_score = clf.decision_function(X_test)
    else:
        print("⚠️ Model lacks probabilistic scoring capability.")
        return
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    avg_precision = average_precision_score(y_test, y_score)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color="blue", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall Curve (AP = {avg_precision:.3f})")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def plot_partial_dependence(model, X_train, features, feature_names=None, grid_resolution=20):
    """
    Generate Partial Dependence Plots to show how features influence model predictions.
    Works best with tree-based models.
    """
    est = model.named_steps["model"]
    if not hasattr(est, "feature_importances_") and not hasattr(est, "tree_"):
        print("⚠️ Partial dependence plots best suited for tree-based models (e.g., RandomForest, XGBoost).")
        return
    PartialDependenceDisplay.from_estimator(
        est, X_train, features, feature_names=feature_names, grid_resolution=grid_resolution
    )
    plt.tight_layout()
    plt.show()