import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_actual_vs_predicted(model, X, y, title="Actual vs Predicted Revenue"):
    """Compare actual vs predicted dependent variable."""
    y_pred = model.predict(X)
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y, y=y_pred, alpha=0.6)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.grid(True)
    plt.show()
    return y_pred

def plot_coefficients(model, X, top_n=15):
    """Bar chart of top features from MMM coefficients."""
    coefs = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.params
    }).sort_values(by="Coefficient", ascending=False)

    plt.figure(figsize=(10, 5))
    sns.barplot(
        y="Feature",
        x="Coefficient",
        data=coefs.head(top_n),
        palette="viridis"
    )
    plt.title(f"Top {top_n} Positive Coefficients")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    sns.barplot(
        y="Feature",
        x="Coefficient",
        data=coefs.tail(top_n),
        palette="rocket"
    )
    plt.title(f"Top {top_n} Negative Coefficients")
    plt.tight_layout()
    plt.show()

def plot_channel_elasticities(model, X, base_col_substrings=None):
    """
    Estimates and plots channel elasticities relative to average spend/effort.
    base_col_substrings: list of feature name fragments that define a group (e.g. ["totals_hits", "totals_pageviews"])
    """
    if base_col_substrings is None:
        base_col_substrings = ["totals_hits", "totals_pageviews", "totals_newVisits"]

    coefs = model.params
    elasticity = {}
    for col in X.columns:
        for base in base_col_substrings:
            if base in col:
                elasticity[col] = coefs[col] * X[col].mean()

    if not elasticity:
        print("No matching channel features found for elasticity plot.")
        return

    elas_df = pd.DataFrame({
        "Feature": elasticity.keys(),
        "Elasticity": elasticity.values()
    }).sort_values(by="Elasticity", ascending=False)

    plt.figure(figsize=(10, 5))
    sns.barplot(x="Elasticity", y="Feature", data=elas_df, palette="coolwarm")
    plt.title("Channel Elasticities (Estimated Impact on log_revenue)")
    plt.tight_layout()
    plt.show()
