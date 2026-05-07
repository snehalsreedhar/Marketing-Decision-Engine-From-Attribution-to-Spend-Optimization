import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def align_feature_influence(shap_values, X_sample, mmm_model, markov_df):
    """
    Combine SHAP feature importance, MMM coefficients, and Markov attribution shares.

    Parameters:
        shap_values: SHAP result object from explain_model_with_shap()
        X_sample: DataFrame used for SHAP calculation
        mmm_model: fitted statsmodels OLS model from run_mmm()
        markov_df: Markov attribution output from run_markov_attribution()

    Returns:
        combined DataFrame ranking features that are consistently impactful.
    """

    # --- Step 1. Aggregate SHAP importance ---
    shap_importance = pd.DataFrame({
        "Feature": X_sample.columns,
        "SHAP_Importance": shap_values.abs.mean(0).values
    }).sort_values(by="SHAP_Importance", ascending=False)

    # --- Step 2. Extract MMM coefficients ---
    mmm_coeffs = mmm_model.params.reset_index()
    mmm_coeffs.columns = ["Feature", "MMM_Coefficient"]

    # --- Step 3. Add Attribution (Markov) shares ---
    markov = markov_df.rename(columns={"channel": "Feature", "share": "Markov_Share"})

    # --- Step 4. Merge all sources ---
    merged = shap_importance.merge(mmm_coeffs, on="Feature", how="outer")
    merged = merged.merge(markov, on="Feature", how="outer")

    # Normalize scales for comparability
    for col in ["SHAP_Importance", "MMM_Coefficient", "Markov_Share"]:
        if col in merged:
            merged[col] = (merged[col] - merged[col].min()) / (merged[col].max() - merged[col].min())

    merged["Composite_Score"] = merged[["SHAP_Importance", "MMM_Coefficient", "Markov_Share"]].mean(axis=1)
    merged = merged.sort_values(by="Composite_Score", ascending=False).fillna(0)
    return merged


def plot_cross_model_alignment(merged_df, top_n=15):
    """
    Visualizes the top aligned drivers across MMM, SHAP, and Attribution models.
    """
    top_df = merged_df.head(top_n)
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        top_df.set_index("Feature")[["SHAP_Importance", "MMM_Coefficient", "Markov_Share"]],
        annot=True, cmap="YlGnBu", fmt=".2f"
    )
    plt.title("Cross-Model Feature Alignment (Normalized)")
    plt.xlabel("Model Source")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()
