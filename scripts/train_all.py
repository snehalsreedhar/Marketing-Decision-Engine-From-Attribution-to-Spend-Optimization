import pandas as pd
from src.mmm_model import run_mmm
from src.attribution_model import run_rule_based_attribution
from src.predictive_model import run_predictive_model
from src.predictive_explainability import explain_model_with_shap

def run_all(path="notebooks/outputs/analysed_data.xlsx"):
    data = pd.read_excel(path)

    # --- Marketing Mix Modeling ---
    mmm_model = run_mmm(data)

    # --- Rule-based Attribution ---
    attribution_df = run_rule_based_attribution(
        data, channel_col="channelGrouping", revenue_col="revenue"
    )
    print("\n✅ Rule-based attribution executed successfully.")
    print(attribution_df.head())

    # --- Predictive Modeling ---
    pred_model, metrics, X_test, y_test = run_predictive_model(
        data, target="converted"
    )
    X_sample = X_test.sample(200, random_state=42)
    shap_values = explain_model_with_shap(pred_model, X_sample)

    return {
        "mmm_model": mmm_model,
        "attribution": attribution_df,
        "predictive_model": pred_model,
        "metrics": metrics,
        "shap": shap_values,
    }
