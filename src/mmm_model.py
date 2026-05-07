import pandas as pd
import numpy as np
import statsmodels.api as sm
from itertools import product

def adstock_transform(x, decay):
    """ Apply adstock transformation. """
    result = np.zeros(len(x))
    for i in range(len(x)):
        result[i] = x.iloc[i] + decay * result[i - 1] if i > 0 else x.iloc[i]
    return result

def saturation_transform(x, alpha=1.0, beta=0.5):
    """ Apply saturation transformation (S-curve). """
    return alpha * (x ** beta)

def tune_adstock_saturation(data, var, y, decay_grid, beta_grid):
    """ 
    Tunes decay and beta parameters by minimizing OLS residual sum of squares (RSS).
    Returns best decay, beta, and transformed series.
    """
    best_rss, best_decay, best_beta = np.inf, None, None
    best_transformed = None

def tune_adstock_saturation(data, var, y, decay_grid, beta_grid):
    """
    Tunes decay and beta parameters by minimizing OLS residual sum of squares.
    Returns best decay, beta, and transformed series.
    """
    from itertools import product
    # Clean input
    if var not in data.columns:
        raise ValueError(f"{var} not found in data.")
    x = data[var].replace([np.inf, -np.inf], np.nan).fillna(0)
    best_rss = np.inf
    best_decay = None
    best_beta = None
    best_transformed = np.zeros_like(x)
    # Loop over candidate grid
    for decay, beta in product(decay_grid, beta_grid):
        try:
            ad = adstock_transform(x, decay)
            sat = saturation_transform(ad, beta=beta)
            # Replace invalid values before OLS
            sat = pd.Series(sat).replace([np.inf, -np.inf], np.nan).fillna(0)
            X = sm.add_constant(sat)
            model = sm.OLS(y, X).fit()
            rss = ((model.resid) ** 2).sum()
            if rss < best_rss:
                best_rss = rss
                best_decay = decay
                best_beta = beta
                best_transformed = sat
        except Exception as e:
            # skip invalid parameter combinations silently
            continue
    if best_decay is None or best_beta is None:
        raise RuntimeError(f"Failed to tune parameters for {var}; input may be all zeros or NaNs.")
    print(f"🔧 {var}: best decay={best_decay:.2f}, beta={best_beta:.2f} (RSS={best_rss:.2f})")
    return best_transformed, best_decay, best_beta

def run_mmm(data: pd.DataFrame, dependent_var: str = "log_revenue"):
    """
    Advanced MMM with automatic adstock/saturation tuning per driver.
    Applies log1p transform to revenue when modeling log_revenue.
    """

    # --- Handle missing or zero revenue safely ---
    if dependent_var == "log_revenue":
        if "revenue" in data.columns:
            data["log_revenue"] = np.log1p(data["revenue"])
            print("Applied log1p transform to revenue for stable MMM fit.")
        else:
            raise ValueError(
                "Expected column 'revenue' when using dependent_var='log_revenue'."
            )

    if dependent_var not in data.columns:
        raise ValueError(f"Dependent variable '{dependent_var}' not found in dataset.")

    numeric_features = [
        "totals_hits",
        "totals_pageviews",
        "totals_newVisits",
        "hour",
        "is_peak",
        "is_weekend",
        "is_returning_visitor",
    ]

    categorical_features = [
        "channelGrouping",
        "socialEng",
        "device_br",
        "device_category",
    ]

    numeric_features = [col for col in numeric_features if col in data.columns]
    categorical_features = [col for col in categorical_features if col in data.columns]

    df = data.dropna(subset=[dependent_var]).copy()
    y = df[dependent_var]

    # --- Tune adstock/saturation only for key numeric marketing metrics ---
    transformed_cols = []
    for col in ["totals_hits", "totals_pageviews", "totals_newVisits"]:
        if col in df.columns:
            df[f"{col}_trans"], decay, beta = tune_adstock_saturation(
                df,
                col,
                y,
                decay_grid=np.linspace(0.1, 0.9, 9),
                beta_grid=np.linspace(0.3, 1.0, 8),
            )
            transformed_cols.append(f"{col}_trans")

    # --- Build final feature set ---
    X = df[numeric_features + transformed_cols + categorical_features]
    X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

    # --- Add intercept ---
    X = sm.add_constant(X)

    # --- Force all numeric types, convert booleans to int ---
    X = X.copy()
    bool_cols = X.select_dtypes(include=["bool"]).columns
    X[bool_cols] = X[bool_cols].astype(int)

    # Convert entire frame to numeric
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    y = pd.to_numeric(y, errors="coerce").fillna(0)

    # Drop any invalid rows
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X.loc[mask]
    y = y.loc[mask]

    # Cast everything to float for Statsmodels
    X = X.astype(float)
    y = y.astype(float)

    print("Cleaned dtypes:\n", X.dtypes.value_counts())

    # --- Fit OLS ---
    model = sm.OLS(y, X).fit()

    print(f"\n✅ Tuned MMM trained successfully with {X.shape[1] - 1} predictors.\n")
    print(model.summary())

    return model
