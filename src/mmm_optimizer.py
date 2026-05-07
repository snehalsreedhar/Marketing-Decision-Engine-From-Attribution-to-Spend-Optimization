import numpy as np
import pandas as pd
from scipy.optimize import minimize

def optimize_budget(model, data, spend_vars, total_budget):
    """
    Optimize budget allocation across channels using MMM coefficients.

    Args:
        model: fitted OLS MMM model
        data: dataframe used for modeling
        spend_vars: list of columns representing marketing channel spend or proxy
        total_budget: numeric value (total spend constraint)

    Returns:
        optimization result (dict) with optimal spends and predicted revenue
    """

    # Extract coefficients for relevant variables
    coefs = model.params[spend_vars].to_numpy()
    baseline = model.params["const"]

    # Initial guess = average current spend
    x0 = data[spend_vars].mean().to_numpy()
    x0 = x0 / x0.sum() * total_budget

    # Bounds: each channel gets at least 0, at most 3x its current level
    bounds = [(0, 3 * val) for val in data[spend_vars].mean()]

    # Objective = negative predicted revenue (we minimize)
    def objective(x):
        return -(baseline + np.dot(coefs, x))

    # Constraint: total spend fixed
    cons = ({'type': 'eq', 'fun': lambda x: x.sum() - total_budget})

    result = minimize(objective, x0, bounds=bounds, constraints=cons, method="SLSQP")

    optimal_spends = pd.DataFrame({
        "Channel": spend_vars,
        "Optimal_Spend": result.x,
        "Current_Avg_Spend": data[spend_vars].mean().values
    })
    optimal_spends["Allocation_%"] = 100 * optimal_spends["Optimal_Spend"] / total_budget

    optimal_revenue = -result.fun

    print("✅ Optimization complete.")
    return {
        "optimal_spends": optimal_spends.sort_values(by="Allocation_%", ascending=False),
        "predicted_revenue": optimal_revenue
    }
