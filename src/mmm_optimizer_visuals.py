import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_optimal_allocation(optimal_spends):
    """Visualize optimal vs. current marketing spend per channel."""
    df = optimal_spends.copy()
    df = df.melt(
        id_vars=["Channel"], 
        value_vars=["Optimal_Spend", "Current_Avg_Spend"], 
        var_name="Type", 
        value_name="Spend"
    )

    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x="Spend", y="Channel", hue="Type", palette="Set2")
    plt.title("Optimal vs. Current Marketing Spend Allocation")
    plt.xlabel("Spend (units)")
    plt.ylabel("Channel")
    plt.tight_layout()
    plt.show()


def plot_allocation_percentages(optimal_spends):
    """Pie chart of budget allocation by channel."""
    plt.figure(figsize=(6, 6))
    plt.pie(
        optimal_spends["Allocation_%"], 
        labels=optimal_spends["Channel"], 
        autopct="%1.1f%%", 
        startangle=140, 
        colors=sns.color_palette("coolwarm", len(optimal_spends))
    )
    plt.title("Optimal Marketing Budget Split (%)")
    plt.show()


def show_uplift_comparison(model, data, spend_vars, optimized_values):
    """
    Estimate and visualize the uplift in predicted revenue
    between current average spend and optimal spend.
    """
    coefs = model.params[spend_vars].to_numpy()
    baseline = model.params["const"]
    current_spends = data[spend_vars].mean().to_numpy()

    current_rev = baseline + (coefs @ current_spends)
    optimal_rev = baseline + (coefs @ optimized_values)
    uplift = optimal_rev - current_rev
    uplift_pct = (uplift / current_rev) * 100

    plt.figure(figsize=(5, 5))
    bars = plt.bar(["Current", "Optimized"], [current_rev, optimal_rev], color=["#ffb347", "#77dd77"])
    plt.title(f"Predicted Revenue Uplift = {uplift_pct:.2f}%")
    plt.ylabel("Predicted log_revenue")
    plt.bar_label(bars, fmt="%.2f")
    plt.tight_layout()
    plt.show()

    print(f"📈 Uplift: {uplift:.4f} ({uplift_pct:.2f}% increase from current)")
