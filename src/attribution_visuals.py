import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def compare_attribution_models(rule_based_df, markov_df):
    """
    Compare channel credit shares between Rule-Based and Markov models.
    Both inputs must include columns ['channel', 'share'] (or ['channelGrouping', 'share_%']).
    """

    # Normalize column names for consistency
    rb = rule_based_df.copy()
    rb.columns = [c.lower().replace("%", "").strip() for c in rb.columns]
    mk = markov_df.copy()
    mk.columns = [c.lower().replace("%", "").strip() for c in mk.columns]

    # Rename for merge clarity
    rb = rb.rename(columns={"channelgrouping": "channel", "share": "rule_share"})
    mk = mk.rename(columns={"channelgrouping": "channel", "share": "markov_share"})

    merged = pd.merge(rb, mk, on="channel", how="outer").fillna(0)

    # Barplot comparison
    plt.figure(figsize=(10, 6))
    merged = merged.melt(id_vars="channel", value_vars=["rule_share", "markov_share"], 
                         var_name="Model", value_name="Attribution_Share")
    sns.barplot(data=merged, x="Attribution_Share", y="channel", hue="Model")
    plt.title("Attribution Comparison: Rule-Based vs Markov Model")
    plt.xlabel("Attribution Share (%)")
    plt.ylabel("Channel")
    plt.legend(title="Model Type")
    plt.tight_layout()
    plt.show()


def plot_single_attribution(attribution_df, model_name="Markov Attribution"):
    """
    Simple horizontal bar chart of one attribution model's results.
    """
    df = attribution_df.copy()
    cols = [c.lower() for c in df.columns]
    df.columns = cols

    share_col = "share" if "share" in cols else "share_"
    chan_col = "channel" if "channel" in cols else "channelgrouping"

    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x=share_col, y=chan_col, palette="viridis")
    plt.title(f"{model_name} — Channel Credit Distribution")
    plt.xlabel("Credit Share (%)")
    plt.ylabel("Channel")
    plt.tight_layout()
    plt.show()
