import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# 1️⃣ Rule-based Attribution (direct or proportional)
# ---------------------------------------------------------------------
def run_rule_based_attribution(data, channel_col="channelGrouping", revenue_col="revenue"):
    """
    Simple baseline attribution — assigns 100% credit to the last touchpoint
    or splits revenue equally among all channels in a path.
    """
    grouped = (
        data.groupby(channel_col)[revenue_col]
        .sum()
        .reset_index()
        .sort_values(by=revenue_col, ascending=False)
    )
    grouped["share_%"] = 100 * grouped[revenue_col] / grouped[revenue_col].sum()
    return grouped