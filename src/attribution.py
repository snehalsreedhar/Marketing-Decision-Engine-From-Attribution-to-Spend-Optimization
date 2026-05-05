def last_click_attribution(df):
    # Sort by user and time
    df = df.sort_values(['fullVisitorId', 'visitStartTime'])

    # Keep last interaction per user
    last_touch = df.groupby('fullVisitorId').tail(1)

    # Aggregate revenue by channel
    attribution = last_touch.groupby('channelGrouping')['revenue'].sum().reset_index()

    attribution = attribution.sort_values(by='revenue', ascending=False)

    return attribution